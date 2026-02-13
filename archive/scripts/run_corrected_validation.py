#!/usr/bin/env python3
"""
Corrected Validation Script for P3 Re-Baseline
================================================

Runs spatial model validation with CORRECTED LS parameters:
- T1_artery = 1650.0 ms (ASL consensus, Alsop 2015) -- was 1850.0
- alpha_BS1 = 1.0 (for synthetic data with no background suppression)

Also adds:
- Multi-SNR sweep: [2, 3, 5, 8, 10, 15, 20, 25]
- Multiple smoothed-LS sigma values: [0.5, 1.0, 2.0, 3.0]
- Tissue-stratified metrics (GM, WM, pathology)
- Statistical significance testing (Wilcoxon)
- Bootstrap CIs on all metrics

Usage:
    # Validate Exp 00 (Baseline SpatialASL)
    python run_corrected_validation.py \
        --run_dir amplitude_ablation_v1/00_Baseline_SpatialASL \
        --output_dir corrected_validation/exp00

    # Validate Exp 14 (ATT_Rebalanced)
    python run_corrected_validation.py \
        --run_dir amplitude_ablation_v2/14_ATT_Rebalanced \
        --output_dir corrected_validation/exp14

    # Quick test (fewer phantoms, single SNR)
    python run_corrected_validation.py \
        --run_dir amplitude_ablation_v1/00_Baseline_SpatialASL \
        --output_dir corrected_validation/exp00_quick \
        --quick
"""

import sys
import os
import json
import logging
import argparse
import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================
# CORRECTED LS PARAMETERS
# ============================================================
# These override whatever the training config says.
# For synthetic data, alpha_BS1=1.0 because phantoms have no BS.
# T1_artery corrected from 1850 -> 1650 per ASL consensus (Alsop 2015).
CORRECTED_LS_PARAMS = {
    'T1_artery': 1650.0,    # CORRECTED from 1850.0
    'T_tau': 1800.0,
    'alpha_PCASL': 0.85,
    'alpha_VSASL': 0.56,
    'T2_factor': 1.0,
    'alpha_BS1': 1.0,        # Synthetic data: no background suppression
}


def load_models(run_dir: Path, device: torch.device):
    """Load trained ensemble models from a run directory."""
    import yaml
    from spatial_asl_network import SpatialASLNet
    from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet

    # Load configs
    config_yaml = run_dir / 'config.yaml'
    research_config = run_dir / 'research_config.json'

    training_config = {}
    if config_yaml.exists():
        with open(config_yaml) as f:
            full_config = yaml.safe_load(f)
            training_config = full_config.get('training', {})

    config = {}
    if research_config.exists():
        with open(research_config) as f:
            config = json.load(f)

    # Load norm_stats
    norm_stats_path = run_dir / 'norm_stats.json'
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    # Model parameters
    model_class_name = training_config.get('model_class_name',
                                            config.get('model_class_name', 'SpatialASLNet'))
    hidden_sizes = training_config.get('hidden_sizes',
                                       config.get('hidden_sizes', [32, 64, 128, 256]))
    pld_values = config.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000])
    normalization_mode = config.get('normalization_mode', 'global_scale')
    global_scale_factor = config.get('global_scale_factor', 10.0)

    # Find model files
    models_dir = run_dir / 'trained_models'
    model_files = sorted(list(models_dir.glob('ensemble_model_*.pt')))
    if not model_files:
        raise FileNotFoundError(f"No model files in {models_dir}")

    logger.info(f"Loading {len(model_files)} {model_class_name} models...")

    models = []
    for mp in model_files:
        if model_class_name == 'AmplitudeAwareSpatialASLNet':
            model = AmplitudeAwareSpatialASLNet(
                n_plds=len(pld_values),
                features=hidden_sizes,
                use_film_at_bottleneck=training_config.get('use_film_at_bottleneck', True),
                use_film_at_decoder=training_config.get('use_film_at_decoder', True),
                use_amplitude_output_modulation=training_config.get('use_amplitude_output_modulation', True),
            )
        else:
            model = SpatialASLNet(n_plds=len(pld_values), features=hidden_sizes)

        state_dict = torch.load(mp, map_location=device, weights_only=False)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
        logger.info(f"  Loaded {mp.name}")

    return models, norm_stats, pld_values, normalization_mode, global_scale_factor


def run_nn_inference(models, input_tensor, norm_stats, device):
    """Run ensemble NN inference and denormalize outputs."""
    with torch.no_grad():
        cbf_maps, att_maps = [], []
        for model in models:
            cbf_pred, att_pred, _, _ = model(input_tensor)
            cbf_denorm = cbf_pred * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
            att_denorm = att_pred * norm_stats['y_std_att'] + norm_stats['y_mean_att']
            cbf_denorm = torch.clamp(cbf_denorm, min=0.0, max=200.0)
            att_denorm = torch.clamp(att_denorm, min=0.0, max=5000.0)
            cbf_maps.append(cbf_denorm.cpu().numpy())
            att_maps.append(att_denorm.cpu().numpy())

    nn_cbf = np.mean(cbf_maps, axis=0)[0, 0]  # (H, W)
    nn_att = np.mean(att_maps, axis=0)[0, 0]
    return nn_cbf, nn_att


def run_ls_fitting(noisy_signals, plds, ls_params, sample_indices, smoothed_signals=None):
    """
    Run LS fitting on sampled voxels.

    Returns (ls_cbf, ls_att) arrays for the sample_indices.
    If smoothed_signals is provided, also returns (sls_cbf, sls_att).
    """
    from utils import get_grid_search_initial_guess
    from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

    pldti = np.column_stack([plds, plds])
    n_plds = len(plds)

    ls_cbf_list, ls_att_list = [], []
    sls_cbf_list, sls_att_list = [], []

    for idx in sample_indices:
        i, j = idx

        # Raw LS
        voxel_signal = noisy_signals[:, i, j]
        try:
            init_guess = get_grid_search_initial_guess(voxel_signal, plds, ls_params)
            signal_reshaped = voxel_signal.reshape((n_plds, 2), order='F')
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                pldti, signal_reshaped, init_guess, **ls_params
            )
            ls_cbf_list.append(beta[0] * 6000.0)
            ls_att_list.append(beta[1])
        except Exception:
            ls_cbf_list.append(np.nan)
            ls_att_list.append(np.nan)

        # Smoothed LS
        if smoothed_signals is not None:
            voxel_signal_s = smoothed_signals[:, i, j]
            try:
                init_guess = get_grid_search_initial_guess(voxel_signal_s, plds, ls_params)
                signal_reshaped = voxel_signal_s.reshape((n_plds, 2), order='F')
                beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                    pldti, signal_reshaped, init_guess, **ls_params
                )
                sls_cbf_list.append(beta[0] * 6000.0)
                sls_att_list.append(beta[1])
            except Exception:
                sls_cbf_list.append(np.nan)
                sls_att_list.append(np.nan)

    result = {
        'ls_cbf': np.array(ls_cbf_list),
        'ls_att': np.array(ls_att_list),
    }
    if smoothed_signals is not None:
        result['sls_cbf'] = np.array(sls_cbf_list)
        result['sls_att'] = np.array(sls_att_list)
    return result


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap CI for the mean."""
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(42)
    n = len(data)
    boot_means = np.array([np.mean(data[rng.randint(0, n, size=n)]) for _ in range(n_bootstrap)])
    alpha = 1.0 - ci
    return (float(np.mean(data)),
            float(np.percentile(boot_means, 100 * alpha / 2)),
            float(np.percentile(boot_means, 100 * (1 - alpha / 2))))


def bootstrap_ci_winrate(nn_errors, ls_errors, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap CI for win rate."""
    valid = ~(np.isnan(nn_errors) | np.isnan(ls_errors))
    nn_err = nn_errors[valid]
    ls_err = ls_errors[valid]
    if len(nn_err) == 0:
        return (np.nan, np.nan, np.nan)
    nn_wins = (nn_err < ls_err).astype(float)
    rng = np.random.RandomState(42)
    n = len(nn_wins)
    boot_rates = np.array([np.mean(nn_wins[rng.randint(0, n, size=n)]) for _ in range(n_bootstrap)])
    alpha = 1.0 - ci
    return (float(np.mean(nn_wins)),
            float(np.percentile(boot_rates, 100 * alpha / 2)),
            float(np.percentile(boot_rates, 100 * (1 - alpha / 2))))


def wilcoxon_test(nn_errors, ls_errors):
    """Run Wilcoxon signed-rank test on paired errors."""
    from scipy.stats import wilcoxon
    valid = ~(np.isnan(nn_errors) | np.isnan(ls_errors))
    nn_err = nn_errors[valid]
    ls_err = ls_errors[valid]
    if len(nn_err) < 10:
        return {'p_value': np.nan, 'statistic': np.nan, 'n': len(nn_err)}
    diff = ls_err - nn_err
    nonzero = diff[diff != 0]
    if len(nonzero) < 10:
        return {'p_value': np.nan, 'statistic': np.nan, 'n': len(nn_err)}
    stat, p = wilcoxon(nn_err, ls_err, alternative='two-sided')
    # Effect size (Cohen's d)
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    return {'p_value': float(p), 'statistic': float(stat), 'effect_size': float(d), 'n': len(nn_err)}


def classify_tissue(cbf_map, att_map, tissue_map, pathologies):
    """
    Classify each voxel into tissue categories for stratified metrics.

    Returns dict mapping tissue_name -> boolean mask (H, W).

    Categories:
    - gray_matter: tissue_map == 1, no pathology overlap
    - white_matter: tissue_map == 2, no pathology overlap
    - csf: tissue_map == 3
    - pathology: voxels within any pathology region
    - boundary: voxels within 2 pixels of a tissue boundary
    """
    from scipy.ndimage import binary_dilation

    h, w = cbf_map.shape
    masks = {}

    # Pathology mask from metadata
    pathology_mask = np.zeros((h, w), dtype=bool)
    for p in pathologies:
        cy, cx = p['center']
        radius = p['radius']
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        pathology_mask |= (dist <= radius * 1.2)  # Slightly expanded to catch rim

    # Boundary mask: where adjacent pixels have different tissue types
    boundary_mask = np.zeros((h, w), dtype=bool)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(tissue_map, di, axis=0), dj, axis=1)
        boundary_mask |= (tissue_map != shifted)
    # Dilate boundary by 1 pixel
    boundary_mask = binary_dilation(boundary_mask, iterations=1)

    # Brain mask (non-background)
    brain_mask = tissue_map > 0

    masks['gray_matter'] = (tissue_map == 1) & ~pathology_mask & ~boundary_mask
    masks['white_matter'] = (tissue_map == 2) & ~pathology_mask & ~boundary_mask
    masks['pathology'] = pathology_mask & brain_mask
    masks['boundary'] = boundary_mask & brain_mask & ~pathology_mask
    masks['all_brain'] = brain_mask

    return masks


def run_single_snr(snr_value, n_phantoms, phantom_size, models, norm_stats,
                   pld_values, normalization_mode, global_scale_factor,
                   ls_params, smooth_sigmas, device):
    """
    Run validation at a single SNR level.

    Returns a dict with all collected voxel-level data for metric computation.
    """
    from enhanced_simulation import RealisticASLSimulator, SpatialPhantomGenerator
    from asl_simulation import ASLParameters

    params = ASLParameters(
        T1_artery=ls_params['T1_artery'],
        T_tau=ls_params['T_tau'],
        alpha_PCASL=ls_params['alpha_PCASL'],
        alpha_VSASL=ls_params['alpha_VSASL'],
    )
    simulator = RealisticASLSimulator(params=params)
    phantom_gen = SpatialPhantomGenerator(size=phantom_size, pve_sigma=1.0)
    plds = np.array(pld_values, dtype=np.float64)
    n_plds = len(plds)

    # Collectors
    collectors = {
        'nn_cbf': [], 'nn_att': [],
        'true_cbf': [], 'true_att': [],
        'tissue_labels': [],  # per-voxel tissue category
    }
    # LS collectors (at sampled voxels only)
    ls_collectors = {
        'ls_cbf': [], 'ls_att': [],
        'ls_true_cbf': [], 'ls_true_att': [],
        'nn_at_ls_cbf': [], 'nn_at_ls_att': [],
    }
    # Smoothed-LS collectors per sigma
    sls_collectors = {}
    for sigma in smooth_sigmas:
        sls_collectors[sigma] = {
            'sls_cbf': [], 'sls_att': [],
            'sls_true_cbf': [], 'sls_true_att': [],
            'nn_at_sls_cbf': [], 'nn_at_sls_att': [],
        }

    for phantom_idx in tqdm(range(n_phantoms), desc=f"SNR={snr_value}"):
        np.random.seed(phantom_idx)
        true_cbf_map, true_att_map, metadata = phantom_gen.generate_phantom(include_pathology=True)
        tissue_map = metadata['tissue_map']
        pathologies = metadata.get('pathologies', [])

        mask = (true_cbf_map > 1.0).astype(np.float32)
        brain_mask_bool = mask > 0

        # Generate clean signals
        signals = np.zeros((n_plds * 2, phantom_size, phantom_size), dtype=np.float32)
        for i in range(phantom_size):
            for j in range(phantom_size):
                if mask[i, j] > 0:
                    cbf, att = true_cbf_map[i, j], true_att_map[i, j]
                    pcasl = simulator._generate_pcasl_signal(
                        plds, att, cbf, params.T1_artery, params.T_tau, params.alpha_PCASL)
                    vsasl = simulator._generate_vsasl_signal(
                        plds, att, cbf, params.T1_artery, params.alpha_VSASL)
                    signals[:n_plds, i, j] = pcasl
                    signals[n_plds:, i, j] = vsasl

        # Add noise
        ref_signal = simulator._compute_reference_signal()
        noise_sd = ref_signal / snr_value
        noise_rng = np.random.RandomState(phantom_idx * 1000 + int(snr_value * 10))
        noise = noise_sd * noise_rng.randn(*signals.shape).astype(np.float32)
        noisy_signals = signals + noise

        # Normalize for NN
        noisy_signals_scaled = noisy_signals * 100.0
        if normalization_mode == 'global_scale':
            noisy_signals_normalized = noisy_signals_scaled * global_scale_factor
        else:
            temporal_mean = np.mean(noisy_signals_scaled, axis=0, keepdims=True)
            temporal_std = np.std(noisy_signals_scaled, axis=0, keepdims=True) + 1e-6
            noisy_signals_normalized = (noisy_signals_scaled - temporal_mean) / temporal_std

        # NN inference
        input_tensor = torch.from_numpy(
            noisy_signals_normalized[np.newaxis, ...]).float().to(device)
        nn_cbf, nn_att = run_nn_inference(models, input_tensor, norm_stats, device)

        # Classify tissue for each brain voxel
        tissue_masks = classify_tissue(true_cbf_map, true_att_map, tissue_map, pathologies)

        # Collect NN results for all brain voxels with tissue labels
        for i in range(phantom_size):
            for j in range(phantom_size):
                if brain_mask_bool[i, j]:
                    collectors['nn_cbf'].append(nn_cbf[i, j])
                    collectors['nn_att'].append(nn_att[i, j])
                    collectors['true_cbf'].append(true_cbf_map[i, j])
                    collectors['true_att'].append(true_att_map[i, j])
                    # Assign tissue label
                    if tissue_masks['pathology'][i, j]:
                        collectors['tissue_labels'].append('pathology')
                    elif tissue_masks['boundary'][i, j]:
                        collectors['tissue_labels'].append('boundary')
                    elif tissue_masks['gray_matter'][i, j]:
                        collectors['tissue_labels'].append('gray_matter')
                    elif tissue_masks['white_matter'][i, j]:
                        collectors['tissue_labels'].append('white_matter')
                    else:
                        collectors['tissue_labels'].append('other')

        # LS fitting on subsampled voxels
        brain_indices = np.argwhere(mask > 0)
        sample_indices = brain_indices[::10]

        # Prepare smoothed signals for each sigma
        smoothed_by_sigma = {}
        for sigma in smooth_sigmas:
            smoothed = np.zeros_like(noisy_signals)
            for ch in range(noisy_signals.shape[0]):
                smoothed[ch] = gaussian_filter(noisy_signals[ch], sigma=sigma)
            smoothed_by_sigma[sigma] = smoothed

        # Run LS on raw signals
        ls_result = run_ls_fitting(noisy_signals, plds, ls_params, sample_indices)

        for k, idx in enumerate(sample_indices):
            i, j = idx
            if not np.isnan(ls_result['ls_cbf'][k]):
                ls_collectors['ls_cbf'].append(ls_result['ls_cbf'][k])
                ls_collectors['ls_att'].append(ls_result['ls_att'][k])
                ls_collectors['ls_true_cbf'].append(true_cbf_map[i, j])
                ls_collectors['ls_true_att'].append(true_att_map[i, j])
                ls_collectors['nn_at_ls_cbf'].append(nn_cbf[i, j])
                ls_collectors['nn_at_ls_att'].append(nn_att[i, j])

        # Run smoothed-LS for each sigma
        for sigma in smooth_sigmas:
            sls_result = run_ls_fitting(
                noisy_signals, plds, ls_params, sample_indices,
                smoothed_signals=smoothed_by_sigma[sigma])

            for k, idx in enumerate(sample_indices):
                i, j = idx
                if not np.isnan(sls_result.get('sls_cbf', [np.nan])[k] if k < len(sls_result.get('sls_cbf', [])) else np.nan):
                    sls_collectors[sigma]['sls_cbf'].append(sls_result['sls_cbf'][k])
                    sls_collectors[sigma]['sls_att'].append(sls_result['sls_att'][k])
                    sls_collectors[sigma]['sls_true_cbf'].append(true_cbf_map[i, j])
                    sls_collectors[sigma]['sls_true_att'].append(true_att_map[i, j])
                    sls_collectors[sigma]['nn_at_sls_cbf'].append(nn_cbf[i, j])
                    sls_collectors[sigma]['nn_at_sls_att'].append(nn_att[i, j])

    # Convert to arrays
    for key in ['nn_cbf', 'nn_att', 'true_cbf', 'true_att']:
        collectors[key] = np.array(collectors[key])
    for key in ls_collectors:
        ls_collectors[key] = np.array(ls_collectors[key])
    for sigma in smooth_sigmas:
        for key in sls_collectors[sigma]:
            sls_collectors[sigma][key] = np.array(sls_collectors[sigma][key])

    return collectors, ls_collectors, sls_collectors


def compute_metrics(collectors, ls_collectors, sls_collectors, smooth_sigmas, snr_value):
    """Compute all metrics for a single SNR level."""
    results = {'snr': snr_value}

    nn_cbf = collectors['nn_cbf']
    nn_att = collectors['nn_att']
    true_cbf = collectors['true_cbf']
    true_att = collectors['true_att']
    tissue_labels = collectors['tissue_labels']

    # NN metrics (full brain)
    nn_cbf_err = np.abs(nn_cbf - true_cbf)
    nn_att_err = np.abs(nn_att - true_att)

    cbf_mae, cbf_lo, cbf_hi = bootstrap_ci(nn_cbf_err)
    att_mae, att_lo, att_hi = bootstrap_ci(nn_att_err)

    results['nn'] = {
        'cbf_mae': cbf_mae, 'cbf_mae_ci': [cbf_lo, cbf_hi],
        'cbf_bias': float(np.mean(nn_cbf - true_cbf)),
        'att_mae': att_mae, 'att_mae_ci': [att_lo, att_hi],
        'att_bias': float(np.mean(nn_att - true_att)),
        'n_voxels': len(nn_cbf),
    }

    # LS metrics (at sampled voxels)
    if len(ls_collectors['ls_cbf']) > 0:
        ls_cbf_err = np.abs(ls_collectors['ls_cbf'] - ls_collectors['ls_true_cbf'])
        ls_att_err = np.abs(ls_collectors['ls_att'] - ls_collectors['ls_true_att'])
        nn_at_ls_cbf_err = np.abs(ls_collectors['nn_at_ls_cbf'] - ls_collectors['ls_true_cbf'])
        nn_at_ls_att_err = np.abs(ls_collectors['nn_at_ls_att'] - ls_collectors['ls_true_att'])

        ls_cbf_mae, ls_cbf_lo, ls_cbf_hi = bootstrap_ci(ls_cbf_err)
        ls_att_mae, ls_att_lo, ls_att_hi = bootstrap_ci(ls_att_err)

        cbf_wr, cbf_wr_lo, cbf_wr_hi = bootstrap_ci_winrate(nn_at_ls_cbf_err, ls_cbf_err)
        att_wr, att_wr_lo, att_wr_hi = bootstrap_ci_winrate(nn_at_ls_att_err, ls_att_err)

        cbf_wilcoxon = wilcoxon_test(nn_at_ls_cbf_err, ls_cbf_err)
        att_wilcoxon = wilcoxon_test(nn_at_ls_att_err, ls_att_err)

        results['ls'] = {
            'cbf_mae': ls_cbf_mae, 'cbf_mae_ci': [ls_cbf_lo, ls_cbf_hi],
            'cbf_bias': float(np.mean(ls_collectors['ls_cbf'] - ls_collectors['ls_true_cbf'])),
            'att_mae': ls_att_mae, 'att_mae_ci': [ls_att_lo, ls_att_hi],
            'att_bias': float(np.mean(ls_collectors['ls_att'] - ls_collectors['ls_true_att'])),
            'n_voxels': len(ls_collectors['ls_cbf']),
        }
        results['comparison'] = {
            'cbf_win_rate': cbf_wr, 'cbf_win_rate_ci': [cbf_wr_lo, cbf_wr_hi],
            'att_win_rate': att_wr, 'att_win_rate_ci': [att_wr_lo, att_wr_hi],
            'cbf_wilcoxon': cbf_wilcoxon,
            'att_wilcoxon': att_wilcoxon,
        }

    # Smoothed-LS metrics per sigma
    results['smoothed_ls'] = {}
    for sigma in smooth_sigmas:
        sc = sls_collectors[sigma]
        if len(sc['sls_cbf']) > 0:
            sls_cbf_err = np.abs(sc['sls_cbf'] - sc['sls_true_cbf'])
            sls_att_err = np.abs(sc['sls_att'] - sc['sls_true_att'])
            nn_at_sls_cbf_err = np.abs(sc['nn_at_sls_cbf'] - sc['sls_true_cbf'])
            nn_at_sls_att_err = np.abs(sc['nn_at_sls_att'] - sc['sls_true_att'])

            sls_cbf_mae, sls_cbf_lo, sls_cbf_hi = bootstrap_ci(sls_cbf_err)
            sls_att_mae, sls_att_lo, sls_att_hi = bootstrap_ci(sls_att_err)
            cbf_wr, cbf_wr_lo, cbf_wr_hi = bootstrap_ci_winrate(nn_at_sls_cbf_err, sls_cbf_err)
            att_wr, att_wr_lo, att_wr_hi = bootstrap_ci_winrate(nn_at_sls_att_err, sls_att_err)

            results['smoothed_ls'][f'sigma_{sigma}'] = {
                'cbf_mae': sls_cbf_mae, 'cbf_mae_ci': [sls_cbf_lo, sls_cbf_hi],
                'att_mae': sls_att_mae, 'att_mae_ci': [sls_att_lo, sls_att_hi],
                'cbf_win_rate': cbf_wr, 'cbf_win_rate_ci': [cbf_wr_lo, cbf_wr_hi],
                'att_win_rate': att_wr, 'att_win_rate_ci': [att_wr_lo, att_wr_hi],
                'n_voxels': len(sc['sls_cbf']),
            }

    # Tissue-stratified metrics
    results['tissue_stratified'] = {}
    tissue_labels_arr = np.array(tissue_labels)
    for tissue_name in ['gray_matter', 'white_matter', 'pathology', 'boundary']:
        tissue_mask = tissue_labels_arr == tissue_name
        n_tissue = np.sum(tissue_mask)
        if n_tissue < 10:
            continue

        t_nn_cbf_err = np.abs(nn_cbf[tissue_mask] - true_cbf[tissue_mask])
        t_nn_att_err = np.abs(nn_att[tissue_mask] - true_att[tissue_mask])

        t_cbf_mae, t_cbf_lo, t_cbf_hi = bootstrap_ci(t_nn_cbf_err)
        t_att_mae, t_att_lo, t_att_hi = bootstrap_ci(t_nn_att_err)

        results['tissue_stratified'][tissue_name] = {
            'nn_cbf_mae': t_cbf_mae, 'nn_cbf_mae_ci': [t_cbf_lo, t_cbf_hi],
            'nn_cbf_bias': float(np.mean(nn_cbf[tissue_mask] - true_cbf[tissue_mask])),
            'nn_att_mae': t_att_mae, 'nn_att_mae_ci': [t_att_lo, t_att_hi],
            'nn_att_bias': float(np.mean(nn_att[tissue_mask] - true_att[tissue_mask])),
            'n_voxels': int(n_tissue),
            'mean_true_cbf': float(np.mean(true_cbf[tissue_mask])),
            'mean_true_att': float(np.mean(true_att[tissue_mask])),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Corrected validation with multi-SNR and tissue stratification")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Quick mode: SNR=10 only, 10 phantoms")
    parser.add_argument("--phantom_size", type=int, default=64)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load models
    models, norm_stats, pld_values, normalization_mode, global_scale_factor = \
        load_models(run_dir, device)

    # Configuration
    if args.quick:
        snr_values = [10]
        n_phantoms = {10: 10}
        smooth_sigmas = [2.0]
    else:
        snr_values = [2, 3, 5, 8, 10, 15, 20, 25]
        n_phantoms = {s: 20 for s in snr_values}
        n_phantoms[10] = 50  # More phantoms at the primary SNR
        smooth_sigmas = [0.5, 1.0, 2.0, 3.0]

    logger.info(f"SNR values: {snr_values}")
    logger.info(f"Smoothed-LS sigmas: {smooth_sigmas}")
    logger.info(f"LS params (CORRECTED): {CORRECTED_LS_PARAMS}")

    # Run validation at each SNR
    all_results = {}
    start_time = time.time()

    for snr in snr_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running SNR={snr} with {n_phantoms[snr]} phantoms...")
        logger.info(f"{'='*60}")

        collectors, ls_collectors, sls_collectors = run_single_snr(
            snr_value=snr,
            n_phantoms=n_phantoms[snr],
            phantom_size=args.phantom_size,
            models=models,
            norm_stats=norm_stats,
            pld_values=pld_values,
            normalization_mode=normalization_mode,
            global_scale_factor=global_scale_factor,
            ls_params=CORRECTED_LS_PARAMS,
            smooth_sigmas=smooth_sigmas,
            device=device,
        )

        metrics = compute_metrics(collectors, ls_collectors, sls_collectors, smooth_sigmas, snr)
        all_results[str(snr)] = metrics

        # Log summary
        logger.info(f"  NN CBF MAE: {metrics['nn']['cbf_mae']:.2f} "
                    f"[{metrics['nn']['cbf_mae_ci'][0]:.2f}, {metrics['nn']['cbf_mae_ci'][1]:.2f}]")
        logger.info(f"  NN ATT MAE: {metrics['nn']['att_mae']:.2f} "
                    f"[{metrics['nn']['att_mae_ci'][0]:.2f}, {metrics['nn']['att_mae_ci'][1]:.2f}]")
        if 'ls' in metrics:
            logger.info(f"  LS CBF MAE: {metrics['ls']['cbf_mae']:.2f} "
                        f"[{metrics['ls']['cbf_mae_ci'][0]:.2f}, {metrics['ls']['cbf_mae_ci'][1]:.2f}]")
            logger.info(f"  LS ATT MAE: {metrics['ls']['att_mae']:.2f}")
            logger.info(f"  CBF Win Rate: {metrics['comparison']['cbf_win_rate']:.1%} "
                        f"[{metrics['comparison']['cbf_win_rate_ci'][0]:.1%}, "
                        f"{metrics['comparison']['cbf_win_rate_ci'][1]:.1%}]")
            logger.info(f"  CBF Wilcoxon p={metrics['comparison']['cbf_wilcoxon']['p_value']:.4g}")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save full results
    output = {
        'experiment': run_dir.name,
        'model_path': str(run_dir),
        'ls_params': CORRECTED_LS_PARAMS,
        'snr_results': all_results,
        'config': {
            'snr_values': snr_values,
            'n_phantoms': n_phantoms,
            'smooth_sigmas': smooth_sigmas,
            'phantom_size': args.phantom_size,
        },
        'runtime_seconds': elapsed,
    }

    json_path = output_dir / 'corrected_baseline_comparison.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)

    logger.info(f"\nResults saved to {json_path}")

    # Print summary table
    logger.info("\n" + "=" * 100)
    logger.info("MULTI-SNR SUMMARY (CORRECTED LS: T1_artery=1650)")
    logger.info("=" * 100)
    header = f"{'SNR':>5} | {'NN CBF MAE':>15} | {'LS CBF MAE':>15} | {'CBF WinRate':>15} | {'CBF p-val':>10} | {'NN ATT MAE':>15} | {'ATT WinRate':>15}"
    logger.info(header)
    logger.info("-" * len(header))

    for snr in snr_values:
        m = all_results[str(snr)]
        nn_cbf_str = f"{m['nn']['cbf_mae']:.2f}"
        nn_att_str = f"{m['nn']['att_mae']:.1f}"

        if 'ls' in m:
            ls_cbf_str = f"{m['ls']['cbf_mae']:.2f}"
            cbf_wr_str = f"{m['comparison']['cbf_win_rate']:.1%}"
            cbf_p_str = f"{m['comparison']['cbf_wilcoxon']['p_value']:.2g}"
            att_wr_str = f"{m['comparison']['att_win_rate']:.1%}"
        else:
            ls_cbf_str = "N/A"
            cbf_wr_str = "N/A"
            cbf_p_str = "N/A"
            att_wr_str = "N/A"

        logger.info(f"{snr:>5} | {nn_cbf_str:>15} | {ls_cbf_str:>15} | {cbf_wr_str:>15} | {cbf_p_str:>10} | {nn_att_str:>15} | {att_wr_str:>15}")

    logger.info("=" * 100)
    logger.info("Done!")


if __name__ == '__main__':
    main()

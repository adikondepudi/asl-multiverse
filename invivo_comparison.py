#!/usr/bin/env python3
"""
In-vivo NN vs LS comparison — the real-world test.

Trains a model from scratch, runs NN on actual patient data, compares to
cached LS results using metrics that don't require ground truth:
  - GM/WM mean CBF and ATT (physiological plausibility)
  - CoV within tissue (consistency)
  - GM/WM contrast ratio
  - Spatial smoothness (gradient magnitude)
  - Cross-subject consistency

Both methods get the EXACT same input. No rigging.

Usage:
    # First run: compute LS baselines (slow, ~5 min per subject)
    python invivo_comparison.py --compute-ls --device mps

    # Ralph loop iteration (fast, uses cached LS)
    python invivo_comparison.py --device mps

    # Quick iteration on 3 subjects
    python invivo_comparison.py --device mps --subjects 20231030_MR1_A152 20231003_MR1_A142 20231016_MR1_A147

    # Skip training, reload saved model
    python invivo_comparison.py --device mps --skip-training --model-path invivo_results/trained_model.pt
"""
import argparse
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings('ignore')

from simulation.asl_simulation import ASLParameters
from simulation.enhanced_simulation import RealisticASLSimulator, SpatialPhantomGenerator
from simulation.noise_engine import NoiseInjector
from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
from models.spatial_asl_network import MaskedSpatialLoss
from baselines.multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from utils.helpers import get_grid_search_initial_guess


# ============================================================================
# Training (same approach as ralph loop, but 5 PLDs to match in-vivo)
# ============================================================================

def load_training_config():
    """Load training config from config/invivo_experiment.yaml if it exists,
    otherwise return defaults."""
    config_path = Path('config/invivo_experiment.yaml')
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        cfg = {}
        for section in ['simulation', 'data', 'training']:
            if section in raw:
                cfg.update(raw[section])
        if 'simulation' in raw and 'domain_randomization' in raw['simulation']:
            cfg['domain_randomization'] = raw['simulation']['domain_randomization']
        if 'noise_config' in raw:
            cfg['noise_config'] = raw['noise_config']
        return cfg
    return None


def train_model_for_invivo(device, n_samples=3000, n_epochs=30, seed=42):
    """Train a model with 5 PLDs matching in-vivo data."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load config overrides
    cfg = load_training_config() or {}

    plds = np.array(cfg.get('pld_values', [500, 1000, 1500, 2000, 2500]), dtype=np.float32)
    n_plds = len(plds)

    # Physics with domain randomization
    base_params = ASLParameters(
        T1_artery=cfg.get('T1_artery', 1650.0),
        T_tau=cfg.get('T_tau', 1800.0),
        alpha_PCASL=cfg.get('alpha_PCASL', 0.85),
        alpha_VSASL=cfg.get('alpha_VSASL', 0.56),
        T2_factor=cfg.get('T2_factor', 1.0),
        alpha_BS1=cfg.get('alpha_BS1', 1.0),
    )

    domain_rand_cfg = cfg.get('domain_randomization', {})
    domain_rand = {
        'T1_artery_range': domain_rand_cfg.get('T1_artery_range', [1550.0, 2150.0]),
        'alpha_PCASL_range': domain_rand_cfg.get('alpha_PCASL_range', [0.75, 0.95]),
        'alpha_VSASL_range': domain_rand_cfg.get('alpha_VSASL_range', [0.40, 0.70]),
        'alpha_BS1_range': domain_rand_cfg.get('alpha_BS1_range', [0.85, 1.0]),
        'T_tau_perturb': domain_rand_cfg.get('T_tau_perturb', 0.10),
    }
    use_domain_rand = domain_rand_cfg.get('enabled', True)

    phantom_gen = SpatialPhantomGenerator(size=64, pve_sigma=1.0)
    plds_bc = plds[:, np.newaxis, np.newaxis]
    lambda_b = 0.90

    print(f"[Train] Generating {n_samples} training samples (domain_rand={use_domain_rand})...")
    signals_list, targets_list = [], []

    for i in range(n_samples):
        cbf_map, att_map, _ = phantom_gen.generate_phantom(include_pathology=True)

        if use_domain_rand:
            t1_b = np.random.uniform(*domain_rand['T1_artery_range'])
            alpha_bs1 = np.random.uniform(*domain_rand['alpha_BS1_range'])
            alpha_p = np.random.uniform(*domain_rand['alpha_PCASL_range']) * (alpha_bs1 ** 4)
            alpha_v = np.random.uniform(*domain_rand['alpha_VSASL_range']) * (alpha_bs1 ** 3)
            tau = base_params.T_tau * (1 + np.random.uniform(
                -domain_rand['T_tau_perturb'], domain_rand['T_tau_perturb']))
        else:
            t1_b = base_params.T1_artery
            alpha_p = base_params.alpha_PCASL * (base_params.alpha_BS1 ** 4)
            alpha_v = base_params.alpha_VSASL * (base_params.alpha_BS1 ** 3)
            tau = base_params.T_tau

        t2_f = base_params.T2_factor

        att_bc = att_map[np.newaxis, :, :].astype(np.float32)
        cbf_bc = (cbf_map / 6000.0)[np.newaxis, :, :].astype(np.float32)

        # PCASL signal
        mask_arrived = (plds_bc >= att_bc)
        mask_transit = (plds_bc < att_bc) & (plds_bc >= (att_bc - tau))
        sig_p_arrived = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                         np.exp(-plds_bc / t1_b) * (1 - np.exp(-tau / t1_b)) * t2_f) / lambda_b
        sig_p_transit = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                         (np.exp(-att_bc / t1_b) - np.exp(-(tau + plds_bc) / t1_b)) * t2_f) / lambda_b
        pcasl_sig = np.zeros_like(plds_bc * cbf_bc)
        pcasl_sig[mask_arrived] = sig_p_arrived[mask_arrived]
        pcasl_sig[mask_transit] = sig_p_transit[mask_transit]

        # VSASL signal
        sib = 1.0
        mask_vs_arrived = (plds_bc > att_bc)
        sig_v_early = (2 * alpha_v * cbf_bc * sib * (plds_bc / 1000.0) *
                       np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
        sig_v_late = (2 * alpha_v * cbf_bc * sib * (att_bc / 1000.0) *
                      np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
        vsasl_sig = np.where(mask_vs_arrived, sig_v_late, sig_v_early)

        clean_signal = np.concatenate([pcasl_sig, vsasl_sig], axis=0).astype(np.float32)
        target = np.stack([cbf_map, att_map], axis=0).astype(np.float32)
        signals_list.append(clean_signal)
        targets_list.append(target)

    signals = np.array(signals_list, dtype=np.float32) * 100.0  # M0 scaling
    targets = np.array(targets_list, dtype=np.float32)

    # Norm stats from brain voxels
    cbf_all = targets[:, 0, :, :]
    att_all = targets[:, 1, :, :]
    brain_mask = cbf_all > 1.0
    norm_stats = {
        'y_mean_cbf': float(np.mean(cbf_all[brain_mask])),
        'y_std_cbf': float(np.std(cbf_all[brain_mask])),
        'y_mean_att': float(np.mean(att_all[brain_mask])),
        'y_std_att': float(np.std(att_all[brain_mask])),
    }
    print(f"  Norm stats: CBF {norm_stats['y_mean_cbf']:.1f}+/-{norm_stats['y_std_cbf']:.1f}, "
          f"ATT {norm_stats['y_mean_att']:.0f}+/-{norm_stats['y_std_att']:.0f}")

    # Create model
    features = cfg.get('hidden_sizes', [32, 64, 128, 256])
    model = AmplitudeAwareSpatialASLNet(
        n_plds=n_plds, features=features,
        use_film_at_bottleneck=cfg.get('use_film_at_bottleneck', True),
        use_film_at_decoder=cfg.get('use_film_at_decoder', True),
        use_amplitude_output_modulation=cfg.get('use_amplitude_output_modulation', True),
    ).to(device)

    loss_fn = MaskedSpatialLoss(
        loss_type=cfg.get('loss_type', 'l1'),
        dc_weight=cfg.get('dc_weight', 0.0),
        att_scale=cfg.get('att_scale', 1.0),
        cbf_weight=cfg.get('cbf_weight', 1.0),
        att_weight=cfg.get('att_weight', 1.0),
        norm_stats=norm_stats,
        variance_weight=cfg.get('variance_weight', 0.01),
    )

    # Noise injector
    noise_cfg = {
        'noise_config': cfg.get('noise_config', {'snr_range': [2.0, 25.0]}),
        'data_noise_components': cfg.get('data_noise_components', ['thermal']),
        'noise_type': cfg.get('noise_type', 'gaussian'),
    }
    noise_injector = NoiseInjector(noise_cfg)
    simulator = RealisticASLSimulator(params=base_params)
    ref_signal = simulator._compute_reference_signal()
    scalings = simulator.compute_tr_noise_scaling(plds)
    pld_scaling = {'PCASL': scalings['PCASL'], 'VSASL': scalings['VSASL']}

    global_scale = cfg.get('global_scale_factor', 10.0)
    batch_size = cfg.get('batch_size', 64)
    lr = cfg.get('learning_rate', 0.003)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg.get('weight_decay', 0.0001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    # Train/val split
    n = len(signals)
    n_val = max(1, int(n * 0.1))
    n_train = n - n_val
    perm = np.random.permutation(n)
    train_idx = perm[:n_train]

    train_signals = torch.from_numpy(signals[train_idx]).float()
    train_targets = targets[train_idx]

    clean_fraction = 0.15

    print(f"[Train] Training for {n_epochs} epochs (lr={lr}, bs={batch_size})...")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        perm_train = np.random.permutation(n_train)
        use_noise = epoch >= int(n_epochs * clean_fraction)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm_train[start:end]

            raw_sig = train_signals[idx].to(device)
            cbf_t = torch.from_numpy(train_targets[idx, 0:1]).float().to(device)
            att_t = torch.from_numpy(train_targets[idx, 1:2]).float().to(device)

            masks = []
            for j in range(len(idx)):
                m = np.mean(np.abs(train_signals[idx[j]].numpy()), axis=0)
                masks.append((m > np.percentile(m, 5)).astype(np.float32))
            mask_t = torch.from_numpy(np.array(masks)[:, np.newaxis]).float().to(device)

            if use_noise:
                noisy_sig = noise_injector.apply_noise(raw_sig, ref_signal, pld_scaling)
            else:
                noisy_sig = raw_sig

            normalized = torch.clamp(noisy_sig * global_scale, -15.0, 15.0)
            optimizer.zero_grad()
            pred_cbf, pred_att, log_var_cbf, log_var_att = model(normalized)
            loss_dict = loss_fn(pred_cbf.float(), pred_att.float(), cbf_t, att_t, mask_t,
                                normalized, log_var_cbf=log_var_cbf.float(), log_var_att=log_var_att.float())
            loss = loss_dict['total_loss']
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss = {avg_loss:.4f}")

    return model, norm_stats


# ============================================================================
# In-vivo inference
# ============================================================================

def run_nn_on_subject(model, subject_dir, norm_stats, device, global_scale=10.0):
    """Run spatial NN inference on one subject's npy data."""
    signals = np.load(subject_dir / 'high_snr_signals.npy')
    brain_mask = np.load(subject_dir / 'brain_mask.npy')

    H, W, Z = brain_mask.shape
    n_plds = 5

    # Reshape to spatial: (n_voxels, 10) -> (H, W, Z, 10) -> (Z, 10, H, W)
    signals_3d = signals.reshape(H, W, Z, 2 * n_plds)
    signals_spatial = np.transpose(signals_3d, (2, 3, 0, 1)).astype(np.float32)

    # Apply M0 scaling + global scale (matching training)
    signals_spatial = signals_spatial * 100.0 * global_scale
    signals_spatial = np.clip(signals_spatial, -15.0, 15.0)

    model.eval()
    cbf_vol = np.zeros((Z, H, W), dtype=np.float32)
    att_vol = np.zeros((Z, H, W), dtype=np.float32)

    with torch.no_grad():
        for z in range(Z):
            inp = torch.from_numpy(signals_spatial[z:z+1]).float().to(device)
            _, _, h, w = inp.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                inp = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')

            pred_cbf, pred_att, _, _ = model(inp)
            pred_cbf = pred_cbf[:, :, :h, :w]
            pred_att = pred_att[:, :, :h, :w]

            cbf_vol[z] = pred_cbf[0, 0].cpu().numpy() * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
            att_vol[z] = pred_att[0, 0].cpu().numpy() * norm_stats['y_std_att'] + norm_stats['y_mean_att']

    cbf_vol = np.clip(np.transpose(cbf_vol, (1, 2, 0)), 0, 200)
    att_vol = np.clip(np.transpose(att_vol, (1, 2, 0)), 0, 5000)
    return cbf_vol, att_vol


def run_ls_on_subject(subject_dir):
    """Run LS fitting on one subject. Uses consensus physics params."""
    signals = np.load(subject_dir / 'high_snr_signals.npy')
    brain_mask = np.load(subject_dir / 'brain_mask.npy')
    plds = np.load(subject_dir / 'plds.npy').astype(np.float64)

    H, W, Z = brain_mask.shape
    brain_flat = brain_mask.flatten()
    brain_idx = np.where(brain_flat)[0]

    ls_params = {
        'T1_artery': 1650.0, 'T_tau': 1800.0, 'T2_factor': 1.0,
        'alpha_BS1': 0.93, 'alpha_PCASL': 0.85, 'alpha_VSASL': 0.56,
    }
    pldti = np.column_stack([plds, plds])
    n_plds = len(plds)

    cbf_flat = np.full(brain_flat.shape, np.nan, dtype=np.float32)
    att_flat = np.full(brain_flat.shape, np.nan, dtype=np.float32)
    n_fit, n_fail = 0, 0

    for vi in brain_idx:
        sig_pcasl = signals[vi, :n_plds]
        sig_vsasl = signals[vi, n_plds:]
        signal_1d = np.concatenate([sig_pcasl, sig_vsasl])
        observed = signal_1d.reshape((n_plds, 2), order='F')

        try:
            init = get_grid_search_initial_guess(signal_1d, plds, ls_params)
        except Exception:
            init = [30.0 / 6000.0, 1500.0]
        try:
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                pldti, observed, init,
                ls_params['T1_artery'], ls_params['T_tau'], ls_params['T2_factor'],
                ls_params['alpha_BS1'], ls_params['alpha_PCASL'], ls_params['alpha_VSASL'])
            cbf_val, att_val = beta[0] * 6000.0, beta[1]
            if np.isfinite(cbf_val) and np.isfinite(att_val):
                cbf_flat[vi] = np.clip(cbf_val, 0, 200)
                att_flat[vi] = np.clip(att_val, 0, 5000)
                n_fit += 1
            else:
                n_fail += 1
        except Exception:
            n_fail += 1

    print(f"    LS fit: {n_fit}/{n_fit+n_fail} succeeded ({100*n_fit/max(n_fit+n_fail,1):.1f}%)")
    return cbf_flat.reshape(H, W, Z), att_flat.reshape(H, W, Z)


# ============================================================================
# Comparison metrics
# ============================================================================

def compute_metrics(cbf, att, brain_mask, gm_mask):
    """Compute comparison metrics for one method on one subject."""
    wm_mask = brain_mask & ~gm_mask
    metrics = {}

    gm_cbf = cbf[gm_mask & np.isfinite(cbf)]
    if len(gm_cbf) > 10:
        metrics['gm_cbf_mean'] = float(np.mean(gm_cbf))
        metrics['gm_cbf_std'] = float(np.std(gm_cbf))
        metrics['gm_cbf_median'] = float(np.median(gm_cbf))
        metrics['gm_cbf_cov'] = float(np.std(gm_cbf) / max(np.mean(gm_cbf), 0.1) * 100)

    wm_cbf = cbf[wm_mask & np.isfinite(cbf)]
    if len(wm_cbf) > 10:
        metrics['wm_cbf_mean'] = float(np.mean(wm_cbf))
        metrics['wm_cbf_std'] = float(np.std(wm_cbf))
        metrics['wm_cbf_cov'] = float(np.std(wm_cbf) / max(np.mean(wm_cbf), 0.1) * 100)

    if 'gm_cbf_mean' in metrics and 'wm_cbf_mean' in metrics and metrics['wm_cbf_mean'] > 0.1:
        metrics['gm_wm_ratio'] = metrics['gm_cbf_mean'] / metrics['wm_cbf_mean']

    gm_att = att[gm_mask & np.isfinite(att)]
    if len(gm_att) > 10:
        metrics['gm_att_mean'] = float(np.mean(gm_att))
        metrics['gm_att_std'] = float(np.std(gm_att))
        metrics['gm_att_cov'] = float(np.std(gm_att) / max(np.mean(gm_att), 0.1) * 100)

    wm_att = att[wm_mask & np.isfinite(att)]
    if len(wm_att) > 10:
        metrics['wm_att_mean'] = float(np.mean(wm_att))
        metrics['wm_att_std'] = float(np.std(wm_att))

    # Spatial smoothness: mean gradient magnitude in brain
    brain_cbf = cbf.copy()
    brain_cbf[~brain_mask | ~np.isfinite(brain_cbf)] = 0
    grad_x = np.diff(brain_cbf, axis=0)
    grad_y = np.diff(brain_cbf, axis=1)
    mask_x = brain_mask[:-1, :, :] & brain_mask[1:, :, :]
    mask_y = brain_mask[:, :-1, :] & brain_mask[:, 1:, :]
    if mask_x.sum() > 0 and mask_y.sum() > 0:
        metrics['spatial_smoothness'] = float(
            (np.mean(np.abs(grad_x[mask_x])) + np.mean(np.abs(grad_y[mask_y]))) / 2)

    return metrics


def evaluate_pass_fail(all_results):
    """Evaluate pass/fail criteria and return structured results."""
    subjects = sorted(all_results.keys())
    n = len(subjects)

    nn_gm_cbfs, ls_gm_cbfs = [], []
    nn_gm_covs, ls_gm_covs = [], []
    nn_smooths, ls_smooths = [], []
    nn_gm_wm_ratios, ls_gm_wm_ratios = [], []
    nn_gm_atts, ls_gm_atts = [], []

    per_subject_pass = True

    for subj in subjects:
        nn = all_results[subj].get('nn', {})
        ls = all_results[subj].get('ls', {})

        gm_cbf = nn.get('gm_cbf_mean', 0)
        gm_wm = nn.get('gm_wm_ratio', 0)
        gm_att = nn.get('gm_att_mean', 0)

        # Per-subject physiological plausibility
        if not (15 < gm_cbf < 120):
            per_subject_pass = False
        if not (1.2 < gm_wm < 6.0):
            per_subject_pass = False
        if not (400 < gm_att < 3000):
            per_subject_pass = False

        if 'gm_cbf_mean' in nn: nn_gm_cbfs.append(nn['gm_cbf_mean'])
        if 'gm_cbf_mean' in ls: ls_gm_cbfs.append(ls['gm_cbf_mean'])
        if 'gm_cbf_cov' in nn: nn_gm_covs.append(nn['gm_cbf_cov'])
        if 'gm_cbf_cov' in ls: ls_gm_covs.append(ls['gm_cbf_cov'])
        if 'spatial_smoothness' in nn: nn_smooths.append(nn['spatial_smoothness'])
        if 'spatial_smoothness' in ls: ls_smooths.append(ls['spatial_smoothness'])
        if 'gm_wm_ratio' in nn: nn_gm_wm_ratios.append(nn['gm_wm_ratio'])
        if 'gm_wm_ratio' in ls: ls_gm_wm_ratios.append(ls['gm_wm_ratio'])
        if 'gm_att_mean' in nn: nn_gm_atts.append(nn['gm_att_mean'])
        if 'gm_att_mean' in ls: ls_gm_atts.append(ls['gm_att_mean'])

    # Aggregate checks
    nn_cov_avg = np.mean(nn_gm_covs) if nn_gm_covs else 999
    ls_cov_avg = np.mean(ls_gm_covs) if ls_gm_covs else 999
    nn_smooth_avg = np.mean(nn_smooths) if nn_smooths else 999
    ls_smooth_avg = np.mean(ls_smooths) if ls_smooths else 999
    nn_cbf_avg = np.mean(nn_gm_cbfs) if nn_gm_cbfs else 0
    ls_cbf_avg = np.mean(ls_gm_cbfs) if ls_gm_cbfs else 0
    nn_ratio_avg = np.mean(nn_gm_wm_ratios) if nn_gm_wm_ratios else 0
    nn_att_avg = np.mean(nn_gm_atts) if nn_gm_atts else 0

    checks = {
        'physio_plausible': per_subject_pass,
        'nn_cov_beats_ls': nn_cov_avg < ls_cov_avg,
        'nn_smoother_than_ls': nn_smooth_avg < ls_smooth_avg,
        'nn_cov_avg': nn_cov_avg,
        'ls_cov_avg': ls_cov_avg,
        'nn_smooth_avg': nn_smooth_avg,
        'ls_smooth_avg': ls_smooth_avg,
        'nn_cbf_avg': nn_cbf_avg,
        'ls_cbf_avg': ls_cbf_avg,
        'nn_ratio_avg': nn_ratio_avg,
        'nn_att_avg': nn_att_avg,
        'cross_subj_cbf_std_nn': float(np.std(nn_gm_cbfs)) if nn_gm_cbfs else 999,
        'cross_subj_cbf_std_ls': float(np.std(ls_gm_cbfs)) if ls_gm_cbfs else 999,
    }

    # Overall pass: physiologically plausible AND (CoV OR smoothness beats LS)
    checks['overall_pass'] = (
        checks['physio_plausible'] and
        (checks['nn_cov_beats_ls'] or checks['nn_smoother_than_ls'])
    )

    return checks


def print_results(all_results, checks):
    """Print formatted comparison table."""
    subjects = sorted(all_results.keys())

    print(f"\n{'='*90}")
    print(f" IN-VIVO NN vs LS COMPARISON ({len(subjects)} subjects)")
    print(f"{'='*90}")

    print(f"\n{'Subject':<22} | {'Meth':4} | {'GM CBF':>7} {'CoV%':>6} | "
          f"{'WM CBF':>7} | {'GM/WM':>5} | {'GM ATT':>7} | {'Smooth':>7}")
    print("-" * 90)

    for subj in subjects:
        nn, ls = all_results[subj].get('nn', {}), all_results[subj].get('ls', {})

        def fmt(m, key, f='{:.1f}'):
            return f.format(m[key]) if key in m else '  N/A'

        print(f"{subj:<22} | {'NN':4} | {fmt(nn,'gm_cbf_mean'):>7} {fmt(nn,'gm_cbf_cov'):>5}% | "
              f"{fmt(nn,'wm_cbf_mean'):>7} | {fmt(nn,'gm_wm_ratio','{:.2f}'):>5} | "
              f"{fmt(nn,'gm_att_mean','{:.0f}'):>7} | {fmt(nn,'spatial_smoothness','{:.2f}'):>7}")
        print(f"{'':22} | {'LS':4} | {fmt(ls,'gm_cbf_mean'):>7} {fmt(ls,'gm_cbf_cov'):>5}% | "
              f"{fmt(ls,'wm_cbf_mean'):>7} | {fmt(ls,'gm_wm_ratio','{:.2f}'):>5} | "
              f"{fmt(ls,'gm_att_mean','{:.0f}'):>7} | {fmt(ls,'spatial_smoothness','{:.2f}'):>7}")
        print("-" * 90)

    # Pass/fail summary
    print(f"\n  PASS/FAIL CHECKS:")
    print(f"    Physiological plausibility:  {'PASS' if checks['physio_plausible'] else 'FAIL'}")
    print(f"    NN CoV < LS CoV:             {'PASS' if checks['nn_cov_beats_ls'] else 'FAIL'} "
          f"(NN={checks['nn_cov_avg']:.1f}% vs LS={checks['ls_cov_avg']:.1f}%)")
    print(f"    NN smoother than LS:         {'PASS' if checks['nn_smoother_than_ls'] else 'FAIL'} "
          f"(NN={checks['nn_smooth_avg']:.2f} vs LS={checks['ls_smooth_avg']:.2f})")
    print(f"    Cross-subj CBF std:          NN={checks['cross_subj_cbf_std_nn']:.1f} vs LS={checks['cross_subj_cbf_std_ls']:.1f}")
    print(f"    NN GM CBF avg:               {checks['nn_cbf_avg']:.1f} (expected 40-80)")
    print(f"    NN GM/WM ratio avg:          {checks['nn_ratio_avg']:.2f} (expected 2.0-3.5)")
    print(f"    NN GM ATT avg:               {checks['nn_att_avg']:.0f}ms (expected 800-1800)")

    verdict = "PASS" if checks['overall_pass'] else "FAIL"
    print(f"\n    OVERALL: {verdict}")
    print(f"{'='*90}\n")


def save_results(all_results, checks, output_dir):
    """Save JSON and append to experiment log."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Latest results
    result = {
        'timestamp': datetime.now().isoformat(),
        'checks': checks,
        'per_subject': all_results,
    }
    with open(output_dir / 'latest_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Append to log
    log_path = output_dir / 'experiment_log.md'
    is_new = not log_path.exists()
    with open(log_path, 'a') as f:
        if is_new:
            f.write("# In-Vivo Experiment Log\n\n")
        f.write(f"## {result['timestamp']}\n\n")
        f.write(f"- **GM CBF**: NN={checks['nn_cbf_avg']:.1f}, LS={checks['ls_cbf_avg']:.1f}\n")
        f.write(f"- **GM CoV**: NN={checks['nn_cov_avg']:.1f}%, LS={checks['ls_cov_avg']:.1f}%\n")
        f.write(f"- **Smoothness**: NN={checks['nn_smooth_avg']:.2f}, LS={checks['ls_smooth_avg']:.2f}\n")
        f.write(f"- **GM/WM ratio**: NN={checks['nn_ratio_avg']:.2f}\n")
        f.write(f"- **GM ATT**: NN={checks['nn_att_avg']:.0f}ms\n")
        f.write(f"- **Physio plausible**: {'PASS' if checks['physio_plausible'] else 'FAIL'}\n")
        f.write(f"- **CoV beats LS**: {'PASS' if checks['nn_cov_beats_ls'] else 'FAIL'}\n")
        f.write(f"- **Smoother than LS**: {'PASS' if checks['nn_smoother_than_ls'] else 'FAIL'}\n")
        f.write(f"- **OVERALL**: {'PASS' if checks['overall_pass'] else 'FAIL'}\n")
        f.write("\n---\n\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='In-vivo NN vs LS comparison')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--data-dir', default='data/invivo_processed_npy')
    parser.add_argument('--ls-cache-dir', default='invivo_comparison_results',
                        help='Directory with cached LS results (ls_cbf.npy, ls_att.npy per subject)')
    parser.add_argument('--n-samples', type=int, default=3000)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--output-dir', default='invivo_results')
    parser.add_argument('--compute-ls', action='store_true',
                        help='Compute LS baselines (slow). Only needed once.')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    ls_cache_dir = Path(args.ls_cache_dir)

    # Subject selection
    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if args.subjects:
        subject_dirs = [d for d in subject_dirs if d.name in args.subjects]

    # Phase 0: Compute LS baselines if requested
    if args.compute_ls:
        print(f"\n[Phase 0] Computing LS baselines for {len(subject_dirs)} subjects...")
        for subj_dir in subject_dirs:
            subj_id = subj_dir.name
            cache_path = ls_cache_dir / subj_id / 'ls_cbf.npy'
            if cache_path.exists():
                print(f"  {subj_id}: cached, skipping")
                continue
            print(f"  {subj_id}: fitting...")
            ls_cbf, ls_att = run_ls_on_subject(subj_dir)
            subj_cache = ls_cache_dir / subj_id
            subj_cache.mkdir(parents=True, exist_ok=True)
            np.save(subj_cache / 'ls_cbf.npy', ls_cbf)
            np.save(subj_cache / 'ls_att.npy', ls_att)
        print("  LS baselines cached.")
        if not args.skip_training:
            print("  (Continuing to training...)")

    # Phase 1: Train model
    if args.skip_training and args.model_path:
        print(f"\n[Phase 1] Loading saved model from {args.model_path}")
        saved = torch.load(args.model_path, map_location='cpu', weights_only=False)
        cfg = load_training_config() or {}
        features = cfg.get('hidden_sizes', [32, 64, 128, 256])
        model = AmplitudeAwareSpatialASLNet(
            n_plds=5, features=features,
            use_film_at_bottleneck=cfg.get('use_film_at_bottleneck', True),
            use_film_at_decoder=cfg.get('use_film_at_decoder', True),
            use_amplitude_output_modulation=cfg.get('use_amplitude_output_modulation', True),
        ).to(device)
        model.load_state_dict(saved['model_state_dict'])
        norm_stats = saved['norm_stats']
    else:
        print(f"\n[Phase 1] Training model from scratch...")
        t_train = time.time()
        model, norm_stats = train_model_for_invivo(
            device, n_samples=args.n_samples, n_epochs=args.n_epochs, seed=args.seed)
        print(f"  Training: {time.time()-t_train:.0f}s")
        torch.save({'model_state_dict': model.state_dict(), 'norm_stats': norm_stats},
                   output_dir / 'trained_model.pt')

    # Phase 2: Run on subjects
    print(f"\n[Phase 2] Evaluating {len(subject_dirs)} subjects...")
    all_results = {}

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        brain_mask = np.load(subj_dir / 'brain_mask.npy')
        gm_mask = np.load(subj_dir / 'gm_mask.npy')

        # NN inference
        t_nn = time.time()
        nn_cbf, nn_att = run_nn_on_subject(model, subj_dir, norm_stats, device)
        nn_time = time.time() - t_nn

        # Load cached LS (or compute if not cached)
        ls_cache = ls_cache_dir / subj_id
        if (ls_cache / 'ls_cbf.npy').exists():
            ls_cbf = np.load(ls_cache / 'ls_cbf.npy')
            ls_att = np.load(ls_cache / 'ls_att.npy')
        else:
            print(f"    {subj_id}: LS not cached, computing...")
            ls_cbf, ls_att = run_ls_on_subject(subj_dir)
            ls_cache.mkdir(parents=True, exist_ok=True)
            np.save(ls_cache / 'ls_cbf.npy', ls_cbf)
            np.save(ls_cache / 'ls_att.npy', ls_att)

        nn_metrics = compute_metrics(nn_cbf, nn_att, brain_mask, gm_mask)
        ls_metrics = compute_metrics(ls_cbf, ls_att, brain_mask, gm_mask)
        nn_metrics['inference_time_s'] = nn_time

        all_results[subj_id] = {'nn': nn_metrics, 'ls': ls_metrics}

        # Save maps
        subj_out = output_dir / subj_id
        subj_out.mkdir(parents=True, exist_ok=True)
        np.save(subj_out / 'nn_cbf.npy', nn_cbf)
        np.save(subj_out / 'nn_att.npy', nn_att)

    # Phase 3: Evaluate and print
    checks = evaluate_pass_fail(all_results)
    print_results(all_results, checks)
    save_results(all_results, checks, output_dir)
    print(f"Results: {output_dir}/latest_results.json")
    print(f"Log: {output_dir}/experiment_log.md")


if __name__ == '__main__':
    main()

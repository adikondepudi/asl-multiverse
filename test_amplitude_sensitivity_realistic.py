#!/usr/bin/env python3
"""
Realistic Amplitude Sensitivity Test for Spatial ASL Models
============================================================

Tests whether a trained spatial model (SpatialASLNet or AmplitudeAwareSpatialASLNet)
can track true CBF values using realistic synthetic ASL signals, rather than
random Gaussian noise inputs.

Two tests are performed:

Test 1 - CBF Linearity:
    Generates uniform 64x64 phantoms with known CBF values
    [10, 20, 40, 60, 80, 100, 120, 150] ml/100g/min at fixed ATT=1500ms.
    Measures whether predicted CBF tracks true CBF linearly.
    Reports R^2 and slope of linear fit (ideal: R^2=1.0, slope=1.0).

Test 2 - Amplitude Scaling:
    Takes a fixed CBF=60 phantom and scales the input signals by
    [0.5, 0.75, 1.0, 1.5, 2.0]. Checks if predicted CBF scales
    proportionally (amplitude-aware) or stays constant (amplitude-invariant).

Usage:
    python test_amplitude_sensitivity_realistic.py --run_dir <path_to_model> --output_dir results/
"""

import sys
import os
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config(run_dir: Path) -> dict:
    """Load training config from run directory.

    Tries config.yaml first, then falls back to research_config.json.
    Returns a flat dict with keys used by the rest of the script.
    """
    import yaml

    config = {}

    # Try config.yaml (primary)
    config_yaml_path = run_dir / 'config.yaml'
    if config_yaml_path.exists():
        with open(config_yaml_path, 'r') as f:
            full_config = yaml.safe_load(f)
        training = full_config.get('training', {})
        data = full_config.get('data', {})
        simulation = full_config.get('simulation', {})

        config['model_class_name'] = training.get('model_class_name', 'SpatialASLNet')
        config['hidden_sizes'] = training.get('hidden_sizes', [32, 64, 128, 256])
        config['use_film_at_bottleneck'] = training.get('use_film_at_bottleneck', True)
        config['use_film_at_decoder'] = training.get('use_film_at_decoder', True)
        config['use_amplitude_output_modulation'] = training.get('use_amplitude_output_modulation', True)
        config['pld_values'] = data.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000])
        config['normalization_mode'] = data.get('normalization_mode', 'global_scale')
        config['global_scale_factor'] = data.get('global_scale_factor', 10.0)
        config['T1_artery'] = simulation.get('T1_artery', 1650.0)  # 3T consensus (Alsop 2015)
        config['T_tau'] = simulation.get('T_tau', 1800.0)
        config['alpha_PCASL'] = simulation.get('alpha_PCASL', 0.85)
        config['alpha_VSASL'] = simulation.get('alpha_VSASL', 0.56)
        config['T2_factor'] = simulation.get('T2_factor', 1.0)
        config['alpha_BS1'] = simulation.get('alpha_BS1', 1.0)

        logger.info(f"Loaded config from {config_yaml_path}")
        return config

    # Fallback to research_config.json
    rc_path = run_dir / 'research_config.json'
    if rc_path.exists():
        with open(rc_path, 'r') as f:
            rc = json.load(f)
        config['model_class_name'] = rc.get('model_class_name', 'SpatialASLNet')
        config['hidden_sizes'] = rc.get('hidden_sizes', [32, 64, 128, 256])
        config['use_film_at_bottleneck'] = rc.get('use_film_at_bottleneck', True)
        config['use_film_at_decoder'] = rc.get('use_film_at_decoder', True)
        config['use_amplitude_output_modulation'] = rc.get('use_amplitude_output_modulation', True)
        config['pld_values'] = rc.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000])
        config['normalization_mode'] = rc.get('normalization_mode', 'global_scale')
        config['global_scale_factor'] = rc.get('global_scale_factor', 10.0)
        config['T1_artery'] = rc.get('T1_artery', 1650.0)  # 3T consensus (Alsop 2015)
        config['T_tau'] = rc.get('T_tau', 1800.0)
        config['alpha_PCASL'] = rc.get('alpha_PCASL', 0.85)
        config['alpha_VSASL'] = rc.get('alpha_VSASL', 0.56)
        config['T2_factor'] = rc.get('T2_factor', 1.0)
        config['alpha_BS1'] = rc.get('alpha_BS1', 1.0)
        logger.info(f"Loaded config from {rc_path}")
        return config

    raise FileNotFoundError(
        f"No config.yaml or research_config.json found in {run_dir}"
    )


def load_norm_stats(run_dir: Path) -> dict:
    """Load normalization statistics from norm_stats.json."""
    norm_path = run_dir / 'norm_stats.json'
    if not norm_path.exists():
        raise FileNotFoundError(f"norm_stats.json not found in {run_dir}")
    with open(norm_path, 'r') as f:
        norm_stats = json.load(f)
    logger.info(f"Loaded norm_stats: y_mean_cbf={norm_stats['y_mean_cbf']:.2f}, "
                f"y_std_cbf={norm_stats['y_std_cbf']:.2f}, "
                f"y_mean_att={norm_stats['y_mean_att']:.2f}, "
                f"y_std_att={norm_stats['y_std_att']:.2f}")
    return norm_stats


def load_ensemble(run_dir: Path, config: dict, device: torch.device) -> List[torch.nn.Module]:
    """Load trained spatial model ensemble from run directory."""
    from spatial_asl_network import SpatialASLNet
    from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet

    models_dir = run_dir / 'trained_models'
    if not models_dir.exists():
        models_dir = run_dir

    model_files = sorted(list(models_dir.glob('ensemble_model_*.pt')))
    if not model_files:
        raise FileNotFoundError(f"No ensemble_model_*.pt files found in {models_dir}")

    logger.info(f"Found {len(model_files)} model files in {models_dir}")

    model_class_name = config['model_class_name']
    features = config['hidden_sizes']
    pld_values = config['pld_values']
    n_plds = len(pld_values)

    # Ensure 4 levels for U-Net
    while len(features) < 4:
        features.insert(0, max(1, features[0] // 2))

    loaded_models = []
    for mp in model_files:
        logger.info(f"  Loading {mp.name}...")

        if model_class_name == 'AmplitudeAwareSpatialASLNet':
            model = AmplitudeAwareSpatialASLNet(
                n_plds=n_plds,
                features=features,
                use_film_at_bottleneck=config.get('use_film_at_bottleneck', True),
                use_film_at_decoder=config.get('use_film_at_decoder', True),
                use_amplitude_output_modulation=config.get('use_amplitude_output_modulation', True),
            )
        elif model_class_name == 'DualEncoderSpatialASLNet':
            from spatial_asl_network import DualEncoderSpatialASLNet
            model = DualEncoderSpatialASLNet(n_plds=n_plds, features=features)
        else:
            model = SpatialASLNet(n_plds=n_plds, features=features)

        try:
            state_dict = torch.load(mp, map_location=device)
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            loaded_models.append(model)
        except Exception as e:
            logger.error(f"  Failed to load {mp.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if not loaded_models:
        raise RuntimeError("All models failed to load!")

    logger.info(f"Successfully loaded {len(loaded_models)} {model_class_name} models")
    return loaded_models


def generate_uniform_phantom(cbf_value: float, att_value: float,
                              plds: np.ndarray, config: dict,
                              size: int = 64, snr: float = 10.0,
                              seed: int = 0) -> Tuple[np.ndarray, float, float]:
    """
    Generate a uniform 64x64 phantom with known CBF and ATT.

    Returns:
        signals: (2*n_plds, size, size) noisy preprocessed signals
        true_cbf: the ground truth CBF in ml/100g/min
        true_att: the ground truth ATT in ms
    """
    from asl_simulation import ASLParameters, ASLSimulator

    params = ASLParameters(
        T1_artery=config.get('T1_artery', 1650.0)  # 3T consensus (Alsop 2015),
        T_tau=config.get('T_tau', 1800.0),
        alpha_PCASL=config.get('alpha_PCASL', 0.85),
        alpha_VSASL=config.get('alpha_VSASL', 0.56),
        T2_factor=config.get('T2_factor', 1.0),
        alpha_BS1=config.get('alpha_BS1', 1.0),
    )
    simulator = ASLSimulator(params=params)

    n_plds = len(plds)

    # Generate clean signals for one voxel
    pcasl = simulator._generate_pcasl_signal(
        plds, att_value, cbf_value,
        params.T1_artery, params.T_tau, params.alpha_PCASL
    )
    vsasl = simulator._generate_vsasl_signal(
        plds, att_value, cbf_value,
        params.T1_artery, params.alpha_VSASL
    )

    # Tile to create uniform 2D phantom: (2*n_plds, size, size)
    clean_signal_1d = np.concatenate([pcasl, vsasl])  # (2*n_plds,)
    clean_signals = np.tile(clean_signal_1d[:, np.newaxis, np.newaxis], (1, size, size))

    # Add Rician noise
    ref_signal = simulator._compute_reference_signal()
    noise_sd = ref_signal / snr

    rng = np.random.RandomState(seed)
    noise_real = noise_sd * rng.randn(*clean_signals.shape)
    noise_imag = noise_sd * rng.randn(*clean_signals.shape)
    noisy_signals = np.sqrt((clean_signals + noise_real)**2 + noise_imag**2).astype(np.float32)

    return noisy_signals, cbf_value, att_value


def preprocess_signals(signals: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply the same preprocessing as the training pipeline.

    Pipeline:
        1. M0_SCALE = 100.0  (SpatialDataset)
        2. global_scale_factor (trainer _process_batch_on_gpu)
    """
    M0_SCALE = 100.0
    scaled = signals * M0_SCALE

    normalization_mode = config.get('normalization_mode', 'global_scale')
    global_scale_factor = config.get('global_scale_factor', 10.0)

    if normalization_mode == 'global_scale':
        preprocessed = scaled * global_scale_factor
    else:
        # Per-pixel z-score (legacy -- destroys CBF info)
        temporal_mean = np.mean(scaled, axis=0, keepdims=True)
        temporal_std = np.std(scaled, axis=0, keepdims=True) + 1e-6
        preprocessed = (scaled - temporal_mean) / temporal_std

    return preprocessed


@torch.no_grad()
def run_ensemble_inference(models: List[torch.nn.Module],
                            input_tensor: torch.Tensor,
                            norm_stats: dict,
                            device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference through the model ensemble and denormalize predictions.

    Returns:
        cbf_map: (H, W) predicted CBF in ml/100g/min
        att_map: (H, W) predicted ATT in ms
    """
    cbf_maps = []
    att_maps = []

    for model in models:
        cbf_pred, att_pred, _, _ = model(input_tensor)

        # Denormalize from z-score to raw units
        cbf_denorm = cbf_pred * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
        att_denorm = att_pred * norm_stats['y_std_att'] + norm_stats['y_mean_att']

        # Apply physical constraints
        cbf_denorm = torch.clamp(cbf_denorm, min=0.0, max=300.0)
        att_denorm = torch.clamp(att_denorm, min=0.0, max=5000.0)

        cbf_maps.append(cbf_denorm.cpu().numpy())
        att_maps.append(att_denorm.cpu().numpy())

    # Ensemble average: each is (1, 1, H, W) -> mean -> (H, W)
    cbf_avg = np.mean(cbf_maps, axis=0)[0, 0]
    att_avg = np.mean(att_maps, axis=0)[0, 0]

    return cbf_avg, att_avg


def run_cbf_linearity_test(models: List[torch.nn.Module],
                            config: dict,
                            norm_stats: dict,
                            device: torch.device,
                            n_repeats: int = 5) -> dict:
    """
    Test 1: CBF Linearity

    Generate uniform phantoms with known CBF values and check if
    predicted CBF tracks true CBF linearly.
    """
    logger.info("=" * 60)
    logger.info("TEST 1: CBF Linearity")
    logger.info("=" * 60)

    cbf_levels = [10, 20, 40, 60, 80, 100, 120, 150]
    fixed_att = 1500.0  # ms (isolate CBF sensitivity)
    plds = np.array(config['pld_values'], dtype=np.float64)

    results = {
        'true_cbf': [],
        'pred_cbf_mean': [],
        'pred_cbf_std': [],
        'pred_att_mean': [],
        'pred_att_std': [],
    }

    for cbf_val in cbf_levels:
        cbf_preds_all = []
        att_preds_all = []

        for repeat in range(n_repeats):
            # Generate phantom with unique noise seed
            noisy_signals, _, _ = generate_uniform_phantom(
                cbf_value=float(cbf_val),
                att_value=fixed_att,
                plds=plds,
                config=config,
                size=64,
                snr=10.0,
                seed=repeat * 1000 + cbf_val
            )

            # Preprocess
            preprocessed = preprocess_signals(noisy_signals, config)

            # Create input tensor: (1, C, H, W)
            input_tensor = torch.from_numpy(preprocessed[np.newaxis, ...]).float().to(device)

            # Run inference
            cbf_map, att_map = run_ensemble_inference(models, input_tensor, norm_stats, device)

            # Take mean over the central region (avoid edge effects)
            margin = 8
            cbf_center = cbf_map[margin:-margin, margin:-margin]
            att_center = att_map[margin:-margin, margin:-margin]

            cbf_preds_all.append(np.mean(cbf_center))
            att_preds_all.append(np.mean(att_center))

        pred_cbf_mean = float(np.mean(cbf_preds_all))
        pred_cbf_std = float(np.std(cbf_preds_all))
        pred_att_mean = float(np.mean(att_preds_all))
        pred_att_std = float(np.std(att_preds_all))

        results['true_cbf'].append(cbf_val)
        results['pred_cbf_mean'].append(pred_cbf_mean)
        results['pred_cbf_std'].append(pred_cbf_std)
        results['pred_att_mean'].append(pred_att_mean)
        results['pred_att_std'].append(pred_att_std)

        logger.info(
            f"  CBF={cbf_val:>4d} -> Pred CBF={pred_cbf_mean:>7.2f} +/- {pred_cbf_std:>5.2f}, "
            f"Pred ATT={pred_att_mean:>7.1f} +/- {pred_att_std:>5.1f}"
        )

    # Compute R^2 and linear fit
    true_cbf = np.array(results['true_cbf'], dtype=np.float64)
    pred_cbf = np.array(results['pred_cbf_mean'], dtype=np.float64)

    # Linear regression: pred = slope * true + intercept
    slope, intercept = np.polyfit(true_cbf, pred_cbf, 1)

    # R^2
    ss_res = np.sum((pred_cbf - (slope * true_cbf + intercept))**2)
    ss_tot = np.sum((pred_cbf - np.mean(pred_cbf))**2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)

    # Also compute R^2 relative to identity line (pred vs true)
    ss_res_identity = np.sum((pred_cbf - true_cbf)**2)
    ss_tot_identity = np.sum((true_cbf - np.mean(true_cbf))**2)
    r_squared_identity = 1.0 - ss_res_identity / (ss_tot_identity + 1e-12)

    results['linear_fit'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared),
        'r_squared_vs_identity': float(r_squared_identity),
    }

    logger.info(f"\n  Linear Fit Results:")
    logger.info(f"    Slope:     {slope:.4f} (ideal: 1.0)")
    logger.info(f"    Intercept: {intercept:.2f} (ideal: 0.0)")
    logger.info(f"    R^2 (fit): {r_squared:.4f}")
    logger.info(f"    R^2 (identity): {r_squared_identity:.4f}")

    # Determine if model is amplitude-sensitive
    # A flat model would have slope near 0 and low R^2
    is_sensitive = slope > 0.3 and r_squared > 0.8
    results['amplitude_sensitive'] = bool(is_sensitive)
    logger.info(f"    Amplitude Sensitive: {'YES' if is_sensitive else 'NO'}")

    return results


def run_amplitude_scaling_test(models: List[torch.nn.Module],
                                config: dict,
                                norm_stats: dict,
                                device: torch.device,
                                n_repeats: int = 5) -> dict:
    """
    Test 2: Amplitude Scaling

    Take a fixed CBF=60 phantom and scale the input signals by various factors.
    Check if predicted CBF scales proportionally (amplitude-aware) or stays constant.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: Amplitude Scaling")
    logger.info("=" * 60)

    fixed_cbf = 60.0
    fixed_att = 1500.0
    scale_factors = [0.5, 0.75, 1.0, 1.5, 2.0]
    plds = np.array(config['pld_values'], dtype=np.float64)

    results = {
        'scale_factors': scale_factors,
        'pred_cbf_mean': [],
        'pred_cbf_std': [],
        'pred_att_mean': [],
        'pred_att_std': [],
    }

    # Use base prediction at scale=1.0 for ratio computation
    base_cbf_pred = None

    for scale in scale_factors:
        cbf_preds_all = []
        att_preds_all = []

        for repeat in range(n_repeats):
            # Generate phantom (same noise pattern across scales for fair comparison)
            noisy_signals, _, _ = generate_uniform_phantom(
                cbf_value=fixed_cbf,
                att_value=fixed_att,
                plds=plds,
                config=config,
                size=64,
                snr=10.0,
                seed=repeat * 1000 + 999
            )

            # Scale the signals BEFORE preprocessing
            scaled_signals = noisy_signals * scale

            # Preprocess
            preprocessed = preprocess_signals(scaled_signals, config)

            # Create input tensor
            input_tensor = torch.from_numpy(preprocessed[np.newaxis, ...]).float().to(device)

            # Run inference
            cbf_map, att_map = run_ensemble_inference(models, input_tensor, norm_stats, device)

            margin = 8
            cbf_center = cbf_map[margin:-margin, margin:-margin]
            att_center = att_map[margin:-margin, margin:-margin]

            cbf_preds_all.append(np.mean(cbf_center))
            att_preds_all.append(np.mean(att_center))

        pred_cbf_mean = float(np.mean(cbf_preds_all))
        pred_cbf_std = float(np.std(cbf_preds_all))
        pred_att_mean = float(np.mean(att_preds_all))
        pred_att_std = float(np.std(att_preds_all))

        results['pred_cbf_mean'].append(pred_cbf_mean)
        results['pred_cbf_std'].append(pred_cbf_std)
        results['pred_att_mean'].append(pred_att_mean)
        results['pred_att_std'].append(pred_att_std)

        if scale == 1.0:
            base_cbf_pred = pred_cbf_mean

        logger.info(
            f"  Scale={scale:.2f} -> Pred CBF={pred_cbf_mean:>7.2f} +/- {pred_cbf_std:>5.2f}, "
            f"Pred ATT={pred_att_mean:>7.1f} +/- {pred_att_std:>5.1f}"
        )

    # Compute sensitivity ratio (as in the original ablation)
    pred_cbf_arr = np.array(results['pred_cbf_mean'])
    if base_cbf_pred is not None and abs(base_cbf_pred) > 1e-6:
        # Ratio of predicted CBF at max scale to min scale
        ratio_max_min = pred_cbf_arr[-1] / (pred_cbf_arr[0] + 1e-6)
        # Compare with true ratio (should be 2.0/0.5 = 4.0 for linear scaling)
        expected_ratio = scale_factors[-1] / scale_factors[0]
    else:
        ratio_max_min = 0.0
        expected_ratio = scale_factors[-1] / scale_factors[0]

    results['sensitivity_ratio'] = float(ratio_max_min)
    results['expected_ratio'] = float(expected_ratio)

    # Linear fit of pred_cbf vs scale_factor
    scales_arr = np.array(scale_factors)
    if len(scales_arr) >= 2:
        slope_s, intercept_s = np.polyfit(scales_arr, pred_cbf_arr, 1)
        ss_res_s = np.sum((pred_cbf_arr - (slope_s * scales_arr + intercept_s))**2)
        ss_tot_s = np.sum((pred_cbf_arr - np.mean(pred_cbf_arr))**2)
        r2_s = 1.0 - ss_res_s / (ss_tot_s + 1e-12)
    else:
        slope_s, intercept_s, r2_s = 0.0, 0.0, 0.0

    results['scale_fit'] = {
        'slope': float(slope_s),
        'intercept': float(intercept_s),
        'r_squared': float(r2_s),
    }

    is_scale_sensitive = ratio_max_min > 2.0  # At least 2x change for 4x input change
    results['scale_sensitive'] = bool(is_scale_sensitive)

    logger.info(f"\n  Scaling Results:")
    logger.info(f"    Prediction ratio (2.0x / 0.5x input): {ratio_max_min:.2f} "
                f"(expected {expected_ratio:.1f} for linear)")
    logger.info(f"    Scale fit slope: {slope_s:.2f}, R^2: {r2_s:.4f}")
    logger.info(f"    Scale Sensitive: {'YES' if is_scale_sensitive else 'NO'}")

    return results


def generate_plots(linearity_results: dict, scaling_results: dict,
                   config: dict, output_dir: Path):
    """Generate diagnostic plots and save to output directory."""
    model_name = config.get('model_class_name', 'Unknown')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Amplitude Sensitivity: {model_name}", fontsize=14, fontweight='bold')

    # --- Plot 1: Predicted vs True CBF ---
    ax = axes[0]
    true_cbf = np.array(linearity_results['true_cbf'])
    pred_cbf = np.array(linearity_results['pred_cbf_mean'])
    pred_std = np.array(linearity_results['pred_cbf_std'])

    ax.errorbar(true_cbf, pred_cbf, yerr=pred_std, fmt='o-', capsize=4,
                color='blue', linewidth=2, markersize=8, label='Model Predictions')

    # Identity line
    max_val = max(np.max(true_cbf), np.max(pred_cbf)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Identity (y=x)')

    # Linear fit line
    fit = linearity_results['linear_fit']
    x_fit = np.linspace(0, max_val, 100)
    ax.plot(x_fit, fit['slope'] * x_fit + fit['intercept'], 'r-', alpha=0.7,
            label=f"Fit: y={fit['slope']:.2f}x + {fit['intercept']:.1f}")

    ax.set_xlabel('True CBF (ml/100g/min)', fontsize=11)
    ax.set_ylabel('Predicted CBF (ml/100g/min)', fontsize=11)
    ax.set_title(f"CBF Linearity\nR$^2$={fit['r_squared']:.3f}, "
                 f"slope={fit['slope']:.3f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # --- Plot 2: Predicted ATT at different CBF levels ---
    ax = axes[1]
    pred_att = np.array(linearity_results['pred_att_mean'])
    pred_att_std = np.array(linearity_results['pred_att_std'])

    ax.errorbar(true_cbf, pred_att, yerr=pred_att_std, fmt='s-', capsize=4,
                color='green', linewidth=2, markersize=8)
    ax.axhline(y=1500.0, color='k', linestyle='--', alpha=0.5, label='True ATT=1500ms')
    ax.set_xlabel('True CBF (ml/100g/min)', fontsize=11)
    ax.set_ylabel('Predicted ATT (ms)', fontsize=11)
    ax.set_title('ATT Stability Across CBF Levels', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Amplitude Scaling Test ---
    ax = axes[2]
    scales = np.array(scaling_results['scale_factors'])
    pred_cbf_s = np.array(scaling_results['pred_cbf_mean'])
    pred_std_s = np.array(scaling_results['pred_cbf_std'])

    ax.errorbar(scales, pred_cbf_s, yerr=pred_std_s, fmt='D-', capsize=4,
                color='purple', linewidth=2, markersize=8, label='Model Predictions')

    # Expected linear scaling reference
    base_idx = list(scaling_results['scale_factors']).index(1.0)
    base_val = pred_cbf_s[base_idx]
    ax.plot(scales, base_val * scales, 'k--', alpha=0.5, label='Linear Scaling')

    # Constant prediction reference
    ax.axhline(y=base_val, color='gray', linestyle=':', alpha=0.5, label='Constant (no sensitivity)')

    ax.set_xlabel('Input Amplitude Scale Factor', fontsize=11)
    ax.set_ylabel('Predicted CBF (ml/100g/min)', fontsize=11)
    ax.set_title(f"Amplitude Scaling\nRatio(2x/0.5x)={scaling_results['sensitivity_ratio']:.2f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'amplitude_sensitivity_realistic.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test amplitude sensitivity using realistic ASL inputs"
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--output_dir', type=str, default='results/amplitude_sensitivity',
                        help='Output directory for results')
    parser.add_argument('--snr', type=float, default=10.0,
                        help='Signal-to-noise ratio for phantom generation (default: 10)')
    parser.add_argument('--n_repeats', type=int, default=5,
                        help='Number of noise repeats per condition (default: 5)')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('cpu')  # MPS can be unstable for these models
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Load config, norm_stats, and models
    try:
        config = load_config(run_dir)
        norm_stats = load_norm_stats(run_dir)
        models = load_ensemble(run_dir, config, device)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return 1

    model_class = config.get('model_class_name', 'Unknown')
    logger.info(f"\nModel: {model_class}")
    logger.info(f"Normalization: {config.get('normalization_mode', 'unknown')}")
    logger.info(f"PLDs: {config['pld_values']}")

    # Run Test 1: CBF Linearity
    linearity_results = run_cbf_linearity_test(
        models, config, norm_stats, device, n_repeats=args.n_repeats
    )

    # Run Test 2: Amplitude Scaling
    scaling_results = run_amplitude_scaling_test(
        models, config, norm_stats, device, n_repeats=args.n_repeats
    )

    # Generate plots
    generate_plots(linearity_results, scaling_results, config, output_dir)

    # Assemble final results
    final_results = {
        'model_class': model_class,
        'run_dir': str(run_dir),
        'normalization_mode': config.get('normalization_mode', 'unknown'),
        'global_scale_factor': config.get('global_scale_factor', None),
        'pld_values': config['pld_values'],
        'snr': args.snr,
        'n_repeats': args.n_repeats,
        'test1_cbf_linearity': linearity_results,
        'test2_amplitude_scaling': scaling_results,
        'summary': {
            'cbf_linearity_r2': linearity_results['linear_fit']['r_squared'],
            'cbf_linearity_slope': linearity_results['linear_fit']['slope'],
            'cbf_linearity_r2_vs_identity': linearity_results['linear_fit']['r_squared_vs_identity'],
            'amplitude_sensitive_linearity': linearity_results['amplitude_sensitive'],
            'amplitude_scaling_ratio': scaling_results['sensitivity_ratio'],
            'amplitude_scale_sensitive': scaling_results['scale_sensitive'],
        }
    }

    # Save results to JSON
    json_path = output_dir / 'amplitude_sensitivity_realistic.json'
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Saved results to {json_path}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {model_class}")
    logger.info(f"Normalization: {config.get('normalization_mode', 'unknown')}")
    logger.info("")
    logger.info("Test 1 - CBF Linearity:")
    logger.info(f"  Slope:  {linearity_results['linear_fit']['slope']:.4f} (ideal: 1.0)")
    logger.info(f"  R^2:    {linearity_results['linear_fit']['r_squared']:.4f} (ideal: 1.0)")
    logger.info(f"  R^2 vs identity: {linearity_results['linear_fit']['r_squared_vs_identity']:.4f}")
    logger.info(f"  Amplitude Sensitive: "
                f"{'YES' if linearity_results['amplitude_sensitive'] else 'NO'}")
    logger.info("")
    logger.info("Test 2 - Amplitude Scaling:")
    logger.info(f"  Ratio (2x/0.5x):  {scaling_results['sensitivity_ratio']:.2f} "
                f"(expected: {scaling_results['expected_ratio']:.1f} for linear)")
    logger.info(f"  Scale Sensitive: "
                f"{'YES' if scaling_results['scale_sensitive'] else 'NO'}")
    logger.info("")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())

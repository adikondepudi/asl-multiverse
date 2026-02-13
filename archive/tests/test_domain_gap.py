#!/usr/bin/env python3
"""
Domain Gap Test Script
======================
Tests whether domain randomization during training helps generalization
to unseen physics parameter settings.

Generates two test sets:
- Set A (training-matched): Standard physics parameters
- Set B (domain-shifted): Out-of-distribution physics parameters

Evaluates a trained spatial model on both sets and reports degradation ratio.
A domain-robust model should have MAE_B / MAE_A close to 1.0.

Usage:
    python test_domain_gap.py --run_dir <path> --output_dir results/domain_gap/
"""

import sys
import os
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Imports from the project
# ------------------------------------------------------------------
from asl_simulation import ASLParameters, ASLSimulator
from enhanced_simulation import SpatialPhantomGenerator
from spatial_asl_network import SpatialASLNet, DualEncoderSpatialASLNet
from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_model_ensemble(run_dir: Path, device: torch.device):
    """
    Load a trained spatial model ensemble from *run_dir*.

    Returns:
        models: list of nn.Module in eval mode
        config: dict from research_config.json
        norm_stats: dict from norm_stats.json
        training_config: dict from config.yaml training section
    """
    # 1. Load research config
    config_path = run_dir / 'research_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"research_config.json not found in {run_dir}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. Load norm stats
    norm_stats_path = run_dir / 'norm_stats.json'
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"norm_stats.json not found in {run_dir}")
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    # 3. Load training config (config.yaml)
    training_config = {}
    config_yaml_path = run_dir / 'config.yaml'
    if config_yaml_path.exists():
        with open(config_yaml_path, 'r') as f:
            full_yaml = yaml.safe_load(f)
            training_config = full_yaml.get('training', {})

    # 4. Determine model class and architecture
    model_class_name = training_config.get('model_class_name',
                                            config.get('model_class_name', 'SpatialASLNet'))
    hidden_sizes = training_config.get('hidden_sizes',
                                        config.get('hidden_sizes', [32, 64, 128, 256]))
    features = sorted(hidden_sizes) if len(hidden_sizes) >= 4 else [32, 64, 128, 256]
    while len(features) < 4:
        features.insert(0, max(1, features[0] // 2))

    plds = np.array(config.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000]))
    n_plds = len(plds)

    # 5. Find model files
    models_dir = run_dir / 'trained_models'
    if not models_dir.exists():
        models_dir = run_dir
    model_files = sorted(list(models_dir.glob('ensemble_model_*.pt')))
    if not model_files:
        raise FileNotFoundError(f"No ensemble_model_*.pt found in {models_dir}")

    logger.info(f"Loading {len(model_files)} {model_class_name} models from {models_dir}")

    # 6. Load each model
    loaded_models = []
    for mp in model_files:
        if model_class_name == 'AmplitudeAwareSpatialASLNet':
            model = AmplitudeAwareSpatialASLNet(
                n_plds=n_plds,
                features=features,
                use_film_at_bottleneck=training_config.get('use_film_at_bottleneck', True),
                use_film_at_decoder=training_config.get('use_film_at_decoder', True),
                use_amplitude_output_modulation=training_config.get('use_amplitude_output_modulation', True),
            )
        elif model_class_name == 'DualEncoderSpatialASLNet':
            model = DualEncoderSpatialASLNet(n_plds=n_plds, features=features)
        else:
            model = SpatialASLNet(n_plds=n_plds, features=features)

        state_dict = torch.load(mp, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        loaded_models.append(model)

    return loaded_models, config, norm_stats, training_config


def generate_spatial_test_set(
    plds: np.ndarray,
    physics: Dict,
    n_phantoms: int = 50,
    phantom_size: int = 64,
    snr: float = 10.0,
    seed_offset: int = 0,
) -> Dict:
    """
    Generate a spatial test set with specific physics parameters.

    Args:
        plds: PLD values in ms (1D array)
        physics: Dict with keys T1_artery, alpha_PCASL, alpha_VSASL, T_tau, alpha_BS1
        n_phantoms: Number of phantoms to generate
        phantom_size: Spatial size of each phantom (phantom_size x phantom_size)
        snr: Signal-to-noise ratio for noise injection
        seed_offset: Seed offset for reproducibility

    Returns:
        Dict with 'signals', 'cbf_maps', 'att_maps', 'masks' (all lists of arrays)
    """
    params = ASLParameters(
        T1_artery=physics['T1_artery'],
        T_tau=physics.get('T_tau', 1800.0),
        alpha_PCASL=physics['alpha_PCASL'],
        alpha_VSASL=physics['alpha_VSASL'],
        alpha_BS1=physics.get('alpha_BS1', 1.0),
    )
    simulator = ASLSimulator(params=params)
    phantom_gen = SpatialPhantomGenerator(size=phantom_size, pve_sigma=1.0)

    signals_list = []
    cbf_maps = []
    att_maps = []
    masks = []

    for idx in range(n_phantoms):
        np.random.seed(seed_offset + idx)

        # Use standard phantom generation (consistent between sets A and B
        # so only the physics parameters change)
        cbf_map, att_map, _ = phantom_gen.generate_phantom(include_pathology=True)

        mask = (cbf_map > 1.0).astype(np.float32)

        # Generate clean signals voxel-by-voxel
        n_plds = len(plds)
        sig = np.zeros((n_plds * 2, phantom_size, phantom_size), dtype=np.float32)

        for i in range(phantom_size):
            for j in range(phantom_size):
                if mask[i, j] > 0:
                    cbf_val = float(cbf_map[i, j])
                    att_val = float(att_map[i, j])
                    pcasl = simulator._generate_pcasl_signal(
                        plds, att_val, cbf_val,
                        params.T1_artery, params.T_tau, params.alpha_PCASL
                    )
                    vsasl = simulator._generate_vsasl_signal(
                        plds, att_val, cbf_val,
                        params.T1_artery, params.alpha_VSASL
                    )
                    sig[:n_plds, i, j] = pcasl
                    sig[n_plds:, i, j] = vsasl

        # Add Rician noise
        mean_sig = np.mean(np.abs(sig[sig != 0])) if np.any(sig != 0) else 1e-6
        sigma = mean_sig / snr
        noise_r = np.random.normal(0, sigma, sig.shape).astype(np.float32)
        noise_i = np.random.normal(0, sigma, sig.shape).astype(np.float32)
        noisy_sig = np.sqrt((sig + noise_r)**2 + noise_i**2)

        signals_list.append(noisy_sig)
        cbf_maps.append(cbf_map)
        att_maps.append(att_map)
        masks.append(mask)

    return {
        'signals': signals_list,
        'cbf_maps': cbf_maps,
        'att_maps': att_maps,
        'masks': masks,
    }


def evaluate_on_test_set(
    models: List[torch.nn.Module],
    test_set: Dict,
    norm_stats: Dict,
    config: Dict,
    device: torch.device,
) -> Dict:
    """
    Run ensemble inference on a test set and compute CBF / ATT metrics.

    Returns:
        Dict with MAE, Bias, RMSE, and per-phantom metrics for CBF and ATT.
    """
    normalization_mode = config.get('normalization_mode', 'per_curve')
    global_scale_factor = config.get('global_scale_factor', 1.0)
    M0_SCALE = 100.0

    all_cbf_errors = []
    all_att_errors = []
    all_cbf_biases = []
    all_att_biases = []
    per_phantom_cbf_mae = []
    per_phantom_att_mae = []

    for idx in range(len(test_set['signals'])):
        noisy_signals = test_set['signals'][idx]
        cbf_true = test_set['cbf_maps'][idx]
        att_true = test_set['att_maps'][idx]
        mask = test_set['masks'][idx]

        # Apply M0 scaling
        scaled = noisy_signals * M0_SCALE

        # Apply normalization matching training
        if normalization_mode == 'global_scale':
            normalized = scaled * global_scale_factor
        else:
            temporal_mean = np.mean(scaled, axis=0, keepdims=True)
            temporal_std = np.std(scaled, axis=0, keepdims=True) + 1e-6
            normalized = (scaled - temporal_mean) / temporal_std

        input_tensor = torch.from_numpy(normalized[np.newaxis, ...]).float().to(device)

        with torch.no_grad():
            cbf_preds, att_preds = [], []
            for model in models:
                cbf_out, att_out, _, _ = model(input_tensor)
                cbf_denorm = cbf_out * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
                att_denorm = att_out * norm_stats['y_std_att'] + norm_stats['y_mean_att']
                cbf_denorm = torch.clamp(cbf_denorm, min=0.0, max=200.0)
                att_denorm = torch.clamp(att_denorm, min=0.0, max=5000.0)
                cbf_preds.append(cbf_denorm.cpu().numpy())
                att_preds.append(att_denorm.cpu().numpy())

            cbf_pred = np.mean(cbf_preds, axis=0)[0, 0]  # (H, W)
            att_pred = np.mean(att_preds, axis=0)[0, 0]

        # Compute errors on brain voxels
        brain = mask > 0
        if brain.sum() == 0:
            continue

        cbf_err = np.abs(cbf_pred[brain] - cbf_true[brain])
        att_err = np.abs(att_pred[brain] - att_true[brain])
        cbf_bias = cbf_pred[brain] - cbf_true[brain]
        att_bias = att_pred[brain] - att_true[brain]

        all_cbf_errors.extend(cbf_err.tolist())
        all_att_errors.extend(att_err.tolist())
        all_cbf_biases.extend(cbf_bias.tolist())
        all_att_biases.extend(att_bias.tolist())
        per_phantom_cbf_mae.append(float(np.mean(cbf_err)))
        per_phantom_att_mae.append(float(np.mean(att_err)))

    all_cbf_errors = np.array(all_cbf_errors)
    all_att_errors = np.array(all_att_errors)
    all_cbf_biases = np.array(all_cbf_biases)
    all_att_biases = np.array(all_att_biases)

    return {
        'CBF_MAE': float(np.mean(all_cbf_errors)),
        'CBF_Bias': float(np.mean(all_cbf_biases)),
        'CBF_RMSE': float(np.sqrt(np.mean(all_cbf_errors**2))),
        'ATT_MAE': float(np.mean(all_att_errors)),
        'ATT_Bias': float(np.mean(all_att_biases)),
        'ATT_RMSE': float(np.sqrt(np.mean(all_att_errors**2))),
        'per_phantom_CBF_MAE': per_phantom_cbf_mae,
        'per_phantom_ATT_MAE': per_phantom_att_mae,
        'n_phantoms': len(per_phantom_cbf_mae),
        'n_voxels': len(all_cbf_errors),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test domain gap: evaluate model robustness to shifted physics parameters."
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to trained model run directory')
    parser.add_argument('--output_dir', type=str, default='results/domain_gap',
                        help='Output directory for results JSON')
    parser.add_argument('--n_phantoms', type=int, default=50,
                        help='Number of phantoms per test set (default 50)')
    parser.add_argument('--snr', type=float, default=10.0,
                        help='SNR for test data (default 10.0)')
    parser.add_argument('--phantom_size', type=int, default=64,
                        help='Spatial size of phantoms (default 64)')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('cpu')  # MPS can be unstable
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # --- Load model ---
    logger.info(f"Loading model from {run_dir}")
    models, config, norm_stats, training_config = load_model_ensemble(run_dir, device)
    plds = np.array(config.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000]), dtype=np.float64)
    logger.info(f"Loaded {len(models)} ensemble members, PLDs: {plds}")

    # --- Define physics parameter sets ---
    physics_A = {
        'T1_artery': 1650.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T_tau': 1800.0,
        'alpha_BS1': 1.0,
        'label': 'Training-Matched (Standard)',
    }
    physics_B = {
        'T1_artery': 1450.0,       # Low hematocrit
        'alpha_PCASL': 0.70,       # Poor labeling
        'alpha_VSASL': 0.45,       # Reduced VS efficiency
        'T_tau': 1800.0,
        'alpha_BS1': 1.0,
        'label': 'Domain-Shifted (Low Hematocrit / Poor Labeling)',
    }

    # --- Generate test sets ---
    logger.info("=" * 60)
    logger.info("Generating test sets...")
    logger.info(f"  Set A: {physics_A['label']}")
    logger.info(f"    T1_artery={physics_A['T1_artery']}, alpha_PCASL={physics_A['alpha_PCASL']}, "
                f"alpha_VSASL={physics_A['alpha_VSASL']}")
    logger.info(f"  Set B: {physics_B['label']}")
    logger.info(f"    T1_artery={physics_B['T1_artery']}, alpha_PCASL={physics_B['alpha_PCASL']}, "
                f"alpha_VSASL={physics_B['alpha_VSASL']}")

    test_set_A = generate_spatial_test_set(
        plds=plds,
        physics=physics_A,
        n_phantoms=args.n_phantoms,
        phantom_size=args.phantom_size,
        snr=args.snr,
        seed_offset=5000,
    )
    logger.info(f"  Set A generated: {args.n_phantoms} phantoms")

    test_set_B = generate_spatial_test_set(
        plds=plds,
        physics=physics_B,
        n_phantoms=args.n_phantoms,
        phantom_size=args.phantom_size,
        snr=args.snr,
        seed_offset=5000,  # Same seeds so phantoms are identical, only physics differ
    )
    logger.info(f"  Set B generated: {args.n_phantoms} phantoms")

    # --- Evaluate ---
    logger.info("=" * 60)
    logger.info("Evaluating on Set A (training-matched)...")
    metrics_A = evaluate_on_test_set(models, test_set_A, norm_stats, config, device)
    logger.info(f"  CBF MAE: {metrics_A['CBF_MAE']:.2f} ml/100g/min")
    logger.info(f"  ATT MAE: {metrics_A['ATT_MAE']:.0f} ms")

    logger.info("Evaluating on Set B (domain-shifted)...")
    metrics_B = evaluate_on_test_set(models, test_set_B, norm_stats, config, device)
    logger.info(f"  CBF MAE: {metrics_B['CBF_MAE']:.2f} ml/100g/min")
    logger.info(f"  ATT MAE: {metrics_B['ATT_MAE']:.0f} ms")

    # --- Compute degradation ratios ---
    cbf_degradation = metrics_B['CBF_MAE'] / max(metrics_A['CBF_MAE'], 1e-6)
    att_degradation = metrics_B['ATT_MAE'] / max(metrics_A['ATT_MAE'], 1e-6)

    # --- Report ---
    logger.info("=" * 60)
    logger.info("DOMAIN GAP RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} {'Set A (Matched)':<20} {'Set B (Shifted)':<20} {'Degradation':>12}")
    logger.info("-" * 72)
    logger.info(f"{'CBF MAE':<20} {metrics_A['CBF_MAE']:<20.2f} {metrics_B['CBF_MAE']:<20.2f} {cbf_degradation:>10.2f}x")
    logger.info(f"{'CBF Bias':<20} {metrics_A['CBF_Bias']:<20.2f} {metrics_B['CBF_Bias']:<20.2f} {'':>12}")
    logger.info(f"{'ATT MAE':<20} {metrics_A['ATT_MAE']:<20.0f} {metrics_B['ATT_MAE']:<20.0f} {att_degradation:>10.2f}x")
    logger.info(f"{'ATT Bias':<20} {metrics_A['ATT_Bias']:<20.0f} {metrics_B['ATT_Bias']:<20.0f} {'':>12}")
    logger.info("-" * 72)

    if cbf_degradation < 1.5:
        logger.info("CBF: Model is DOMAIN-ROBUST (degradation < 1.5x)")
    elif cbf_degradation < 2.0:
        logger.info("CBF: Model shows MODERATE domain sensitivity (1.5-2.0x)")
    else:
        logger.info(f"CBF: Model is DOMAIN-SENSITIVE (degradation {cbf_degradation:.1f}x)")

    if att_degradation < 1.5:
        logger.info("ATT: Model is DOMAIN-ROBUST (degradation < 1.5x)")
    elif att_degradation < 2.0:
        logger.info("ATT: Model shows MODERATE domain sensitivity (1.5-2.0x)")
    else:
        logger.info(f"ATT: Model is DOMAIN-SENSITIVE (degradation {att_degradation:.1f}x)")

    # --- Save results ---
    results = {
        'run_dir': str(run_dir),
        'model_class': training_config.get('model_class_name', config.get('model_class_name', 'SpatialASLNet')),
        'domain_randomization_enabled': training_config.get('domain_randomization', config.get('domain_randomization', {})).get('enabled', False)
            if isinstance(training_config.get('domain_randomization', config.get('domain_randomization', {})), dict)
            else False,
        'test_config': {
            'n_phantoms': args.n_phantoms,
            'snr': args.snr,
            'phantom_size': args.phantom_size,
        },
        'physics_A': {k: v for k, v in physics_A.items() if k != 'label'},
        'physics_A_label': physics_A['label'],
        'physics_B': {k: v for k, v in physics_B.items() if k != 'label'},
        'physics_B_label': physics_B['label'],
        'metrics_A': {k: v for k, v in metrics_A.items() if not k.startswith('per_phantom')},
        'metrics_B': {k: v for k, v in metrics_B.items() if not k.startswith('per_phantom')},
        'degradation': {
            'CBF_MAE_ratio': float(cbf_degradation),
            'ATT_MAE_ratio': float(att_degradation),
        },
        'interpretation': {
            'CBF_robust': cbf_degradation < 1.5,
            'ATT_robust': att_degradation < 1.5,
        },
        'per_phantom_detail': {
            'A_CBF_MAE': metrics_A['per_phantom_CBF_MAE'],
            'A_ATT_MAE': metrics_A['per_phantom_ATT_MAE'],
            'B_CBF_MAE': metrics_B['per_phantom_CBF_MAE'],
            'B_ATT_MAE': metrics_B['per_phantom_ATT_MAE'],
        },
    }

    results_path = output_dir / 'domain_gap_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n--- [CANCELLED] User stopped the script. ---")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! [FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

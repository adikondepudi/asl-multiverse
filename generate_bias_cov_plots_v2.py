#!/usr/bin/env python3
"""
Generate Bias & CoV (Coefficient of Variation) plots for ASL parameter estimation.

Produces publication-ready figures comparing NN models vs LS baseline across:
  Sweep A: Fixed CBF=50, ATT varying 500→3000 ms
  Sweep B: Fixed ATT=1500, CBF varying 20→120 ml/100g/min

Usage:
    python generate_bias_cov_plots_v2.py --output-dir bias_cov_results_v2
    python generate_bias_cov_plots_v2.py --output-dir bias_cov_results_v2 --snr 3.0 5.0 10.0
    python generate_bias_cov_plots_v2.py --output-dir bias_cov_results_v2 --n-phantoms 10 --n-ls-realizations 1000
"""

import argparse
import json
import sys
import os
import time
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation.asl_simulation import ASLParameters, ASLSimulator, _generate_pcasl_signal_jit, _generate_vsasl_signal_jit
from models.spatial_asl_network import SpatialASLNet, DualEncoderSpatialASLNet, CapacityMatchedSpatialASLNet
from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
from utils.helpers import get_grid_search_initial_guess
from baselines.multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

# ── Physics constants (match training configs) ──────────────────────────────
PLDS = np.array([500, 1000, 1500, 2000, 2500, 3000], dtype=np.float64)
T1_ARTERY = 1850.0  # All trained models used this
T_TAU = 1800.0
ALPHA_PCASL = 0.85
ALPHA_VSASL = 0.56
ALPHA_BS1 = 1.0
T2_FACTOR = 1.0
T_SAT_VS = 2000.0
PHANTOM_SIZE = 64
MASK_RADIUS = 28

# ── Model registry ──────────────────────────────────────────────────────────
MODELS = [
    {
        'name': 'Spatial U-Net (Exp 00)',
        'dir': 'amplitude_ablation_v1/00_Baseline_SpatialASL',
        'color': '#1f77b4',
        'marker': 'o',
    },
    {
        'name': 'Amplitude-Aware (Exp 02)',
        'dir': 'amplitude_ablation_v1/02_AmpAware_Full',
        'color': '#d62728',
        'marker': 's',
    },
    {
        'name': 'ATT-Rebalanced (Exp 14)',
        'dir': 'amplitude_ablation_v2/14_ATT_Rebalanced',
        'color': '#2ca02c',
        'marker': '^',
    },
]

# ── Model class registry for loading ────────────────────────────────────────
MODEL_CLASS_MAP = {
    'SpatialASLNet': SpatialASLNet,
    'AmplitudeAwareSpatialASLNet': AmplitudeAwareSpatialASLNet,
    'DualEncoderSpatialASLNet': DualEncoderSpatialASLNet,
    'CapacityMatchedSpatialASLNet': CapacityMatchedSpatialASLNet,
}


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model_ensemble(run_dir, device):
    """Load a spatial model ensemble from a run directory."""
    run_dir = Path(run_dir)
    config_path = run_dir / 'config.yaml'
    norm_stats_path = run_dir / 'norm_stats.json'
    models_dir = run_dir / 'trained_models'

    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    training_config = full_config.get('training', {})
    data_config = full_config.get('data', {})

    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    model_class_name = training_config.get('model_class_name', 'SpatialASLNet')
    hidden_sizes = training_config.get('hidden_sizes', [32, 64, 128, 256])
    n_plds = len(data_config.get('pld_values', PLDS))

    model_files = sorted(models_dir.glob('ensemble_model_*.pt'))
    if not model_files:
        raise FileNotFoundError(f"No model files in {models_dir}")

    models = []
    for mf in model_files:
        if model_class_name == 'AmplitudeAwareSpatialASLNet':
            model = AmplitudeAwareSpatialASLNet(
                n_plds=n_plds,
                features=hidden_sizes,
                use_film_at_bottleneck=training_config.get('use_film_at_bottleneck', True),
                use_film_at_decoder=training_config.get('use_film_at_decoder', True),
                use_amplitude_output_modulation=training_config.get('use_amplitude_output_modulation', True),
            )
        elif model_class_name == 'DualEncoderSpatialASLNet':
            model = DualEncoderSpatialASLNet(n_plds=n_plds, features=hidden_sizes)
        elif model_class_name == 'CapacityMatchedSpatialASLNet':
            model = CapacityMatchedSpatialASLNet(n_plds=n_plds, features=hidden_sizes)
        else:
            model = SpatialASLNet(n_plds=n_plds, features=hidden_sizes)

        state = torch.load(mf, map_location=device, weights_only=False)
        sd = state['model_state_dict'] if 'model_state_dict' in state else state
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        models.append(model)

    global_scale_factor = data_config.get('global_scale_factor', 10.0)
    normalization_mode = data_config.get('normalization_mode', 'global_scale')

    return models, norm_stats, global_scale_factor, normalization_mode


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Signal Generation
# ═══════════════════════════════════════════════════════════════════════════

def make_brain_mask(size=PHANTOM_SIZE, radius=MASK_RADIUS):
    """Circular brain mask to avoid U-Net edge effects."""
    yy, xx = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    return ((xx - cx)**2 + (yy - cy)**2 <= radius**2).astype(np.float32)


def generate_clean_phantom(cbf, att, mask):
    """Generate clean 12-channel phantom (6 PCASL + 6 VSASL) for uniform params."""
    plds = PLDS.copy()
    cbf_cgs = cbf / 6000.0
    alpha1 = ALPHA_PCASL * (ALPHA_BS1**4)
    alpha2 = ALPHA_VSASL * (ALPHA_BS1**3)

    pcasl = _generate_pcasl_signal_jit(plds, att, cbf_cgs, T1_ARTERY, T_TAU, alpha1, T2_FACTOR)
    vsasl = _generate_vsasl_signal_jit(plds, att, cbf_cgs, T1_ARTERY, alpha2, T2_FACTOR, T_SAT_VS)

    # Shape: (12, H, W) — broadcast uniform signal across all brain voxels
    n_plds = len(plds)
    signals = np.zeros((n_plds * 2, PHANTOM_SIZE, PHANTOM_SIZE), dtype=np.float32)
    for c in range(n_plds):
        signals[c] = pcasl[c] * mask
        signals[n_plds + c] = vsasl[c] * mask
    return signals


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 & 4: NN Inference
# ═══════════════════════════════════════════════════════════════════════════

def nn_inference(clean_signals, mask, models, norm_stats, global_scale_factor,
                 normalization_mode, snr, n_phantoms, device, rng):
    """Run NN inference over n_phantoms noise realizations, return (cbf_preds, att_preds)."""
    simulator = ASLSimulator(ASLParameters(
        T1_artery=T1_ARTERY, T_tau=T_TAU,
        alpha_PCASL=ALPHA_PCASL, alpha_VSASL=ALPHA_VSASL,
    ))
    ref_signal = simulator._compute_reference_signal()
    noise_sd = ref_signal / snr

    brain_idx = mask > 0
    n_brain = int(brain_idx.sum())

    all_cbf = []
    all_att = []

    for pi in range(n_phantoms):
        noise = noise_sd * rng.randn(*clean_signals.shape).astype(np.float32)
        noisy = clean_signals + noise

        # Preprocessing: M0 scale × 100, then global_scale
        noisy_scaled = noisy * 100.0
        if normalization_mode == 'global_scale':
            noisy_norm = noisy_scaled * global_scale_factor
        else:
            t_mean = np.mean(noisy_scaled, axis=0, keepdims=True)
            t_std = np.std(noisy_scaled, axis=0, keepdims=True) + 1e-6
            noisy_norm = (noisy_scaled - t_mean) / t_std

        inp = torch.from_numpy(noisy_norm[np.newaxis]).float().to(device)

        with torch.no_grad():
            cbf_maps, att_maps = [], []
            for model in models:
                cbf_pred, att_pred, _, _ = model(inp)
                cbf_d = cbf_pred * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
                att_d = att_pred * norm_stats['y_std_att'] + norm_stats['y_mean_att']
                cbf_d = torch.clamp(cbf_d, 0.0, 200.0)
                att_d = torch.clamp(att_d, 0.0, 5000.0)
                cbf_maps.append(cbf_d.cpu().numpy())
                att_maps.append(att_d.cpu().numpy())

            cbf_ens = np.mean(cbf_maps, axis=0)[0, 0]  # (H, W)
            att_ens = np.mean(att_maps, axis=0)[0, 0]

        all_cbf.append(cbf_ens[brain_idx])
        all_att.append(att_ens[brain_idx])

    return np.concatenate(all_cbf), np.concatenate(all_att)


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: LS Fitting
# ═══════════════════════════════════════════════════════════════════════════

def _fit_single_voxel(args):
    """Worker for parallel LS fitting."""
    signal_1d, plds_flat, ls_params, pldti = args
    try:
        init = get_grid_search_initial_guess(signal_1d, plds_flat, ls_params)
        signal_reshaped = signal_1d.reshape((len(plds_flat), 2), order='F')
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
            pldti, signal_reshaped, init, **ls_params
        )
        cbf = beta[0] * 6000.0
        att = beta[1]
        if np.isfinite(cbf) and np.isfinite(att):
            return cbf, att
    except Exception:
        pass
    return None


def ls_inference(clean_signals_1d, snr, n_realizations, rng):
    """Run LS fitting over n_realizations of 1D noise, return (cbf_array, att_array)."""
    simulator = ASLSimulator(ASLParameters(
        T1_artery=T1_ARTERY, T_tau=T_TAU,
        alpha_PCASL=ALPHA_PCASL, alpha_VSASL=ALPHA_VSASL,
    ))
    ref_signal = simulator._compute_reference_signal()
    noise_sd = ref_signal / snr

    plds_flat = PLDS.copy()
    pldti = np.column_stack([plds_flat, plds_flat])
    ls_params = {
        'T1_artery': T1_ARTERY, 'T_tau': T_TAU,
        'alpha_PCASL': ALPHA_PCASL, 'alpha_VSASL': ALPHA_VSASL,
        'T2_factor': T2_FACTOR, 'alpha_BS1': ALPHA_BS1,
    }

    tasks = []
    for _ in range(n_realizations):
        noise = noise_sd * rng.randn(len(clean_signals_1d)).astype(np.float64)
        noisy = clean_signals_1d + noise
        tasks.append((noisy, plds_flat, ls_params, pldti))

    cbf_list, att_list = [], []
    # Use multiprocessing for speed
    n_workers = min(8, os.cpu_count() or 1)
    with Pool(n_workers) as pool:
        results = pool.map(_fit_single_voxel, tasks)

    for r in results:
        if r is not None:
            cbf_list.append(r[0])
            att_list.append(r[1])

    return np.array(cbf_list), np.array(att_list)


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_bias_cov(preds, true_val):
    """Return (bias, cov_percent). CoV = std/true * 100."""
    bias = np.mean(preds) - true_val
    if true_val != 0:
        cov = np.std(preds) / abs(true_val) * 100.0
    else:
        cov = np.nan
    return bias, cov


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Plotting
# ═══════════════════════════════════════════════════════════════════════════

def make_bias_cov_figure(data, x_values, x_label, x_param_name, snr, output_dir):
    """Create a 2x2 figure: CBF Bias, CBF CoV, ATT Bias, ATT CoV."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(f'Bias & CoV vs {x_param_name} (SNR = {snr})', fontsize=14, fontweight='bold')

    panels = [
        (axes[0, 0], 'cbf_bias', f'CBF Bias (ml/100g/min)', True),
        (axes[0, 1], 'cbf_cov', f'CBF CoV (%)', False),
        (axes[1, 0], 'att_bias', f'ATT Bias (ms)', True),
        (axes[1, 1], 'att_cov', f'ATT CoV (%)', False),
    ]

    for ax, key, ylabel, show_zero in panels:
        for model_data in data:
            ax.plot(x_values, model_data[key],
                    color=model_data['color'],
                    marker=model_data['marker'],
                    markersize=5,
                    linewidth=1.8,
                    label=model_data['name'],
                    linestyle=model_data.get('linestyle', '-'))
        if show_zero:
            ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    slug = x_param_name.lower().replace(' ', '_')
    for fmt in ['pdf', 'png']:
        path = output_dir / f'bias_cov_vs_{slug}_snr{snr}.{fmt}'
        fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: bias_cov_vs_{slug}_snr{snr}.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate Bias & CoV plots for ASL models')
    parser.add_argument('--output-dir', type=str, default='bias_cov_results_v2')
    parser.add_argument('--snr', type=float, nargs='+', default=[3.0, 5.0, 10.0],
                        help='SNR levels to evaluate')
    parser.add_argument('--n-phantoms', type=int, default=10,
                        help='Number of spatial noise realizations per sweep point (NN)')
    parser.add_argument('--n-ls-realizations', type=int, default=1000,
                        help='Number of 1D noise realizations per sweep point (LS)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', type=str, nargs='*', default=None,
                        help='Additional model dirs to include (format: name:dir)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).parent.resolve()

    # Device — support MPS (Apple Silicon), CUDA, and CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Build model list (built-in + any extra from CLI)
    model_list = list(MODELS)
    if args.models:
        extra_colors = ['#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
        extra_markers = ['D', 'v', 'P', '*']
        for i, spec in enumerate(args.models):
            name, dir_path = spec.split(':', 1)
            model_list.append({
                'name': name,
                'dir': dir_path,
                'color': extra_colors[i % len(extra_colors)],
                'marker': extra_markers[i % len(extra_markers)],
            })

    # Sweep definitions
    att_sweep = np.arange(500, 3001, 100)    # 26 points
    cbf_sweep = np.arange(20, 121, 5)        # 21 points
    fixed_cbf = 50.0
    fixed_att = 1500.0

    mask = make_brain_mask()

    # ── Load all NN models ──────────────────────────────────────────────
    print("\n=== Loading NN models ===")
    loaded_models = []
    for minfo in model_list:
        run_dir = project_root / minfo['dir']
        if not run_dir.exists():
            print(f"  SKIP {minfo['name']}: {run_dir} not found")
            continue
        print(f"  Loading {minfo['name']} from {run_dir.name}...")
        try:
            models, norm_stats, gsf, norm_mode = load_model_ensemble(run_dir, device)
            loaded_models.append({
                'name': minfo['name'],
                'color': minfo['color'],
                'marker': minfo['marker'],
                'models': models,
                'norm_stats': norm_stats,
                'global_scale_factor': gsf,
                'normalization_mode': norm_mode,
            })
            print(f"    Loaded {len(models)} ensemble members ({norm_mode}, gsf={gsf})")
        except Exception as e:
            print(f"  ERROR loading {minfo['name']}: {e}")

    if not loaded_models:
        print("\nERROR: No models loaded successfully. Check paths.")
        sys.exit(1)

    # ── Run sweeps for each SNR ─────────────────────────────────────────
    all_csv_rows = []

    for snr in args.snr:
        print(f"\n{'='*60}")
        print(f"  SNR = {snr}")
        print(f"{'='*60}")

        rng = np.random.RandomState(args.seed + int(snr * 10))

        # ─── Sweep A: vary ATT, fixed CBF ────────────────────────────
        print(f"\n--- Sweep A: CBF={fixed_cbf}, ATT={att_sweep[0]}→{att_sweep[-1]} ---")

        sweep_a_data = []  # one dict per model (including LS)

        # NN models
        for minfo in loaded_models:
            print(f"  NN: {minfo['name']}...")
            cbf_biases, cbf_covs, att_biases, att_covs = [], [], [], []
            for att_val in att_sweep:
                clean = generate_clean_phantom(fixed_cbf, att_val, mask)
                cbf_preds, att_preds = nn_inference(
                    clean, mask, minfo['models'], minfo['norm_stats'],
                    minfo['global_scale_factor'], minfo['normalization_mode'],
                    snr, args.n_phantoms, device, rng,
                )
                cb, cc = compute_bias_cov(cbf_preds, fixed_cbf)
                ab, ac = compute_bias_cov(att_preds, att_val)
                cbf_biases.append(cb); cbf_covs.append(cc)
                att_biases.append(ab); att_covs.append(ac)

                all_csv_rows.append({
                    'sweep': 'A', 'model': minfo['name'], 'snr': snr,
                    'x_param': 'ATT', 'x_value': att_val,
                    'cbf_bias': cb, 'cbf_cov': cc,
                    'att_bias': ab, 'att_cov': ac,
                })

            sweep_a_data.append({
                'name': minfo['name'], 'color': minfo['color'],
                'marker': minfo['marker'],
                'cbf_bias': cbf_biases, 'cbf_cov': cbf_covs,
                'att_bias': att_biases, 'att_cov': att_covs,
            })

        # LS baseline
        print("  LS: Least Squares...")
        ls_cbf_b, ls_cbf_c, ls_att_b, ls_att_c = [], [], [], []
        for att_val in att_sweep:
            # 1D clean signal for LS
            cbf_cgs = fixed_cbf / 6000.0
            alpha1 = ALPHA_PCASL * (ALPHA_BS1**4)
            alpha2 = ALPHA_VSASL * (ALPHA_BS1**3)
            pcasl = _generate_pcasl_signal_jit(PLDS, att_val, cbf_cgs, T1_ARTERY, T_TAU, alpha1, T2_FACTOR)
            vsasl = _generate_vsasl_signal_jit(PLDS, att_val, cbf_cgs, T1_ARTERY, alpha2, T2_FACTOR, T_SAT_VS)
            clean_1d = np.concatenate([pcasl, vsasl])

            cbf_p, att_p = ls_inference(clean_1d, snr, args.n_ls_realizations, rng)
            if len(cbf_p) > 0:
                cb, cc = compute_bias_cov(cbf_p, fixed_cbf)
                ab, ac = compute_bias_cov(att_p, att_val)
            else:
                cb, cc, ab, ac = np.nan, np.nan, np.nan, np.nan
            ls_cbf_b.append(cb); ls_cbf_c.append(cc)
            ls_att_b.append(ab); ls_att_c.append(ac)

            all_csv_rows.append({
                'sweep': 'A', 'model': 'Least Squares', 'snr': snr,
                'x_param': 'ATT', 'x_value': att_val,
                'cbf_bias': cb, 'cbf_cov': cc,
                'att_bias': ab, 'att_cov': ac,
            })

        sweep_a_data.append({
            'name': 'Least Squares', 'color': '#7f7f7f',
            'marker': 'x', 'linestyle': '--',
            'cbf_bias': ls_cbf_b, 'cbf_cov': ls_cbf_c,
            'att_bias': ls_att_b, 'att_cov': ls_att_c,
        })

        make_bias_cov_figure(sweep_a_data, att_sweep, 'True ATT (ms)', 'ATT', snr, output_dir)

        # ─── Sweep B: vary CBF, fixed ATT ────────────────────────────
        print(f"\n--- Sweep B: ATT={fixed_att}, CBF={cbf_sweep[0]}→{cbf_sweep[-1]} ---")

        sweep_b_data = []

        # NN models
        for minfo in loaded_models:
            print(f"  NN: {minfo['name']}...")
            cbf_biases, cbf_covs, att_biases, att_covs = [], [], [], []
            for cbf_val in cbf_sweep:
                clean = generate_clean_phantom(cbf_val, fixed_att, mask)
                cbf_preds, att_preds = nn_inference(
                    clean, mask, minfo['models'], minfo['norm_stats'],
                    minfo['global_scale_factor'], minfo['normalization_mode'],
                    snr, args.n_phantoms, device, rng,
                )
                cb, cc = compute_bias_cov(cbf_preds, cbf_val)
                ab, ac = compute_bias_cov(att_preds, fixed_att)
                cbf_biases.append(cb); cbf_covs.append(cc)
                att_biases.append(ab); att_covs.append(ac)

                all_csv_rows.append({
                    'sweep': 'B', 'model': minfo['name'], 'snr': snr,
                    'x_param': 'CBF', 'x_value': cbf_val,
                    'cbf_bias': cb, 'cbf_cov': cc,
                    'att_bias': ab, 'att_cov': ac,
                })

            sweep_b_data.append({
                'name': minfo['name'], 'color': minfo['color'],
                'marker': minfo['marker'],
                'cbf_bias': cbf_biases, 'cbf_cov': cbf_covs,
                'att_bias': att_biases, 'att_cov': att_covs,
            })

        # LS baseline
        print("  LS: Least Squares...")
        ls_cbf_b, ls_cbf_c, ls_att_b, ls_att_c = [], [], [], []
        for cbf_val in cbf_sweep:
            cbf_cgs = cbf_val / 6000.0
            alpha1 = ALPHA_PCASL * (ALPHA_BS1**4)
            alpha2 = ALPHA_VSASL * (ALPHA_BS1**3)
            pcasl = _generate_pcasl_signal_jit(PLDS, fixed_att, cbf_cgs, T1_ARTERY, T_TAU, alpha1, T2_FACTOR)
            vsasl = _generate_vsasl_signal_jit(PLDS, fixed_att, cbf_cgs, T1_ARTERY, alpha2, T2_FACTOR, T_SAT_VS)
            clean_1d = np.concatenate([pcasl, vsasl])

            cbf_p, att_p = ls_inference(clean_1d, snr, args.n_ls_realizations, rng)
            if len(cbf_p) > 0:
                cb, cc = compute_bias_cov(cbf_p, cbf_val)
                ab, ac = compute_bias_cov(att_p, fixed_att)
            else:
                cb, cc, ab, ac = np.nan, np.nan, np.nan, np.nan
            ls_cbf_b.append(cb); ls_cbf_c.append(cc)
            ls_att_b.append(ab); ls_att_c.append(ac)

            all_csv_rows.append({
                'sweep': 'B', 'model': 'Least Squares', 'snr': snr,
                'x_param': 'CBF', 'x_value': cbf_val,
                'cbf_bias': cb, 'cbf_cov': cc,
                'att_bias': ab, 'att_cov': ac,
            })

        sweep_b_data.append({
            'name': 'Least Squares', 'color': '#7f7f7f',
            'marker': 'x', 'linestyle': '--',
            'cbf_bias': ls_cbf_b, 'cbf_cov': ls_cbf_c,
            'att_bias': ls_att_b, 'att_cov': ls_att_c,
        })

        make_bias_cov_figure(sweep_b_data, cbf_sweep, 'True CBF (ml/100g/min)', 'CBF', snr, output_dir)

    # ── Step 8: Data Export ──────────────────────────────────────────────
    # CSV
    csv_path = output_dir / 'bias_cov_data.csv'
    with open(csv_path, 'w') as f:
        cols = ['sweep', 'model', 'snr', 'x_param', 'x_value',
                'cbf_bias', 'cbf_cov', 'att_bias', 'att_cov']
        f.write(','.join(cols) + '\n')
        for row in all_csv_rows:
            vals = [str(row[c]) for c in cols]
            f.write(','.join(vals) + '\n')
    print(f"\nSaved CSV: {csv_path}")

    # JSON with metadata
    json_path = output_dir / 'bias_cov_data.json'
    meta = {
        'description': 'Bias and CoV analysis for ASL parameter estimation',
        'physics': {
            'T1_artery': T1_ARTERY, 'T_tau': T_TAU,
            'alpha_PCASL': ALPHA_PCASL, 'alpha_VSASL': ALPHA_VSASL,
            'alpha_BS1': ALPHA_BS1, 'PLDs': list(PLDS),
        },
        'sweep_A': {
            'fixed_CBF': fixed_cbf,
            'ATT_range': [int(att_sweep[0]), int(att_sweep[-1])],
        },
        'sweep_B': {
            'fixed_ATT': fixed_att,
            'CBF_range': [int(cbf_sweep[0]), int(cbf_sweep[-1])],
        },
        'n_phantoms_nn': args.n_phantoms,
        'n_realizations_ls': args.n_ls_realizations,
        'phantom_size': PHANTOM_SIZE,
        'mask_radius': MASK_RADIUS,
        'snr_levels': args.snr,
        'seed': args.seed,
        'models': [m['name'] for m in loaded_models] + ['Least Squares'],
        'data': all_csv_rows,
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    print(f"\nDone. All outputs in: {output_dir}/")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Detailed v7 evaluation: Bias, CoV, Bland-Altman, binned accuracy, tissue stratification.

Generates publication-style figures:
  1. Bias & CoV vs True CBF (binned)
  2. Bias & CoV vs True ATT (binned)
  3. Bias & CoV sweep: vary ATT @ fixed CBF, vary CBF @ fixed ATT
  4. Per-tissue-type metrics (GM, WM, pathology)
  5. Bland-Altman plots
  6. nRMSE & MAE vs SNR
  7. Error distribution histograms
  8. Linearity (scatter + regression) at multiple SNR

Usage:
    python3 amplitude_ablation_v7/evaluate_v7_detailed.py
"""

import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

V7_DIR = PROJECT_ROOT / 'amplitude_ablation_v7'
RESULTS_DIR = V7_DIR / 'v7_evaluation_results' / 'detailed'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PHANTOM_DIR = V7_DIR / 'test_phantoms'
SNR_LEVELS = [2, 3, 5, 10, 15, 25]
TISSUE_NAMES = {1: 'Gray Matter', 2: 'White Matter', 3: 'CSF / Boundary'}
TISSUE_COLORS = {1: '#E53935', 2: '#1E88E5', 3: '#43A047'}

EXPERIMENTS = {
    'A_Baseline_SpatialASL': {
        'label': 'Baseline SpatialASLNet', 'short': 'Baseline',
        'color': '#2196F3', 'marker': 'o',
    },
    'B_AmplitudeAware': {
        'label': 'AmplitudeAware', 'short': 'AmpAware',
        'color': '#F44336', 'marker': 's',
    },
}

LS_STYLE = {'label': 'Corrected LS', 'short': 'LS',
            'color': '#4CAF50', 'marker': '^', 'linestyle': '--'}


# ============================================================
# Model Loading
# ============================================================

def load_all_models():
    """Load ensembles for both experiments."""
    models = {}
    norm_stats = {}
    configs = {}

    for exp_name in EXPERIMENTS:
        exp_dir = V7_DIR / exp_name
        with open(exp_dir / 'research_config.json') as f:
            config = json.load(f)
        with open(exp_dir / 'norm_stats.json') as f:
            ns = json.load(f)
        configs[exp_name] = config
        norm_stats[exp_name] = ns

        model_class = config['model_class_name']
        features = config.get('hidden_sizes', [32, 64, 128, 256])
        n_plds = len(config['pld_values'])

        if model_class == 'AmplitudeAwareSpatialASLNet':
            from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
            def make_model(n=n_plds, f=features, c=config):
                return AmplitudeAwareSpatialASLNet(
                    n_plds=n, features=f,
                    use_film_at_bottleneck=c.get('use_film_at_bottleneck', True),
                    use_film_at_decoder=c.get('use_film_at_decoder', True),
                    use_amplitude_output_modulation=c.get('use_amplitude_output_modulation', True),
                )
        else:
            from models.spatial_asl_network import SpatialASLNet
            def make_model(n=n_plds, f=features):
                return SpatialASLNet(n_plds=n, features=f)

        ensemble = []
        for mp in sorted((exp_dir / 'trained_models').glob('ensemble_model_*.pt')):
            model = make_model()
            sd = torch.load(mp, map_location='cpu', weights_only=False)
            model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
            model.eval()
            ensemble.append(model)
        models[exp_name] = ensemble
        print(f"Loaded {len(ensemble)} models for {exp_name}")

    return models, norm_stats, configs


@torch.no_grad()
def predict_phantom(signals, models, norm_stats):
    """Run ensemble inference on scaled signals. Returns denormalized CBF, ATT."""
    input_tensor = torch.from_numpy(signals * 100.0 * 10.0).unsqueeze(0).float()

    cbf_preds, att_preds = [], []
    for model in models:
        cbf_n, att_n, _, _ = model(input_tensor)
        cbf_preds.append(cbf_n[0, 0].numpy())
        att_preds.append(att_n[0, 0].numpy())

    cbf = np.clip(np.mean(cbf_preds, axis=0) * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf'], 0, 300)
    att = np.clip(np.mean(att_preds, axis=0) * norm_stats['y_std_att'] + norm_stats['y_mean_att'], 0, 5000)
    return cbf, att


# ============================================================
# LS Fitting (subsampled)
# ============================================================

def run_ls_on_phantom(signals, cbf_true, att_true, plds, mask, subsample_rate=0.1):
    """Run corrected LS fitting on subsampled brain voxels."""
    from baselines.multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
    from utils.helpers import get_grid_search_initial_guess

    T1_ARTERY = 1650.0
    T_TAU = 1800.0
    ALPHA_PCASL = 0.85
    ALPHA_VSASL = 0.56
    ALPHA_BS1 = 1.0
    T2_FACTOR = 1.0

    asl_params = {
        'T1_artery': T1_ARTERY, 'T_tau': T_TAU,
        'alpha_PCASL': ALPHA_PCASL, 'alpha_VSASL': ALPHA_VSASL,
        'alpha_BS1': ALPHA_BS1, 'T2_factor': T2_FACTOR,
    }

    num_plds = len(plds)
    pldti = np.column_stack([plds, plds])
    h, w = signals.shape[1], signals.shape[2]

    # Scale signals for LS (same as NN but LS expects raw scale)
    # Phantoms store raw signals (~0.01 range), no additional scaling needed for LS
    raw_signals = signals.copy()

    brain_indices = np.argwhere(mask > 0.5)
    rng = np.random.RandomState(42)
    n_sample = max(1, int(len(brain_indices) * subsample_rate))
    sample_idx = rng.choice(len(brain_indices), n_sample, replace=False)
    sample_indices = brain_indices[sample_idx]

    cbf_ls = np.full((h, w), np.nan)
    att_ls = np.full((h, w), np.nan)

    for idx in sample_indices:
        i, j = idx
        voxel_signal = raw_signals[:, i, j]

        try:
            init_guess = get_grid_search_initial_guess(voxel_signal, plds, asl_params)
            signal_reshaped = voxel_signal.reshape((num_plds, 2), order='F')
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                pldti, signal_reshaped, init_guess,
                T1_ARTERY, T_TAU, T2_FACTOR, ALPHA_BS1,
                ALPHA_PCASL, ALPHA_VSASL
            )
            cbf_ls[i, j] = beta[0] * 6000.0
            att_ls[i, j] = beta[1]
        except Exception:
            continue

    return cbf_ls, att_ls


# ============================================================
# Data Collection
# ============================================================

def collect_all_predictions(models_dict, norm_stats_dict, configs_dict,
                            n_phantoms=100, snr_levels=None, run_ls=True, ls_subsample=0.1):
    """
    Collect per-voxel predictions across all phantoms, SNRs, and models.

    Returns a nested dict:
        data[exp_name][snr] = {
            'cbf_true': [], 'att_true': [], 'tissue': [],
            'cbf_pred': [], 'att_pred': [],
        }
        data['LS'][snr] = { ... }
    """
    if snr_levels is None:
        snr_levels = SNR_LEVELS

    phantom_files = sorted(PHANTOM_DIR.glob('phantom_*.npz'))[:n_phantoms]
    print(f"\nCollecting predictions from {len(phantom_files)} phantoms...")

    plds = np.array(list(configs_dict.values())[0]['pld_values'])

    # Initialize storage
    data = {}
    for exp_name in EXPERIMENTS:
        data[exp_name] = {snr: defaultdict(list) for snr in snr_levels}
    if run_ls:
        data['LS'] = {snr: defaultdict(list) for snr in snr_levels}

    for pi, pf in enumerate(phantom_files):
        if pi % 20 == 0:
            print(f"  Phantom {pi}/{len(phantom_files)}...")

        phantom = np.load(pf)
        cbf_true = phantom['cbf_map']
        att_true = phantom['att_map']
        tissue_map = phantom['tissue_map']
        mask = cbf_true > 0.5  # brain voxels

        for snr in snr_levels:
            noisy = phantom[f'noisy_snr_{snr}']

            # NN predictions
            for exp_name in EXPERIMENTS:
                cbf_pred, att_pred = predict_phantom(
                    noisy, models_dict[exp_name], norm_stats_dict[exp_name]
                )
                d = data[exp_name][snr]
                d['cbf_true'].append(cbf_true[mask])
                d['att_true'].append(att_true[mask])
                d['tissue'].append(tissue_map[mask])
                d['cbf_pred'].append(cbf_pred[mask])
                d['att_pred'].append(att_pred[mask])

            # LS predictions (subsampled)
            if run_ls:
                cbf_ls, att_ls = run_ls_on_phantom(
                    noisy, cbf_true, att_true, plds, mask, subsample_rate=ls_subsample
                )
                ls_valid = ~np.isnan(cbf_ls) & mask
                d = data['LS'][snr]
                d['cbf_true'].append(cbf_true[ls_valid])
                d['att_true'].append(att_true[ls_valid])
                d['tissue'].append(tissue_map[ls_valid])
                d['cbf_pred'].append(cbf_ls[ls_valid])
                d['att_pred'].append(att_ls[ls_valid])

    # Concatenate
    for method in data:
        for snr in data[method]:
            for key in data[method][snr]:
                data[method][snr][key] = np.concatenate(data[method][snr][key])

    return data


# ============================================================
# Figure 1: Bias & CoV vs True CBF (binned)
# ============================================================

def plot_binned_bias_cov_cbf(data, snr_to_plot=10):
    """2x2: CBF Bias vs true CBF, CBF CoV vs true CBF, ATT Bias vs true CBF, ATT CoV vs true CBF."""

    cbf_bins = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 80), (80, 120)]
    bin_centers = [(b[0]+b[1])/2 for b in cbf_bins]
    bin_labels = [f'{b[0]}-{b[1]}' for b in cbf_bins]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Bias & CoV vs True CBF Range (SNR={snr_to_plot})', fontsize=14, fontweight='bold')

    methods = list(EXPERIMENTS.keys()) + (['LS'] if 'LS' in data else [])
    method_styles = {**EXPERIMENTS, 'LS': LS_STYLE}

    panels = [
        (axes[0, 0], 'cbf', 'bias', 'CBF Bias (ml/100g/min)'),
        (axes[0, 1], 'cbf', 'cov', 'CBF CoV (%)'),
        (axes[1, 0], 'att', 'bias', 'ATT Bias (ms)'),
        (axes[1, 1], 'att', 'cov', 'ATT CoV (%)'),
    ]

    for ax, param, metric, ylabel in panels:
        for method in methods:
            if snr_to_plot not in data.get(method, {}):
                continue
            d = data[method][snr_to_plot]
            style = method_styles[method]

            vals = []
            for lo, hi in cbf_bins:
                in_bin = (d['cbf_true'] >= lo) & (d['cbf_true'] < hi)
                if in_bin.sum() < 10:
                    vals.append(np.nan)
                    continue

                pred = d[f'{param}_pred'][in_bin]
                true = d[f'{param}_true'][in_bin]

                if metric == 'bias':
                    vals.append(np.mean(pred - true))
                else:  # cov
                    mean_true = np.mean(true)
                    if abs(mean_true) > 1e-3:
                        vals.append(np.std(pred - true) / abs(mean_true) * 100)
                    else:
                        vals.append(np.nan)

            ax.plot(bin_centers, vals, f'-{style["marker"]}',
                    color=style['color'], label=style.get('short', style['label']),
                    linewidth=2, markersize=7,
                    linestyle=style.get('linestyle', '-'))

        if metric == 'bias':
            ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xlabel('True CBF Range (ml/100g/min)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(bin_labels, fontsize=9, rotation=20)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'binned_bias_cov_vs_cbf_snr{snr_to_plot}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: binned_bias_cov_vs_cbf_snr{snr_to_plot}.png")


# ============================================================
# Figure 2: Bias & CoV vs True ATT (binned)
# ============================================================

def plot_binned_bias_cov_att(data, snr_to_plot=10):
    """2x2: metrics vs true ATT range."""

    att_bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 4000)]
    bin_centers = [(b[0]+b[1])/2 for b in att_bins]
    bin_labels = [f'{b[0]}-{b[1]}' for b in att_bins]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Bias & CoV vs True ATT Range (SNR={snr_to_plot})', fontsize=14, fontweight='bold')

    methods = list(EXPERIMENTS.keys()) + (['LS'] if 'LS' in data else [])
    method_styles = {**EXPERIMENTS, 'LS': LS_STYLE}

    panels = [
        (axes[0, 0], 'cbf', 'bias', 'CBF Bias (ml/100g/min)'),
        (axes[0, 1], 'cbf', 'cov', 'CBF CoV (%)'),
        (axes[1, 0], 'att', 'bias', 'ATT Bias (ms)'),
        (axes[1, 1], 'att', 'cov', 'ATT CoV (%)'),
    ]

    for ax, param, metric, ylabel in panels:
        for method in methods:
            if snr_to_plot not in data.get(method, {}):
                continue
            d = data[method][snr_to_plot]
            style = method_styles[method]

            vals = []
            for lo, hi in att_bins:
                in_bin = (d['att_true'] >= lo) & (d['att_true'] < hi)
                if in_bin.sum() < 10:
                    vals.append(np.nan)
                    continue

                pred = d[f'{param}_pred'][in_bin]
                true = d[f'{param}_true'][in_bin]

                if metric == 'bias':
                    vals.append(np.mean(pred - true))
                else:
                    mean_true = np.mean(true)
                    if abs(mean_true) > 1e-3:
                        vals.append(np.std(pred - true) / abs(mean_true) * 100)
                    else:
                        vals.append(np.nan)

            ax.plot(bin_centers, vals, f'-{style["marker"]}',
                    color=style['color'], label=style.get('short', style['label']),
                    linewidth=2, markersize=7,
                    linestyle=style.get('linestyle', '-'))

        if metric == 'bias':
            ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xlabel('True ATT Range (ms)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(bin_labels, fontsize=8, rotation=25)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'binned_bias_cov_vs_att_snr{snr_to_plot}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: binned_bias_cov_vs_att_snr{snr_to_plot}.png")


# ============================================================
# Figure 3: Per-tissue metrics
# ============================================================

def plot_tissue_stratified(data, snr_to_plot=10):
    """Bar chart: MAE, Bias, CoV by tissue type for each model."""

    methods = list(EXPERIMENTS.keys()) + (['LS'] if 'LS' in data else [])
    method_styles = {**EXPERIMENTS, 'LS': LS_STYLE}
    tissue_ids = [1, 2]  # GM and WM only (enough voxels)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Tissue-Stratified Metrics (SNR={snr_to_plot})', fontsize=14, fontweight='bold')

    metrics = [
        ('MAE', lambda p, t: np.mean(np.abs(p - t))),
        ('Bias', lambda p, t: np.mean(p - t)),
        ('CoV (%)', lambda p, t: np.std(p - t) / max(abs(np.mean(t)), 1e-3) * 100),
    ]

    for row, param in enumerate(['cbf', 'att']):
        param_label = 'CBF (ml/100g/min)' if param == 'cbf' else 'ATT (ms)'

        for col, (metric_name, metric_fn) in enumerate(metrics):
            ax = axes[row, col]
            x = np.arange(len(tissue_ids))
            width = 0.25
            n_methods = len(methods)

            for mi, method in enumerate(methods):
                if snr_to_plot not in data.get(method, {}):
                    continue
                d = data[method][snr_to_plot]
                style = method_styles[method]

                vals = []
                for tid in tissue_ids:
                    in_tissue = d['tissue'] == tid
                    if in_tissue.sum() < 10:
                        vals.append(0)
                        continue
                    pred = d[f'{param}_pred'][in_tissue]
                    true = d[f'{param}_true'][in_tissue]
                    vals.append(metric_fn(pred, true))

                offset = (mi - n_methods / 2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width * 0.9,
                              label=style.get('short', style['label']),
                              color=style['color'], alpha=0.75)

            if metric_name == 'Bias':
                ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

            ax.set_title(f'{param.upper()} {metric_name}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([TISSUE_NAMES[t] for t in tissue_ids])
            ax.set_ylabel(f'{metric_name} ({param_label.split("(")[1]}' if '(' in param_label else metric_name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'tissue_stratified_snr{snr_to_plot}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: tissue_stratified_snr{snr_to_plot}.png")


# ============================================================
# Figure 4: Bland-Altman plots
# ============================================================

def plot_bland_altman(data, snr_to_plot=10):
    """Bland-Altman plots for CBF and ATT, each model."""

    nn_methods = list(EXPERIMENTS.keys())
    n_methods = len(nn_methods)

    fig, axes = plt.subplots(2, n_methods, figsize=(7 * n_methods, 10))
    fig.suptitle(f'Bland-Altman Analysis (SNR={snr_to_plot})', fontsize=14, fontweight='bold')

    for col, method in enumerate(nn_methods):
        if snr_to_plot not in data.get(method, {}):
            continue
        d = data[method][snr_to_plot]
        style = EXPERIMENTS[method]

        for row, (param, label, units) in enumerate([
            ('cbf', 'CBF', 'ml/100g/min'),
            ('att', 'ATT', 'ms'),
        ]):
            ax = axes[row, col]

            pred = d[f'{param}_pred']
            true = d[f'{param}_true']

            mean_val = (pred + true) / 2
            diff = pred - true

            bias = np.mean(diff)
            loa_upper = bias + 1.96 * np.std(diff)
            loa_lower = bias - 1.96 * np.std(diff)

            # Subsample for plotting
            n_plot = min(20000, len(mean_val))
            idx = np.random.RandomState(42).choice(len(mean_val), n_plot, replace=False)

            ax.scatter(mean_val[idx], diff[idx], alpha=0.05, s=2, c=style['color'])
            ax.axhline(bias, color='red', linewidth=1.5, label=f'Bias: {bias:.1f}')
            ax.axhline(loa_upper, color='orange', linewidth=1, linestyle='--',
                        label=f'+1.96 SD: {loa_upper:.1f}')
            ax.axhline(loa_lower, color='orange', linewidth=1, linestyle='--',
                        label=f'-1.96 SD: {loa_lower:.1f}')

            ax.set_xlabel(f'Mean {label} ({units})')
            ax.set_ylabel(f'Difference (Pred - True) ({units})')
            ax.set_title(f'{style["short"]} - {label}', fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'bland_altman_snr{snr_to_plot}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: bland_altman_snr{snr_to_plot}.png")


# ============================================================
# Figure 5: MAE & nRMSE vs SNR
# ============================================================

def plot_metrics_vs_snr(data):
    """Line plots: MAE, nRMSE, nBias, CoV vs SNR for all methods."""

    methods = list(EXPERIMENTS.keys()) + (['LS'] if 'LS' in data else [])
    method_styles = {**EXPERIMENTS, 'LS': LS_STYLE}

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('Metrics vs SNR (All Methods)', fontsize=14, fontweight='bold')

    metric_defs = [
        ('MAE', lambda p, t: np.mean(np.abs(p - t))),
        ('nRMSE (%)', lambda p, t: np.sqrt(np.mean((p - t)**2)) / max(abs(np.mean(t)), 1) * 100),
        ('nBias (%)', lambda p, t: np.mean(p - t) / max(abs(np.mean(t)), 1) * 100),
        ('CoV (%)', lambda p, t: np.std(p) / max(abs(np.mean(t)), 1) * 100),
    ]

    for row, (param, param_label) in enumerate([('cbf', 'CBF'), ('att', 'ATT')]):
        for col, (metric_name, metric_fn) in enumerate(metric_defs):
            ax = axes[row, col]

            for method in methods:
                style = method_styles[method]
                vals = []
                for snr in SNR_LEVELS:
                    if snr not in data.get(method, {}):
                        vals.append(np.nan)
                        continue
                    d = data[method][snr]
                    vals.append(metric_fn(d[f'{param}_pred'], d[f'{param}_true']))

                ax.plot(SNR_LEVELS, vals, f'-{style["marker"]}',
                        color=style['color'], label=style.get('short', style['label']),
                        linewidth=2, markersize=7,
                        linestyle=style.get('linestyle', '-'))

            if 'Bias' in metric_name:
                ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

            ax.set_title(f'{param_label} {metric_name}', fontweight='bold')
            ax.set_xlabel('SNR')
            ax.set_ylabel(metric_name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(SNR_LEVELS)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'metrics_vs_snr.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: metrics_vs_snr.png")


# ============================================================
# Figure 6: Error distributions
# ============================================================

def plot_error_distributions(data, snr_to_plot=10):
    """Histograms of prediction errors for CBF and ATT."""

    nn_methods = list(EXPERIMENTS.keys())

    fig, axes = plt.subplots(2, len(nn_methods), figsize=(7 * len(nn_methods), 8))
    fig.suptitle(f'Error Distributions (SNR={snr_to_plot})', fontsize=14, fontweight='bold')

    for col, method in enumerate(nn_methods):
        d = data[method][snr_to_plot]
        style = EXPERIMENTS[method]

        for row, (param, label, units, xlim) in enumerate([
            ('cbf', 'CBF', 'ml/100g/min', (-40, 40)),
            ('att', 'ATT', 'ms', (-800, 800)),
        ]):
            ax = axes[row, col]
            errors = d[f'{param}_pred'] - d[f'{param}_true']

            ax.hist(errors, bins=100, range=xlim, alpha=0.7, color=style['color'],
                    density=True, edgecolor='white', linewidth=0.3)
            ax.axvline(0, color='black', linewidth=1, linestyle=':')

            mean_err = np.mean(errors)
            std_err = np.std(errors)
            median_err = np.median(errors)

            ax.axvline(mean_err, color='red', linewidth=1.5, linestyle='-',
                        label=f'Mean: {mean_err:.1f}')
            ax.axvline(median_err, color='orange', linewidth=1.5, linestyle='--',
                        label=f'Median: {median_err:.1f}')

            ax.set_title(f'{style["short"]} - {label} Error\nSD={std_err:.1f} {units}',
                         fontweight='bold', fontsize=11)
            ax.set_xlabel(f'Error ({units})')
            ax.set_ylabel('Density')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'error_distributions_snr{snr_to_plot}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: error_distributions_snr{snr_to_plot}.png")


# ============================================================
# Figure 7: Linearity (scatter + regression) at multiple SNR
# ============================================================

def plot_linearity_multi_snr(data):
    """Scatter plots at multiple SNR levels showing linearity."""

    nn_methods = list(EXPERIMENTS.keys())
    snrs_to_show = [3, 10, 25]

    for param, label, units, lim in [('cbf', 'CBF', 'ml/100g/min', (0, 150)),
                                      ('att', 'ATT', 'ms', (0, 4000))]:
        fig, axes = plt.subplots(len(snrs_to_show), len(nn_methods),
                                  figsize=(7 * len(nn_methods), 5 * len(snrs_to_show)))
        fig.suptitle(f'{label} Linearity at Multiple SNR Levels', fontsize=14, fontweight='bold')

        for row, snr in enumerate(snrs_to_show):
            for col, method in enumerate(nn_methods):
                ax = axes[row, col]
                d = data[method][snr]
                style = EXPERIMENTS[method]

                true = d[f'{param}_true']
                pred = d[f'{param}_pred']

                n_plot = min(25000, len(true))
                idx = np.random.RandomState(42).choice(len(true), n_plot, replace=False)

                ax.scatter(true[idx], pred[idx], alpha=0.04, s=2, c=style['color'])
                ax.plot(lim, lim, 'k--', linewidth=1.5, alpha=0.7)

                # Linear regression
                valid = (true > lim[0] + 5) & (pred > lim[0]) & (pred < lim[1])
                if valid.sum() > 100:
                    coeffs = np.polyfit(true[valid], pred[valid], 1)
                    x_fit = np.linspace(lim[0] + 5, lim[1] * 0.9, 100)
                    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=2)

                    # R^2 vs identity
                    ss_res = np.sum((pred[valid] - true[valid])**2)
                    ss_tot = np.sum((true[valid] - np.mean(true[valid]))**2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                    mae = np.mean(np.abs(pred - true))
                    ax.text(0.05, 0.92,
                            f'y={coeffs[0]:.2f}x+{coeffs[1]:.0f}\nR²={r2:.3f}\nMAE={mae:.1f}',
                            transform=ax.transAxes, fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.set_xlabel(f'True {label} ({units})')
                ax.set_ylabel(f'Predicted {label}')
                if row == 0:
                    ax.set_title(f'{style["short"]}', fontweight='bold', fontsize=12)
                if col == 0:
                    ax.set_ylabel(f'SNR={snr}\nPredicted {label}', fontsize=11)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f'linearity_{param}_multi_snr.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: linearity_{param}_multi_snr.png")


# ============================================================
# Figure 8: Win rate by CBF/ATT bin
# ============================================================

def plot_win_rate_by_bin(data, snr_to_plot=10):
    """Win rate vs LS stratified by CBF or ATT range."""
    if 'LS' not in data:
        return

    cbf_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120)]
    att_bins = [(0, 750), (750, 1250), (1250, 1750), (1750, 2500), (2500, 4000)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Win Rate vs LS by Parameter Range (SNR={snr_to_plot})', fontsize=14, fontweight='bold')

    for row, (param, param_label, bins, bin_key) in enumerate([
        ('cbf', 'CBF', cbf_bins, 'cbf'),
        ('att', 'ATT', att_bins, 'att'),
    ]):
        for col, (eval_param, eval_label) in enumerate([('cbf', 'CBF'), ('att', 'ATT')]):
            ax = axes[row, col]
            bin_labels = [f'{b[0]}-{b[1]}' for b in bins]
            x = np.arange(len(bins))

            ls_d = data['LS'][snr_to_plot]

            for mi, (method, style) in enumerate(EXPERIMENTS.items()):
                nn_d = data[method][snr_to_plot]
                win_rates = []

                for lo, hi in bins:
                    # Use bin_key to select range, but need matching indices
                    # LS has subsampled voxels, so we compute per-method
                    nn_bin = (nn_d[f'{bin_key}_true'] >= lo) & (nn_d[f'{bin_key}_true'] < hi)
                    ls_bin = (ls_d[f'{bin_key}_true'] >= lo) & (ls_d[f'{bin_key}_true'] < hi)

                    if nn_bin.sum() < 10 or ls_bin.sum() < 10:
                        win_rates.append(np.nan)
                        continue

                    nn_mae = np.mean(np.abs(nn_d[f'{eval_param}_pred'][nn_bin] - nn_d[f'{eval_param}_true'][nn_bin]))
                    ls_mae = np.mean(np.abs(ls_d[f'{eval_param}_pred'][ls_bin] - ls_d[f'{eval_param}_true'][ls_bin]))

                    # Win = NN has lower MAE
                    win_rates.append(100 * (nn_mae < ls_mae))

                width = 0.35
                offset = (mi - 0.5) * width
                ax.bar(x + offset, win_rates, width * 0.9, label=style['short'],
                       color=style['color'], alpha=0.7)

            ax.axhline(50, color='gray', linewidth=1, linestyle=':', alpha=0.7)
            ax.set_title(f'{eval_label} Win Rate by {param_label} Range', fontweight='bold')
            ax.set_xlabel(f'True {param_label} Range')
            ax.set_ylabel('NN Wins (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, fontsize=9, rotation=20)
            ax.set_ylim(0, 110)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'win_rate_by_bin_snr{snr_to_plot}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: win_rate_by_bin_snr{snr_to_plot}.png")


# ============================================================
# Figure 9: Multi-SNR Bias & CoV vs CBF (comprehensive)
# ============================================================

def plot_bias_cov_multi_snr(data):
    """3-SNR x 2-metric figure for CBF and ATT."""

    cbf_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120)]
    bin_centers = [(b[0]+b[1])/2 for b in cbf_bins]
    snrs_show = [3, 10, 25]

    methods = list(EXPERIMENTS.keys()) + (['LS'] if 'LS' in data else [])
    method_styles = {**EXPERIMENTS, 'LS': LS_STYLE}

    for param, param_label, bins_list, units in [
        ('cbf', 'CBF', cbf_bins, 'ml/100g/min'),
        ('att', 'ATT', [(0, 750), (750, 1250), (1250, 1750), (1750, 2500), (2500, 4000)], 'ms'),
    ]:
        centers = [(b[0]+b[1])/2 for b in bins_list]
        blabels = [f'{b[0]}-{b[1]}' for b in bins_list]

        fig, axes = plt.subplots(len(snrs_show), 2, figsize=(14, 5 * len(snrs_show)))
        fig.suptitle(f'{param_label} Bias & CoV Across SNR Levels', fontsize=14, fontweight='bold')

        for row, snr in enumerate(snrs_show):
            for col, (metric, ylabel) in enumerate([
                ('bias', f'{param_label} Bias ({units})'),
                ('cov', f'{param_label} CoV (%)'),
            ]):
                ax = axes[row, col]

                for method in methods:
                    if snr not in data.get(method, {}):
                        continue
                    d = data[method][snr]
                    style = method_styles[method]

                    vals = []
                    for lo, hi in bins_list:
                        in_bin = (d[f'{param}_true'] >= lo) & (d[f'{param}_true'] < hi)
                        if in_bin.sum() < 10:
                            vals.append(np.nan)
                            continue
                        pred = d[f'{param}_pred'][in_bin]
                        true = d[f'{param}_true'][in_bin]
                        if metric == 'bias':
                            vals.append(np.mean(pred - true))
                        else:
                            mean_true = np.mean(true)
                            vals.append(np.std(pred - true) / max(abs(mean_true), 1) * 100)

                    ax.plot(centers, vals, f'-{style["marker"]}',
                            color=style['color'], label=style.get('short', style['label']),
                            linewidth=2, markersize=7,
                            linestyle=style.get('linestyle', '-'))

                if metric == 'bias':
                    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

                ax.set_title(f'SNR={snr}', fontweight='bold')
                ax.set_xlabel(f'True {param_label} Range')
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(centers)
                ax.set_xticklabels(blabels, fontsize=8, rotation=25)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f'bias_cov_multi_snr_{param}.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: bias_cov_multi_snr_{param}.png")


# ============================================================
# Summary table
# ============================================================

def print_detailed_summary(data):
    """Print comprehensive summary tables."""

    methods = list(EXPERIMENTS.keys()) + (['LS'] if 'LS' in data else [])
    method_styles = {**EXPERIMENTS, 'LS': LS_STYLE}

    lines = []
    lines.append("=" * 100)
    lines.append("DETAILED V7 EVALUATION SUMMARY")
    lines.append("=" * 100)

    # Table 1: Overall metrics by SNR
    for param, label in [('cbf', 'CBF'), ('att', 'ATT')]:
        lines.append(f"\n### {label} Metrics by SNR\n")
        header = f"{'SNR':>4}"
        for method in methods:
            short = method_styles[method].get('short', method_styles[method]['label'])
            header += f" | {'MAE':>7} {'Bias':>7} {'nRMSE':>7} {'CoV':>7}  [{short}]"
        lines.append(header)
        lines.append("-" * len(header))

        for snr in SNR_LEVELS:
            row = f"{snr:>4}"
            for method in methods:
                if snr not in data.get(method, {}):
                    row += f" | {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7}"
                    continue
                d = data[method][snr]
                pred = d[f'{param}_pred']
                true = d[f'{param}_true']
                mae = np.mean(np.abs(pred - true))
                bias = np.mean(pred - true)
                nrmse = np.sqrt(np.mean((pred - true)**2)) / max(abs(np.mean(true)), 1) * 100
                cov = np.std(pred) / max(abs(np.mean(true)), 1) * 100
                row += f" | {mae:>7.1f} {bias:>7.1f} {nrmse:>7.1f} {cov:>7.1f}"
            lines.append(row)

    # Table 2: Per-tissue at SNR=10
    lines.append(f"\n### Per-Tissue Metrics (SNR=10)\n")
    for tid in [1, 2]:
        lines.append(f"\n  {TISSUE_NAMES[tid]}:")
        for method in methods:
            if 10 not in data.get(method, {}):
                continue
            d = data[method][10]
            style = method_styles[method]
            in_tissue = d['tissue'] == tid
            if in_tissue.sum() < 10:
                continue

            for param, label in [('cbf', 'CBF'), ('att', 'ATT')]:
                pred = d[f'{param}_pred'][in_tissue]
                true = d[f'{param}_true'][in_tissue]
                mae = np.mean(np.abs(pred - true))
                bias = np.mean(pred - true)
                lines.append(f"    {style.get('short', style['label']):>12} {label}: MAE={mae:.1f}, Bias={bias:.1f}")

    report = "\n".join(lines)
    with open(RESULTS_DIR / 'detailed_summary.txt', 'w') as f:
        f.write(report)
    print(report)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("V7 Detailed Evaluation")
    print("=" * 60)

    # Load models
    models_dict, norm_stats_dict, configs_dict = load_all_models()

    # Collect all predictions (this is the slow step)
    data = collect_all_predictions(
        models_dict, norm_stats_dict, configs_dict,
        n_phantoms=100, snr_levels=SNR_LEVELS,
        run_ls=True, ls_subsample=0.05,  # 5% subsample for LS speed
    )

    # Save raw data for reuse
    print("\nSaving collected data...")
    save_data = {}
    for method in data:
        save_data[method] = {}
        for snr in data[method]:
            save_data[method][snr] = {k: v for k, v in data[method][snr].items()}
    np.savez_compressed(RESULTS_DIR / 'collected_predictions.npz',
                        **{f'{m}_snr{s}_{k}': data[m][s][k]
                           for m in data for s in data[m] for k in data[m][s]})

    # Generate all figures
    print("\n--- Generating Figures ---\n")

    for snr in [3, 10, 25]:
        plot_binned_bias_cov_cbf(data, snr_to_plot=snr)
        plot_binned_bias_cov_att(data, snr_to_plot=snr)
        plot_tissue_stratified(data, snr_to_plot=snr)
        plot_bland_altman(data, snr_to_plot=snr)
        plot_error_distributions(data, snr_to_plot=snr)
        plot_win_rate_by_bin(data, snr_to_plot=snr)

    plot_metrics_vs_snr(data)
    plot_linearity_multi_snr(data)
    plot_bias_cov_multi_snr(data)

    # Summary
    print_detailed_summary(data)

    print(f"\nAll detailed results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for V7 Evaluation
======================================================
Creates figures from phantom and in-vivo evaluation results.

Usage:
    python amplitude_ablation_v7/generate_publication_figures.py \
        --results-dir amplitude_ablation_v7/v7_evaluation_results \
        --phantom-dir amplitude_ablation_v7/test_phantoms \
        --output-dir amplitude_ablation_v7/v7_evaluation_results/figures
"""

import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Style ─────────────────────────────────────────────────────────────────

COLORS = {
    'NN': '#1f77b4',
    'LS': '#d62728',
    'nn_1rep': '#1f77b4',
    'nn_4rep': '#2ca02c',
    'ls_1rep': '#d62728',
    'ls_4rep': '#7f7f7f',
}
LINE_STYLES = {'NN': '-', 'LS': '--'}
MARKERS = {'NN': 'o', 'LS': 's'}

def setup_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def save_figure(fig, output_dir, name):
    """Save figure as PNG and PDF."""
    fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png, {name}.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: nBias / CoV / nRMSE vs SNR
# ═══════════════════════════════════════════════════════════════════════════

def figure_1_bias_cov_nrmse(phantom_results, output_dir):
    """
    2x3 grid: rows = CBF, ATT; cols = nBias, CoV, nRMSE.
    X-axis = SNR levels, lines = NN vs LS.
    """
    print("Generating Figure 1: nBias/CoV/nRMSE vs SNR...")

    snr_levels = sorted(phantom_results.get('snr_levels', []))
    if not snr_levels:
        # Try to extract from per-SNR results
        snr_keys = [k for k in phantom_results.keys() if k.startswith('snr_')]
        snr_levels = sorted([float(k.split('_')[1]) for k in snr_keys])

    if not snr_levels:
        print("  SKIP: No SNR data found in phantom results.")
        return

    # Extract metrics per SNR
    nn_data = {'cbf_nbias': [], 'cbf_cov': [], 'cbf_nrmse': [],
               'att_nbias': [], 'att_cov': [], 'att_nrmse': [],
               'cbf_nbias_std': [], 'cbf_cov_std': [], 'cbf_nrmse_std': [],
               'att_nbias_std': [], 'att_cov_std': [], 'att_nrmse_std': []}
    ls_data = {k: [] for k in nn_data}

    for snr in snr_levels:
        snr_key = f'snr_{snr}' if f'snr_{snr}' in phantom_results else f'snr_{int(snr)}'
        if snr_key not in phantom_results:
            snr_key = f'snr_{snr:.1f}'
        if snr_key not in phantom_results:
            continue

        snr_result = phantom_results[snr_key]

        for method, data_dict in [('nn', nn_data), ('ls', ls_data)]:
            prefix = method
            for param in ['cbf', 'att']:
                # Try both casing conventions: nBias/CoV/nRMSE (eval script) and nbias/cov/nrmse
                for metric_base, metric_keys in [
                    ('nbias', ['nBias', 'nbias']),
                    ('cov', ['CoV', 'cov']),
                    ('nrmse', ['nRMSE', 'nrmse']),
                ]:
                    val = np.nan
                    std = 0.0
                    for mk in metric_keys:
                        val_key = f'{prefix}_{param}_{mk}'
                        if val_key in snr_result and snr_result[val_key] is not None:
                            val = snr_result[val_key]
                            break
                    for mk in metric_keys:
                        std_key = f'{prefix}_{param}_{mk}_std'
                        if std_key in snr_result and snr_result[std_key] is not None:
                            std = snr_result[std_key]
                            break

                    data_dict[f'{param}_{metric_base}'].append(val if val is not None else np.nan)
                    data_dict[f'{param}_{metric_base}_std'].append(std if std is not None else 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Simulated Phantom Evaluation', fontsize=15, fontweight='bold')

    panels = [
        (0, 0, 'cbf_nbias', 'CBF nBias (%)'),
        (0, 1, 'cbf_cov', 'CBF CoV (%)'),
        (0, 2, 'cbf_nrmse', 'CBF nRMSE (%)'),
        (1, 0, 'att_nbias', 'ATT nBias (%)'),
        (1, 1, 'att_cov', 'ATT CoV (%)'),
        (1, 2, 'att_nrmse', 'ATT nRMSE (%)'),
    ]

    for row, col, key, ylabel in panels:
        ax = axes[row, col]

        nn_vals = np.array(nn_data[key])
        nn_errs = np.array(nn_data[f'{key}_std'])
        ls_vals = np.array(ls_data[key])
        ls_errs = np.array(ls_data[f'{key}_std'])

        ax.errorbar(snr_levels, nn_vals, yerr=nn_errs, color=COLORS['NN'],
                     marker='o', markersize=5, linewidth=2, capsize=4,
                     label='NN', linestyle='-')
        ax.errorbar(snr_levels, ls_vals, yerr=ls_errs, color=COLORS['LS'],
                     marker='s', markersize=5, linewidth=2, capsize=4,
                     label='LS', linestyle='--')

        ax.set_xlabel('SNR')
        ax.set_ylabel(ylabel)

        if 'nbias' in key:
            ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

        ax.legend(loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_dir, 'fig1_bias_cov_nrmse_vs_snr')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Example Phantom Maps
# ═══════════════════════════════════════════════════════════════════════════

def figure_2_phantom_maps(phantom_dir, phantom_results, output_dir, snr=5):
    """
    2x4 grid showing Ground Truth, NN, LS, |Error NN| for CBF and ATT.
    """
    print(f"Generating Figure 2: Example phantom maps (SNR={snr})...")

    # Find phantom file
    phantom_files = sorted(Path(phantom_dir).glob('phantom_*.npz'))
    if not phantom_files:
        print("  SKIP: No phantom files found.")
        return

    phantom_path = phantom_files[0]
    phantom_data = np.load(phantom_path)

    cbf_true = phantom_data.get('cbf_map', None)
    att_true = phantom_data.get('att_map', None)
    mask = phantom_data.get('mask', None)

    if cbf_true is None or att_true is None:
        print("  SKIP: Phantom missing cbf_map or att_map.")
        return

    # Try to load predictions from results
    snr_key = f'snr_{snr}' if f'snr_{snr}' in phantom_results else f'snr_{int(snr)}'
    nn_cbf = phantom_results.get(snr_key, {}).get('nn_cbf_map', None)
    nn_att = phantom_results.get(snr_key, {}).get('nn_att_map', None)
    ls_cbf = phantom_results.get(snr_key, {}).get('ls_cbf_map', None)
    ls_att = phantom_results.get(snr_key, {}).get('ls_att_map', None)

    # Also try loading from saved npz files
    maps_dir = Path(phantom_dir).parent / 'v7_evaluation_results' / 'phantom_maps'
    if nn_cbf is None and maps_dir.exists():
        for f in maps_dir.glob(f'*snr{snr}*.npz'):
            d = np.load(f)
            if 'nn_cbf' in d:
                nn_cbf = d['nn_cbf']
                nn_att = d['nn_att']
                ls_cbf = d.get('ls_cbf', None)
                ls_att = d.get('ls_att', None)
                break

    if nn_cbf is None:
        print("  SKIP: No NN prediction maps found. Run evaluate_realistic_phantoms.py first.")
        return

    if mask is not None:
        mask_2d = mask.astype(bool)
    else:
        mask_2d = np.ones(cbf_true.shape, dtype=bool)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Phantom Evaluation (SNR = {snr})', fontsize=15, fontweight='bold')

    # CBF row
    cbf_vmin, cbf_vmax = 0, np.percentile(cbf_true[mask_2d], 99) * 1.2 if mask_2d.any() else 100
    cbf_panels = [
        (cbf_true, 'Ground Truth'),
        (nn_cbf, 'NN Prediction'),
        (ls_cbf, 'LS Prediction'),
        (np.abs(nn_cbf - cbf_true) if nn_cbf is not None else None, '|Error| NN'),
    ]

    for col, (data, title) in enumerate(cbf_panels):
        ax = axes[0, col]
        if data is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'CBF: {title}')
            ax.axis('off')
            continue

        disp = np.where(mask_2d, data, np.nan)
        if col == 3:  # Error map
            im = ax.imshow(disp.T, cmap='hot', vmin=0,
                           vmax=np.nanpercentile(disp, 95) if np.any(np.isfinite(disp)) else 20,
                           origin='lower')
        else:
            im = ax.imshow(disp.T, cmap='hot', vmin=cbf_vmin, vmax=cbf_vmax, origin='lower')
        ax.set_title(f'CBF: {title}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='ml/100g/min' if col < 3 else '')

    # ATT row
    att_vmin = 500
    att_vmax = np.percentile(att_true[mask_2d], 99) * 1.1 if mask_2d.any() else 3000
    att_panels = [
        (att_true, 'Ground Truth'),
        (nn_att, 'NN Prediction'),
        (ls_att, 'LS Prediction'),
        (np.abs(nn_att - att_true) if nn_att is not None else None, '|Error| NN'),
    ]

    for col, (data, title) in enumerate(att_panels):
        ax = axes[1, col]
        if data is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'ATT: {title}')
            ax.axis('off')
            continue

        disp = np.where(mask_2d, data, np.nan)
        if col == 3:
            im = ax.imshow(disp.T, cmap='hot', vmin=0,
                           vmax=np.nanpercentile(disp, 95) if np.any(np.isfinite(disp)) else 500,
                           origin='lower')
        else:
            im = ax.imshow(disp.T, cmap='viridis', vmin=att_vmin, vmax=att_vmax, origin='lower')
        ax.set_title(f'ATT: {title}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='ms' if col < 3 else '')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_dir, 'fig2_phantom_maps')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: In-Vivo Comparison
# ═══════════════════════════════════════════════════════════════════════════

def figure_3_invivo_comparison(invivo_results, results_dir, output_dir):
    """
    For one subject, show mid-slice:
    Row 1: LS(4rep) CBF, NN(1rep) CBF, LS(1rep) CBF, |NN(1rep)-ref|, |LS(1rep)-ref|
    Row 2: Same for ATT
    """
    print("Generating Figure 3: In-vivo comparison...")

    subjects = invivo_results.get('subjects', [])
    if not subjects:
        print("  SKIP: No in-vivo subjects found.")
        return

    # Pick first subject with valid results
    subject = subjects[0]
    sid = subject.get('subject', 'unknown')

    # Load maps
    maps_dir = results_dir / 'invivo_maps' / sid
    if not maps_dir.exists():
        print(f"  SKIP: No maps found for {sid} at {maps_dir}")
        return

    try:
        nn_1rep_cbf = np.load(maps_dir / 'nn_1rep_cbf.npy')
        nn_1rep_att = np.load(maps_dir / 'nn_1rep_att.npy')
        ls_4rep_cbf = np.load(maps_dir / 'ls_4rep_cbf.npy')
        ls_4rep_att = np.load(maps_dir / 'ls_4rep_att.npy')
        ls_1rep_cbf = np.load(maps_dir / 'ls_1rep_cbf.npy')
        ls_1rep_att = np.load(maps_dir / 'ls_1rep_att.npy')
        brain_mask = np.load(maps_dir / 'brain_mask.npy') > 0
    except FileNotFoundError as e:
        print(f"  SKIP: Missing map file: {e}")
        return

    n_slices = brain_mask.shape[2]
    mid_slice = n_slices // 2

    # Find slice with most brain voxels
    brain_counts = [brain_mask[:, :, s].sum() for s in range(n_slices)]
    mid_slice = np.argmax(brain_counts)

    mask_s = brain_mask[:, :, mid_slice]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'In-Vivo Comparison: {sid} (Slice {mid_slice})', fontsize=15, fontweight='bold')

    # CBF row
    cbf_ref = ls_4rep_cbf[:, :, mid_slice]
    cbf_nn = nn_1rep_cbf[:, :, mid_slice]
    cbf_ls = ls_1rep_cbf[:, :, mid_slice]

    cbf_vmin, cbf_vmax = 0, 100

    cbf_panels = [
        (cbf_ref, 'LS(4-rep) [ref]', 'hot', cbf_vmin, cbf_vmax),
        (cbf_nn, 'NN(1-rep)', 'hot', cbf_vmin, cbf_vmax),
        (cbf_ls, 'LS(1-rep)', 'hot', cbf_vmin, cbf_vmax),
        (np.abs(cbf_nn - cbf_ref), '|NN(1rep) - ref|', 'hot', 0, 30),
        (np.abs(cbf_ls - cbf_ref), '|LS(1rep) - ref|', 'hot', 0, 30),
    ]

    for col, (data, title, cmap, vmin, vmax) in enumerate(cbf_panels):
        ax = axes[0, col]
        disp = np.where(mask_s & np.isfinite(data), data, np.nan)
        im = ax.imshow(disp.T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'CBF: {title}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='ml/100g/min')

    # ATT row
    att_ref = ls_4rep_att[:, :, mid_slice]
    att_nn = nn_1rep_att[:, :, mid_slice]
    att_ls = ls_1rep_att[:, :, mid_slice]

    att_vmin, att_vmax = 500, 3000

    att_panels = [
        (att_ref, 'LS(4-rep) [ref]', 'viridis', att_vmin, att_vmax),
        (att_nn, 'NN(1-rep)', 'viridis', att_vmin, att_vmax),
        (att_ls, 'LS(1-rep)', 'viridis', att_vmin, att_vmax),
        (np.abs(att_nn - att_ref), '|NN(1rep) - ref|', 'hot', 0, 500),
        (np.abs(att_ls - att_ref), '|LS(1rep) - ref|', 'hot', 0, 500),
    ]

    for col, (data, title, cmap, vmin, vmax) in enumerate(att_panels):
        ax = axes[1, col]
        disp = np.where(mask_s & np.isfinite(data), data, np.nan)
        im = ax.imshow(disp.T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'ATT: {title}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='ms')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_dir, 'fig3_invivo_comparison')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Win Rate vs SNR
# ═══════════════════════════════════════════════════════════════════════════

def figure_4_win_rate(phantom_results, output_dir):
    """Bar chart of win rates grouped by SNR."""
    print("Generating Figure 4: Win rate vs SNR...")

    snr_keys = sorted([k for k in phantom_results.keys() if k.startswith('snr_')],
                      key=lambda k: float(k.split('_')[1]))

    if not snr_keys:
        print("  SKIP: No SNR data found.")
        return

    snr_labels = []
    cbf_wrs = []
    att_wrs = []

    for sk in snr_keys:
        snr_val = float(sk.split('_')[1])
        snr_labels.append(f'{snr_val:g}')

        cbf_wr = phantom_results[sk].get('nn_cbf_win_rate',
                 phantom_results[sk].get('cbf_win_rate', np.nan))
        att_wr = phantom_results[sk].get('nn_att_win_rate',
                 phantom_results[sk].get('att_win_rate', np.nan))

        cbf_wrs.append(cbf_wr if cbf_wr is not None else np.nan)
        att_wrs.append(att_wr if att_wr is not None else np.nan)

    if all(np.isnan(v) for v in cbf_wrs):
        print("  SKIP: All win rates are NaN.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(snr_labels))
    width = 0.35

    bars_cbf = ax.bar(x - width/2, cbf_wrs, width, label='CBF Win Rate',
                       color=COLORS['NN'], alpha=0.8)
    bars_att = ax.bar(x + width/2, att_wrs, width, label='ATT Win Rate',
                       color='#ff7f0e', alpha=0.8)

    ax.axhline(50, color='gray', linestyle='--', linewidth=1.5, label='Chance (50%)')

    ax.set_xlabel('SNR')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('NN vs LS Win Rate by SNR')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set_ylim(0, 100)
    ax.legend()

    # Add value labels on bars
    for bar in bars_cbf:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}%',
                    ha='center', va='bottom', fontsize=9)
    for bar in bars_att:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}%',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig4_win_rate_vs_snr')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Speed Comparison
# ═══════════════════════════════════════════════════════════════════════════

def figure_5_speed(phantom_results, invivo_results, output_dir):
    """Bar chart comparing NN vs LS inference time (log scale)."""
    print("Generating Figure 5: Speed comparison...")

    nn_time = None
    ls_time = None

    # Try phantom results first
    if phantom_results:
        nn_time = phantom_results.get('nn_mean_time_per_phantom')
        ls_time = phantom_results.get('ls_mean_time_per_phantom')

        if nn_time is None:
            # Try aggregating from SNR levels
            nn_times = []
            ls_times = []
            for k, v in phantom_results.items():
                if k.startswith('snr_') and isinstance(v, dict):
                    if 'nn_time' in v:
                        nn_times.append(v['nn_time'])
                    if 'ls_time' in v:
                        ls_times.append(v['ls_time'])
            if nn_times:
                nn_time = np.mean(nn_times)
            if ls_times:
                ls_time = np.mean(ls_times)

    # Try in-vivo results
    if invivo_results and (nn_time is None or ls_time is None):
        agg = invivo_results.get('aggregate', {})
        if nn_time is None:
            nn_time = agg.get('nn_1rep_mean_time')
        if ls_time is None:
            ls_time = agg.get('ls_4rep_mean_time')

    if nn_time is None or ls_time is None:
        print("  SKIP: Timing data not available.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    methods = ['NN', 'LS']
    times = [nn_time, ls_time]
    colors = [COLORS['NN'], COLORS['LS']]

    bars = ax.bar(methods, times, color=colors, alpha=0.8, width=0.5)

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Inference Speed Comparison')
    ax.set_yscale('log')

    # Add value labels
    for bar, t in zip(bars, times):
        label = f'{t:.2f}s' if t < 1 else f'{t:.1f}s'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                label, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Speedup annotation
    if ls_time > 0 and nn_time > 0:
        speedup = ls_time / nn_time
        ax.text(0.5, 0.95, f'NN is {speedup:.0f}x faster',
                ha='center', va='top', transform=ax.transAxes,
                fontsize=13, fontweight='bold', color='#2ca02c',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, output_dir, 'fig5_speed_comparison')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality figures from V7 evaluation results.')
    parser.add_argument('--results-dir', type=str,
                        default=str(PROJECT_ROOT / 'amplitude_ablation_v7' / 'v7_evaluation_results'),
                        help='Directory containing result JSON files')
    parser.add_argument('--phantom-dir', type=str,
                        default=str(PROJECT_ROOT / 'amplitude_ablation_v7' / 'test_phantoms'),
                        help='Directory containing phantom .npz files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures (default: results-dir/figures)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    phantom_dir = Path(args.phantom_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    print(f"Results directory: {results_dir}")
    print(f"Phantom directory: {phantom_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load phantom results - try multiple possible filenames and locations
    phantom_results = {}
    phantom_json_candidates = [
        results_dir / 'phantom_results.json',
        results_dir / 'phantom_eval_summary.json',
    ]
    # Also search model directories for eval results
    v7_dir = PROJECT_ROOT / 'amplitude_ablation_v7'
    for exp in ['B_AmplitudeAware', 'A_Baseline_SpatialASL']:
        phantom_json_candidates.append(v7_dir / exp / 'phantom_eval_results' / 'phantom_eval_summary.json')

    for candidate in phantom_json_candidates:
        if candidate.exists():
            with open(candidate) as f:
                loaded = json.load(f)
            # The eval script nests results under 'summary' key
            if 'summary' in loaded:
                phantom_results = loaded['summary']
            else:
                phantom_results = loaded
            print(f"Loaded phantom results from {candidate}")
            break
    else:
        print(f"WARNING: No phantom results found. Phantom figures will be skipped.")

    # Load in-vivo results
    invivo_results = {}
    invivo_json = results_dir / 'invivo_results.json'
    if invivo_json.exists():
        with open(invivo_json) as f:
            invivo_results = json.load(f)
        print(f"Loaded in-vivo results from {invivo_json}")
    else:
        print(f"WARNING: {invivo_json} not found. In-vivo figures will be skipped.")

    print()
    generated = []

    # Figure 1: nBias/CoV/nRMSE vs SNR
    if phantom_results:
        try:
            figure_1_bias_cov_nrmse(phantom_results, output_dir)
            generated.append('fig1_bias_cov_nrmse_vs_snr')
        except Exception as e:
            print(f"  ERROR generating Figure 1: {e}")

    # Figure 2: Example phantom maps
    if phantom_results and phantom_dir.exists():
        try:
            figure_2_phantom_maps(phantom_dir, phantom_results, output_dir)
            generated.append('fig2_phantom_maps')
        except Exception as e:
            print(f"  ERROR generating Figure 2: {e}")

    # Figure 3: In-vivo comparison
    if invivo_results:
        try:
            figure_3_invivo_comparison(invivo_results, results_dir, output_dir)
            generated.append('fig3_invivo_comparison')
        except Exception as e:
            print(f"  ERROR generating Figure 3: {e}")

    # Figure 4: Win rate vs SNR
    if phantom_results:
        try:
            figure_4_win_rate(phantom_results, output_dir)
            generated.append('fig4_win_rate_vs_snr')
        except Exception as e:
            print(f"  ERROR generating Figure 4: {e}")

    # Figure 5: Speed comparison
    if phantom_results or invivo_results:
        try:
            figure_5_speed(phantom_results, invivo_results, output_dir)
            generated.append('fig5_speed_comparison')
        except Exception as e:
            print(f"  ERROR generating Figure 5: {e}")

    print(f"\n{'='*50}")
    print(f"Generated {len(generated)} figures:")
    for name in generated:
        print(f"  - {name}")
    if not generated:
        print("  None (run evaluation scripts first)")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()

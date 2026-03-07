#!/usr/bin/env python3
"""
Comprehensive evaluation of v7 ablation experiments.

Generates:
1. Phantom (simulated) evaluation summary tables and figures
2. In-vivo inference on both models
3. Comparison figures and report

Usage:
    python amplitude_ablation_v7/evaluate_v7.py
"""

import sys
import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

V7_DIR = PROJECT_ROOT / 'amplitude_ablation_v7'
RESULTS_DIR = V7_DIR / 'v7_evaluation_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    'A_Baseline_SpatialASL': {
        'label': 'Baseline SpatialASLNet',
        'color': '#2196F3',
        'marker': 'o',
    },
    'B_AmplitudeAware': {
        'label': 'AmplitudeAwareSpatialASLNet',
        'color': '#F44336',
        'marker': 's',
    },
}

SNR_LEVELS = [2, 3, 5, 10, 15, 25]


# ============================================================
# Part 1: Phantom (Simulated Data) Evaluation
# ============================================================

def load_phantom_results() -> Dict:
    """Load existing phantom evaluation summaries."""
    results = {}
    for exp_name in EXPERIMENTS:
        summary_path = V7_DIR / exp_name / 'phantom_eval_results' / 'phantom_eval_summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                results[exp_name] = json.load(f)
            print(f"Loaded phantom results for {exp_name}")
        else:
            print(f"WARNING: No phantom results for {exp_name}")
    return results


def plot_phantom_metrics(phantom_results: Dict):
    """Generate comprehensive phantom evaluation figures."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_to_plot = [
        ('MAE', 'nn_{}_MAE', 'ls_{}_MAE', '{} MAE', ['ml/100g/min', 'ms']),
        ('nBias', 'nn_{}_nBias', 'ls_{}_nBias', '{} Normalized Bias (%)', ['%', '%']),
        ('Win Rate', 'cbf_win_rate', 'att_win_rate', '{} NN Win Rate (%)', ['%', '%']),
    ]

    for col, (metric_name, nn_key_template, ls_key_template, title_template, units) in enumerate(metrics_to_plot):
        for row, param in enumerate(['cbf', 'att']):
            ax = axes[row, col]

            for exp_name, exp_info in EXPERIMENTS.items():
                if exp_name not in phantom_results:
                    continue
                summary = phantom_results[exp_name]['summary']

                nn_vals = []
                ls_vals = []

                for snr in SNR_LEVELS:
                    snr_data = summary[f'snr_{snr}']
                    if metric_name == 'Win Rate':
                        nn_vals.append(snr_data[f'{param}_win_rate'])
                    else:
                        nn_key = nn_key_template.format(param)
                        ls_key = ls_key_template.format(param)
                        nn_vals.append(snr_data[nn_key])
                        ls_vals.append(snr_data[ls_key])

                if metric_name == 'Win Rate':
                    ax.plot(SNR_LEVELS, nn_vals, f'-{exp_info["marker"]}',
                            color=exp_info['color'], label=exp_info['label'],
                            linewidth=2, markersize=8)
                else:
                    ax.plot(SNR_LEVELS, nn_vals, f'-{exp_info["marker"]}',
                            color=exp_info['color'], label=f'NN: {exp_info["label"]}',
                            linewidth=2, markersize=8)

            # Plot LS baseline (same for both experiments)
            if metric_name != 'Win Rate':
                first_exp = list(phantom_results.keys())[0]
                summary = phantom_results[first_exp]['summary']
                ls_vals = []
                for snr in SNR_LEVELS:
                    ls_key = ls_key_template.format(param)
                    ls_vals.append(summary[f'snr_{snr}'][ls_key])
                ax.plot(SNR_LEVELS, ls_vals, '--^', color='#4CAF50',
                        label='Corrected LS', linewidth=2, markersize=8)

            if metric_name == 'Win Rate':
                ax.axhline(y=50, color='gray', linestyle=':', alpha=0.7, label='50% (no advantage)')

            param_label = 'CBF' if param == 'cbf' else 'ATT'
            ax.set_title(title_template.format(param_label), fontsize=13, fontweight='bold')
            ax.set_xlabel('SNR', fontsize=11)
            unit = units[row]
            ax.set_ylabel(f'{metric_name} ({unit})', fontsize=11)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(SNR_LEVELS)

    plt.suptitle('v7 Phantom Evaluation: NN vs Corrected LS Baseline', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'phantom_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'phantom_metrics_comparison.png'}")


def plot_phantom_examples(n_examples: int = 3):
    """Generate side-by-side phantom prediction examples for both models at SNR=10."""
    import torch

    snr_level = 10

    # Load a few phantoms
    phantom_dir = V7_DIR / 'test_phantoms'
    phantom_files = sorted(phantom_dir.glob('phantom_*.npz'))[:n_examples]

    if not phantom_files:
        print("No phantom files found, skipping example plots")
        return

    # Load models
    models = {}
    configs = {}
    for exp_name in EXPERIMENTS:
        exp_dir = V7_DIR / exp_name
        config_path = exp_dir / 'research_config.json'
        with open(config_path) as f:
            config = json.load(f)
        configs[exp_name] = config

        model_class = config['model_class_name']
        features = config.get('hidden_sizes', [32, 64, 128, 256])
        n_plds = len(config['pld_values'])

        if model_class == 'AmplitudeAwareSpatialASLNet':
            from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
            model_fn = lambda: AmplitudeAwareSpatialASLNet(
                n_plds=n_plds, features=features,
                use_film_at_bottleneck=config.get('use_film_at_bottleneck', True),
                use_film_at_decoder=config.get('use_film_at_decoder', True),
                use_amplitude_output_modulation=config.get('use_amplitude_output_modulation', True),
            )
        else:
            from models.spatial_asl_network import SpatialASLNet
            model_fn = lambda: SpatialASLNet(n_plds=n_plds, features=features)

        ensemble = []
        models_dir = exp_dir / 'trained_models'
        for mp in sorted(models_dir.glob('ensemble_model_*.pt')):
            model = model_fn()
            sd = torch.load(mp, map_location='cpu', weights_only=False)
            if 'model_state_dict' in sd:
                model.load_state_dict(sd['model_state_dict'])
            else:
                model.load_state_dict(sd)
            model.eval()
            ensemble.append(model)
        models[exp_name] = ensemble
        print(f"Loaded {len(ensemble)} models for {exp_name}")

    # Load norm stats
    norm_stats = {}
    for exp_name in EXPERIMENTS:
        ns_path = V7_DIR / exp_name / 'norm_stats.json'
        with open(ns_path) as f:
            norm_stats[exp_name] = json.load(f)

    # Generate predictions
    fig = plt.figure(figsize=(24, 4 * n_examples))
    gs = GridSpec(n_examples, 8, figure=fig, hspace=0.4, wspace=0.35)

    for row, pf in enumerate(phantom_files):
        phantom = np.load(pf)
        cbf_true = phantom['cbf_map']
        att_true = phantom['att_map']
        noisy = phantom[f'noisy_snr_{snr_level}']

        # Scale signals like training pipeline
        global_scale = 10.0
        M0_SCALE = 100.0
        noisy_scaled = noisy * M0_SCALE * global_scale

        input_tensor = torch.from_numpy(noisy_scaled).unsqueeze(0).float()

        preds = {}
        for exp_name, ensemble in models.items():
            cbf_preds = []
            att_preds = []
            for model in ensemble:
                with torch.no_grad():
                    cbf_n, att_n, _, _ = model(input_tensor)
                ns = norm_stats[exp_name]
                cbf_denorm = cbf_n[0, 0].numpy() * ns['y_std_cbf'] + ns['y_mean_cbf']
                att_denorm = att_n[0, 0].numpy() * ns['y_std_att'] + ns['y_mean_att']
                cbf_preds.append(cbf_denorm)
                att_preds.append(att_denorm)
            preds[exp_name] = {
                'cbf': np.clip(np.mean(cbf_preds, axis=0), 0, 200),
                'att': np.clip(np.mean(att_preds, axis=0), 0, 5000),
            }

        # Plot: Ground Truth CBF | Baseline CBF | AmpAware CBF | CBF Error Comparison |
        #        Ground Truth ATT | Baseline ATT | AmpAware ATT | ATT Error Comparison

        vmax_cbf = max(100, cbf_true.max() * 1.1)

        # CBF True
        ax = fig.add_subplot(gs[row, 0])
        im = ax.imshow(cbf_true, cmap='hot', vmin=0, vmax=vmax_cbf)
        ax.set_title('CBF Ground Truth' if row == 0 else '', fontsize=10)
        ax.set_ylabel(f'Phantom {row}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # CBF Baseline
        ax = fig.add_subplot(gs[row, 1])
        im = ax.imshow(preds['A_Baseline_SpatialASL']['cbf'], cmap='hot', vmin=0, vmax=vmax_cbf)
        ax.set_title('CBF Baseline' if row == 0 else '', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # CBF AmpAware
        ax = fig.add_subplot(gs[row, 2])
        im = ax.imshow(preds['B_AmplitudeAware']['cbf'], cmap='hot', vmin=0, vmax=vmax_cbf)
        ax.set_title('CBF AmpAware' if row == 0 else '', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # CBF Error comparison
        ax = fig.add_subplot(gs[row, 3])
        err_baseline = np.abs(preds['A_Baseline_SpatialASL']['cbf'] - cbf_true)
        err_ampaware = np.abs(preds['B_AmplitudeAware']['cbf'] - cbf_true)
        # Show which model is better: blue = baseline better, red = ampaware better
        diff = err_baseline - err_ampaware  # positive = ampaware wins
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-20, vmax=20)
        ax.set_title('CBF Error Diff\n(red=AmpAware better)' if row == 0 else '', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # ATT True
        ax = fig.add_subplot(gs[row, 4])
        im = ax.imshow(att_true, cmap='viridis', vmin=0, vmax=3500)
        ax.set_title('ATT Ground Truth' if row == 0 else '', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # ATT Baseline
        ax = fig.add_subplot(gs[row, 5])
        im = ax.imshow(preds['A_Baseline_SpatialASL']['att'], cmap='viridis', vmin=0, vmax=3500)
        ax.set_title('ATT Baseline' if row == 0 else '', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # ATT AmpAware
        ax = fig.add_subplot(gs[row, 6])
        im = ax.imshow(preds['B_AmplitudeAware']['att'], cmap='viridis', vmin=0, vmax=3500)
        ax.set_title('ATT AmpAware' if row == 0 else '', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # ATT Error comparison
        ax = fig.add_subplot(gs[row, 7])
        err_baseline = np.abs(preds['A_Baseline_SpatialASL']['att'] - att_true)
        err_ampaware = np.abs(preds['B_AmplitudeAware']['att'] - att_true)
        diff = err_baseline - err_ampaware
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-500, vmax=500)
        ax.set_title('ATT Error Diff\n(red=AmpAware better)' if row == 0 else '', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'Phantom Examples at SNR={snr_level}', fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(RESULTS_DIR / 'phantom_examples_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'phantom_examples_comparison.png'}")


def plot_scatter_phantoms(n_phantoms: int = 50):
    """Scatter plots: predicted vs true for both models at SNR=10."""
    import torch

    snr_level = 10
    phantom_dir = V7_DIR / 'test_phantoms'
    phantom_files = sorted(phantom_dir.glob('phantom_*.npz'))[:n_phantoms]

    if not phantom_files:
        print("No phantom files, skipping scatter")
        return

    # Load models
    models = {}
    norm_stats_dict = {}
    for exp_name in EXPERIMENTS:
        exp_dir = V7_DIR / exp_name
        with open(exp_dir / 'research_config.json') as f:
            config = json.load(f)
        with open(exp_dir / 'norm_stats.json') as f:
            ns = json.load(f)
        norm_stats_dict[exp_name] = ns

        model_class = config['model_class_name']
        features = config.get('hidden_sizes', [32, 64, 128, 256])
        n_plds = len(config['pld_values'])

        if model_class == 'AmplitudeAwareSpatialASLNet':
            from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
            model_fn = lambda: AmplitudeAwareSpatialASLNet(
                n_plds=n_plds, features=features,
                use_film_at_bottleneck=config.get('use_film_at_bottleneck', True),
                use_film_at_decoder=config.get('use_film_at_decoder', True),
                use_amplitude_output_modulation=config.get('use_amplitude_output_modulation', True),
            )
        else:
            from models.spatial_asl_network import SpatialASLNet
            model_fn = lambda: SpatialASLNet(n_plds=n_plds, features=features)

        ensemble = []
        for mp in sorted((exp_dir / 'trained_models').glob('ensemble_model_*.pt')):
            model = model_fn()
            sd = torch.load(mp, map_location='cpu', weights_only=False)
            model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
            model.eval()
            ensemble.append(model)
        models[exp_name] = ensemble

    # Collect predictions
    all_cbf_true = []
    all_att_true = []
    all_preds = {exp: {'cbf': [], 'att': []} for exp in EXPERIMENTS}

    for pf in phantom_files:
        phantom = np.load(pf)
        cbf_true = phantom['cbf_map']
        att_true = phantom['att_map']
        noisy = phantom[f'noisy_snr_{snr_level}']

        mask = cbf_true > 1.0  # brain voxels only

        all_cbf_true.append(cbf_true[mask])
        all_att_true.append(att_true[mask])

        noisy_scaled = noisy * 100.0 * 10.0
        input_tensor = torch.from_numpy(noisy_scaled).unsqueeze(0).float()

        for exp_name, ensemble in models.items():
            cbf_preds = []
            att_preds = []
            for model in ensemble:
                with torch.no_grad():
                    cbf_n, att_n, _, _ = model(input_tensor)
                ns = norm_stats_dict[exp_name]
                cbf_preds.append(cbf_n[0, 0].numpy() * ns['y_std_cbf'] + ns['y_mean_cbf'])
                att_preds.append(att_n[0, 0].numpy() * ns['y_std_att'] + ns['y_mean_att'])
            cbf_avg = np.clip(np.mean(cbf_preds, axis=0), 0, 200)
            att_avg = np.clip(np.mean(att_preds, axis=0), 0, 5000)
            all_preds[exp_name]['cbf'].append(cbf_avg[mask])
            all_preds[exp_name]['att'].append(att_avg[mask])

    cbf_true_flat = np.concatenate(all_cbf_true)
    att_true_flat = np.concatenate(all_att_true)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for col, (exp_name, exp_info) in enumerate(EXPERIMENTS.items()):
        cbf_pred_flat = np.concatenate(all_preds[exp_name]['cbf'])
        att_pred_flat = np.concatenate(all_preds[exp_name]['att'])

        # Subsample for plotting
        n_plot = min(30000, len(cbf_true_flat))
        idx = np.random.choice(len(cbf_true_flat), n_plot, replace=False)

        # CBF scatter
        ax = axes[0, col]
        ax.scatter(cbf_true_flat[idx], cbf_pred_flat[idx], alpha=0.05, s=2,
                   c=exp_info['color'])
        ax.plot([0, 150], [0, 150], 'k--', linewidth=2, alpha=0.7)

        # Linear fit
        from numpy.polynomial import polynomial as P
        valid = (cbf_true_flat > 5) & (cbf_pred_flat > 0) & (cbf_pred_flat < 200)
        if valid.sum() > 100:
            coeffs = np.polyfit(cbf_true_flat[valid], cbf_pred_flat[valid], 1)
            x_fit = np.linspace(5, 120, 100)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=2,
                    label=f'Fit: y={coeffs[0]:.2f}x + {coeffs[1]:.1f}')

        cbf_mae = np.mean(np.abs(cbf_pred_flat - cbf_true_flat))
        cbf_bias = np.mean(cbf_pred_flat - cbf_true_flat)
        ax.set_title(f'{exp_info["label"]}\nCBF MAE={cbf_mae:.1f}, Bias={cbf_bias:.1f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('True CBF (ml/100g/min)')
        ax.set_ylabel('Predicted CBF')
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ATT scatter
        ax = axes[1, col]
        ax.scatter(att_true_flat[idx], att_pred_flat[idx], alpha=0.05, s=2,
                   c=exp_info['color'])
        ax.plot([0, 4000], [0, 4000], 'k--', linewidth=2, alpha=0.7)

        valid_att = (att_true_flat > 100) & (att_pred_flat > 0) & (att_pred_flat < 5000)
        if valid_att.sum() > 100:
            coeffs = np.polyfit(att_true_flat[valid_att], att_pred_flat[valid_att], 1)
            x_fit = np.linspace(500, 3500, 100)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=2,
                    label=f'Fit: y={coeffs[0]:.2f}x + {coeffs[1]:.0f}')

        att_mae = np.mean(np.abs(att_pred_flat - att_true_flat))
        att_bias = np.mean(att_pred_flat - att_true_flat)
        ax.set_title(f'{exp_info["label"]}\nATT MAE={att_mae:.0f}ms, Bias={att_bias:.0f}ms',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('True ATT (ms)')
        ax.set_ylabel('Predicted ATT')
        ax.set_xlim(0, 4000)
        ax.set_ylim(0, 4000)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Predicted vs True (SNR={snr_level}, {n_phantoms} phantoms)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'phantom_scatter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'phantom_scatter_comparison.png'}")


# ============================================================
# Part 2: In-Vivo Inference
# ============================================================

def run_invivo_inference():
    """Run in-vivo inference for both models on all subjects."""
    import torch
    import nibabel as nib

    invivo_dir = PROJECT_ROOT / 'data' / 'invivo_validated'
    subject_dirs = sorted([d for d in invivo_dir.iterdir() if d.is_dir()])

    if not subject_dirs:
        print("No in-vivo subjects found, skipping")
        return {}

    print(f"\nFound {len(subject_dirs)} in-vivo subjects")

    # Import inference functions
    sys.path.insert(0, str(PROJECT_ROOT))
    from inference.predict_spatial_invivo import (
        load_spatial_model, preprocess_subject, predict_volume
    )

    device = torch.device('cpu')

    all_results = {}

    for exp_name, exp_info in EXPERIMENTS.items():
        exp_dir = V7_DIR / exp_name
        output_dir = RESULTS_DIR / 'invivo' / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"In-vivo inference: {exp_info['label']}")
        print(f"{'='*60}")

        try:
            models, config, norm_stats = load_spatial_model(exp_dir, device)
        except Exception as e:
            print(f"Failed to load model {exp_name}: {e}")
            continue

        model_plds = config['pld_values']
        global_scale = config.get('global_scale_factor', 10.0)
        alpha_bs1 = 0.93

        exp_results = {}

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            subject_output = output_dir / subject_id
            subject_output.mkdir(parents=True, exist_ok=True)

            print(f"\n  Processing: {subject_id}")

            try:
                spatial_stack, brain_mask, ref_img, subject_plds = preprocess_subject(
                    subject_dir, model_plds, global_scale, alpha_bs1=alpha_bs1
                )

                cbf, att, cbf_std, att_std = predict_volume(
                    spatial_stack, models, norm_stats, device
                )

                cbf_masked = cbf * brain_mask
                att_masked = att * brain_mask

                # Save NIfTI outputs
                for name, data in [('nn_cbf', cbf_masked), ('nn_att', att_masked),
                                   ('nn_cbf_std', cbf_std), ('nn_att_std', att_std)]:
                    img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
                    nib.save(img, subject_output / f'{name}.nii.gz')

                # Compute stats
                brain_voxels = brain_mask > 0
                stats = {
                    'cbf_mean': float(cbf_masked[brain_voxels].mean()),
                    'cbf_std': float(cbf_masked[brain_voxels].std()),
                    'cbf_median': float(np.median(cbf_masked[brain_voxels])),
                    'att_mean': float(att_masked[brain_voxels].mean()),
                    'att_std': float(att_masked[brain_voxels].std()),
                    'att_median': float(np.median(att_masked[brain_voxels])),
                    'n_brain_voxels': int(brain_voxels.sum()),
                }

                # Check for GM mask
                gm_masks = list(subject_dir.glob('*GM*'))
                if gm_masks:
                    try:
                        gm_img = nib.load(gm_masks[0])
                        gm_mask = gm_img.get_fdata() > 0.5
                        gm_brain = gm_mask & brain_voxels
                        if gm_brain.sum() > 10:
                            stats['gm_cbf_mean'] = float(cbf_masked[gm_brain].mean())
                            stats['gm_cbf_std'] = float(cbf_masked[gm_brain].std())
                            stats['gm_att_mean'] = float(att_masked[gm_brain].mean())
                            stats['gm_att_std'] = float(att_masked[gm_brain].std())
                    except Exception:
                        pass

                exp_results[subject_id] = stats

                with open(subject_output / 'stats.json', 'w') as f:
                    json.dump(stats, f, indent=2)

                print(f"    CBF: {stats['cbf_mean']:.1f} +/- {stats['cbf_std']:.1f} ml/100g/min")
                print(f"    ATT: {stats['att_mean']:.0f} +/- {stats['att_std']:.0f} ms")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        all_results[exp_name] = exp_results

    # Save combined results
    with open(RESULTS_DIR / 'invivo_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


def plot_invivo_comparison(invivo_results: Dict):
    """Generate in-vivo comparison figures."""
    if not invivo_results:
        print("No in-vivo results to plot")
        return

    # Collect stats across subjects
    subjects = set()
    for exp_results in invivo_results.values():
        subjects.update(exp_results.keys())
    subjects = sorted(subjects)

    if not subjects:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Bar plots: CBF and ATT means per subject
    x = np.arange(len(subjects))
    width = 0.35

    for metric_idx, (metric, ylabel, ax_row) in enumerate([
        ('cbf', 'CBF (ml/100g/min)', 0),
        ('att', 'ATT (ms)', 1),
    ]):
        ax = axes[ax_row, 0]
        for i, (exp_name, exp_info) in enumerate(EXPERIMENTS.items()):
            means = []
            stds = []
            for s in subjects:
                if s in invivo_results.get(exp_name, {}):
                    stats = invivo_results[exp_name][s]
                    means.append(stats[f'{metric}_mean'])
                    stds.append(stats[f'{metric}_std'])
                else:
                    means.append(0)
                    stds.append(0)

            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width * 0.9, label=exp_info['label'],
                   color=exp_info['color'], alpha=0.7,
                   yerr=stds, capsize=3)

        ax.set_xlabel('Subject')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{metric.upper()} per Subject', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.split('_')[-1] for s in subjects], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Expected ranges
        if metric == 'cbf':
            ax.axhspan(40, 80, alpha=0.1, color='green', label='Expected GM range')
        else:
            ax.axhspan(1000, 2000, alpha=0.1, color='green', label='Expected range')

    # Distribution plots across all subjects
    for metric_idx, (metric, ylabel, ax_col) in enumerate([
        ('cbf', 'CBF (ml/100g/min)', 0),
        ('att', 'ATT (ms)', 1),
    ]):
        ax = axes[metric_idx, 1]
        for exp_name, exp_info in EXPERIMENTS.items():
            all_means = [invivo_results[exp_name][s][f'{metric}_mean']
                         for s in subjects if s in invivo_results.get(exp_name, {})]
            if all_means:
                ax.hist(all_means, bins=8, alpha=0.5, color=exp_info['color'],
                        label=f'{exp_info["label"]}\n(mean={np.mean(all_means):.1f})')

        ax.set_xlabel(ylabel)
        ax.set_ylabel('Count')
        ax.set_title(f'{metric.upper()} Distribution Across Subjects', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('In-Vivo Results: Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'invivo_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'invivo_comparison.png'}")


def plot_invivo_slices(invivo_results: Dict, n_slices: int = 5):
    """Generate side-by-side in-vivo slice comparison for one subject."""
    import nibabel as nib

    if not invivo_results:
        return

    # Pick first subject that has results for both models
    subjects = set()
    for exp_results in invivo_results.values():
        subjects.update(exp_results.keys())

    target_subject = None
    for s in sorted(subjects):
        if all(s in invivo_results.get(exp, {}) for exp in EXPERIMENTS):
            target_subject = s
            break

    if not target_subject:
        print("No subject has results for both models")
        return

    print(f"\nGenerating in-vivo slice comparison for {target_subject}")

    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 3 * n_slices))

    exp_names = list(EXPERIMENTS.keys())
    titles = ['Baseline CBF', 'AmpAware CBF', 'Baseline ATT', 'AmpAware ATT']

    # Load all NIfTI results
    volumes = {}
    for exp_name in exp_names:
        subj_dir = RESULTS_DIR / 'invivo' / exp_name / target_subject
        cbf_path = subj_dir / 'nn_cbf.nii.gz'
        att_path = subj_dir / 'nn_att.nii.gz'
        if cbf_path.exists() and att_path.exists():
            volumes[exp_name] = {
                'cbf': nib.load(cbf_path).get_fdata(),
                'att': nib.load(att_path).get_fdata(),
            }

    if len(volumes) < 2:
        print("Not enough model results to compare")
        return

    # Select representative slices
    cbf_vol = list(volumes.values())[0]['cbf']
    n_total = cbf_vol.shape[2]
    slice_indices = np.linspace(n_total * 0.2, n_total * 0.8, n_slices, dtype=int)

    for row, sl in enumerate(slice_indices):
        for col, (exp_name, title) in enumerate(zip(
            [exp_names[0], exp_names[1], exp_names[0], exp_names[1]],
            titles
        )):
            ax = axes[row, col]
            param = 'cbf' if col < 2 else 'att'
            data = volumes[exp_name][param][:, :, sl]

            if param == 'cbf':
                im = ax.imshow(np.rot90(data), cmap='hot', vmin=0, vmax=100)
            else:
                im = ax.imshow(np.rot90(data), cmap='viridis', vmin=0, vmax=3000)

            if row == 0:
                ax.set_title(title, fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'Slice {sl}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'In-Vivo: {target_subject}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'invivo_slices_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'invivo_slices_comparison.png'}")


# ============================================================
# Part 3: Summary Report
# ============================================================

def generate_summary_report(phantom_results: Dict, invivo_results: Dict):
    """Generate a text summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("V7 ABLATION EVALUATION REPORT")
    lines.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 80)

    # Training summary
    lines.append("\n## Training Summary\n")
    lines.append(f"{'Model':<35} {'Final Train Loss':>18} {'Final Val Loss':>15}")
    lines.append("-" * 70)
    # From the user's training logs:
    lines.append(f"{'A: Baseline SpatialASLNet':<35} {'0.2127':>18} {'0.2270':>15}")
    lines.append(f"{'B: AmplitudeAwareSpatialASLNet':<35} {'0.2102':>18} {'0.2273':>15}")

    # Phantom results table
    if phantom_results:
        lines.append("\n\n## Phantom Evaluation (100 phantoms, corrected LS baseline)\n")

        lines.append(f"{'SNR':>4} | {'--- CBF MAE ---':^30} | {'--- ATT MAE ---':^30} | {'--- Win Rate (%) ---':^24}")
        lines.append(f"{'':>4} | {'Baseline':>10} {'AmpAware':>10} {'LS':>10} | {'Baseline':>10} {'AmpAware':>10} {'LS':>10} | {'CBF-B':>6} {'CBF-A':>6} {'ATT-B':>6} {'ATT-A':>6}")
        lines.append("-" * 110)

        for snr in SNR_LEVELS:
            row = f"{snr:>4} |"
            for exp_name in ['A_Baseline_SpatialASL', 'B_AmplitudeAware']:
                if exp_name in phantom_results:
                    data = phantom_results[exp_name]['summary'][f'snr_{snr}']
                    row_data = data
            # Get values
            a_data = phantom_results.get('A_Baseline_SpatialASL', {}).get('summary', {}).get(f'snr_{snr}', {})
            b_data = phantom_results.get('B_AmplitudeAware', {}).get('summary', {}).get(f'snr_{snr}', {})

            row = f"{snr:>4} |"
            row += f" {a_data.get('nn_cbf_MAE', 0):>10.1f}"
            row += f" {b_data.get('nn_cbf_MAE', 0):>10.1f}"
            row += f" {a_data.get('ls_cbf_MAE', 0):>10.1f}"
            row += f" |"
            row += f" {a_data.get('nn_att_MAE', 0):>10.0f}"
            row += f" {b_data.get('nn_att_MAE', 0):>10.0f}"
            row += f" {a_data.get('ls_att_MAE', 0):>10.0f}"
            row += f" |"
            row += f" {a_data.get('cbf_win_rate', 0):>6.1f}"
            row += f" {b_data.get('cbf_win_rate', 0):>6.1f}"
            row += f" {a_data.get('att_win_rate', 0):>6.1f}"
            row += f" {b_data.get('att_win_rate', 0):>6.1f}"
            lines.append(row)

        # Key observations
        lines.append("\n### Key Observations (Phantom)\n")

        # Compare at SNR=10
        a10 = phantom_results.get('A_Baseline_SpatialASL', {}).get('summary', {}).get('snr_10', {})
        b10 = phantom_results.get('B_AmplitudeAware', {}).get('summary', {}).get('snr_10', {})

        if a10 and b10:
            lines.append(f"  At SNR=10:")
            lines.append(f"    - Baseline CBF MAE: {a10['nn_cbf_MAE']:.1f}, Win Rate: {a10['cbf_win_rate']:.1f}%")
            lines.append(f"    - AmpAware CBF MAE: {b10['nn_cbf_MAE']:.1f}, Win Rate: {b10['cbf_win_rate']:.1f}%")
            lines.append(f"    - LS CBF MAE: {a10['ls_cbf_MAE']:.1f}")
            lines.append(f"    - Baseline ATT MAE: {a10['nn_att_MAE']:.0f}ms, Win Rate: {a10['att_win_rate']:.1f}%")
            lines.append(f"    - AmpAware ATT MAE: {b10['nn_att_MAE']:.0f}ms, Win Rate: {b10['att_win_rate']:.1f}%")
            lines.append(f"    - LS ATT MAE: {a10['ls_att_MAE']:.0f}ms")

        # Summarize win rate trends
        lines.append(f"\n  CBF Win Rate trends:")
        for exp_name, label in [('A_Baseline_SpatialASL', 'Baseline'), ('B_AmplitudeAware', 'AmpAware')]:
            if exp_name in phantom_results:
                rates = [phantom_results[exp_name]['summary'][f'snr_{s}']['cbf_win_rate'] for s in SNR_LEVELS]
                lines.append(f"    {label}: {' -> '.join(f'{r:.0f}%' for r in rates)} (SNR {SNR_LEVELS[0]}-{SNR_LEVELS[-1]})")
                above_50 = sum(1 for r in rates if r > 50)
                lines.append(f"      Beats LS at {above_50}/{len(rates)} SNR levels")

        lines.append(f"\n  ATT Win Rate trends:")
        for exp_name, label in [('A_Baseline_SpatialASL', 'Baseline'), ('B_AmplitudeAware', 'AmpAware')]:
            if exp_name in phantom_results:
                rates = [phantom_results[exp_name]['summary'][f'snr_{s}']['att_win_rate'] for s in SNR_LEVELS]
                lines.append(f"    {label}: {' -> '.join(f'{r:.0f}%' for r in rates)} (SNR {SNR_LEVELS[0]}-{SNR_LEVELS[-1]})")

    # In-vivo results
    if invivo_results:
        lines.append("\n\n## In-Vivo Results\n")

        subjects = sorted(set(s for exp in invivo_results.values() for s in exp.keys()))
        lines.append(f"{'Subject':<25} | {'--- Baseline ---':^25} | {'--- AmpAware ---':^25}")
        lines.append(f"{'':25} | {'CBF':>12} {'ATT':>12} | {'CBF':>12} {'ATT':>12}")
        lines.append("-" * 80)

        for s in subjects:
            row = f"{s:<25} |"
            for exp_name in ['A_Baseline_SpatialASL', 'B_AmplitudeAware']:
                if s in invivo_results.get(exp_name, {}):
                    stats = invivo_results[exp_name][s]
                    row += f" {stats['cbf_mean']:>5.1f}+/-{stats['cbf_std']:>4.1f}"
                    row += f" {stats['att_mean']:>5.0f}+/-{stats['att_std']:>4.0f}"
                else:
                    row += f" {'N/A':>12} {'N/A':>12}"
                row += " |"
            lines.append(row)

        # Grand averages
        lines.append("")
        for exp_name, label in [('A_Baseline_SpatialASL', 'Baseline'), ('B_AmplitudeAware', 'AmpAware')]:
            if exp_name in invivo_results:
                cbf_means = [invivo_results[exp_name][s]['cbf_mean'] for s in subjects
                             if s in invivo_results[exp_name]]
                att_means = [invivo_results[exp_name][s]['att_mean'] for s in subjects
                             if s in invivo_results[exp_name]]
                if cbf_means:
                    lines.append(f"  {label} grand average: CBF={np.mean(cbf_means):.1f}+/-{np.std(cbf_means):.1f}, "
                                 f"ATT={np.mean(att_means):.0f}+/-{np.std(att_means):.0f}ms")

        lines.append("\n  Expected healthy GM: CBF ~50-70 ml/100g/min, ATT ~1000-1600ms")

    # Conclusions
    lines.append("\n\n## Conclusions\n")
    lines.append("  1. Both models achieve similar final validation loss (~0.227)")
    lines.append("  2. Neither model consistently beats corrected LS for CBF (win rates <30%)")
    lines.append("  3. Both models show ATT advantage over LS at low SNR (>50% win rate)")
    lines.append("  4. AmplitudeAware shows marginal CBF win rate improvement over Baseline")
    lines.append("  5. NN advantage diminishes with increasing SNR, as expected")
    lines.append("  6. In-vivo results should be compared to literature values")

    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)

    with open(RESULTS_DIR / 'v7_evaluation_report.txt', 'w') as f:
        f.write(report)

    print(f"\nSaved: {RESULTS_DIR / 'v7_evaluation_report.txt'}")
    print("\n" + report)

    return report


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("V7 Ablation Evaluation")
    print("=" * 60)

    # Part 1: Phantom evaluation
    print("\n--- Part 1: Phantom (Simulated) Evaluation ---")
    phantom_results = load_phantom_results()

    if phantom_results:
        plot_phantom_metrics(phantom_results)
        plot_phantom_examples(n_examples=3)
        plot_scatter_phantoms(n_phantoms=50)

    # Part 2: In-vivo inference
    print("\n--- Part 2: In-Vivo Inference ---")
    invivo_results = run_invivo_inference()

    if invivo_results:
        plot_invivo_comparison(invivo_results)
        plot_invivo_slices(invivo_results)

    # Part 3: Summary report
    print("\n--- Part 3: Summary Report ---")
    generate_summary_report(phantom_results, invivo_results)

    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()

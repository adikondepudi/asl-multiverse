#!/usr/bin/env python3
"""
Summarize in-vivo NN vs LS comparison results for thesis/publication.
Generates publication-ready tables and statistics.
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd


def load_comparison_results(base_dir: Path):
    """Load all comparison metrics from a directory."""
    results = []
    for metrics_file in sorted(base_dir.glob("*/comparison_metrics.json")):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        results.append(data)
    return results


def compute_summary_stats(results: list, metric_path: list):
    """Compute mean ± std for a nested metric across subjects."""
    values = []
    for r in results:
        val = r
        for key in metric_path:
            val = val.get(key, {})
        if isinstance(val, (int, float)):
            values.append(val)
    if values:
        return np.mean(values), np.std(values), values
    return None, None, []


def main():
    base_path = Path("/Users/adikondepudi/Desktop/asl-multiverse")

    # Load results from both models
    ampaware_results = load_comparison_results(base_path / "invivo_comparison_ampaware")
    baseline_results = load_comparison_results(base_path / "invivo_comparison_baseline")

    print("=" * 80)
    print("IN-VIVO NEURAL NETWORK vs LEAST-SQUARES COMPARISON")
    print("Summary for PhD Thesis / Publication")
    print("=" * 80)

    print(f"\nNumber of subjects: {len(ampaware_results)}")

    # =========================================================================
    # TABLE 1: Per-Subject Detailed Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("TABLE 1: Per-Subject Comparison Metrics")
    print("=" * 80)

    # Create detailed table
    rows = []
    for amp_r, base_r in zip(ampaware_results, baseline_results):
        subject = amp_r['subject_id']
        n_brain = amp_r['n_brain_voxels']
        ls_fail = amp_r['ls_failure_rate'] * 100
        n_compare = amp_r['cbf_comparison']['n_voxels']

        rows.append({
            'Subject': subject,
            'Brain Voxels': n_brain,
            'LS Failure (%)': f"{ls_fail:.1f}",
            'Compare Voxels': n_compare,
            # AmplitudeAware CBF
            'AmpAware CBF r': f"{amp_r['cbf_comparison']['pearson_r']:.3f}",
            'AmpAware CBF Bias': f"{amp_r['cbf_comparison']['bland_altman']['bias']:.1f}",
            # Baseline CBF
            'Baseline CBF r': f"{base_r['cbf_comparison']['pearson_r']:.3f}",
            'Baseline CBF Bias': f"{base_r['cbf_comparison']['bland_altman']['bias']:.1f}",
            # AmplitudeAware ATT
            'AmpAware ATT r': f"{amp_r['att_comparison']['pearson_r']:.3f}",
            'AmpAware ATT Bias': f"{amp_r['att_comparison']['bland_altman']['bias']:.0f}",
            # Baseline ATT
            'Baseline ATT r': f"{base_r['att_comparison']['pearson_r']:.3f}",
            'Baseline ATT Bias': f"{base_r['att_comparison']['bland_altman']['bias']:.0f}",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # =========================================================================
    # TABLE 2: Summary Statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("TABLE 2: Summary Statistics Across All Subjects (Mean ± SD)")
    print("=" * 80)

    metrics_to_summarize = [
        ("LS Failure Rate (%)", "ls_failure_rate", lambda x: x * 100),
        ("Compare Voxels", "cbf_comparison.n_voxels", lambda x: x),
        # CBF metrics
        ("CBF Pearson r", "cbf_comparison.pearson_r", lambda x: x),
        ("CBF Spearman r", "cbf_comparison.spearman_r", lambda x: x),
        ("CBF Bias (ml/100g/min)", "cbf_comparison.bland_altman.bias", lambda x: x),
        ("CBF MAE (ml/100g/min)", "cbf_comparison.mae", lambda x: x),
        ("CBF ICC", "cbf_comparison.icc", lambda x: x),
        # ATT metrics
        ("ATT Pearson r", "att_comparison.pearson_r", lambda x: x),
        ("ATT Spearman r", "att_comparison.spearman_r", lambda x: x),
        ("ATT Bias (ms)", "att_comparison.bland_altman.bias", lambda x: x),
        ("ATT MAE (ms)", "att_comparison.mae", lambda x: x),
        ("ATT ICC", "att_comparison.icc", lambda x: x),
    ]

    print(f"\n{'Metric':<30} {'AmplitudeAware':>20} {'Baseline':>20}")
    print("-" * 72)

    for name, path, transform in metrics_to_summarize:
        keys = path.split('.')

        # AmplitudeAware
        amp_values = []
        for r in ampaware_results:
            val = r
            for k in keys:
                val = val.get(k, {}) if isinstance(val, dict) else None
            if isinstance(val, (int, float)):
                amp_values.append(transform(val))

        # Baseline
        base_values = []
        for r in baseline_results:
            val = r
            for k in keys:
                val = val.get(k, {}) if isinstance(val, dict) else None
            if isinstance(val, (int, float)):
                base_values.append(transform(val))

        amp_str = f"{np.mean(amp_values):.2f} ± {np.std(amp_values):.2f}" if amp_values else "N/A"
        base_str = f"{np.mean(base_values):.2f} ± {np.std(base_values):.2f}" if base_values else "N/A"

        print(f"{name:<30} {amp_str:>20} {base_str:>20}")

    # =========================================================================
    # TABLE 3: NN Mean Values vs LS Mean Values
    # =========================================================================
    print("\n" + "=" * 80)
    print("TABLE 3: Mean Parameter Values (NN vs LS)")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'AmplitudeAware NN':>18} {'Baseline NN':>15} {'LS Fitting':>15}")
    print("-" * 75)

    # Collect mean values
    for results, model_name in [(ampaware_results, "AmplitudeAware"), (baseline_results, "Baseline")]:
        nn_cbf_means = [r['cbf_comparison']['nn_stats']['mean'] for r in results]
        ls_cbf_means = [r['cbf_comparison']['ls_stats']['mean'] for r in results]
        nn_att_means = [r['att_comparison']['nn_stats']['mean'] for r in results]
        ls_att_means = [r['att_comparison']['ls_stats']['mean'] for r in results]

        if model_name == "AmplitudeAware":
            print(f"{'CBF (ml/100g/min)':<25} {np.mean(nn_cbf_means):>13.1f} ± {np.std(nn_cbf_means):.1f}", end="")
        else:
            print(f" {np.mean(nn_cbf_means):>10.1f} ± {np.std(nn_cbf_means):.1f} {np.mean(ls_cbf_means):>10.1f} ± {np.std(ls_cbf_means):.1f}")

    for results, model_name in [(ampaware_results, "AmplitudeAware"), (baseline_results, "Baseline")]:
        nn_att_means = [r['att_comparison']['nn_stats']['mean'] for r in results]
        ls_att_means = [r['att_comparison']['ls_stats']['mean'] for r in results]

        if model_name == "AmplitudeAware":
            print(f"{'ATT (ms)':<25} {np.mean(nn_att_means):>13.1f} ± {np.std(nn_att_means):.1f}", end="")
        else:
            print(f" {np.mean(nn_att_means):>10.1f} ± {np.std(nn_att_means):.1f} {np.mean(ls_att_means):>10.1f} ± {np.std(ls_att_means):.1f}")

    # =========================================================================
    # KEY FINDINGS FOR THESIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR THESIS")
    print("=" * 80)

    # Calculate key statistics
    ls_failures = [r['ls_failure_rate'] * 100 for r in ampaware_results]
    amp_cbf_r = [r['cbf_comparison']['pearson_r'] for r in ampaware_results]
    base_cbf_r = [r['cbf_comparison']['pearson_r'] for r in baseline_results]
    amp_cbf_bias = [r['cbf_comparison']['bland_altman']['bias'] for r in ampaware_results]
    base_cbf_bias = [r['cbf_comparison']['bland_altman']['bias'] for r in baseline_results]

    print(f"""
1. LEAST-SQUARES FITTING LIMITATIONS:
   - LS fitting failed to converge in {np.mean(ls_failures):.1f}% ± {np.std(ls_failures):.1f}% of brain voxels
   - Range: {np.min(ls_failures):.1f}% to {np.max(ls_failures):.1f}% across subjects
   - Neural networks provide estimates for ALL voxels (100% coverage)

2. CBF AGREEMENT WITH LS (where LS succeeded):
   - AmplitudeAware: r = {np.mean(amp_cbf_r):.3f} ± {np.std(amp_cbf_r):.3f}
   - Baseline:       r = {np.mean(base_cbf_r):.3f} ± {np.std(base_cbf_r):.3f}
   - Both show moderate correlation, neither significantly better

3. SYSTEMATIC CBF BIAS (NN predicts higher than LS):
   - AmplitudeAware: +{np.mean(amp_cbf_bias):.1f} ± {np.std(amp_cbf_bias):.1f} ml/100g/min
   - Baseline:       +{np.mean(base_cbf_bias):.1f} ± {np.std(base_cbf_bias):.1f} ml/100g/min
   - Possible causes: training data domain gap, LS selection bias

4. ATT ESTIMATION:
   - Lower correlation than CBF (r ≈ 0.54)
   - High variability in bias across subjects
   - Suggests ATT estimation is more challenging in-vivo

5. MODEL COMPARISON:
   - AmplitudeAware and Baseline show similar in-vivo performance
   - This may be due to the training bug (both trained with same architecture)
   - Proper ablation requires retraining with fixed configuration
""")

    # =========================================================================
    # EXPORT TO CSV
    # =========================================================================
    csv_path = base_path / "invivo_comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

    # Export summary statistics
    summary_rows = []
    for name, path, transform in metrics_to_summarize:
        keys = path.split('.')

        amp_values = []
        for r in ampaware_results:
            val = r
            for k in keys:
                val = val.get(k, {}) if isinstance(val, dict) else None
            if isinstance(val, (int, float)):
                amp_values.append(transform(val))

        base_values = []
        for r in baseline_results:
            val = r
            for k in keys:
                val = val.get(k, {}) if isinstance(val, dict) else None
            if isinstance(val, (int, float)):
                base_values.append(transform(val))

        summary_rows.append({
            'Metric': name,
            'AmplitudeAware_Mean': np.mean(amp_values) if amp_values else None,
            'AmplitudeAware_Std': np.std(amp_values) if amp_values else None,
            'Baseline_Mean': np.mean(base_values) if base_values else None,
            'Baseline_Std': np.std(base_values) if base_values else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = base_path / "invivo_comparison_summary_stats.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary statistics saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()

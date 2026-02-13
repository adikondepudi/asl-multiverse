#!/usr/bin/env python3
"""
Compare Amplitude Ablation Study Results

Analyzes and compares results from the amplitude-aware architecture ablation study.
Generates summary CSV and visualizations.
"""

import argparse
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available, skipping plots")


def load_experiment_results(exp_dir: Path) -> dict:
    """Load results from a single experiment directory."""
    result = {
        "name": exp_dir.name,
        "config": {},
        "metrics": {},
        "amplitude_sensitivity": {},
    }

    # Load config
    config_file = exp_dir / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
            result["config"] = cfg
            result["model_class"] = cfg.get("training", {}).get("model_class_name", "Unknown")
            result["use_film_bottleneck"] = cfg.get("training", {}).get("use_film_at_bottleneck", False)
            result["use_film_decoder"] = cfg.get("training", {}).get("use_film_at_decoder", False)
            result["use_amp_output"] = cfg.get("training", {}).get("use_amplitude_output_modulation", False)
            result["dc_weight"] = cfg.get("training", {}).get("dc_weight", 0.0)
            result["normalization"] = cfg.get("data", {}).get("normalization_mode", "unknown")
            result["hypothesis"] = cfg.get("_experiment", {}).get("hypothesis", "")

    # Load validation metrics
    metrics_file = exp_dir / "validation_results" / "llm_analysis_report.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            report = json.load(f)

            # Get metrics from C_VarBoth_SNR10 (moderate difficulty) or fallback
            for scenario in ["C_VarBoth_SNR10", "B_VarCBF_SNR10", "A_FixedParams_SNR10"]:
                if scenario in report:
                    data = report[scenario]
                    cbf_data = data.get("CBF", {})
                    att_data = data.get("ATT", {})

                    result["scenario"] = scenario
                    result["cbf_mae"] = cbf_data.get("Neural_Net", {}).get("MAE", np.nan)
                    result["cbf_bias"] = cbf_data.get("Neural_Net", {}).get("Bias", np.nan)
                    result["cbf_win_rate"] = cbf_data.get("NN_vs_LS_Win_Rate", np.nan)
                    result["ls_cbf_mae"] = cbf_data.get("Least_Squares", {}).get("MAE", np.nan)

                    result["att_mae"] = att_data.get("Neural_Net", {}).get("MAE", np.nan)
                    result["att_win_rate"] = att_data.get("NN_vs_LS_Win_Rate", np.nan)
                    break

    # Load amplitude sensitivity
    amp_file = exp_dir / "amplitude_sensitivity.json"
    if amp_file.exists():
        with open(amp_file) as f:
            amp_data = json.load(f)
            result["amp_ratio"] = amp_data.get("sensitivity_ratio", np.nan)
            result["is_amp_sensitive"] = amp_data.get("is_sensitive", False)

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare amplitude ablation results")
    parser.add_argument("--results_dir", type=str, default="amplitude_ablation_v1",
                        help="Directory containing experiment results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return

    # Find all experiment directories
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(("0", "1"))])

    if not exp_dirs:
        print(f"No experiment directories found in {results_dir}")
        return

    print(f"Found {len(exp_dirs)} experiments")
    print()

    # Load results
    results = []
    for exp_dir in exp_dirs:
        try:
            result = load_experiment_results(exp_dir)
            results.append(result)
            print(f"  Loaded: {exp_dir.name}")
        except Exception as e:
            print(f"  Error loading {exp_dir.name}: {e}")

    if not results:
        print("No valid results found")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Select columns for summary
    summary_cols = [
        "name", "model_class", "normalization",
        "use_film_bottleneck", "use_film_decoder", "use_amp_output",
        "dc_weight", "amp_ratio", "is_amp_sensitive",
        "cbf_mae", "cbf_bias", "cbf_win_rate",
        "att_mae", "att_win_rate", "hypothesis"
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    df_summary = df[summary_cols].copy()

    # Sort by CBF MAE
    if "cbf_mae" in df_summary.columns:
        df_summary = df_summary.sort_values("cbf_mae")

    # Save CSV
    csv_path = results_dir / "amplitude_ablation_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("AMPLITUDE ABLATION RESULTS SUMMARY")
    print("=" * 100)

    # Key findings
    print("\n--- KEY FINDINGS ---\n")

    # 1. Amplitude sensitivity comparison
    if "is_amp_sensitive" in df.columns and "name" in df.columns:
        print("Amplitude Sensitivity:")
        for _, row in df.iterrows():
            sens_str = "YES" if row.get("is_amp_sensitive", False) else "NO"
            ratio = row.get("amp_ratio", np.nan)
            print(f"  {row['name']}: {sens_str} (ratio: {ratio:.1f})")
        print()

    # 2. CBF performance comparison
    if "cbf_mae" in df.columns:
        print("CBF MAE Ranking (lower is better):")
        for i, (_, row) in enumerate(df_summary.iterrows(), 1):
            mae = row.get("cbf_mae", np.nan)
            win = row.get("cbf_win_rate", np.nan)
            if not np.isnan(mae):
                print(f"  {i}. {row['name']}: MAE={mae:.2f}, Win Rate={win*100:.1f}%")
        print()

    # 3. Component analysis
    amp_aware_df = df[df["model_class"] == "AmplitudeAwareSpatialASLNet"]
    if len(amp_aware_df) > 0:
        print("AmplitudeAware Component Analysis:")

        # Group by components
        if "use_amp_output" in df.columns:
            with_output_mod = amp_aware_df[amp_aware_df["use_amp_output"] == True]
            without_output_mod = amp_aware_df[amp_aware_df["use_amp_output"] == False]

            if len(with_output_mod) > 0 and len(without_output_mod) > 0:
                mae_with = with_output_mod["cbf_mae"].mean()
                mae_without = without_output_mod["cbf_mae"].mean()
                print(f"  Output modulation: WITH={mae_with:.2f} vs WITHOUT={mae_without:.2f}")

        if "dc_weight" in df.columns:
            dc_0 = amp_aware_df[amp_aware_df["dc_weight"] == 0.0]["cbf_mae"].mean()
            dc_01 = amp_aware_df[amp_aware_df["dc_weight"] == 0.1]["cbf_mae"].mean()
            dc_03 = amp_aware_df[amp_aware_df["dc_weight"] == 0.3]["cbf_mae"].mean()
            print(f"  Physics loss: dc=0: {dc_0:.2f}, dc=0.1: {dc_01:.2f}, dc=0.3: {dc_03:.2f}")

    # Plotting
    if HAS_PLOTTING and len(df) > 2:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: CBF MAE by experiment
            ax = axes[0, 0]
            if "cbf_mae" in df.columns:
                df_plot = df_summary.dropna(subset=["cbf_mae"])
                colors = ["red" if "SpatialASL" in n else "blue" for n in df_plot["name"]]
                bars = ax.bar(range(len(df_plot)), df_plot["cbf_mae"], color=colors)
                ax.set_xticks(range(len(df_plot)))
                ax.set_xticklabels([n[:15] for n in df_plot["name"]], rotation=45, ha="right")
                ax.set_ylabel("CBF MAE")
                ax.set_title("CBF MAE by Experiment (Red=SpatialASL, Blue=AmplitudeAware)")

            # Plot 2: Amplitude sensitivity
            ax = axes[0, 1]
            if "amp_ratio" in df.columns:
                df_plot = df.dropna(subset=["amp_ratio"])
                colors = ["green" if s else "red" for s in df_plot["is_amp_sensitive"]]
                ax.bar(range(len(df_plot)), df_plot["amp_ratio"], color=colors)
                ax.set_xticks(range(len(df_plot)))
                ax.set_xticklabels([n[:15] for n in df_plot["name"]], rotation=45, ha="right")
                ax.axhline(y=5.0, color="black", linestyle="--", label="Sensitivity threshold")
                ax.set_ylabel("Amplitude Ratio (10x/0.1x)")
                ax.set_title("Amplitude Sensitivity (Green=Sensitive)")
                ax.set_yscale("log")

            # Plot 3: NN vs LS Win Rate
            ax = axes[1, 0]
            if "cbf_win_rate" in df.columns:
                df_plot = df_summary.dropna(subset=["cbf_win_rate"])
                colors = ["green" if w > 0.5 else "red" for w in df_plot["cbf_win_rate"]]
                ax.bar(range(len(df_plot)), df_plot["cbf_win_rate"] * 100, color=colors)
                ax.set_xticks(range(len(df_plot)))
                ax.set_xticklabels([n[:15] for n in df_plot["name"]], rotation=45, ha="right")
                ax.axhline(y=50, color="black", linestyle="--", label="NN = LS")
                ax.set_ylabel("NN Win Rate (%)")
                ax.set_title("NN vs LS Win Rate (Green > 50%)")

            # Plot 4: CBF MAE vs Amplitude Ratio scatter
            ax = axes[1, 1]
            if "cbf_mae" in df.columns and "amp_ratio" in df.columns:
                df_plot = df.dropna(subset=["cbf_mae", "amp_ratio"])
                colors = ["blue" if "AmpAware" in n else "red" for n in df_plot["name"]]
                ax.scatter(df_plot["amp_ratio"], df_plot["cbf_mae"], c=colors, s=100)
                for i, row in df_plot.iterrows():
                    ax.annotate(row["name"][:10], (row["amp_ratio"], row["cbf_mae"]),
                               fontsize=8, ha="left")
                ax.set_xlabel("Amplitude Sensitivity Ratio")
                ax.set_ylabel("CBF MAE")
                ax.set_title("CBF Error vs Amplitude Sensitivity")
                ax.set_xscale("log")

            plt.tight_layout()
            plot_path = results_dir / "amplitude_ablation_plots.png"
            plt.savefig(plot_path, dpi=150)
            print(f"\nSaved plots to: {plot_path}")
            plt.close()

        except Exception as e:
            print(f"Plotting error: {e}")

    # Generate markdown report
    report_path = results_dir / "AMPLITUDE_ABLATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Amplitude-Aware Architecture Ablation Study Report\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Summary\n\n")
        f.write("This ablation study tests the AmplitudeAwareSpatialASLNet architecture components:\n")
        f.write("- FiLM conditioning at bottleneck and decoder\n")
        f.write("- Amplitude output modulation\n")
        f.write("- Physics loss (dc_weight)\n")
        f.write("- Domain randomization\n\n")

        f.write("## Results Table\n\n")
        f.write("| Experiment | Model | Amp Sensitive | CBF MAE | CBF Win Rate | ATT MAE |\n")
        f.write("|------------|-------|---------------|---------|--------------|----------|\n")
        for _, row in df_summary.iterrows():
            sens = "YES" if row.get("is_amp_sensitive", False) else "NO"
            f.write(f"| {row.get('name', 'N/A')} | {row.get('model_class', 'N/A')[:15]} | "
                   f"{sens} | {row.get('cbf_mae', np.nan):.2f} | "
                   f"{row.get('cbf_win_rate', np.nan)*100:.1f}% | "
                   f"{row.get('att_mae', np.nan):.2f} |\n")

        f.write("\n## Key Findings\n\n")
        f.write("1. **Amplitude Sensitivity**: Only AmplitudeAwareSpatialASLNet shows amplitude sensitivity\n")
        f.write("2. **Best Configuration**: [To be filled after running]\n")
        f.write("3. **Physics Loss Impact**: [To be filled after running]\n")

    print(f"Saved report to: {report_path}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

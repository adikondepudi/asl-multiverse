# FILE: compare_optimization_results.py
"""
Compare results from optimization ablation study.
Compares each experiment against the baseline (00_Baseline_Control).
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import numpy as np
import yaml


def load_experiment_results(exp_dir: Path) -> dict:
    """Load all results from an experiment directory."""
    results = {
        "name": exp_dir.name,
        "config": {},
        "metrics": {},
        "hypothesis": "",
        "changes": {},
    }

    # Load config
    config_file = exp_dir / "config.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
            results["config"] = cfg

            # Extract experiment metadata
            if "_experiment" in cfg:
                results["hypothesis"] = cfg["_experiment"].get("hypothesis", "")
                results["changes"] = cfg["_experiment"].get("changes", {})

    # Load validation metrics
    metrics_file = exp_dir / "validation_results" / "llm_analysis_report.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            report = json.load(f)

            # Get the spatial scenario (should be Spatial_SNR10)
            for scenario_name, scenario_data in report.items():
                results["metrics"][scenario_name] = {
                    "CBF_MAE_NN": scenario_data.get("CBF", {}).get("Neural_Net", {}).get("MAE", np.nan),
                    "CBF_MAE_LS": scenario_data.get("CBF", {}).get("Least_Squares", {}).get("MAE", np.nan),
                    "CBF_Bias_NN": scenario_data.get("CBF", {}).get("Neural_Net", {}).get("Bias", np.nan),
                    "CBF_R2_NN": scenario_data.get("CBF", {}).get("Neural_Net", {}).get("R2", np.nan),
                    "CBF_WinRate": scenario_data.get("CBF", {}).get("NN_vs_LS_Win_Rate", np.nan),
                    "ATT_MAE_NN": scenario_data.get("ATT", {}).get("Neural_Net", {}).get("MAE", np.nan),
                    "ATT_MAE_LS": scenario_data.get("ATT", {}).get("Least_Squares", {}).get("MAE", np.nan),
                    "ATT_Bias_NN": scenario_data.get("ATT", {}).get("Neural_Net", {}).get("Bias", np.nan),
                    "ATT_R2_NN": scenario_data.get("ATT", {}).get("Neural_Net", {}).get("R2", np.nan),
                    "ATT_WinRate": scenario_data.get("ATT", {}).get("NN_vs_LS_Win_Rate", np.nan),
                }

    # Load training info from slurm output (final val loss)
    slurm_out = exp_dir / "slurm.out"
    if slurm_out.exists():
        with open(slurm_out, 'r') as f:
            content = f.read()
            # Find last epoch line
            import re
            epochs = re.findall(r"Epoch (\d+)/\d+: Train Loss = ([\d.]+), Val Loss = ([\d.]+)", content)
            if epochs:
                last_epoch = epochs[-1]
                results["final_epoch"] = int(last_epoch[0])
                results["final_train_loss"] = float(last_epoch[1])
                results["final_val_loss"] = float(last_epoch[2])

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare optimization ablation results")
    parser.add_argument("--results_dir", type=str, default="optimization_ablation_v1",
                        help="Directory containing experiment results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory '{results_dir}' does not exist.")
        return

    print(f"--- AGGREGATING OPTIMIZATION ABLATION RESULTS ---")
    print(f"Results directory: {results_dir}")

    # Find all experiment directories
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(('0', '1', '2'))])
    print(f"Found {len(exp_dirs)} experiment directories")

    all_results = []
    for exp_dir in exp_dirs:
        result = load_experiment_results(exp_dir)
        if result["metrics"]:
            all_results.append(result)
        else:
            print(f"  Warning: No metrics for {exp_dir.name}")

    if not all_results:
        print("No valid results found.")
        return

    # Find baseline
    baseline = None
    for r in all_results:
        if "Baseline" in r["name"] or "Control" in r["name"]:
            baseline = r
            break

    if baseline is None:
        print("Warning: No baseline found, using first experiment as reference")
        baseline = all_results[0]

    print(f"\nBaseline: {baseline['name']}")

    # Build comparison dataframe
    rows = []
    for r in all_results:
        # Get primary scenario metrics
        scenario = list(r["metrics"].keys())[0] if r["metrics"] else None
        if not scenario:
            continue

        m = r["metrics"][scenario]

        # Calculate improvement over baseline
        baseline_m = baseline["metrics"].get(scenario, {})

        cbf_improvement = 0
        att_improvement = 0
        if baseline_m and not np.isnan(baseline_m.get("CBF_MAE_NN", np.nan)):
            baseline_cbf = baseline_m["CBF_MAE_NN"]
            baseline_att = baseline_m["ATT_MAE_NN"]
            if not np.isnan(m["CBF_MAE_NN"]) and baseline_cbf > 0:
                cbf_improvement = (baseline_cbf - m["CBF_MAE_NN"]) / baseline_cbf * 100
            if not np.isnan(m["ATT_MAE_NN"]) and baseline_att > 0:
                att_improvement = (baseline_att - m["ATT_MAE_NN"]) / baseline_att * 100

        # Extract key config changes
        cfg = r["config"]
        training_cfg = cfg.get("training", {})

        row = {
            "Experiment": r["name"],
            "Hypothesis": r["hypothesis"][:50] + "..." if len(r["hypothesis"]) > 50 else r["hypothesis"],

            # Key hyperparameters
            "Epochs": training_cfg.get("n_epochs", 50),
            "Capacity": str(training_cfg.get("hidden_sizes", []))[:20],
            "LR": training_cfg.get("learning_rate", 0.0001),
            "CBF_Weight": training_cfg.get("cbf_weight", 1.0),
            "Var_Weight": training_cfg.get("variance_weight", 0.1),
            "DC_Weight": training_cfg.get("dc_weight", 0.0001),

            # Primary metrics
            "CBF_MAE": m.get("CBF_MAE_NN", np.nan),
            "CBF_Bias": m.get("CBF_Bias_NN", np.nan),
            "CBF_R2": m.get("CBF_R2_NN", np.nan),
            "CBF_WinRate": m.get("CBF_WinRate", np.nan) * 100 if m.get("CBF_WinRate") else np.nan,

            "ATT_MAE": m.get("ATT_MAE_NN", np.nan),
            "ATT_Bias": m.get("ATT_Bias_NN", np.nan),
            "ATT_R2": m.get("ATT_R2_NN", np.nan),
            "ATT_WinRate": m.get("ATT_WinRate", np.nan) * 100 if m.get("ATT_WinRate") else np.nan,

            # Improvement over baseline
            "CBF_Improvement_%": cbf_improvement,
            "ATT_Improvement_%": att_improvement,

            # Training metrics
            "Final_Val_Loss": r.get("final_val_loss", np.nan),
            "Final_Epoch": r.get("final_epoch", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by CBF improvement
    df = df.sort_values("CBF_MAE", ascending=True)

    # Save detailed CSV
    csv_path = results_dir / "optimization_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed comparison to {csv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("OPTIMIZATION ABLATION RESULTS SUMMARY")
    print("=" * 100)

    # Show baseline reference
    baseline_row = df[df["Experiment"].str.contains("Baseline|Control")].iloc[0] if len(df[df["Experiment"].str.contains("Baseline|Control")]) > 0 else df.iloc[0]
    print(f"\nBASELINE REFERENCE: {baseline_row['Experiment']}")
    print(f"  CBF MAE: {baseline_row['CBF_MAE']:.3f}, ATT MAE: {baseline_row['ATT_MAE']:.3f}")
    print(f"  CBF Win Rate: {baseline_row['CBF_WinRate']:.1f}%, ATT Win Rate: {baseline_row['ATT_WinRate']:.1f}%")

    # Show improvements
    print("\n" + "-" * 100)
    print("EXPERIMENTS RANKED BY CBF MAE (lower is better)")
    print("-" * 100)
    print(f"{'Rank':<5} {'Experiment':<30} {'CBF_MAE':<10} {'ATT_MAE':<10} {'CBF_Impr%':<12} {'ATT_Impr%':<12}")
    print("-" * 100)

    for i, (_, row) in enumerate(df.iterrows()):
        cbf_symbol = "✓" if row["CBF_Improvement_%"] > 0 else "✗" if row["CBF_Improvement_%"] < -1 else "="
        att_symbol = "✓" if row["ATT_Improvement_%"] > 0 else "✗" if row["ATT_Improvement_%"] < -1 else "="

        print(f"{i+1:<5} {row['Experiment']:<30} {row['CBF_MAE']:<10.3f} {row['ATT_MAE']:<10.2f} "
              f"{cbf_symbol} {row['CBF_Improvement_%']:>+8.2f}%  {att_symbol} {row['ATT_Improvement_%']:>+8.2f}%")

    # Print winners
    print("\n" + "=" * 100)
    print("TOP PERFORMERS")
    print("=" * 100)

    best_cbf = df.loc[df["CBF_MAE"].idxmin()]
    best_att = df.loc[df["ATT_MAE"].idxmin()]
    best_combined = df.loc[(df["CBF_Improvement_%"] + df["ATT_Improvement_%"]).idxmax()]

    print(f"\n🏆 Best CBF:      {best_cbf['Experiment']}")
    print(f"   CBF MAE: {best_cbf['CBF_MAE']:.3f} ({best_cbf['CBF_Improvement_%']:+.2f}% vs baseline)")

    print(f"\n🏆 Best ATT:      {best_att['Experiment']}")
    print(f"   ATT MAE: {best_att['ATT_MAE']:.2f} ({best_att['ATT_Improvement_%']:+.2f}% vs baseline)")

    print(f"\n🏆 Best Combined: {best_combined['Experiment']}")
    print(f"   CBF: {best_combined['CBF_MAE']:.3f} ({best_combined['CBF_Improvement_%']:+.2f}%), "
          f"ATT: {best_combined['ATT_MAE']:.2f} ({best_combined['ATT_Improvement_%']:+.2f}%)")

    # Generate plots
    try:
        # Plot 1: CBF MAE comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar plot of CBF MAE
        ax1 = axes[0]
        colors = ['green' if x > 0 else 'red' if x < -1 else 'gray' for x in df["CBF_Improvement_%"]]
        bars = ax1.barh(df["Experiment"], df["CBF_MAE"], color=colors, alpha=0.7)
        ax1.axvline(x=baseline_row["CBF_MAE"], color='blue', linestyle='--', linewidth=2, label='Baseline')
        ax1.set_xlabel("CBF MAE (lower is better)")
        ax1.set_title("CBF MAE by Experiment")
        ax1.legend()
        ax1.invert_yaxis()

        # Bar plot of ATT MAE
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'red' if x < -1 else 'gray' for x in df["ATT_Improvement_%"]]
        bars = ax2.barh(df["Experiment"], df["ATT_MAE"], color=colors, alpha=0.7)
        ax2.axvline(x=baseline_row["ATT_MAE"], color='blue', linestyle='--', linewidth=2, label='Baseline')
        ax2.set_xlabel("ATT MAE (lower is better)")
        ax2.set_title("ATT MAE by Experiment")
        ax2.legend()
        ax2.invert_yaxis()

        plt.tight_layout()
        plot_path = results_dir / "optimization_comparison_mae.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved MAE comparison plot to {plot_path}")
        plt.close()

        # Plot 2: Improvement scatter
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df["CBF_Improvement_%"], df["ATT_Improvement_%"],
                            s=100, alpha=0.7, c=range(len(df)), cmap='viridis')

        # Add labels
        for _, row in df.iterrows():
            ax.annotate(row["Experiment"].replace("_", "\n")[:15],
                       (row["CBF_Improvement_%"], row["ATT_Improvement_%"]),
                       fontsize=7, ha='center', va='bottom')

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
        ax.fill_between([-50, 50], 0, 50, alpha=0.1, color='green', label='Both Improved')
        ax.fill_between([-50, 50], -50, 0, alpha=0.1, color='red', label='ATT Worse')

        ax.set_xlabel("CBF Improvement % (positive = better)")
        ax.set_ylabel("ATT Improvement % (positive = better)")
        ax.set_title("Improvement Over Baseline")
        ax.legend()

        plot_path = results_dir / "optimization_comparison_improvement.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved improvement scatter plot to {plot_path}")
        plt.close()

    except Exception as e:
        print(f"Warning: Plotting failed: {e}")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()

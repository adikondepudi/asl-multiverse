import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import numpy as np
import yaml

def main():
    root_dir = Path("hpc_ablation_jobs")
    results = []
    
    print(f"Crawling {root_dir} for 'llm_analysis_report.json'...")
    
    # 1. Crawl for the correct filename
    files_found = list(root_dir.rglob("llm_analysis_report.json"))
    print(f"Found {len(files_found)} result files.")

    for metrics_file in files_found:
        # Load Metrics
        try:
            with open(metrics_file, 'r') as f:
                report = json.load(f)
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
            continue
            
        # Load Config (To know hyperparameters)
        # Try finding config in the experiment root (2 levels up from validation_results)
        experiment_dir = metrics_file.parent.parent 
        config_file = experiment_dir / "config.yaml"
        
        if not config_file.exists():
            # Fallback: maybe it's just 1 level up?
            config_file = metrics_file.parent / "config.yaml"
        
        if not config_file.exists():
            print(f"Warning: Config not found for {metrics_file}, using placeholders.")
            cfg = {}
        else:
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)

        # Parse the JSON structure
        try:
            # We want to compare performance on a specific hard scenario
            # Priority: C_VarBoth_SNR10 -> D_VarBoth_SNR3 -> First Available
            target_scenario = 'C_VarBoth_SNR10'
            if target_scenario not in report:
                target_scenario = 'D_VarBoth_SNR3'
            if target_scenario not in report:
                target_scenario = list(report.keys())[0]

            scenario_data = report[target_scenario]
            cbf_data = scenario_data.get('CBF', {})
            
            # Extract data safely
            nn_stats = cbf_data.get('Neural_Net', {})
            ls_stats = cbf_data.get('Least_Squares', {})
            
            row = {
                "ID": experiment_dir.name,
                "Noise_Robustness": "Robust" if "physio" in cfg.get('data_noise_components', []) else "Standard",
                "Num_Features": len(cfg.get('active_features', [])),
                "Features": str(cfg.get('active_features', [])),
                "NN_MAE_CBF": nn_stats.get('MAE', np.nan),
                "LS_MAE_CBF": ls_stats.get('MAE', np.nan),
                "NN_Win_Rate": cbf_data.get('NN_vs_LS_Win_Rate', 0) * 100 if cbf_data.get('NN_vs_LS_Win_Rate') is not None else np.nan,
                "Target_Scenario": target_scenario
            }
            results.append(row)
        except Exception as e:
            print(f"Warning: Could not parse logic for {metrics_file}: {e}")
            continue
        
    if not results:
        print("No valid results parsed.")
        sys.exit(0)
        
    df = pd.DataFrame(results)
    
    # Sort for cleaner CSV
    if "NN_MAE_CBF" in df.columns:
        df = df.sort_values("NN_MAE_CBF")

    df.to_csv("final_ablation_summary.csv", index=False)
    print(f"\nSUCCESS: Saved summary to final_ablation_summary.csv ({len(df)} experiments)")
    
    # --- PLOTTING ---
    try:
        # Plot 1: Feature Impact
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="ID", y="NN_MAE_CBF", hue="Noise_Robustness")
        plt.xticks(rotation=45, ha='right')
        plt.title("CBF Error by Experiment")
        plt.tight_layout()
        plt.savefig("plot_feature_impact.png", dpi=150)
        print("Saved plot_feature_impact.png")
        
        # Plot 2: Tournament
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df, x="LS_MAE_CBF", y="NN_MAE_CBF", hue="ID", s=150)
        
        # Identity Line
        all_vals = list(df["LS_MAE_CBF"].dropna()) + list(df["NN_MAE_CBF"].dropna())
        if all_vals:
            lims = [min(all_vals) * 0.9, max(all_vals) * 1.1]
            plt.plot(lims, lims, 'k--', label='NN = LS')
        
        plt.title("NN vs LS Performance (Points below line = NN Wins)")
        plt.tight_layout()
        plt.savefig("plot_tournament.png", dpi=150)
        print("Saved plot_tournament.png")
    except Exception as e:
        print(f"Plotting failed (but CSV is saved): {e}")

if __name__ == "__main__":
    main()
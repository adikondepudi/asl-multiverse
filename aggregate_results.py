# aggregate_results.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

def aggregate_results(study_dir: Path):
    """
    Aggregates results from a 2D ablation study and generates summary heatmaps.
    """
    all_results = []
    
    print(f"Scanning for results in: {study_dir}")

    # Regex to capture both stage 1 and stage 2 weights from the directory name
    dir_pattern = re.compile(r'pinn_s1_(\d+\.?\d*)_s2_(\d+\.?\d*)')

    for run_dir in study_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        match = dir_pattern.match(run_dir.name)
        if not match:
            continue
            
        pinn_weight_s1 = float(match.group(1))
        pinn_weight_s2 = float(match.group(2))

        results_csv = run_dir / 'benchmark_performance_summary.csv'
        
        if results_csv.exists():
            print(f"  - Found results for S1={pinn_weight_s1}, S2={pinn_weight_s2} in {run_dir.name}")
            df = pd.read_csv(results_csv)
            df['pinn_weight_s1'] = pinn_weight_s1
            df['pinn_weight_s2'] = pinn_weight_s2
            all_results.append(df)
        else:
            print(f"  - WARNING: No 'benchmark_performance_summary.csv' found in {run_dir.name}")

    if not all_results:
        print("No results found to aggregate. Exiting.")
        return

    summary_df = pd.concat(all_results, ignore_index=True)
    summary_csv_path = study_dir / 'ablation_summary_2D.csv'
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.3f')
    print(f"\nAggregated 2D results saved to: {summary_csv_path}")

    generate_summary_heatmaps(summary_df, study_dir)

def generate_summary_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Generates heatmaps visualizing the impact of PINN weights."""
    print("Generating summary heatmaps...")
    
    df_plot = df[df['method'] == 'NN'].copy()
    if df_plot.empty:
        print("No 'NN' method results found to plot.")
        return

    sns.set_theme(style="white", context="talk")
    
    # Define the metrics we want to visualize
    metrics_to_plot = {
        'cbf_nrmse_perc': 'CBF Accuracy (nRMSE %)',
        'att_nrmse_perc': 'ATT Accuracy (nRMSE %)'
    }

    blue_cmap = sns.color_palette("light:b", as_cmap=True)
    

    for metric_key, metric_title in metrics_to_plot.items():
        # Create a figure with a subplot for each ATT range
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        fig.suptitle(f'Impact of PINN Weights on {metric_title}', fontsize=22, weight='bold')

        att_ranges = sorted(df_plot['att_range_name'].unique())
        
        for i, att_range in enumerate(att_ranges):
            ax = axes[i]
            # Pivot the data to create a 2D grid for the heatmap
            pivot_df = df_plot[df_plot['att_range_name'] == att_range].pivot(
                index='pinn_weight_s2', 
                columns='pinn_weight_s1', 
                values=metric_key
            )
            
            sns.heatmap(pivot_df, ax=ax, annot=True, fmt=".1f", cmap=blue_cmap, 
                        linewidths=.5, cbar_kws={'label': f'{metric_key}'})
            
            ax.set_title(f'{att_range}', fontsize=16)
            ax.set_xlabel('Stage 1 PINN Weight', fontsize=14)
            if i == 0:
                ax.set_ylabel('Stage 2 PINN Weight', fontsize=14)
            else:
                ax.set_ylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = output_dir / f'ablation_heatmap_{metric_key}.png'
        plt.savefig(plot_path, dpi=300)
        print(f"Summary heatmap saved to: {plot_path}")
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate results from a 2D PINN loss ablation study.")
    parser.add_argument("study_dir", type=str, help="Path to the parent directory of the ablation study.")
    args = parser.parse_args()
    
    aggregate_results(Path(args.study_dir))
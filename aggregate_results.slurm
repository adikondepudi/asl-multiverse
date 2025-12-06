#!/bin/bash
#SBATCH --job-name=aggregate_ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=aggregate_results.out
#SBATCH --error=aggregate_results.err

source ~/.bashrc
conda activate asl_multiverse

echo "--- AGGREGATING ALL ABLATION RESULTS ---"
python compare_ablation_results.py

echo "--- AGGREGATION COMPLETE ---"
echo "Results saved to: final_ablation_summary.csv"
echo "Plots saved to: plot_feature_impact.png, plot_tournament.png"

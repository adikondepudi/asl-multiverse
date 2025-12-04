#!/bin/bash
#SBATCH --job-name=exp006_feats3_noise3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=hpc_ablation_jobs/exp006_feats3_noise3/slurm.out
#SBATCH --error=hpc_ablation_jobs/exp006_feats3_noise3/slurm.err

source ~/.bashrc
conda activate asl_multiverse

echo "--- 1. STARTING TRAINING ---"
python main.py hpc_ablation_jobs/exp006_feats3_noise3/config.yaml --stage 2 --output-dir hpc_ablation_jobs/exp006_feats3_noise3

echo "--- 2. AUTO-VALIDATION (NN vs LS) ---"
# This script compares the model against LS on in-vivo data
# and saves 'metrics.json'
python validate.py --run_dir hpc_ablation_jobs/exp006_feats3_noise3 --output_dir hpc_ablation_jobs/exp006_feats3_noise3/validation_results

echo "--- JOB COMPLETE ---"

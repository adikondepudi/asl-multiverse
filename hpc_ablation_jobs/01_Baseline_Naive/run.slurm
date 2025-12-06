#!/bin/bash
#SBATCH --job-name=01_Baseline_Naive
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=hpc_ablation_jobs/01_Baseline_Naive/slurm.out
#SBATCH --error=hpc_ablation_jobs/01_Baseline_Naive/slurm.err

source ~/.bashrc
conda activate asl_multiverse

cd $SLURM_SUBMIT_DIR

echo "=== EXPERIMENT: 01_Baseline_Naive ==="
echo "--- 1. STAGE 1: DENOISING PRE-TRAINING ---"
python main.py hpc_ablation_jobs/01_Baseline_Naive/config.yaml --stage 1 --output-dir hpc_ablation_jobs/01_Baseline_Naive

echo "--- 2. STAGE 2: REGRESSION TRAINING ---"
python main.py hpc_ablation_jobs/01_Baseline_Naive/config.yaml --stage 2 --output-dir hpc_ablation_jobs/01_Baseline_Naive --load-weights-from hpc_ablation_jobs/01_Baseline_Naive

echo "--- 3. AUTO-VALIDATION (NN vs LS) ---"
python validate.py --run_dir hpc_ablation_jobs/01_Baseline_Naive --output_dir hpc_ablation_jobs/01_Baseline_Naive/validation_results

echo "--- JOB COMPLETE ---"

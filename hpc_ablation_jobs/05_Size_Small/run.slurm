#!/bin/bash
#SBATCH --job-name=05_Size_Small
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=hpc_ablation_jobs/05_Size_Small/slurm.out
#SBATCH --error=hpc_ablation_jobs/05_Size_Small/slurm.err

source ~/.bashrc
conda activate asl_multiverse

cd $SLURM_SUBMIT_DIR

echo "=== EXPERIMENT: 05_Size_Small ==="
echo "--- 1. STAGE 1: DENOISING PRE-TRAINING ---"
python main.py hpc_ablation_jobs/05_Size_Small/config.yaml --stage 1 --output-dir hpc_ablation_jobs/05_Size_Small

echo "--- 2. STAGE 2: REGRESSION TRAINING ---"
python main.py hpc_ablation_jobs/05_Size_Small/config.yaml --stage 2 --output-dir hpc_ablation_jobs/05_Size_Small --load-weights-from hpc_ablation_jobs/05_Size_Small

echo "--- 3. AUTO-VALIDATION (NN vs LS) ---"
python validate.py --run_dir hpc_ablation_jobs/05_Size_Small --output_dir hpc_ablation_jobs/05_Size_Small/validation_results

echo "--- JOB COMPLETE ---"

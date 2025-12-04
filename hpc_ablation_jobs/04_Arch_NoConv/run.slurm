#!/bin/bash
#SBATCH --job-name=04_Arch_NoConv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=hpc_ablation_jobs/04_Arch_NoConv/slurm.out
#SBATCH --error=hpc_ablation_jobs/04_Arch_NoConv/slurm.err

source ~/.bashrc
conda activate asl_multiverse

cd $SLURM_SUBMIT_DIR

echo "=== EXPERIMENT: 04_Arch_NoConv ==="
echo "--- 1. STAGE 1: DENOISING PRE-TRAINING ---"
python main.py hpc_ablation_jobs/04_Arch_NoConv/config.yaml --stage 1 --output-dir hpc_ablation_jobs/04_Arch_NoConv

echo "--- 2. STAGE 2: REGRESSION TRAINING ---"
python main.py hpc_ablation_jobs/04_Arch_NoConv/config.yaml --stage 2 --output-dir hpc_ablation_jobs/04_Arch_NoConv --load-weights-from hpc_ablation_jobs/04_Arch_NoConv

echo "--- 3. AUTO-VALIDATION (NN vs LS) ---"
python validate.py --run_dir hpc_ablation_jobs/04_Arch_NoConv --output_dir hpc_ablation_jobs/04_Arch_NoConv/validation_results

echo "--- JOB COMPLETE ---"

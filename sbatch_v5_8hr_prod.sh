#!/bin/bash
#SBATCH --job-name=asl-v5-8hr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_logs/v5_8hr_prod_%j.out
#SBATCH --error=slurm_logs/v5_8hr_prod_%j.err

echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_logs
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
export WANDB_MODE=online

echo "Executing the accelerated v5 two-stage training pipeline script..."
./run_v5_8hr_prod.sh

echo "Job finished at $(date)."
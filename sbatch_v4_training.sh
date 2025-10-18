#!/bin/bash
#SBATCH --job-name=asl-v4-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm_logs/v4_training_%j.out
#SBATCH --error=slurm_logs/v4_training_%j.err

# --- Setup ---
echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_logs

# --- Activate Conda Environment ---
echo "Loading Anaconda module and activating environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# --- Set W&B Mode ---
export WANDB_MODE=online

# --- Execute the Master Pipeline Script ---
echo "Executing the v4 two-stage training pipeline script..."
./run_v4_training_pipeline.sh

echo "Job finished at $(date)."
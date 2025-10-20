#!/bin/bash
#SBATCH --job-name=asl-v5-moe-ft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/v5_finetuning_%j.out
#SBATCH --error=slurm_logs/v5_finetuning_%j.err

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
# IMPORTANT: You must replace the placeholder below with the actual path to your
# completed v4 Stage 1 training run directory before submitting this job.
STAGE1_RUN_DIR="path/to/your/v4_stage1_run_directory" 

echo "Executing the v5 MoE fine-tuning pipeline script..."
./run_v5_MoE_finetuning.sh "${STAGE1_RUN_DIR}"

echo "Job finished at $(date)."
#!/bin/bash
#SBATCH --job-name=asl-v2-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8      # REDUCED: Still plenty for data loading, but much more flexible.
#SBATCH --mem=64G              # REDUCED: Halved, but still a very large and safe amount of RAM.
#SBATCH --time=1-12:00:00        # REDUCED: 36 hours is still very long, but easier to schedule than 48.
#SBATCH --output=slurm_logs/training_%j.out
#SBATCH --error=slurm_logs/training_%j.err

# --- Setup ---
echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_logs

# --- Activate Conda Environment ---
echo "Loading Anaconda module and activating environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# --- Set W&B to Online Mode ---
export WANDB_MODE=online

# --- Define Output Directory ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="training_runs/disentangled_prod_v2_run_${TIMESTAMP}"

# --- Run the Main Training Script ---
echo "Starting training run for DisentangledASLNet v2..."
python main.py config/production_disentangled.yaml "${OUTPUT_DIR}"

echo "Training job finished at $(date)."
echo "To sync W&B logs, navigate to the run directory inside ${OUTPUT_DIR} and run: wandb sync ."
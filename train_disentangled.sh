#!/bin/bash
#SBATCH --job-name=asl-v2-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --cpus-per-task=16    # Request 16 CPUs for data loading
#SBATCH --mem=128G            # Request 128 GB of RAM
#SBATCH --time=2-00:00:00       # Set a long time limit (2 days)
#SBATCH --output=slurm_logs/training_%j.out
#SBATCH --error=slurm_logs/training_%j.err

# --- Setup ---
echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_logs # Create directory for logs if it doesn't exist

# --- Activate Conda Environment using the cluster's module system ---
echo "Loading Anaconda module and activating environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse # Use your environment name

# --- Set W&B to Offline Mode for HPC Stability ---
# RECOMMENDED CHANGE: Use offline mode for better stability on the cluster.
export WANDB_MODE=online

# --- Define Output Directory ---
# This creates a unique, timestamped directory for the results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="training_runs/disentangled_prod_v2_run_${TIMESTAMP}"

# --- Run the Main Training Script ---
echo "Starting training run for DisentangledASLNet v2..."
python main.py config/production_disentangled.yaml "${OUTPUT_DIR}"

echo "Training job finished at $(date)."
echo "To sync W&B logs, navigate to the run directory inside ${OUTPUT_DIR} and run: wandb sync ."
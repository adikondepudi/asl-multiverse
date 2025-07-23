#!/bin/bash
#SBATCH --job-name=asl_multiverse_HPO      # A descriptive name for your HPO job
#SBATCH --partition=gpu                  # Request the GPU partition
#SBATCH --gres=gpu:H100:1                # Request one H100 GPU for fast trials
#SBATCH --cpus-per-task=16               # Request 16 CPU cores for data loaders
#SBATCH --mem=128G                       # Request 128 GB of RAM
#SBATCH --time=05:00:00                  # Request 5 hours (production.yaml has a 4-hour timeout)
#SBATCH --output=slurm_logs/hpo_run_%j.out # Standard output log file
#SBATCH --error=slurm_logs/hpo_run_%j.err  # Standard error log file
#SBATCH --mail-type=BEGIN,END,FAIL         # Receive email notifications
#SBATCH --mail-user=your_email@jh.edu      # <-- IMPORTANT: CHANGE THIS TO YOUR EMAIL

# --- Environment Setup ---
echo "======================================================"
echo "Starting HPO Job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Project Directory: $(pwd)"
echo "======================================================"

# Load necessary modules
module purge
module load anaconda3/2023.09

# Activate your conda environment
source activate asl_multiverse

# Set WANDB to offline mode to prevent network issues on compute nodes
export WANDB_MODE=offline

# Define a unique output directory for this HPO run using the job ID
HPO_OUTPUT_DIR="production_run_hpo_${SLURM_JOB_ID}"
mkdir -p $HPO_OUTPUT_DIR

echo "Running HPO. Results will be in: $HPO_OUTPUT_DIR"

# Run the main script in optimization mode
python main.py config/production.yaml $HPO_OUTPUT_DIR --optimize --study-name "production_hpo_study"

echo "======================================================"
echo "HPO Job finished at $(date)"
echo "======================================================"
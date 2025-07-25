#!/bin/bash
# FILE: run_hpo_a100.sbatch
# (Full and updated content)

#SBATCH --job-name=asl_multiverse_HPO_A100 # A descriptive name for your HPO job
#SBATCH --partition=gpua100                # <-- CORRECTED PARTITION for A100
#SBATCH --gres=gpu:A100:1                  # <-- CORRECTED GPU NAME for A100
#SBATCH --cpus-per-task=16                 # Request 16 CPU cores for data loaders
#SBATCH --mem=128G                         # Request 128 GB of RAM
#SBATCH --time=05:00:00                    # Request 5 hours (production.yaml has a 4-hour timeout)
#SBATCH --output=slurm_logs/hpo_run_%j.out   # Standard output log file
#SBATCH --error=slurm_logs/hpo_run_%j.err    # Standard error log file
#SBATCH --mail-type=BEGIN,END,FAIL           # Receive email notifications
#SBATCH --mail-user=your_email@jh.edu        # <-- IMPORTANT: SET YOUR EMAIL HERE

# --- Environment Setup ---
echo "======================================================"
echo "Starting HPO Job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Project Directory: $(pwd)"
echo "======================================================"

module purge
module load anaconda3/2023.09
source activate asl_multiverse
export WANDB_MODE=offline

# Define a unique output directory for this HPO run using the job ID
HPO_OUTPUT_DIR="production_run_hpo_${SLURM_JOB_ID}"
mkdir -p $HPO_OUTPUT_DIR

# --- If norm_stats.json exists in the main directory, copy it to save time ---
if [ -f "norm_stats.json" ]; then
    echo "Found existing norm_stats.json. Copying to run directory to reuse."
    cp norm_stats.json "${HPO_OUTPUT_DIR}/"
fi

echo "Running HPO. Results will be in: $HPO_OUTPUT_DIR"

# Run the main script in optimization mode
python main.py config/production.yaml $HPO_OUTPUT_DIR --optimize --study-name "production_hpo_study"

echo "======================================================"
echo "HPO Job finished at $(date)"
echo "======================================================"
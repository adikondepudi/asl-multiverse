#!/bin/bash
#SBATCH --job-name=asl-data-gen
#SBATCH --partition=cpu          # Use the CPU partition for this task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48       # Request a lot of CPU cores for parallelism
#SBATCH --mem=128G               # Request a good amount of RAM
#SBATCH --time=12:00:00          # Set a 12-hour time limit (should be plenty)
#SBATCH --output=slurm_logs/data_gen_%j.out
#SBATCH --error=slurm_logs/data_gen_%j.err

# --- Setup ---
echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_logs # Create directory for logs if it doesn't exist

# --- Activate Conda Environment ---
# Make sure this path is correct for your setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asl-multiverse # Use your environment name

# --- Define Output Directory for the Dataset ---
DATASET_DIR="asl_offline_dataset_10M"

# --- Clean up any previous partial run ---
echo "Checking for and removing previous dataset directory: ${DATASET_DIR}"
rm -rf "${DATASET_DIR}"

# --- Run the Data Generation Script ---
echo "Starting offline dataset generation for 10,000,000 samples..."
python generate_offline_dataset.py "${DATASET_DIR}" --total_samples 10000000 --chunk_size 25000

echo "Job finished at $(date)"
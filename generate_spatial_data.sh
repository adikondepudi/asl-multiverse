#!/bin/bash
#SBATCH --job-name=asl-spatial-gen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/spatial_gen_%j.out
#SBATCH --error=slurm_logs/spatial_gen_%j.err

# --- Setup ---
echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_logs

# --- Activate Conda Environment ---
echo "Loading Anaconda module and activating environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# --- Define Output Directory ---
DATASET_DIR="asl_spatial_dataset_100k"

# --- Clean up previous run ---
echo "Removing previous dataset directory: ${DATASET_DIR}"
rm -rf "${DATASET_DIR}"

# --- Run Spatial Data Generation ---
echo "Starting SPATIAL dataset generation (100,000 samples, 64x64)..."
python generate_clean_library.py "${DATASET_DIR}" \
    --spatial \
    --total_samples 100000 \
    --spatial-chunk-size 500 \
    --image-size 64

echo "Job finished at $(date)"

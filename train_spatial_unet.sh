#!/bin/bash
#SBATCH --job-name=spatial-unet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/spatial_train_%j.out
#SBATCH --error=slurm_logs/spatial_train_%j.err

# --- Setup ---
echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_logs

# --- Activate Conda Environment ---
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# --- Define Paths ---
CONFIG_FILE="config/base_template.yaml"
OUTPUT_DIR="spatial_experiment_v1"
DATASET_DIR="asl_spatial_dataset_100k"

# --- Verify dataset exists ---
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory '$DATASET_DIR' not found!"
    echo "Run 'sbatch generate_spatial_data.sh' first."
    exit 1
fi

# --- Run Training ---
echo "Starting Spatial U-Net training..."
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET_DIR}"

python main.py \
    --config "${CONFIG_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --offline_dataset_path "${DATASET_DIR}" \
    --stage 2

echo "Job finished at $(date)"

#!/bin/bash
#SBATCH --job-name=asl-prod-datagen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/prod_datagen_%j.out
#SBATCH --error=slurm_logs/prod_datagen_%j.err

# =============================================================================
# PRODUCTION DATA GENERATION
# =============================================================================
# Generates large-scale spatial phantom dataset for production training
#
# Dataset: 200,000 64x64 spatial phantoms
# Features: Tissue segmentation, pathology, realistic CBF/ATT distributions
# =============================================================================

set -e

echo "============================================"
echo "PRODUCTION DATA GENERATION"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "============================================"

# --- Setup ---
mkdir -p slurm_logs
cd $SLURM_SUBMIT_DIR

# --- Activate Environment ---
echo ""
echo "[1/3] Loading environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# --- Configuration ---
DATASET_NAME="asl_spatial_dataset_200k"
TOTAL_SAMPLES=200000
CHUNK_SIZE=500
IMAGE_SIZE=64

echo ""
echo "[2/3] Configuration:"
echo "  Dataset: ${DATASET_NAME}"
echo "  Samples: ${TOTAL_SAMPLES}"
echo "  Chunk size: ${CHUNK_SIZE}"
echo "  Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Estimated chunks: $((TOTAL_SAMPLES / CHUNK_SIZE))"

# --- Clean previous run (optional - comment out to resume) ---
if [ -d "$DATASET_NAME" ]; then
    echo ""
    echo "WARNING: Dataset directory exists. Removing..."
    rm -rf "$DATASET_NAME"
fi

# --- Generate Data ---
echo ""
echo "[3/3] Generating spatial dataset..."
echo "  This will take ~2-3 hours for 200k samples."
echo ""

python generate_clean_library.py "${DATASET_NAME}" \
    --spatial \
    --total_samples ${TOTAL_SAMPLES} \
    --spatial-chunk-size ${CHUNK_SIZE} \
    --image-size ${IMAGE_SIZE}

# --- Verify ---
echo ""
echo "============================================"
echo "DATA GENERATION COMPLETE"
echo "============================================"
echo "Dataset: ${DATASET_NAME}"
echo "Finished: $(date)"

# Count generated files
NUM_CHUNKS=$(ls -1 ${DATASET_NAME}/spatial_chunk_*.npz 2>/dev/null | wc -l)
echo "Generated chunks: ${NUM_CHUNKS}"
echo "Expected chunks: $((TOTAL_SAMPLES / CHUNK_SIZE))"

if [ "$NUM_CHUNKS" -eq "$((TOTAL_SAMPLES / CHUNK_SIZE))" ]; then
    echo "STATUS: SUCCESS - All chunks generated"
else
    echo "STATUS: WARNING - Chunk count mismatch"
fi
echo "============================================"

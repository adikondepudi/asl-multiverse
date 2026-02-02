#!/bin/bash
#SBATCH --job-name=asl-prod-train
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/prod_train_%j.out
#SBATCH --error=slurm_logs/prod_train_%j.err

# =============================================================================
# PRODUCTION MODEL TRAINING
# =============================================================================
# Full production training with all optimizations:
#   - 500k sample dataset
#   - 5-model ensemble
#   - 200 epochs
#   - A100 GPU for speed
#   - Comprehensive validation
# =============================================================================

set -e

echo "============================================"
echo "PRODUCTION MODEL TRAINING"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

# --- Setup ---
mkdir -p slurm_logs
cd $SLURM_SUBMIT_DIR

# --- Activate Environment ---
echo ""
echo "[1/5] Loading environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# --- Configuration ---
CONFIG_FILE="config/production_v1.yaml"
OUTPUT_DIR="production_model_v1"
DATASET_DIR="asl_spatial_dataset_500k"

echo ""
echo "[2/5] Configuration:"
echo "  Config: ${CONFIG_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Dataset: ${DATASET_DIR}"

# --- Verify Prerequisites ---
echo ""
echo "[3/5] Verifying prerequisites..."

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_DIR}"
    echo "Run 'sbatch generate_production_data.sh' first."
    exit 1
fi

# Count dataset chunks
NUM_CHUNKS=$(ls -1 ${DATASET_DIR}/spatial_chunk_*.npz 2>/dev/null | wc -l)
echo "  Dataset chunks found: ${NUM_CHUNKS}"

if [ "$NUM_CHUNKS" -lt 100 ]; then
    echo "WARNING: Dataset seems small (${NUM_CHUNKS} chunks). Expected ~1000 for 500k samples."
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Copy config to output for reproducibility
cp "${CONFIG_FILE}" "${OUTPUT_DIR}/config.yaml"

# --- Training ---
echo ""
echo "[4/5] Starting training..."
echo "  Training 5-model ensemble for 200 epochs"
echo "  This will take ~24-48 hours on A100"
echo ""

python main.py "${CONFIG_FILE}" \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}" \
    --offline_dataset_path "${DATASET_DIR}"

# --- Validation ---
echo ""
echo "[5/5] Running comprehensive validation..."

python validate.py \
    --run_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}/validation_results"

# --- Summary ---
echo ""
echo "============================================"
echo "TRAINING COMPLETE"
echo "============================================"
echo "Model saved to: ${OUTPUT_DIR}/trained_models/"
echo "Validation results: ${OUTPUT_DIR}/validation_results/"
echo "Finished: $(date)"
echo ""

# Print key metrics if available
if [ -f "${OUTPUT_DIR}/validation_results/llm_analysis_report.json" ]; then
    echo "--- KEY METRICS ---"
    python -c "
import json
with open('${OUTPUT_DIR}/validation_results/llm_analysis_report.json') as f:
    d = json.load(f)
    for scenario, metrics in d.items():
        print(f'\n{scenario}:')
        for param in ['CBF', 'ATT']:
            if param in metrics:
                nn_mae = metrics[param]['Neural_Net']['MAE']
                ls_mae = metrics[param]['Least_Squares']['MAE']
                win = metrics[param]['NN_vs_LS_Win_Rate']
                print(f'  {param}: NN MAE={nn_mae:.2f}, LS MAE={ls_mae:.2f}, Win Rate={win:.1%}')
"
fi

echo "============================================"

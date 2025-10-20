#!/bin/bash
# Master script to run the v5 Mixture of Experts (MoE) fine-tuning pipeline.
set -e # Exit immediately if any command fails

# --- 1. Argument Validation ---
if [ -z "$1" ]; then
    echo "❌ ERROR: Missing mandatory argument."
    echo "This script requires the path to the v4 Stage 1 output directory which contains 'encoder_pretrained.pt'."
    echo "Usage: $0 <path_to_stage1_output_dir>"
    exit 1
fi

if [ ! -f "$1/encoder_pretrained.pt" ]; then
    echo "❌ ERROR: The pretrained encoder file 'encoder_pretrained.pt' was not found in the specified directory: $1"
    exit 1
fi

echo "🚀 STARTING ASL MULTIVERSE V5 MoE FINE-TUNING PIPELINE..."
echo "======================================================"

# --- 2. Define Configurations and Base Name ---
STAGE1_OUTPUT_DIR="$1"
CONFIG_FILE="config/v5_MoE_finetune.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RUN_NAME="prod_v5_MoE_run_${TIMESTAMP}"
OUTPUT_DIR="training_runs/${BASE_RUN_NAME}"
RUN_NAME="${BASE_RUN_NAME}"

# --- 3. Stage 2 (MoE Fine-tuning) Execution ---
echo " "
echo "--- Starting MoE Fine-tuning ---"
echo "Config: ${CONFIG_FILE}"
echo "Loading weights from: ${STAGE1_OUTPUT_DIR}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "W&B Run Name: ${RUN_NAME}"

python main.py "${CONFIG_FILE}" \
    --stage 2 \
    --load-weights-from "${STAGE1_OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --output-dir "${OUTPUT_DIR}"

echo "✅ v5 MoE fine-tuning complete. Final models saved in ${OUTPUT_DIR}"
echo "======================================================"
echo "🎉 V5 FINE-TUNING PIPELINE FINISHED SUCCESSFULLY!"
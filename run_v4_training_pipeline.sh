#!/bin/bash
# Master script to run the full v4 two-stage training pipeline.
set -e # Exit immediately if any command fails

echo "🚀 STARTING ASL MULTIVERSE V4 TRAINING PIPELINE..."
echo "======================================================"

# --- 1. Define Configurations and Base Name ---
STAGE1_CONFIG="config/v4_stage1.yaml"
STAGE2_CONFIG="config/v4_stage2.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RUN_NAME="prod_v4_run_${TIMESTAMP}"

# --- 2. Stage 1: Pre-training from scratch ---
STAGE1_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage1"
STAGE1_RUN_NAME="${BASE_RUN_NAME}_stage1"
echo " "
echo "--- Starting Stage 1: Pre-training ---"
echo "Config: ${STAGE1_CONFIG}"
echo "Output Dir: ${STAGE1_OUTPUT_DIR}"
echo "W&B Run Name: ${STAGE1_RUN_NAME}"

python main.py "${STAGE1_CONFIG}" \
    --stage 1 \
    --run-name "${STAGE1_RUN_NAME}" \
    --output-dir "${STAGE1_OUTPUT_DIR}"

echo "✅ Stage 1 complete. Pre-trained encoder saved in ${STAGE1_OUTPUT_DIR}"
echo "======================================================"

# --- 3. Stage 2: Fine-tuning from pre-trained weights ---
STAGE2_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage2"
STAGE2_RUN_NAME="${BASE_RUN_NAME}_stage2"
echo " "
echo "--- Starting Stage 2: Fine-tuning ---"
echo "Config: ${STAGE2_CONFIG}"
echo "Loading weights from: ${STAGE1_OUTPUT_DIR}"
echo "Output Dir: ${STAGE2_OUTPUT_DIR}"
echo "W&B Run Name: ${STAGE2_RUN_NAME}"

python main.py "${STAGE2_CONFIG}" \
    --stage 2 \
    --load-weights-from "${STAGE1_OUTPUT_DIR}" \
    --run-name "${STAGE2_RUN_NAME}" \
    --output-dir "${STAGE2_OUTPUT_DIR}"

echo "✅ Stage 2 complete. Final fine-tuned models saved in ${STAGE2_OUTPUT_DIR}"
echo "======================================================"
echo "🎉 V4 TRAINING PIPELINE FINISHED SUCCESSFULLY!"
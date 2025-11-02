#!/bin/bash
# Master script to run the FAST TEST of the v5 two-stage training pipeline.
# This script uses minimal settings to quickly verify the code integrity.
set -e # Exit immediately if any command fails

echo "🚀 STARTING ASL MULTIVERSE V5 FAST TEST PIPELINE..."
echo "======================================================"

# --- 1. Define FAST TEST Configurations and Base Name ---
STAGE1_CONFIG="config/v5_stage1_fast_test.yaml"
STAGE2_CONFIG="config/v5_stage2_fast_test.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RUN_NAME="prod_v5_fast_test_${TIMESTAMP}"

# --- 2. Stage 1: Pre-training the Generalist Encoder ---
STAGE1_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage1_pretrain"
STAGE1_RUN_NAME="${BASE_RUN_NAME}_stage1_pretrain"
echo " "
echo "--- Starting Stage 1: Pre-training (FAST TEST) ---"
echo "Config: ${STAGE1_CONFIG}"
echo "Output Dir: ${STAGE1_OUTPUT_DIR}"
echo "W&B Run Name: ${STAGE1_RUN_NAME}"

python main.py "${STAGE1_CONFIG}" \
    --stage 1 \
    --run-name "${STAGE1_RUN_NAME}" \
    --output-dir "${STAGE1_OUTPUT_DIR}"

echo "✅ Stage 1 complete. Pre-trained encoder saved in ${STAGE1_OUTPUT_DIR}"
echo "======================================================"

# --- 3. Stage 2: Fine-tuning the MoE Head ---
STAGE2_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage2_MoE_finetune"
STAGE2_RUN_NAME="${BASE_RUN_NAME}_stage2_MoE_finetune"
echo " "
echo "--- Starting Stage 2: MoE Fine-tuning (FAST TEST) ---"
echo "Config: ${STAGE2_CONFIG}"
echo "Loading pre-trained encoder from: ${STAGE1_OUTPUT_DIR}"
echo "Output Dir: ${STAGE2_OUTPUT_DIR}"
echo "W&B Run Name: ${STAGE2_RUN_NAME}"

python main.py "${STAGE2_CONFIG}" \
    --stage 2 \
    --load-weights-from "${STAGE1_OUTPUT_DIR}" \
    --run-name "${STAGE2_RUN_NAME}" \
    --output-dir "${STAGE2_OUTPUT_DIR}"

echo "✅ Stage 2 complete. Final fine-tuned models saved in ${STAGE2_OUTPUT_DIR}"
echo "======================================================"
echo "🎉 V5 FAST TEST PIPELINE FINISHED SUCCESSFULLY!"
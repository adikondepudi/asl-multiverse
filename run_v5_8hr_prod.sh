#!/bin/bash
# Master script to run the TUNED v5 two-stage training pipeline (8-hour target).
# This version uses a larger data subset to improve generalization and prevent overfitting.
set -e 

echo "🚀 STARTING ASL MULTIVERSE V5 TUNED (8-HOUR TARGET) PIPELINE..."
echo "==================================================================="

# --- 1. Define TUNED Configurations and Base Name ---
STAGE1_CONFIG="config/v5_stage1_tuned_8hr.yaml"
STAGE2_CONFIG="config/v5_stage2_tuned_8hr.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RUN_NAME="prod_v5_tuned_8hr_run_${TIMESTAMP}"

# --- 2. Stage 1: Pre-training the Generalist Encoder ---
STAGE1_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage1_pretrain"
STAGE1_RUN_NAME="${BASE_RUN_NAME}_stage1_pretrain"
echo "--- Starting Stage 1: Pre-training (Tuned) ---"
python main.py "${STAGE1_CONFIG}" --stage 1 --run-name "${STAGE1_RUN_NAME}" --output-dir "${STAGE1_OUTPUT_DIR}"
echo "✅ Stage 1 complete."

# --- 3. Stage 2: Fine-tuning the MoE Head ---
STAGE2_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage2_MoE_finetune"
STAGE2_RUN_NAME="${BASE_RUN_NAME}_stage2_MoE_finetune"
echo "--- Starting Stage 2: MoE Fine-tuning (Tuned) ---"
python main.py "${STAGE2_CONFIG}" --stage 2 --load-weights-from "${STAGE1_OUTPUT_DIR}" --run-name "${STAGE2_RUN_NAME}" --output-dir "${STAGE2_OUTPUT_DIR}"
echo "✅ Stage 2 complete."

echo "🎉 V5 TUNED PIPELINE FINISHED SUCCESSFULLY!"
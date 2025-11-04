#!/bin/bash
# Master script to run the ACCELERATED v5 two-stage training pipeline.
set -e 

echo "🚀 STARTING ASL MULTIVERSE V5 ACCELERATED (8-HOUR TARGET) PIPELINE..."
echo "==================================================================="

STAGE1_CONFIG="config/v5_stage1_8hr_prod.yaml"
STAGE2_CONFIG="config/v5_stage2_8hr_prod.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RUN_NAME="prod_v5_8hr_run_${TIMESTAMP}"

# Stage 1
STAGE1_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage1_pretrain"
STAGE1_RUN_NAME="${BASE_RUN_NAME}_stage1_pretrain"
echo "--- Starting Stage 1: Pre-training (Accelerated) ---"
python main.py "${STAGE1_CONFIG}" --stage 1 --run-name "${STAGE1_RUN_NAME}" --output-dir "${STAGE1_OUTPUT_DIR}"
echo "✅ Stage 1 complete."

# Stage 2
STAGE2_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage2_MoE_finetune"
STAGE2_RUN_NAME="${BASE_RUN_NAME}_stage2_MoE_finetune"
echo "--- Starting Stage 2: MoE Fine-tuning (Accelerated) ---"
python main.py "${STAGE2_CONFIG}" --stage 2 --load-weights-from "${STAGE1_OUTPUT_DIR}" --run-name "${STAGE2_RUN_NAME}" --output-dir "${STAGE2_OUTPUT_DIR}"
echo "✅ Stage 2 complete."

echo "🎉 V5 ACCELERATED PIPELINE FINISHED SUCCESSFULLY!"
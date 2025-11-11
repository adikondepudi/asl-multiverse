#!/bin/bash
# Master script to run the v5 two-stage training pipeline.
# Usage: ./run_pipeline.sh <config_prefix>
# Example: ./run_pipeline.sh prod -> uses the 8-hour production configs
# Example: ./run_pipeline.sh default -> uses the original longer training configs
set -e

CONFIG_TYPE=${1:-prod} # Use 'prod' (8hr) as default if no argument is provided

echo "🚀 STARTING ASL MULTIVERSE V5 PIPELINE (Config: ${CONFIG_TYPE})..."
echo "==================================================================="

# --- 1. Define Configurations based on argument ---
if [ "$CONFIG_TYPE" == "prod" ]; then
    STAGE1_CONFIG="config/v5_stage1_8hr_prod.yaml"
    STAGE2_CONFIG="config/v5_stage2_8hr_prod.yaml"
else
    STAGE1_CONFIG="config/v5_stage1_pretrain.yaml"
    STAGE2_CONFIG="config/v5_stage2_MoE_finetune.yaml"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RUN_NAME="${CONFIG_TYPE}_v5_run_${TIMESTAMP}"

# --- 2. Stage 1: Pre-training the Generalist Encoder ---
STAGE1_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage1_pretrain"
STAGE1_RUN_NAME="${BASE_RUN_NAME}_stage1_pretrain"
echo "--- Starting Stage 1: Pre-training (${CONFIG_TYPE}) ---"
python main.py "${STAGE1_CONFIG}" --stage 1 --run-name "${STAGE1_RUN_NAME}" --output-dir "${STAGE1_OUTPUT_DIR}"
echo "✅ Stage 1 complete."

# --- 3. Stage 2: Fine-tuning the MoE Head ---
STAGE2_OUTPUT_DIR="training_runs/${BASE_RUN_NAME}_stage2_MoE_finetune"
STAGE2_RUN_NAME="${BASE_RUN_NAME}_stage2_MoE_finetune"
echo "--- Starting Stage 2: MoE Fine-tuning (${CONFIG_TYPE}) ---"
python main.py "${STAGE2_CONFIG}" --stage 2 --load-weights-from "${STAGE1_OUTPUT_DIR}" --run-name "${STAGE2_RUN_NAME}" --output-dir "${STAGE2_OUTPUT_DIR}"
echo "✅ Stage 2 complete."

echo "🎉 V5 PIPELINE (${CONFIG_TYPE}) FINISHED SUCCESSFULLY!"
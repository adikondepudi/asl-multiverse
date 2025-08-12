#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# ==============================================================================
# ASL MULTIVERSE: IN-VIVO ANALYSIS & EVALUATION PIPELINE (MODIFIED)
# ==============================================================================
# This script can be run standalone or orchestrated by a master script.
# It accepts the MODEL_ARTIFACTS_DIR as an environment variable.

# --- 1. Define Input Directories ---
# Use the environment variable if it's set, otherwise use a default.
MODEL_ARTIFACTS_DIR=${MODEL_ARTIFACTS_DIR:-"final_training_run_v12"}
RAW_INVIVO_SOURCE_DIR="Multiverse"

# --- 2. Define Output Directories (These will be created specific to the model run) ---
RUN_NAME=$(basename "$MODEL_ARTIFACTS_DIR")
BASE_RESULTS_DIR="./results_all_versions"
VALIDATED_INVIVO_DIR="${BASE_RESULTS_DIR}/${RUN_NAME}/data/invivo_validated"
PROCESSED_INVIVO_DIR="${BASE_RESULTS_DIR}/${RUN_NAME}/data/invivo_processed_npy"
FINAL_MAPS_DIR="${BASE_RESULTS_DIR}/${RUN_NAME}/results/final_invivo_maps"
FINAL_EVAL_DIR="${BASE_RESULTS_DIR}/${RUN_NAME}/results/final_evaluation_metrics"

# --- 3. Pre-flight Checks ---
if [ ! -d "$RAW_INVIVO_SOURCE_DIR" ]; then
    echo "❌ ERROR: Raw data folder '$RAW_INVIVO_SOURCE_DIR' not found."
    echo "Please make sure your 'Multiverse' data folder is in the current directory."
    exit 1
fi

if [ ! -d "$MODEL_ARTIFACTS_DIR" ]; then
    echo "❌ ERROR: Model artifacts folder '$MODEL_ARTIFACTS_DIR' not found."
    exit 1
fi

# --- 4. Pipeline Execution ---
echo "🚀 STARTING ANALYSIS FOR MODEL: $RUN_NAME..."
echo "------------------------------------------------"

# Step 1: Validate and Prepare Raw In-Vivo Data
echo "STEP 1: Validating and copying raw in-vivo subjects..."
python prepare_invivo_data.py "$RAW_INVIVO_SOURCE_DIR" "$VALIDATED_INVIVO_DIR"
echo "✅ STEP 1 COMPLETE."
echo "------------------------------------------------"

# Step 2: Process Validated Data into NumPy Format
echo "STEP 2: Processing validated data into NumPy format..."
python process_invivo_data.py "$VALIDATED_INVIVO_DIR" "$PROCESSED_INVIVO_DIR"
echo "✅ STEP 2 COMPLETE."
echo "------------------------------------------------"

# Step 3: Run Prediction on Processed In-Vivo Data
echo "STEP 3: Generating final parameter maps..."
python predict_on_invivo.py "$PROCESSED_INVIVO_DIR" "$MODEL_ARTIFACTS_DIR" "$FINAL_MAPS_DIR"
echo "✅ STEP 3 COMPLETE."
echo "------------------------------------------------"

# Step 4: Run Final Quantitative Evaluations
echo "STEP 4: Running final quantitative evaluations..."
python run_all_evaluations.py "$FINAL_MAPS_DIR" "$PROCESSED_INVIVO_DIR" "$VALIDATED_INVIVO_DIR" "$MODEL_ARTIFACTS_DIR" "$FINAL_EVAL_DIR"
echo "✅ STEP 4 COMPLETE."
echo "------------------------------------------------"

echo "🎉 ANALYSIS FOR '$RUN_NAME' FINISHED SUCCESSFULLY!"
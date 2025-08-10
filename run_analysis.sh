#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# ==============================================================================
# ASL MULTIVERSE: IN-VIVO ANALYSIS & EVALUATION PIPELINE
# ==============================================================================
# This script assumes training is already complete. It will:
#  1. Process the raw in-vivo data from the 'Multiverse' folder.
#  2. Generate CBF/ATT maps using the pre-trained models.
#  3. Calculate and report the final quantitative metrics.

# --- 1. Define Input Directories (THESE MUST EXIST) ---
RAW_INVIVO_SOURCE_DIR="Multiverse"
MODEL_ARTIFACTS_DIR="final_training_run_v10"

# --- 2. Define Output Directories (These will be created) ---
VALIDATED_INVIVO_DIR="./data/invivo_validated"
PROCESSED_INVIVO_DIR="./data/invivo_processed_npy"
FINAL_MAPS_DIR="./results/final_invivo_maps"
FINAL_EVAL_DIR="./results/final_evaluation_metrics"

# --- 3. Pre-flight Checks ---
if [ ! -d "$RAW_INVIVO_SOURCE_DIR" ]; then
    echo "❌ ERROR: Raw data folder '$RAW_INVIVO_SOURCE_DIR' not found."
    echo "Please make sure your 'Multiverse' data folder is in the current directory."
    exit 1
fi

if [ ! -d "$MODEL_ARTIFACTS_DIR" ]; then
    echo "❌ ERROR: Model artifacts folder '$MODEL_ARTIFACTS_DIR' not found."
    echo "Please make sure your 'final_training_run_v5' folder is in the current directory."
    exit 1
fi

# --- 4. Pipeline Execution ---
echo "🚀 STARTING ASL MULTIVERSE ANALYSIS PIPELINE..."
echo "------------------------------------------------"

# Step 1: Validate and Prepare Raw In-Vivo Data
# Takes the 'Multiverse' directory, validates each subject, and copies only the
# good ones to a clean directory for further processing.
echo "STEP 1: Validating and copying raw in-vivo subjects from '$RAW_INVIVO_SOURCE_DIR'..."
python prepare_invivo_data.py "$RAW_INVIVO_SOURCE_DIR" "$VALIDATED_INVIVO_DIR"
echo "✅ STEP 1 COMPLETE."
echo "------------------------------------------------"

# Step 2: Process Validated Data into NumPy Format
# Converts the validated NIfTI files into flattened NumPy arrays (.npy) which
# are required by the prediction and evaluation scripts.
echo "STEP 2: Processing validated data into NumPy format..."
python process_invivo_data.py "$VALIDATED_INVIVO_DIR" "$PROCESSED_INVIVO_DIR"
echo "✅ STEP 2 COMPLETE."
echo "------------------------------------------------"

# Step 3: Run Prediction on Processed In-Vivo Data
# Takes the trained models and the processed in-vivo data to generate the
# final CBF and ATT parameter maps (e.g., 'nn_from_1_repeat_cbf.nii.gz').
echo "STEP 3: Generating final parameter maps for in-vivo data..."
python predict_on_invivo.py "$PROCESSED_INVIVO_DIR" "$MODEL_ARTIFACTS_DIR" "$FINAL_MAPS_DIR"
echo "✅ STEP 3 COMPLETE."
echo "------------------------------------------------"

# Step 4: Run Final Quantitative Evaluations
# Compares the generated maps (NN vs. LS) and calculates all the final
# metrics, including mean GM values and test-retest CoV.
echo "STEP 4: Running final quantitative evaluations and generating summary tables..."
python run_all_evaluations.py "$FINAL_MAPS_DIR" "$PROCESSED_INVIVO_DIR" "$VALIDATED_INVIVO_DIR" "$MODEL_ARTIFACTS_DIR" "$FINAL_EVAL_DIR"
echo "✅ STEP 4 COMPLETE."
echo "------------------------------------------------"

echo "🎉 ASL MULTIVERSE ANALYSIS FINISHED SUCCESSFULLY!"
echo "Find your final evaluation summaries in: $FINAL_EVAL_DIR"
#!/bin/bash

# =====================================================================================
# ASL-MULTIVERSE MASTER PIPELINE SCRIPT
#
# This script orchestrates the entire workflow from raw data to final evaluation.
#
# USAGE:
#   ./run_pipeline.sh /path/to/original/raw/data /path/to/output/base
#
# EXAMPLE:
#   ./run_pipeline.sh ./raw_subject_data ./pipeline_output
# =====================================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- 0. Configuration ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_raw_data> <path_to_output_base_dir>"
    exit 1
fi

RAW_DATA_DIR="$1"
OUTPUT_BASE_DIR="$2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define directory structure for the pipeline run
VALIDATED_DATA_DIR="${OUTPUT_BASE_DIR}/01_validated_raw_data"
PREPROCESSED_DIR="${OUTPUT_BASE_DIR}/02_preprocessed_data"
OFFLINE_DATASET_DIR="${OUTPUT_BASE_DIR}/03_offline_dataset_10M"
MODEL_RESULTS_DIR="${OUTPUT_BASE_DIR}/04_model_results_${TIMESTAMP}"
INVIVO_MAPS_DIR="${OUTPUT_BASE_DIR}/05_invivo_maps_${TIMESTAMP}"
FINAL_EVAL_DIR="${OUTPUT_BASE_DIR}/06_final_evaluation_${TIMESTAMP}"

# --- 1. Validate and Prepare Raw In-Vivo Data ---
echo -e "\n\n--- STEP 1: VALIDATING AND COPYING RAW DATA ---"
python prepare_invivo_data.py "${RAW_DATA_DIR}" "${VALIDATED_DATA_DIR}"

# --- 2. Preprocess In-Vivo Data into NumPy Format ---
echo -e "\n\n--- STEP 2: PREPROCESSING IN-VIVO DATA ---"
python process_invivo_data.py "${VALIDATED_DATA_DIR}" "${PREPROCESSED_DIR}"

# --- 3. Segment Tissues to Create Gray Matter Masks ---
echo -e "\n\n--- STEP 3: SEGMENTING TISSUES (Requires FSL) ---"
# This step requires the new segment_tissues.py script
python segment_tissues.py "${PREPROCESSED_DIR}"

# --- 4. Generate Large-Scale Offline Dataset for Training ---
echo -e "\n\n--- STEP 4: GENERATING OFFLINE TRAINING DATASET ---"
if [ -d "${OFFLINE_DATASET_DIR}" ]; then
    echo "Offline dataset already exists at ${OFFLINE_DATASET_DIR}. Skipping generation."
else
    # Generate a smaller dataset for a faster demo run. Change --total_samples for a full run.
    python generate_offline_dataset.py "${OFFLINE_DATASET_DIR}" --total_samples 1000000 --chunk_size 25000
fi

# --- 5. Run the Final Production Training ---
echo -e "\n\n--- STEP 5: RUNNING FINAL MODEL TRAINING ---"
# We need to update the config to point to the generated offline dataset
# Using sed to create a temporary, updated config file.
TEMP_CONFIG_PATH="${OUTPUT_BASE_DIR}/temp_production_config.yaml"
sed "s|offline_dataset_path:.*|offline_dataset_path: \"${OFFLINE_DATASET_DIR}\"|" config/production_final.yaml > "${TEMP_CONFIG_PATH}"
python main.py "${TEMP_CONFIG_PATH}" "${MODEL_RESULTS_DIR}"
rm "${TEMP_CONFIG_PATH}" # Clean up

# --- 6. Predict on In-Vivo Data using Trained Model ---
echo -e "\n\n--- STEP 6: GENERATING IN-VIVO CBF/ATT MAPS ---"
python predict_on_invivo.py "${PREPROCESSED_DIR}" "${MODEL_RESULTS_DIR}" "${INVIVO_MAPS_DIR}"

# --- 7. Run Final Quantitative Evaluation ---
echo -e "\n\n--- STEP 7: RUNNING FINAL QUANTITATIVE EVALUATION ---"
# This step requires the new run_all_evaluations.py script
python run_all_evaluations.py "${INVIVO_MAPS_DIR}" "${PREPROCESSED_DIR}" "${VALIDATED_DATA_DIR}" "${MODEL_RESULTS_DIR}" "${FINAL_EVAL_DIR}"

echo -e "\n\n--- ✅ PIPELINE COMPLETED SUCCESSFULLY ---"
echo "Final quantitative results are in: ${FINAL_EVAL_DIR}"
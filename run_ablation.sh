#!/bin/bash
# run_ablation.sh - A script to run a 2D ablation study on PINN loss weights for both training stages.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# 1. Define the PINN weight PAIRS for [Stage 1, Stage 2] you want to test.
#    Use a "S1_WEIGHT,S2_WEIGHT" format.
#    "0.0,0.0" is the crucial "no PINN" baseline.
# In run_ablation.sh

# This set provides a comprehensive exploration of the PINN loss parameter space.
ABLATION_PAIRS=(
    # --- From the First Refined Search ---
    "1.0,0.05"      # Refining the Winner (gentler S2)
    "0.5,0.1"       # Splitting the Difference (interpolating S1)
    "1.0,0.5"       # Stronger Fine-Tuning (stronger S2)
    "0.1,0.0"       # Physics Foundation Only (no S2)
    "2.0,0.1"       # Exploring the Upper Bound (stronger S1)

    # --- From the Second, More Comprehensive Search ---
    "5.0,0.1"       # Finding the Cliff (even stronger S1)
    "1.0,1.0"       # Consistent Physics Nudge (no curriculum effect)
    "0.1,2.0"       # Reverse Curriculum (sanity check)
    "0.0,1.0"       # Data-First, Physics-Second (alternative philosophy)

    # --- Don't forget to include the original champion and baseline in your final analysis ---
    "1.0,0.1"       # The original champion from the first run
    "0.0,0.0"       # The essential no-physics baseline
)

# 2. Choose your base configuration file. Use a debug config for faster testing.
BASE_CONFIG="config/ablation_base.yaml" # Use "config/pinn_strong.yaml" or "default.yaml" for the full run

# 3. Define a parent directory for all ablation results.
STUDY_DIR="pinn_ablation_2D_$(date +%Y%m%d_%H%M%S)"

# --- Execution ---
echo "Starting 2D PINN Loss Ablation Study."
echo "Base config: ${BASE_CONFIG}"
echo "Results will be saved in: ${STUDY_DIR}"
echo "Testing [Stage 1, Stage 2] weight pairs..."
printf " - %s\n" "${ABLATION_PAIRS[@]}"
echo "================================================="

mkdir -p "${STUDY_DIR}"

for PAIR in "${ABLATION_PAIRS[@]}"; do
    # Use Internal Field Separator (IFS) to split the pair by comma
    IFS=',' read -r WEIGHT_S1 WEIGHT_S2 <<< "$PAIR"

    RUN_NAME="pinn_s1_${WEIGHT_S1}_s2_${WEIGHT_S2}"
    RUN_DIR="${STUDY_DIR}/${RUN_NAME}"
    TEMP_CONFIG_FILE="${RUN_DIR}/temp_config.yaml"

    echo ""
    echo "----- Running for [S1, S2] weights = [${WEIGHT_S1}, ${WEIGHT_S2}] -----"
    
    mkdir -p "${RUN_DIR}"
    cp "${BASE_CONFIG}" "${TEMP_CONFIG_FILE}"

    # Modify the temporary config file using yq for both stage weights
    echo "Setting loss_pinn_weight_stage1 = ${WEIGHT_S1}"
    yq e ".training.loss_pinn_weight_stage1 = ${WEIGHT_S1}" -i "${TEMP_CONFIG_FILE}"
    
    echo "Setting loss_pinn_weight_stage2 = ${WEIGHT_S2}"
    yq e ".training.loss_pinn_weight_stage2 = ${WEIGHT_S2}" -i "${TEMP_CONFIG_FILE}"

    # Run the main script with the modified config and specific output directory
    # Note: Assumes you've made the change to main.py to accept a second argument
    WANDB_MODE=offline python main.py "${TEMP_CONFIG_FILE}" "${RUN_DIR}"
    
    echo "----- Finished run for weights [${WEIGHT_S1}, ${WEIGHT_S2}] -----"
    echo "Results are in ${RUN_DIR}"
    echo "================================================="
done

echo "Ablation study complete."
echo "You can now run the aggregation script:"
echo "python aggregate_results.py ${STUDY_DIR}"
#!/bin/bash
# A simple script to launch a single, serial Optuna HPO worker on one GPU.

# --- Configuration ---
# The GPU ID to use. '0' is almost always the correct choice for a single-GPU machine.
GPU_ID=0

# The name of the Optuna study.
STUDY_NAME="production_hpo_study"

# The directory where HPO results will be saved.
OUTPUT_DIR="production_run_hpo"

# The config file to use for the HPO run.
CONFIG_FILE="config/hpo.yaml"
# --- End Configuration ---


# --- Script Logic ---
echo "--- Serial HPO Launcher ---"
echo "Starting HPO worker on GPU ${GPU_ID}..."

mkdir -p $OUTPUT_DIR

# Set environment variables and run the main python script.
# We are no longer running in the background with '&'.
CUDA_VISIBLE_DEVICES=$GPU_ID \
python main.py $CONFIG_FILE $OUTPUT_DIR --optimize --study-name "$STUDY_NAME"

echo "----------------------------------------------------"
echo "HPO run has completed."
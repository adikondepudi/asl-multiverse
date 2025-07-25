#!/bin/bash
# FILE: launch_parallel_hpo.sh
# A script to automatically launch multiple Optuna HPO workers in parallel,
# with each worker assigned to a unique GPU.

# --- Configuration ---
# The total number of GPUs you want to use.
# For your 8x RTX 4070 machine, this should be 8.
NUM_GPUS_TO_USE=8

# The name of the shared Optuna study. All workers will contribute to this study.
STUDY_NAME="production_hpo_study"

# The directory where HPO results (norm_stats, best_params, db file) will be saved.
OUTPUT_DIR="production_run_hpo"

# The config file to use for the HPO run.
CONFIG_FILE="config/hpo.yaml"
# --- End Configuration ---


# --- Script Logic ---
echo "--- Automated Parallel HPO Launcher ---"
echo "Starting ${NUM_GPUS_TO_USE} HPO workers..."

# Create the output directory for the results.
mkdir -p $OUTPUT_DIR

# The main loop.
for (( i=0; i<$NUM_GPUS_TO_USE; i++ ))
do
    echo "Launching worker for GPU ${i}..."

    CACHE_DIR="/tmp/torch_cache_worker_${i}"
    mkdir -p $CACHE_DIR

    # Set environment variables ON THE SAME LINE as the command.
    # This is the most robust way to ensure they apply only to this specific process.
    TORCH_INDUCTOR_CACHE_DIR=$CACHE_DIR \
    CUDA_VISIBLE_DEVICES=$i \
    MAX_JOINT_CAT_BYTES=100000 \
    python main.py $CONFIG_FILE $OUTPUT_DIR --optimize --study-name "$STUDY_NAME" &

    sleep 2
done

echo "----------------------------------------------------"
echo "All ${NUM_GPUS_TO_USE} workers have been launched in the background."
echo "You can monitor their progress using 'htop' or 'nvidia-smi'."
echo "To stop all workers, run: pkill -f 'python main.py'"
echo "You can now safely detach from tmux (Ctrl+B, then D) or close your SSH window."
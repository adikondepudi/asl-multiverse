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

# The main loop. This will iterate from GPU 0 to GPU (NUM_GPUS_TO_USE - 1).
for (( i=0; i<$NUM_GPUS_TO_USE; i++ ))
do
    echo "Launching worker for GPU ${i}..."

    # Set the CUDA_VISIBLE_DEVICES environment variable specifically for the upcoming command.
    # This ensures each background process only sees one unique GPU.
    # The '&' at the end of the command runs the process in the background.
    CUDA_VISIBLE_DEVICES=$i python main.py $CONFIG_FILE $OUTPUT_DIR --optimize --study-name "$STUDY_NAME" &

    # Add a small delay between launches to prevent system/network congestion.
    sleep 2
done

echo "----------------------------------------------------"
echo "All ${NUM_GPUS_TO_USE} workers have been launched in the background."
echo "You can monitor their progress using 'htop' or 'nvidia-smi'."
echo "To see the live output of all workers, you can run:"
echo "tail -f ${OUTPUT_DIR}/slurm_logs/*.out  (Note: This assumes main.py logs to a file; if not, monitor via htop)"
echo "IMPORTANT: The processes will continue to run even if you close your SSH session."
echo "To stop all workers, you can run: pkill -f 'python main.py'"
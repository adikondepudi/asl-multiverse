#!/bin/bash
#SBATCH --job-name=asl_multiverse_final    # Job name for the Final Training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1                  # Use the H100 for the long training run
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00                    # 48 hours for the full training schedule
#SBATCH --output=slurm_logs/final_run_%j.out
#SBATCH --error=slurm_logs/final_run_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@jh.edu      # <-- IMPORTANT: CHANGE THIS TO YOUR EMAIL

# --- Environment Setup ---
echo "======================================================"
echo "Starting Final Training Job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================================"

module purge
module load anaconda3/2023.09
source activate asl_multiverse
export WANDB_MODE=offline

# --- Critical Step: Reuse Artifacts from HPO Run ---
#
# V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V
#   EDIT THIS LINE to point to your completed HPO results directory!
#   For example: HPO_RESULTS_DIR="production_run_hpo_12345"
# V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V
HPO_RESULTS_DIR="production_run_hpo_<JOB_ID>"

FINAL_OUTPUT_DIR="production_run_final_${SLURM_JOB_ID}"
mkdir -p $FINAL_OUTPUT_DIR

if [ -f "${HPO_RESULTS_DIR}/norm_stats.json" ]; then
    echo "Copying norm_stats.json from HPO run to reuse for final training."
    cp "${HPO_RESULTS_DIR}/norm_stats.json" "${FINAL_OUTPUT_DIR}/"
else
    echo "FATAL: norm_stats.json not found in ${HPO_RESULTS_DIR}. Aborting."
    exit 1
fi

echo "Running Final Training. Results will be in: $FINAL_OUTPUT_DIR"

# Run the final training using the OPTIMIZED config file
python main.py config/production_optimized.yaml $FINAL_OUTPUT_DIR

echo "======================================================"
echo "Final Training Job finished at $(date)"
echo "======================================================"
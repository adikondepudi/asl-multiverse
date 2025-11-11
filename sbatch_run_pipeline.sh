#!/bin/bash
#SBATCH --job-name=asl-v5-pipe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_logs/v5_pipeline_%j.out
#SBATCH --error=slurm_logs/v5_pipeline_%j.err

echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_logs
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
export WANDB_MODE=online

# --- Execute the Master Pipeline Script ---
# To run the 8-hour production version:
#   sbatch sbatch_run_pipeline.sh prod
#
# To run the default (longer) version:
#   sbatch sbatch_run_pipeline.sh default
#
# If run without an argument, it defaults to 'prod'.
#   sbatch sbatch_run_pipeline.sh

CONFIG_CHOICE=${1:-prod}
echo "Executing the v5 two-stage training pipeline with config: ${CONFIG_CHOICE}"
./run_pipeline.sh "${CONFIG_CHOICE}"

echo "Job finished at $(date)."
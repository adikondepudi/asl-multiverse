# FILE: sbatch_v5_fast_test.sh
#!/bin/bash
#SBATCH --job-name=asl-v5-fast-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm_logs/v5_fast_test_%j.out
#SBATCH --error=slurm_logs/v5_fast_test_%j.err

# --- Setup ---
echo "Job started on $(hostname) at $(date)"
mkdir -p slurm_logs

# --- Activate Conda Environment ---
echo "Loading Anaconda module and activating environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# --- Set W&B Mode ---
export WANDB_MODE=online

# --- Execute the FAST TEST Pipeline Script ---
echo "Executing the fast test v5 two-stage training pipeline script..."
./run_v5_fast_test.sh

echo "Job finished at $(date)."
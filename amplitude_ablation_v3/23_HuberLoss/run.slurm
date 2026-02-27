#!/bin/bash
#SBATCH --job-name=v3_23
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=amplitude_ablation_v3/23_HuberLoss/slurm_%j.out
#SBATCH --error=amplitude_ablation_v3/23_HuberLoss/slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "EXPERIMENT: 23_HuberLoss"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# --- TRAINING ---
python main.py amplitude_ablation_v3/23_HuberLoss/config.yaml \
    --stage 2 --output-dir amplitude_ablation_v3/23_HuberLoss

# --- VALIDATION ---
python -m validation.validate_spatial amplitude_ablation_v3/23_HuberLoss

echo "============================================"
echo "EXPERIMENT COMPLETE: 23_HuberLoss"
echo "Finished: $(date)"
echo "============================================"

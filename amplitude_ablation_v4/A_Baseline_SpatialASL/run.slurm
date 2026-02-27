#!/bin/bash
#SBATCH --job-name=v4_A_Baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=amplitude_ablation_v4/A_Baseline_SpatialASL/slurm_%j.out
#SBATCH --error=amplitude_ablation_v4/A_Baseline_SpatialASL/slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "EXPERIMENT: v4_A_Baseline_SpatialASL"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# --- TRAINING ---
python main.py config/v4_baseline_spatial.yaml \
    --stage 2 --output-dir amplitude_ablation_v4/A_Baseline_SpatialASL

# --- VALIDATION ---
python -m validation.validate_spatial amplitude_ablation_v4/A_Baseline_SpatialASL

echo "============================================"
echo "EXPERIMENT COMPLETE: v4_A_Baseline_SpatialASL"
echo "Finished: $(date)"
echo "============================================"

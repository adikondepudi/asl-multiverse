#!/bin/bash
#SBATCH --job-name=v6_A_Baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=amplitude_ablation_v6/A_Baseline_SpatialASL/slurm_%j.out
#SBATCH --error=amplitude_ablation_v6/A_Baseline_SpatialASL/slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

EXP_DIR="amplitude_ablation_v6/A_Baseline_SpatialASL"
CONFIG="${EXP_DIR}/config.yaml"

echo "============================================"
echo "EXPERIMENT: v6_A_Baseline_SpatialASL"
echo "  L1 loss, variance_weight=0.5, 5 PLDs, expanded CBF/ATT"
echo "  Speed: 1 ensemble, batch 64, torch.compile, 30 epochs"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

# --- TRAINING ---
python main.py "${CONFIG}" \
    --stage 2 --output-dir "${EXP_DIR}"

# --- VALIDATION ---
python -m validation.validate_spatial "${EXP_DIR}"

echo "============================================"
echo "EXPERIMENT COMPLETE: v6_A_Baseline_SpatialASL"
echo "Finished: $(date)"
echo "============================================"

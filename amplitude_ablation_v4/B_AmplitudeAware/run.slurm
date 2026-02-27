#!/bin/bash
#SBATCH --job-name=v4_B_AmpAware
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=amplitude_ablation_v4/B_AmplitudeAware/slurm_%j.out
#SBATCH --error=amplitude_ablation_v4/B_AmplitudeAware/slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "EXPERIMENT: v4_B_AmplitudeAware"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# --- TRAINING ---
python main.py config/v4_amplitude_aware.yaml \
    --stage 2 --output-dir amplitude_ablation_v4/B_AmplitudeAware

# --- VALIDATION ---
python -m validation.validate_spatial amplitude_ablation_v4/B_AmplitudeAware

echo "============================================"
echo "EXPERIMENT COMPLETE: v4_B_AmplitudeAware"
echo "Finished: $(date)"
echo "============================================"

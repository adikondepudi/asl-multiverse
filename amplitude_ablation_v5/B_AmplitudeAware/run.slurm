#!/bin/bash
#SBATCH --job-name=v5_B_AmpAware
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=amplitude_ablation_v5/B_AmplitudeAware/slurm_%j.out
#SBATCH --error=amplitude_ablation_v5/B_AmplitudeAware/slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "EXPERIMENT: v5_B_AmplitudeAware"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# --- COPY CONFIG (plot script reads config.yaml from run dir) ---
cp config/v5_amplitude_aware.yaml amplitude_ablation_v5/B_AmplitudeAware/config.yaml

# --- TRAINING ---
python main.py config/v5_amplitude_aware.yaml \
    --stage 2 --output-dir amplitude_ablation_v5/B_AmplitudeAware

# --- VALIDATION ---
python -m validation.validate_spatial amplitude_ablation_v5/B_AmplitudeAware

echo "============================================"
echo "EXPERIMENT COMPLETE: v5_B_AmplitudeAware"
echo "Finished: $(date)"
echo "============================================"

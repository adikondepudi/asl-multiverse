#!/bin/bash
#SBATCH --job-name=v4-plots
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=amplitude_ablation_v4/plots_%j.out
#SBATCH --error=amplitude_ablation_v4/plots_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "GENERATING BIAS/COV PLOTS (v4)"
echo "Started: $(date)"
echo "============================================"

# Ensure config.yaml exists in each run dir (plot script needs it)
cp -n config/v4_baseline_spatial.yaml amplitude_ablation_v4/A_Baseline_SpatialASL/config.yaml 2>/dev/null || true
cp -n config/v4_amplitude_aware.yaml amplitude_ablation_v4/B_AmplitudeAware/config.yaml 2>/dev/null || true

python generate_bias_cov_plots_v2.py \
    --output-dir bias_cov_results_v4 \
    --snr 3.0 5.0 10.0 \
    --n-phantoms 10 --n-ls-realizations 1000 \
    --models "Baseline SpatialASLNet:amplitude_ablation_v4/A_Baseline_SpatialASL" \
             "AmplitudeAware:amplitude_ablation_v4/B_AmplitudeAware"

echo "============================================"
echo "PLOTS COMPLETE"
echo "Finished: $(date)"
echo "============================================"

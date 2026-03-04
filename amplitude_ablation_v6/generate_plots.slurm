#!/bin/bash
#SBATCH --job-name=v6-plots
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=amplitude_ablation_v6/plots_%j.out
#SBATCH --error=amplitude_ablation_v6/plots_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "GENERATING BIAS/COV PLOTS (v6)"
echo "5 PLDs [500..2500], CBF sweep to 200"
echo "Started: $(date)"
echo "============================================"

# Verify configs exist in experiment dirs (plot script needs them)
for exp_dir in amplitude_ablation_v6/A_Baseline_SpatialASL amplitude_ablation_v6/B_AmplitudeAware; do
    if [ ! -f "${exp_dir}/config.yaml" ]; then
        echo "ERROR: Missing ${exp_dir}/config.yaml"
        exit 1
    fi
done

python generate_bias_cov_plots_v2.py \
    --output-dir bias_cov_results_v6 \
    --pld-values 500 1000 1500 2000 2500 \
    --snr 3.0 5.0 10.0 \
    --n-phantoms 10 --n-ls-realizations 1000 \
    --models "Baseline SpatialASLNet:amplitude_ablation_v6/A_Baseline_SpatialASL" \
             "AmplitudeAware:amplitude_ablation_v6/B_AmplitudeAware"

echo "============================================"
echo "PLOTS COMPLETE"
echo "Finished: $(date)"
echo "============================================"

#!/bin/bash
#SBATCH --job-name=v6_datagen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=amplitude_ablation_v6/datagen_slurm_%j.out
#SBATCH --error=amplitude_ablation_v6/datagen_slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "DATA GENERATION: asl_spatial_dataset_v6"
echo "V6 changes: 5 PLDs [500..2500], expanded CBF (GM 10-200), ATT (GM 500-3000)"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "============================================"

python -m simulation.generate_clean_library \
    asl_spatial_dataset_v6 \
    --spatial --total_samples 100000 \
    --spatial-chunk-size 500 --image-size 64 --domain-rand \
    --pld-values 500 1000 1500 2000 2500

echo "============================================"
echo "DATA GENERATION COMPLETE"
echo "Finished: $(date)"
echo "============================================"

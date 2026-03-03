#!/bin/bash
#SBATCH --job-name=v5_datagen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=amplitude_ablation_v5/datagen_slurm_%j.out
#SBATCH --error=amplitude_ablation_v5/datagen_slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "DATA GENERATION: asl_spatial_dataset_v5"
echo "Overhaul ranges: GM CBF 10-120, WM CBF 5-50"
echo "  + tumor_hyper CBF 80-200, stroke_core CBF 0-10"
echo "  + alpha_BS1 domain randomization [0.85, 1.0]"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "============================================"

python -m simulation.generate_clean_library \
    asl_spatial_dataset_v5 \
    --spatial --total_samples 100000 \
    --spatial-chunk-size 500 --image-size 64 --domain-rand

echo "============================================"
echo "DATA GENERATION COMPLETE"
echo "Finished: $(date)"
echo "============================================"

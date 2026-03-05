#!/bin/bash
#SBATCH --job-name=v7_datagen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=amplitude_ablation_v7/datagen_slurm_%j.out
#SBATCH --error=amplitude_ablation_v7/datagen_slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "V7 DATA GENERATION"
echo "Reusing V6 data if available, otherwise regenerating"
echo "Started: $(date)"
echo "============================================"

DATASET_DIR="asl_spatial_dataset_v6"

if [ -d "$DATASET_DIR" ]; then
    NUM_CHUNKS=$(ls -1 ${DATASET_DIR}/spatial_chunk_*.npz 2>/dev/null | wc -l)
    if [ "$NUM_CHUNKS" -gt 0 ]; then
        echo "Found existing V6 dataset with ${NUM_CHUNKS} chunks. Skipping generation."
        exit 0
    fi
fi

echo "Generating new spatial dataset..."
python -m simulation.generate_clean_library \
    asl_spatial_dataset_v6 \
    --spatial --total_samples 100000 \
    --spatial-chunk-size 500 --image-size 64 --domain-rand \
    --pld-values 500 1000 1500 2000 2500

echo "============================================"
echo "DATA GENERATION COMPLETE"
echo "Finished: $(date)"
echo "============================================"

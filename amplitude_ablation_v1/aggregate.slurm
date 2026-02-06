#!/bin/bash
#SBATCH --job-name=amp-aggregate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=amplitude_ablation_v1/aggregate_%j.out
#SBATCH --error=amplitude_ablation_v1/aggregate_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

python compare_amplitude_ablation.py --results_dir amplitude_ablation_v1

#!/bin/bash
#SBATCH --job-name=exp001_feats2_noise1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=hpc_ablation_jobs/exp001_feats2_noise1/slurm.out
#SBATCH --error=hpc_ablation_jobs/exp001_feats2_noise1/slurm.err

source ~/.bashrc
conda activate asl_multiverse

cd $SLURM_SUBMIT_DIR

echo "--- 1. STAGE 1: DENOISING PRE-TRAINING ---"
python main.py hpc_ablation_jobs/exp001_feats2_noise1/config.yaml --stage 1 --output-dir hpc_ablation_jobs/exp001_feats2_noise1

echo "--- 2. STAGE 2: REGRESSION TRAINING ---"
python main.py hpc_ablation_jobs/exp001_feats2_noise1/config.yaml --stage 2 --output-dir hpc_ablation_jobs/exp001_feats2_noise1 --load-weights-from hpc_ablation_jobs/exp001_feats2_noise1

echo "--- 3. AUTO-VALIDATION (NN vs LS) ---"
python validate.py --run_dir hpc_ablation_jobs/exp001_feats2_noise1 --output_dir hpc_ablation_jobs/exp001_feats2_noise1/validation_results

echo "--- JOB COMPLETE ---"

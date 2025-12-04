#!/bin/bash
echo 'Launching Ablation Array...'
JOB_0=$(sbatch --parsable hpc_ablation_jobs/exp000_feats2_noise1/run.slurm)
echo "Submitted exp000_feats2_noise1 as Job ID: $JOB_0"
JOB_1=$(sbatch --parsable hpc_ablation_jobs/exp001_feats2_noise1/run.slurm)
echo "Submitted exp001_feats2_noise1 as Job ID: $JOB_1"
JOB_2=$(sbatch --parsable hpc_ablation_jobs/exp002_feats2_noise3/run.slurm)
echo "Submitted exp002_feats2_noise3 as Job ID: $JOB_2"
JOB_3=$(sbatch --parsable hpc_ablation_jobs/exp003_feats2_noise3/run.slurm)
echo "Submitted exp003_feats2_noise3 as Job ID: $JOB_3"
JOB_4=$(sbatch --parsable hpc_ablation_jobs/exp004_feats3_noise1/run.slurm)
echo "Submitted exp004_feats3_noise1 as Job ID: $JOB_4"
JOB_5=$(sbatch --parsable hpc_ablation_jobs/exp005_feats3_noise1/run.slurm)
echo "Submitted exp005_feats3_noise1 as Job ID: $JOB_5"
JOB_6=$(sbatch --parsable hpc_ablation_jobs/exp006_feats3_noise3/run.slurm)
echo "Submitted exp006_feats3_noise3 as Job ID: $JOB_6"
JOB_7=$(sbatch --parsable hpc_ablation_jobs/exp007_feats3_noise3/run.slurm)
echo "Submitted exp007_feats3_noise3 as Job ID: $JOB_7"
JOB_8=$(sbatch --parsable hpc_ablation_jobs/exp008_feats4_noise1/run.slurm)
echo "Submitted exp008_feats4_noise1 as Job ID: $JOB_8"
JOB_9=$(sbatch --parsable hpc_ablation_jobs/exp009_feats4_noise1/run.slurm)
echo "Submitted exp009_feats4_noise1 as Job ID: $JOB_9"
JOB_10=$(sbatch --parsable hpc_ablation_jobs/exp010_feats4_noise3/run.slurm)
echo "Submitted exp010_feats4_noise3 as Job ID: $JOB_10"
JOB_11=$(sbatch --parsable hpc_ablation_jobs/exp011_feats4_noise3/run.slurm)
echo "Submitted exp011_feats4_noise3 as Job ID: $JOB_11"

# Launch Aggregator with Dependency on all jobs
DEPENDENCY="$JOB_0:$JOB_1:$JOB_2:$JOB_3:$JOB_4:$JOB_5:$JOB_6:$JOB_7:$JOB_8:$JOB_9:$JOB_10:$JOB_11"
sbatch --dependency=afterany:${DEPENDENCY} aggregate_results.slurm
echo "Aggregator job submitted with dependency on all experiments"
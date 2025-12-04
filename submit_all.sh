#!/bin/bash
echo '============================================'
echo 'ASL Ablation Study: 10 Targeted Experiments'
echo '============================================'

# Experiment 1: 01_Baseline_Naive
# Hypothesis: How well can we do with just basic stats?
JOB_0=$(sbatch --parsable hpc_ablation_jobs/01_Baseline_Naive/run.slurm)
echo "Submitted 01_Baseline_Naive as Job ID: $JOB_0"

# Experiment 2: 02_Feature_Peak
# Hypothesis: Does adding Peak Height fix the bias?
JOB_1=$(sbatch --parsable hpc_ablation_jobs/02_Feature_Peak/run.slurm)
echo "Submitted 02_Feature_Peak as Job ID: $JOB_1"

# Experiment 3: 03_Feature_Full
# Hypothesis: Does T1 help biological variance?
JOB_2=$(sbatch --parsable hpc_ablation_jobs/03_Feature_Full/run.slurm)
echo "Submitted 03_Feature_Full as Job ID: $JOB_2"

# Experiment 4: 04_Arch_NoConv
# Hypothesis: Do we strictly need the Conv1D, or are scalars enough?
JOB_3=$(sbatch --parsable hpc_ablation_jobs/04_Arch_NoConv/run.slurm)
echo "Submitted 04_Arch_NoConv as Job ID: $JOB_3"

# Experiment 5: 05_Size_Small
# Hypothesis: Can we make it faster?
JOB_4=$(sbatch --parsable hpc_ablation_jobs/05_Size_Small/run.slurm)
echo "Submitted 05_Size_Small as Job ID: $JOB_4"

# Experiment 6: 06_Size_Large
# Hypothesis: Are we underfitting?
JOB_5=$(sbatch --parsable hpc_ablation_jobs/06_Size_Large/run.slurm)
echo "Submitted 06_Size_Large as Job ID: $JOB_5"

# Experiment 7: 07_Robust_Full
# Hypothesis: Does NN beat LS on realistic noise?
JOB_6=$(sbatch --parsable hpc_ablation_jobs/07_Robust_Full/run.slurm)
echo "Submitted 07_Robust_Full as Job ID: $JOB_6"

# Experiment 8: 08_Robust_NoConv
# Hypothesis: Does the Conv1D layer help filter complex noise?
JOB_7=$(sbatch --parsable hpc_ablation_jobs/08_Robust_NoConv/run.slurm)
echo "Submitted 08_Robust_NoConv as Job ID: $JOB_7"

# Experiment 9: 09_Robust_Peak
# Hypothesis: Does Peak height help even more when noise is messy?
JOB_8=$(sbatch --parsable hpc_ablation_jobs/09_Robust_Peak/run.slurm)
echo "Submitted 09_Robust_Peak as Job ID: $JOB_8"

# Experiment 10: 10_Robust_Small
# Hypothesis: Can a small model handle complex noise?
JOB_9=$(sbatch --parsable hpc_ablation_jobs/10_Robust_Small/run.slurm)
echo "Submitted 10_Robust_Small as Job ID: $JOB_9"

# Launch Aggregator with Dependency on all jobs
DEPENDENCY="$JOB_0:$JOB_1:$JOB_2:$JOB_3:$JOB_4:$JOB_5:$JOB_6:$JOB_7:$JOB_8:$JOB_9"
sbatch --dependency=afterany:${DEPENDENCY} aggregate_results.slurm
echo "Aggregator job submitted with dependency on all experiments"
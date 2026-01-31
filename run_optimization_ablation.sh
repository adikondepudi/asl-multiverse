#!/bin/bash
#SBATCH --job-name=opt-orchestrator
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/orchestrator_%j.out
#SBATCH --error=slurm_logs/orchestrator_%j.err

# =============================================================================
# OPTIMIZATION ABLATION - MASTER ORCHESTRATOR (10 experiments)
# =============================================================================
# Usage: sbatch run_optimization_ablation.sh
# =============================================================================

set -e

echo "============================================"
echo "OPTIMIZATION ABLATION (10 experiments)"
echo "Started: $(date)"
echo "============================================"

mkdir -p slurm_logs
cd $SLURM_SUBMIT_DIR

# --- Load Environment ---
echo ""
echo "[1/4] Loading environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# =============================================================================
# STEP 1: Data Generation
# =============================================================================
echo ""
echo "[2/4] Submitting data generation..."

cat > slurm_logs/gen_data.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=opt-datagen
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/datagen_%j.out
#SBATCH --error=slurm_logs/datagen_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "Generating spatial dataset..."
rm -rf asl_spatial_dataset_100k
python generate_clean_library.py asl_spatial_dataset_100k \
    --spatial --total_samples 100000 --spatial-chunk-size 500 --image-size 64

echo "Done: $(date)"
EOF

DATA_JOB=$(sbatch --parsable slurm_logs/gen_data.slurm)
echo "  Data job: $DATA_JOB"

# =============================================================================
# STEP 2: Generate Experiment Configs
# =============================================================================
echo ""
echo "[3/4] Generating experiment configs..."
python setup_optimization_grid.py

# =============================================================================
# STEP 3: Submit Training Jobs
# =============================================================================
echo ""
echo "[4/4] Submitting 10 training experiments..."
echo "  (All depend on data job $DATA_JOB)"
echo ""

JOB_IDS=""
for exp_dir in optimization_ablation_v1/0*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        slurm_script="$exp_dir/run.slurm"

        if [ -f "$slurm_script" ]; then
            JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB "$slurm_script")
            echo "  $exp_name → Job $JOB_ID"

            [ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"
        fi
    fi
done

# =============================================================================
# STEP 4: Aggregation Job
# =============================================================================
echo ""
echo "Submitting aggregation job..."

cat > optimization_ablation_v1/aggregate.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=opt-aggregate
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=optimization_ablation_v1/aggregate_%j.out
#SBATCH --error=optimization_ablation_v1/aggregate_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

python compare_optimization_results.py --results_dir optimization_ablation_v1
EOF

AGG_JOB=$(sbatch --parsable --dependency=afterany:$JOB_IDS optimization_ablation_v1/aggregate.slurm)
echo "  Aggregation → Job $AGG_JOB"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================"
echo "ALL JOBS SUBMITTED"
echo "============================================"
echo "Data generation: $DATA_JOB"
echo "Training jobs:   10 experiments"
echo "Aggregation:     $AGG_JOB"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: optimization_ablation_v1/optimization_comparison.csv"
echo "============================================"

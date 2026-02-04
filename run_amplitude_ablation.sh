#!/bin/bash
#SBATCH --job-name=amp-ablation
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/amp_ablation_orchestrator_%j.out
#SBATCH --error=slurm_logs/amp_ablation_orchestrator_%j.err

# =============================================================================
# AMPLITUDE-AWARE ABLATION - MASTER ORCHESTRATOR (10 experiments)
# =============================================================================

set -e

echo "============================================"
echo "AMPLITUDE-AWARE ABLATION (10 experiments)"
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
# STEP 1: Data Generation (20k samples for quick ablation)
# =============================================================================
echo ""
echo "[2/4] Submitting data generation (20k samples)..."

cat > slurm_logs/gen_ablation_data.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=amp-datagen
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/amp_datagen_%j.out
#SBATCH --error=slurm_logs/amp_datagen_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "Generating spatial dataset for ablation (20k samples)..."
rm -rf asl_spatial_dataset_ablation
python generate_clean_library.py asl_spatial_dataset_ablation \
    --spatial --total_samples 20000 --spatial-chunk-size 500 --image-size 64

echo "Done: $(date)"
EOF

DATA_JOB=$(sbatch --parsable slurm_logs/gen_ablation_data.slurm)
echo "  Data generation job: $DATA_JOB"

# =============================================================================
# STEP 2: Submit Training Jobs
# =============================================================================
echo ""
echo "[3/4] Submitting 10 training experiments..."
echo "  (All depend on data job $DATA_JOB)"
echo ""

JOB_IDS=""

# 00_Baseline_SpatialASL: Control: Standard SpatialASLNet (shows amplitude invariance problem)
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/00_Baseline_SpatialASL/run.slurm)
echo "  00_Baseline_SpatialASL -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 01_PerCurve_Norm: Confirm: per_curve normalization destroys CBF info (should be worse)
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/01_PerCurve_Norm/run.slurm)
echo "  01_PerCurve_Norm -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 02_AmpAware_Full: Full AmplitudeAwareSpatialASLNet with all components
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/02_AmpAware_Full/run.slurm)
echo "  02_AmpAware_Full -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 03_AmpAware_OutputMod_Only: Amplitude output modulation only (no FiLM) - is direct amplitude enough?
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/03_AmpAware_OutputMod_Only/run.slurm)
echo "  03_AmpAware_OutputMod_Only -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 04_AmpAware_FiLM_Only: FiLM conditioning only (no output modulation) - is FiLM alone enough?
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/04_AmpAware_FiLM_Only/run.slurm)
echo "  04_AmpAware_FiLM_Only -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 05_AmpAware_Bottleneck_Only: FiLM at bottleneck only (minimal modification)
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/05_AmpAware_Bottleneck_Only/run.slurm)
echo "  05_AmpAware_Bottleneck_Only -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 06_AmpAware_Physics_0p1: Full AmplitudeAware + physics loss (dc_weight=0.1)
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/06_AmpAware_Physics_0p1/run.slurm)
echo "  06_AmpAware_Physics_0p1 -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 07_AmpAware_Physics_0p3: Full AmplitudeAware + stronger physics loss (dc_weight=0.3)
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/07_AmpAware_Physics_0p3/run.slurm)
echo "  07_AmpAware_Physics_0p3 -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 08_AmpAware_DomainRand: Full AmplitudeAware + domain randomization for generalization
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/08_AmpAware_DomainRand/run.slurm)
echo "  08_AmpAware_DomainRand -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 09_AmpAware_Optimized: Best combination: Full AmplitudeAware + DomainRand (no DC based on prior ablation)
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v1/09_AmpAware_Optimized/run.slurm)
echo "  09_AmpAware_Optimized -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# =============================================================================
# STEP 3: Aggregation Job
# =============================================================================
echo ""
echo "[4/4] Submitting aggregation job..."

cat > amplitude_ablation_v1/aggregate.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=amp-aggregate
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=amplitude_ablation_v1/aggregate_%j.out
#SBATCH --error=amplitude_ablation_v1/aggregate_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

python compare_amplitude_ablation.py --results_dir amplitude_ablation_v1
EOF

AGG_JOB=$(sbatch --parsable --dependency=afterany:$JOB_IDS amplitude_ablation_v1/aggregate.slurm)
echo "  Aggregation -> Job $AGG_JOB"

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
echo "Results: amplitude_ablation_v1/amplitude_ablation_summary.csv"
echo "============================================"

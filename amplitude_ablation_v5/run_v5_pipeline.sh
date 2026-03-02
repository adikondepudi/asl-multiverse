#!/bin/bash
#SBATCH --job-name=v5-orchestrator
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=amplitude_ablation_v5/orchestrator_%j.out
#SBATCH --error=amplitude_ablation_v5/orchestrator_%j.err

# =============================================================================
# V5 EXPANDED-RANGE RETRAINING PIPELINE
# =============================================================================
# Fixes from v4:
#   1. Expanded training CBF/ATT ranges (GM: 30-90/500-2500, WM: 10-40/800-3000)
#      to eliminate CBF bias dip at ATT > 1800ms and CBF regression to mean
#   2. global_scale_factor = 10.0 for both experiments (v4 Baseline had 1.0)
#
# Runs the full pipeline:
#   1. Generate spatial training data (100k samples, expanded ranges)
#   2. Train Baseline SpatialASLNet (50 epochs, 3 ensembles)
#   3. Train AmplitudeAwareSpatialASLNet (100 epochs, 3 ensembles)
#   4. Generate bias/CoV plots comparing both models vs LS
#
# Usage:
#   sbatch amplitude_ablation_v5/run_v5_pipeline.sh             # Full pipeline
#   sbatch amplitude_ablation_v5/run_v5_pipeline.sh --train-only  # Skip data gen
#   sbatch amplitude_ablation_v5/run_v5_pipeline.sh --plots-only  # Skip data+training
#
# Estimated time: ~10-12 hours total
#   Data generation: ~2-4 hours (CPU)
#   Baseline training: ~1-2 hours (GPU)
#   AmplitudeAware training: ~6-7 hours (GPU)
#   Bias/CoV plots: ~1-2 hours (CPU+GPU)
# =============================================================================

set -e

# --- Parse Arguments ---
TRAIN_ONLY=false
PLOTS_ONLY=false
for arg in "$@"; do
    case $arg in
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --plots-only)
            PLOTS_ONLY=true
            shift
            ;;
    esac
done

echo "============================================"
echo "V5 EXPANDED-RANGE RETRAINING PIPELINE"
echo "============================================"
echo "Started: $(date)"
echo "Train only: ${TRAIN_ONLY}"
echo "Plots only: ${PLOTS_ONLY}"
echo "============================================"

mkdir -p slurm_logs
cd $SLURM_SUBMIT_DIR

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

DATASET_DIR="asl_spatial_dataset_v5"

# =============================================================================
# STEP 1: DATA GENERATION
# =============================================================================
if [ "$TRAIN_ONLY" = false ] && [ "$PLOTS_ONLY" = false ]; then
    echo ""
    echo "[1/4] Submitting data generation (100k spatial samples, expanded ranges)..."

    DATA_JOB=$(sbatch --parsable amplitude_ablation_v5/generate_data.slurm)
    echo "  Data generation job: ${DATA_JOB}"
    TRAIN_DEP="--dependency=afterok:${DATA_JOB}"
else
    echo ""
    echo "[1/4] Skipping data generation"

    if [ ! -d "$DATASET_DIR" ]; then
        echo "ERROR: Dataset not found: ${DATASET_DIR}"
        echo "Run without --train-only/--plots-only to generate data first."
        exit 1
    fi
    NUM_CHUNKS=$(ls -1 ${DATASET_DIR}/spatial_chunk_*.npz 2>/dev/null | wc -l)
    echo "  Found existing dataset with ${NUM_CHUNKS} chunks"
    TRAIN_DEP=""
fi

# =============================================================================
# STEP 2: TRAINING (2 experiments in parallel)
# =============================================================================
if [ "$PLOTS_ONLY" = false ]; then
    echo ""
    echo "[2/4] Submitting training jobs..."

    JOB_A=$(sbatch --parsable ${TRAIN_DEP} amplitude_ablation_v5/A_Baseline_SpatialASL/run.slurm)
    echo "  A_Baseline_SpatialASL -> Job ${JOB_A}"

    JOB_B=$(sbatch --parsable ${TRAIN_DEP} amplitude_ablation_v5/B_AmplitudeAware/run.slurm)
    echo "  B_AmplitudeAware      -> Job ${JOB_B}"

    PLOT_DEP="--dependency=afterok:${JOB_A}:${JOB_B}"
else
    echo ""
    echo "[2/4] Skipping training"

    # Verify models exist
    for exp in A_Baseline_SpatialASL B_AmplitudeAware; do
        MODEL_DIR="amplitude_ablation_v5/${exp}/trained_models"
        if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
            echo "ERROR: No trained models in ${MODEL_DIR}"
            echo "Run without --plots-only to train first."
            exit 1
        fi
        echo "  Found models in ${MODEL_DIR}"
    done
    PLOT_DEP=""
fi

# =============================================================================
# STEP 3: BIAS/COV PLOTS
# =============================================================================
echo ""
echo "[3/4] Submitting bias/CoV plot generation..."

PLOT_JOB=$(sbatch --parsable ${PLOT_DEP} amplitude_ablation_v5/generate_plots.slurm)
echo "  Bias/CoV plots -> Job ${PLOT_JOB}"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================"
echo "PIPELINE SUBMITTED SUCCESSFULLY"
echo "============================================"
echo ""
echo "Job Chain:"
if [ "$TRAIN_ONLY" = false ] && [ "$PLOTS_ONLY" = false ]; then
    echo "  1. Data Generation:  ${DATA_JOB}"
    echo "     ├─> 2a. Baseline:      ${JOB_A}"
    echo "     └─> 2b. AmplitudeAware: ${JOB_B}"
    echo "              └─> 3. Plots:  ${PLOT_JOB}"
elif [ "$PLOTS_ONLY" = false ]; then
    echo "  1a. Baseline:      ${JOB_A}"
    echo "  1b. AmplitudeAware: ${JOB_B}"
    echo "       └─> 2. Plots: ${PLOT_JOB}"
else
    echo "  1. Plots: ${PLOT_JOB}"
fi
echo ""
echo "Monitor:  squeue -u \$USER"
echo ""
echo "Output locations:"
echo "  Dataset:  ${DATASET_DIR}/"
echo "  Models A: amplitude_ablation_v5/A_Baseline_SpatialASL/trained_models/"
echo "  Models B: amplitude_ablation_v5/B_AmplitudeAware/trained_models/"
echo "  Plots:    bias_cov_results_v5/"
echo "============================================"

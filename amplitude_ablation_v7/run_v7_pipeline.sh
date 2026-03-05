#!/bin/bash
#SBATCH --job-name=v7-orchestrator
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=amplitude_ablation_v7/orchestrator_%j.out
#SBATCH --error=amplitude_ablation_v7/orchestrator_%j.err

# =============================================================================
# V7 PIPELINE: Publication-Grade Training + Evaluation
# =============================================================================
# Changes from V6:
#   1. 3 ensembles (was 1) — publication-grade uncertainty
#   2. 50 epochs (was 30/40) — more training
#   3. variance_weight: 0.0 (was 0.5) — V6 showed it's not needed
#   4. learning_rate: 0.001 for both models
#   5. 24h SLURM time limit for training (was 3h/5h)
#   6. Evaluation pipeline: test phantoms + realistic phantom eval + figures
#
# Pipeline:
#   1. Generate spatial training data (reuses V6 if available)
#   2. Train Baseline SpatialASLNet (50 epochs, 3 ensembles)
#   3. Train AmplitudeAwareSpatialASLNet (50 epochs, 3 ensembles)
#   4. Evaluate: generate test phantoms + evaluate both models + figures
#
# Usage:
#   sbatch amplitude_ablation_v7/run_v7_pipeline.sh               # Full pipeline
#   sbatch amplitude_ablation_v7/run_v7_pipeline.sh --train-only   # Skip data gen
#   sbatch amplitude_ablation_v7/run_v7_pipeline.sh --plots-only   # Skip data+training
#   sbatch amplitude_ablation_v7/run_v7_pipeline.sh --eval-only    # Only evaluation
# =============================================================================

set -e

# --- Parse Arguments ---
TRAIN_ONLY=false
PLOTS_ONLY=false
EVAL_ONLY=false
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
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
    esac
done

echo "============================================"
echo "V7 PIPELINE: Publication-Grade Training + Evaluation"
echo "============================================"
echo "Started: $(date)"
echo "Train only: ${TRAIN_ONLY}"
echo "Plots only: ${PLOTS_ONLY}"
echo "Eval only:  ${EVAL_ONLY}"
echo "============================================"

mkdir -p slurm_logs
cd $SLURM_SUBMIT_DIR

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

DATASET_DIR="asl_spatial_dataset_v6"

# =============================================================================
# EVAL-ONLY MODE: Just run evaluation scripts
# =============================================================================
if [ "$EVAL_ONLY" = true ]; then
    echo ""
    echo "[EVAL-ONLY] Running evaluation pipeline..."

    # Verify models exist
    for exp in A_Baseline_SpatialASL B_AmplitudeAware; do
        MODEL_DIR="amplitude_ablation_v7/${exp}/trained_models"
        if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
            echo "ERROR: No trained models in ${MODEL_DIR}"
            echo "Run training first."
            exit 1
        fi
        echo "  Found models in ${MODEL_DIR}"
    done

    EVAL_JOB=$(sbatch --parsable <<'EVAL_SCRIPT'
#!/bin/bash
#SBATCH --job-name=v7_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=amplitude_ablation_v7/eval_slurm_%j.out
#SBATCH --error=amplitude_ablation_v7/eval_slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "V7 EVALUATION: $(date)"

python amplitude_ablation_v7/generate_test_phantoms.py
python amplitude_ablation_v7/evaluate_realistic_phantoms.py --model-dir amplitude_ablation_v7/A_Baseline_SpatialASL
python amplitude_ablation_v7/evaluate_realistic_phantoms.py --model-dir amplitude_ablation_v7/B_AmplitudeAware
python amplitude_ablation_v7/generate_publication_figures.py

echo "V7 EVALUATION COMPLETE: $(date)"
EVAL_SCRIPT
)
    echo "  Evaluation job: ${EVAL_JOB}"
    echo ""
    echo "Monitor: squeue -u \$USER"
    exit 0
fi

# =============================================================================
# STEP 1: DATA GENERATION
# =============================================================================
if [ "$TRAIN_ONLY" = false ] && [ "$PLOTS_ONLY" = false ]; then
    echo ""
    echo "[1/4] Submitting data generation (reuses V6 data if available)..."

    DATA_JOB=$(sbatch --parsable amplitude_ablation_v7/generate_data.slurm)
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
    echo "[2/4] Submitting training jobs (3 ensembles, 50 epochs each)..."

    JOB_A=$(sbatch --parsable ${TRAIN_DEP} amplitude_ablation_v7/A_Baseline_SpatialASL/run.slurm)
    echo "  A_Baseline_SpatialASL -> Job ${JOB_A}"

    JOB_B=$(sbatch --parsable ${TRAIN_DEP} amplitude_ablation_v7/B_AmplitudeAware/run.slurm)
    echo "  B_AmplitudeAware      -> Job ${JOB_B}"

    EVAL_DEP="--dependency=afterok:${JOB_A}:${JOB_B}"
else
    echo ""
    echo "[2/4] Skipping training"

    # Verify models exist
    for exp in A_Baseline_SpatialASL B_AmplitudeAware; do
        MODEL_DIR="amplitude_ablation_v7/${exp}/trained_models"
        if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
            echo "ERROR: No trained models in ${MODEL_DIR}"
            echo "Run without --plots-only to train first."
            exit 1
        fi
        echo "  Found models in ${MODEL_DIR}"
    done
    EVAL_DEP=""
fi

# =============================================================================
# STEP 3: EVALUATION (test phantoms + model eval + figures)
# =============================================================================
echo ""
echo "[3/4] Submitting evaluation pipeline..."

EVAL_JOB=$(sbatch --parsable ${EVAL_DEP} <<'EVAL_SCRIPT'
#!/bin/bash
#SBATCH --job-name=v7_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=amplitude_ablation_v7/eval_slurm_%j.out
#SBATCH --error=amplitude_ablation_v7/eval_slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "V7 EVALUATION PIPELINE"
echo "Started: $(date)"
echo "============================================"

echo "[1/4] Generating test phantoms..."
python amplitude_ablation_v7/generate_test_phantoms.py

echo "[2/4] Evaluating Baseline SpatialASLNet..."
python amplitude_ablation_v7/evaluate_realistic_phantoms.py --model-dir amplitude_ablation_v7/A_Baseline_SpatialASL

echo "[3/4] Evaluating AmplitudeAwareSpatialASLNet..."
python amplitude_ablation_v7/evaluate_realistic_phantoms.py --model-dir amplitude_ablation_v7/B_AmplitudeAware

echo "[4/4] Generating publication figures..."
python amplitude_ablation_v7/generate_publication_figures.py

echo "============================================"
echo "V7 EVALUATION COMPLETE"
echo "Finished: $(date)"
echo "============================================"
EVAL_SCRIPT
)
echo "  Evaluation pipeline -> Job ${EVAL_JOB}"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================"
echo "V7 PIPELINE SUBMITTED SUCCESSFULLY"
echo "============================================"
echo ""
echo "Job Chain:"
if [ "$TRAIN_ONLY" = false ] && [ "$PLOTS_ONLY" = false ]; then
    echo "  1. Data Generation:  ${DATA_JOB}"
    echo "     ├─> 2a. Baseline:      ${JOB_A}"
    echo "     └─> 2b. AmplitudeAware: ${JOB_B}"
    echo "              └─> 3. Eval:   ${EVAL_JOB}"
elif [ "$PLOTS_ONLY" = false ]; then
    echo "  1a. Baseline:      ${JOB_A}"
    echo "  1b. AmplitudeAware: ${JOB_B}"
    echo "       └─> 2. Eval:  ${EVAL_JOB}"
else
    echo "  1. Eval: ${EVAL_JOB}"
fi
echo ""
echo "Monitor:  squeue -u \$USER"
echo ""
echo "Output locations:"
echo "  Dataset:  ${DATASET_DIR}/"
echo "  Models A: amplitude_ablation_v7/A_Baseline_SpatialASL/trained_models/"
echo "  Models B: amplitude_ablation_v7/B_AmplitudeAware/trained_models/"
echo "  Eval:     amplitude_ablation_v7/v7_evaluation_results/"
echo "  Figures:  amplitude_ablation_v7/v7_publication_figures/"
echo "============================================"

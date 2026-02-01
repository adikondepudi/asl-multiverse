#!/bin/bash
# =============================================================================
# PRODUCTION PIPELINE ORCHESTRATOR
# =============================================================================
# Master script to run the complete production training pipeline:
#   1. Generate large-scale spatial dataset (500k samples)
#   2. Train 5-model ensemble (200 epochs)
#   3. Run comprehensive validation
#   4. Generate final report
#
# Usage:
#   sbatch run_production_pipeline.sh           # Full pipeline
#   sbatch run_production_pipeline.sh --train-only  # Skip data generation
#
# Estimated time: ~56 hours total
#   - Data generation: ~8 hours
#   - Training: ~48 hours (can vary with GPU)
# =============================================================================

#SBATCH --job-name=asl-prod-orchestrator
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/prod_orchestrator_%j.out
#SBATCH --error=slurm_logs/prod_orchestrator_%j.err

set -e

# --- Parse Arguments ---
TRAIN_ONLY=false
for arg in "$@"; do
    case $arg in
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
    esac
done

echo "============================================"
echo "ASL MULTIVERSE PRODUCTION PIPELINE"
echo "============================================"
echo "Started: $(date)"
echo "Train only mode: ${TRAIN_ONLY}"
echo "============================================"

# --- Setup ---
mkdir -p slurm_logs
cd $SLURM_SUBMIT_DIR

# --- Load Environment ---
echo ""
echo "[1/4] Loading environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# --- Configuration ---
OUTPUT_DIR="production_model_v1"
DATASET_DIR="asl_spatial_dataset_500k"
CONFIG_FILE="config/production_v1.yaml"

# =============================================================================
# STEP 1: DATA GENERATION (Skip if --train-only)
# =============================================================================
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "[2/4] Submitting data generation job..."

    DATA_JOB=$(sbatch --parsable generate_production_data.sh)
    echo "  Data generation job: ${DATA_JOB}"
    echo "  Estimated time: ~8 hours"

    # Training depends on data generation
    DEPENDENCY="--dependency=afterok:${DATA_JOB}"
else
    echo ""
    echo "[2/4] Skipping data generation (--train-only mode)"

    # Verify dataset exists
    if [ ! -d "$DATASET_DIR" ]; then
        echo "ERROR: Dataset not found: ${DATASET_DIR}"
        echo "Run without --train-only to generate data first."
        exit 1
    fi
    NUM_CHUNKS=$(ls -1 ${DATASET_DIR}/spatial_chunk_*.npz 2>/dev/null | wc -l)
    echo "  Found existing dataset with ${NUM_CHUNKS} chunks"

    DEPENDENCY=""
fi

# =============================================================================
# STEP 2: TRAINING
# =============================================================================
echo ""
echo "[3/4] Submitting training job..."

TRAIN_JOB=$(sbatch --parsable ${DEPENDENCY} train_production.sh)
echo "  Training job: ${TRAIN_JOB}"
echo "  Estimated time: ~48 hours"

# =============================================================================
# STEP 3: MULTI-SNR VALIDATION (runs after training)
# =============================================================================
echo ""
echo "[4/4] Submitting extended validation job..."

# Create extended validation script
cat > slurm_logs/extended_validation.slurm << 'VALIDATION_EOF'
#!/bin/bash
#SBATCH --job-name=asl-prod-validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/prod_validate_%j.out
#SBATCH --error=slurm_logs/prod_validate_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

OUTPUT_DIR="production_model_v1"

echo "============================================"
echo "EXTENDED VALIDATION"
echo "Started: $(date)"
echo "============================================"

# Run main validation
echo ""
echo "Running spatial validation (SNR=10)..."
python validate.py \
    --run_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}/validation_results"

# Generate comparison report
echo ""
echo "Generating analysis report..."
python -c "
import json
from pathlib import Path
import numpy as np

output_dir = Path('${OUTPUT_DIR}/validation_results')
report_path = output_dir / 'llm_analysis_report.json'

if report_path.exists():
    with open(report_path) as f:
        results = json.load(f)

    print('='*60)
    print('PRODUCTION MODEL PERFORMANCE SUMMARY')
    print('='*60)

    for scenario, metrics in results.items():
        print(f'\n{scenario}:')
        print('-'*40)

        for param in ['CBF', 'ATT']:
            if param in metrics:
                nn = metrics[param]['Neural_Net']
                ls = metrics[param]['Least_Squares']
                win = metrics[param]['NN_vs_LS_Win_Rate']

                improvement = ls['MAE'] / nn['MAE'] if nn['MAE'] > 0 else 0

                print(f'  {param}:')
                print(f'    NN:  MAE={nn[\"MAE\"]:.2f}, RMSE={nn[\"RMSE\"]:.2f}, R²={nn[\"R2\"]:.3f}')
                print(f'    LS:  MAE={ls[\"MAE\"]:.2f}, RMSE={ls[\"RMSE\"]:.2f}, R²={ls[\"R2\"]:.3f}')
                print(f'    Win Rate: {win:.1%}')
                print(f'    Improvement: {improvement:.1f}x better than LS')

    print('='*60)
else:
    print('WARNING: Validation report not found')
"

echo ""
echo "Validation complete: $(date)"
VALIDATION_EOF

VALID_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} slurm_logs/extended_validation.slurm)
echo "  Validation job: ${VALID_JOB}"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================"
echo "PIPELINE SUBMITTED SUCCESSFULLY"
echo "============================================"
echo ""
echo "Job Chain:"
if [ "$TRAIN_ONLY" = false ]; then
    echo "  1. Data Generation: ${DATA_JOB}"
    echo "     └─> 2. Training: ${TRAIN_JOB}"
else
    echo "  1. Training: ${TRAIN_JOB}"
fi
echo "          └─> 3. Validation: ${VALID_JOB}"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f slurm_logs/prod_train_*.out"
echo ""
echo "Expected completion: ~56 hours from now"
echo ""
echo "Output locations:"
echo "  Models:     ${OUTPUT_DIR}/trained_models/"
echo "  Validation: ${OUTPUT_DIR}/validation_results/"
echo "  Logs:       slurm_logs/"
echo "============================================"

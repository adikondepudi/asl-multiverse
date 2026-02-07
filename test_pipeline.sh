#!/bin/bash
#SBATCH --job-name=asl-pipeline-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/pipeline_test_%j.out
#SBATCH --error=slurm_logs/pipeline_test_%j.err

# =============================================================================
# ASL Pipeline Quick Test Script
# =============================================================================
# Tests the entire training pipeline from data generation to validation.
# Expected runtime: ~30-60 minutes
#
# Usage:
#   sbatch test_pipeline.sh              # Default: spatial mode, quick test
#   sbatch test_pipeline.sh --full       # Full test (~1 hour)
#   sbatch test_pipeline.sh --voxel      # Test voxel-wise model instead
# =============================================================================

set -e  # Exit on error

# --- Parse Arguments ---
MODE="spatial"
QUICK="true"
for arg in "$@"; do
    case $arg in
        --voxel)
            MODE="voxel"
            shift
            ;;
        --full)
            QUICK="false"
            shift
            ;;
    esac
done

# --- Configuration ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_DIR="pipeline_test_${MODE}_${TIMESTAMP}"
DATA_DIR="${TEST_DIR}/test_data"
MODEL_DIR="${TEST_DIR}/trained_model"
VAL_DIR="${TEST_DIR}/validation"
CONFIG_FILE="${TEST_DIR}/test_config.yaml"

if [ "$QUICK" = "true" ]; then
    if [ "$MODE" = "spatial" ]; then
        NUM_SAMPLES=500
        CHUNK_SIZE=100
        IMAGE_SIZE=32
        N_EPOCHS=5
        N_ENSEMBLES=1
        BATCH_SIZE=64
    else
        NUM_SAMPLES=10000
        CHUNK_SIZE=2500
        N_EPOCHS=5
        N_ENSEMBLES=1
        BATCH_SIZE=256
    fi
else
    if [ "$MODE" = "spatial" ]; then
        NUM_SAMPLES=2000
        CHUNK_SIZE=200
        IMAGE_SIZE=48
        N_EPOCHS=15
        N_ENSEMBLES=2
        BATCH_SIZE=128
    else
        NUM_SAMPLES=50000
        CHUNK_SIZE=10000
        N_EPOCHS=15
        N_ENSEMBLES=2
        BATCH_SIZE=256
    fi
fi

# --- Setup ---
echo "============================================================"
echo "ASL PIPELINE TEST - ${MODE^^} MODE"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Quick mode: ${QUICK}"
echo "Test directory: ${TEST_DIR}"
echo "============================================================"

mkdir -p slurm_logs
mkdir -p "${TEST_DIR}"
mkdir -p "${DATA_DIR}"

# --- Activate Conda Environment ---
echo ""
echo "[1/5] Setting up environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# --- Create Config File ---
echo ""
echo "[2/5] Creating test configuration..."

if [ "$MODE" = "spatial" ]; then
cat > "${CONFIG_FILE}" << EOF
training:
  model_class_name: "SpatialASLNet"
  dropout_rate: 0.1
  weight_decay: 0.0001
  learning_rate: 0.0001
  log_var_cbf_min: -3.0
  log_var_cbf_max: 7.0
  log_var_att_min: -3.0
  log_var_att_max: 10.0
  batch_size: ${BATCH_SIZE}
  n_ensembles: ${N_ENSEMBLES}
  n_epochs: ${N_EPOCHS}
  validation_steps_per_epoch: 10
  early_stopping_patience: 5
  early_stopping_min_delta: 0.0
  norm_type: "batch"
  hidden_sizes: [32, 64, 128, 256]
  dc_weight: 0.0  # Data consistency loss weight (0.0 = disabled, 0.0001-0.001 for regularization)

data:
  use_offline_dataset: true
  offline_dataset_path: "${DATA_DIR}"
  num_samples_to_load: ${NUM_SAMPLES}
  pld_values: [500, 1000, 1500, 2000, 2500, 3000]
  active_features: ["mean", "std"]
  data_noise_components: ["thermal"]
  noise_type: "gaussian"
  normalization_mode: "per_curve"

simulation:
  T1_artery: 1650.0  # 3T consensus (Alsop 2015)
  T_tau: 1800.0
  T2_factor: 1.0
  alpha_BS1: 1.0
  alpha_PCASL: 0.85
  alpha_VSASL: 0.56

noise_config:
  snr_range: [3.0, 15.0]

wandb:
  wandb_project: "asl-pipeline-test"
  wandb_entity: null
EOF
else
cat > "${CONFIG_FILE}" << EOF
training:
  model_class_name: "DisentangledASLNet"
  encoder_type: "physics_processor"
  dropout_rate: 0.1
  weight_decay: 0.00001
  learning_rate: 0.001
  log_var_cbf_min: -3.0
  log_var_cbf_max: 7.0
  log_var_att_min: -3.0
  log_var_att_max: 10.0
  batch_size: ${BATCH_SIZE}
  n_ensembles: ${N_ENSEMBLES}
  n_epochs: ${N_EPOCHS}
  steps_per_epoch: 200
  validation_steps_per_epoch: 20
  early_stopping_patience: 5
  early_stopping_min_delta: 0.0
  norm_type: "batch"
  hidden_sizes: [128, 64, 32]
  transformer_d_model_focused: 32
  transformer_nhead_model: 4
  dc_weight: 0.0  # Data consistency loss weight (0.0 = disabled)

data:
  use_offline_dataset: true
  offline_dataset_path: "${DATA_DIR}"
  num_samples_to_load: ${NUM_SAMPLES}
  pld_values: [500, 1000, 1500, 2000, 2500, 3000]
  active_features: ["mean", "std", "peak", "t1_artery"]
  data_noise_components: ["thermal"]
  noise_type: "gaussian"
  normalization_mode: "per_curve"

simulation:
  T1_artery: 1650.0  # 3T consensus (Alsop 2015)
  T_tau: 1800.0
  T2_factor: 1.0
  alpha_BS1: 1.0
  alpha_PCASL: 0.85
  alpha_VSASL: 0.56

noise_config:
  snr_range: [3.0, 15.0]

wandb:
  wandb_project: "asl-pipeline-test"
  wandb_entity: null
EOF
fi

echo "Config saved to: ${CONFIG_FILE}"
echo "  Samples: ${NUM_SAMPLES}"
echo "  Epochs: ${N_EPOCHS}"
echo "  Ensembles: ${N_ENSEMBLES}"

# --- Generate Test Data ---
echo ""
echo "[3/5] Generating test data..."
START_TIME=$(date +%s)

if [ "$MODE" = "spatial" ]; then
    python generate_clean_library.py "${DATA_DIR}" \
        --spatial \
        --total_samples ${NUM_SAMPLES} \
        --spatial-chunk-size ${CHUNK_SIZE} \
        --image-size ${IMAGE_SIZE}
else
    python generate_clean_library.py "${DATA_DIR}" \
        --total_samples ${NUM_SAMPLES} \
        --chunk_size ${CHUNK_SIZE}
fi

END_TIME=$(date +%s)
echo "Data generation completed in $((END_TIME - START_TIME)) seconds"

# Count generated files
if [ "$MODE" = "spatial" ]; then
    NUM_FILES=$(ls -1 ${DATA_DIR}/spatial_chunk_*.npz 2>/dev/null | wc -l)
else
    NUM_FILES=$(ls -1 ${DATA_DIR}/dataset_chunk_*.npz 2>/dev/null | wc -l)
fi
echo "Generated ${NUM_FILES} data chunks"

# --- Train Model ---
echo ""
echo "[4/5] Training model..."
START_TIME=$(date +%s)

python main.py "${CONFIG_FILE}" \
    --stage 2 \
    --output-dir "${MODEL_DIR}" \
    --run-name "pipeline_test_${TIMESTAMP}"

END_TIME=$(date +%s)
TRAIN_TIME=$((END_TIME - START_TIME))
echo "Training completed in ${TRAIN_TIME} seconds ($((TRAIN_TIME / 60)) minutes)"

# Check if models were saved
NUM_MODELS=$(ls -1 ${MODEL_DIR}/trained_models/ensemble_model_*.pt 2>/dev/null | wc -l)
echo "Saved ${NUM_MODELS} model checkpoint(s)"

# --- Run Validation ---
echo ""
echo "[5/5] Running validation..."
START_TIME=$(date +%s)

if [ "$MODE" = "spatial" ]; then
    python validate_spatial.py \
        --run_dir "${MODEL_DIR}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${VAL_DIR}" \
        --max_samples 50
else
    python validate.py \
        --run_dir "${MODEL_DIR}" \
        --output_dir "${VAL_DIR}"
fi

END_TIME=$(date +%s)
echo "Validation completed in $((END_TIME - START_TIME)) seconds"

# --- Summary ---
echo ""
echo "============================================================"
echo "PIPELINE TEST COMPLETE"
echo "============================================================"
echo "Mode: ${MODE}"
echo "Test directory: ${TEST_DIR}"
echo ""
echo "Results:"
echo "  - Config: ${CONFIG_FILE}"
echo "  - Data: ${DATA_DIR} (${NUM_FILES} chunks)"
echo "  - Models: ${MODEL_DIR}/trained_models/ (${NUM_MODELS} models)"
echo "  - Validation: ${VAL_DIR}/"
echo ""

# Check for validation metrics
if [ "$MODE" = "spatial" ] && [ -f "${VAL_DIR}/spatial_metrics.json" ]; then
    echo "Validation Metrics:"
    python -c "
import json
with open('${VAL_DIR}/spatial_metrics.json') as f:
    m = json.load(f)
print(f\"  CBF MAE: {m['CBF']['MAE']:.2f} ml/100g/min\")
print(f\"  ATT MAE: {m['ATT']['MAE']:.0f} ms\")
"
fi

echo ""
echo "Finished: $(date)"
echo "============================================================"

# --- Cleanup prompt ---
echo ""
echo "To clean up test files, run:"
echo "  rm -rf ${TEST_DIR}"

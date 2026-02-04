#!/bin/bash
#SBATCH --job-name=asl-amp-aware
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/amp_aware_%j.out
#SBATCH --error=slurm_logs/amp_aware_%j.err

# =============================================================================
# AMPLITUDE-AWARE SPATIAL ASL NET TRAINING
# =============================================================================
# Trains the AmplitudeAwareSpatialASLNet which preserves signal amplitude
# information through a dedicated pathway, addressing the fundamental limitation
# of GroupNorm destroying CBF-encoded amplitude.
#
# Key architecture features:
#   - AmplitudeFeatureExtractor: extracts 40 amplitude features BEFORE GroupNorm
#   - FiLM conditioning: injects amplitude info into spatial features
#   - Direct amplitude-CBF modulation: CBF = spatial * amplitude_scale
#   - Physics loss enabled for self-supervision
# =============================================================================

set -e

echo "============================================"
echo "AMPLITUDE-AWARE MODEL TRAINING"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

# --- Setup ---
mkdir -p slurm_logs
cd $SLURM_SUBMIT_DIR

# --- Activate Environment ---
echo ""
echo "[1/5] Loading environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# --- Configuration ---
CONFIG_FILE="config/amplitude_aware_spatial.yaml"
OUTPUT_DIR="amplitude_aware_v1"
DATASET_DIR="asl_spatial_dataset_200k"

echo ""
echo "[2/5] Configuration:"
echo "  Config: ${CONFIG_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Dataset: ${DATASET_DIR}"
echo "  Model: AmplitudeAwareSpatialASLNet"

# --- Verify Prerequisites ---
echo ""
echo "[3/5] Verifying prerequisites..."

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_DIR}"
    echo "Run 'sbatch generate_production_data.sh' first."
    exit 1
fi

# Count dataset chunks
NUM_CHUNKS=$(ls -1 ${DATASET_DIR}/spatial_chunk_*.npz 2>/dev/null | wc -l)
echo "  Dataset chunks found: ${NUM_CHUNKS}"

if [ "$NUM_CHUNKS" -lt 100 ]; then
    echo "WARNING: Dataset seems small (${NUM_CHUNKS} chunks). Expected ~400 for 200k samples."
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Copy config to output for reproducibility
cp "${CONFIG_FILE}" "${OUTPUT_DIR}/config.yaml"

# --- Training ---
echo ""
echo "[4/5] Starting training..."
echo "  Model: AmplitudeAwareSpatialASLNet"
echo "  Key features:"
echo "    - Amplitude feature extraction (40 features)"
echo "    - FiLM conditioning at bottleneck and decoder"
echo "    - Direct amplitude-CBF modulation"
echo "    - Physics loss weight: 0.1"
echo "  Training 5-model ensemble for 200 epochs"
echo ""

python main.py "${CONFIG_FILE}" \
    --stage 2 \
    --output-dir "${OUTPUT_DIR}"

# --- Validation ---
echo ""
echo "[5/5] Running comprehensive validation..."

python validate.py \
    --run_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}/validation_results"

# --- Amplitude Sensitivity Test ---
echo ""
echo "Running amplitude sensitivity test..."

python -c "
import torch
import json
import numpy as np
from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AmplitudeAwareSpatialASLNet(
    in_channels=12,
    hidden_sizes=[32, 64, 128, 256],
    use_film_at_bottleneck=True,
    use_film_at_decoder=True,
    use_amplitude_output_modulation=True
).to(device)

# Load trained weights
model_path = '${OUTPUT_DIR}/trained_models/spatial_model_0.pt'
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Test amplitude sensitivity
torch.manual_seed(42)
base_input = torch.randn(1, 12, 64, 64).to(device) * 0.1

scales = [0.1, 1.0, 10.0]
results = {}

with torch.no_grad():
    for scale in scales:
        scaled_input = base_input * scale
        output = model(scaled_input)
        cbf = output[0, 0].mean().item()
        results[f'scale_{scale}'] = cbf

print('Amplitude Sensitivity Test:')
for scale in scales:
    print(f'  Input scale {scale}x: CBF = {results[f\"scale_{scale}\"]:.2f}')

# Check if sensitive (CBF should change with scale)
cbf_ratio = results['scale_10.0'] / max(results['scale_0.1'], 1e-6)
is_sensitive = cbf_ratio > 2.0 and cbf_ratio < 200.0
print(f'  CBF ratio (10x/0.1x): {cbf_ratio:.2f}')
print(f'  Amplitude sensitive: {is_sensitive}')

# Save results
with open('${OUTPUT_DIR}/amplitude_sensitivity.json', 'w') as f:
    json.dump({
        'scales': scales,
        'cbf_predictions': results,
        'cbf_ratio_10x_to_01x': cbf_ratio,
        'is_amplitude_sensitive': is_sensitive
    }, f, indent=2)
"

# --- Summary ---
echo ""
echo "============================================"
echo "TRAINING COMPLETE"
echo "============================================"
echo "Model saved to: ${OUTPUT_DIR}/trained_models/"
echo "Validation results: ${OUTPUT_DIR}/validation_results/"
echo "Amplitude sensitivity: ${OUTPUT_DIR}/amplitude_sensitivity.json"
echo "Finished: $(date)"
echo ""

# Print key metrics if available
if [ -f "${OUTPUT_DIR}/validation_results/llm_analysis_report.json" ]; then
    echo "--- KEY METRICS ---"
    python -c "
import json
with open('${OUTPUT_DIR}/validation_results/llm_analysis_report.json') as f:
    d = json.load(f)
    for scenario, metrics in d.items():
        print(f'\n{scenario}:')
        for param in ['CBF', 'ATT']:
            if param in metrics:
                nn_mae = metrics[param]['Neural_Net']['MAE']
                ls_mae = metrics[param]['Least_Squares']['MAE']
                win = metrics[param]['NN_vs_LS_Win_Rate']
                print(f'  {param}: NN MAE={nn_mae:.2f}, LS MAE={ls_mae:.2f}, Win Rate={win:.1%}')
"
fi

echo ""
echo "============================================"
echo "NEXT STEPS:"
echo "1. Check validation metrics - target CBF MAE < 5, Win Rate > 80%"
echo "2. Verify amplitude sensitivity - CBF should scale with input"
echo "3. Run in vivo validation: python predict_spatial_invivo.py --run_dir ${OUTPUT_DIR}"
echo "============================================"

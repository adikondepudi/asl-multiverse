#!/bin/bash
# Quick test script to verify MAE loss configuration works
# Run this to test the new loss formulation before full training

set -e

echo "=============================================="
echo "Testing MAE Loss Configuration"
echo "=============================================="

# Create a minimal test config
cat > /tmp/test_mae_config.yaml << 'EOF'
training:
  loss_mode: "mae_only"
  mae_weight: 1.0
  nll_weight: 0.0
  loss_weight_cbf: 1.0
  loss_weight_att: 1.0
  learning_rate: 0.001
  hidden_sizes: [64, 32]
  n_ensembles: 1
  training_epochs: 2
  batch_size: 256
  early_stopping_patience: 5
  encoder_type: "physics_processor"

data:
  dataset_path: "asl_clean_library_10M"
  pld_values: [500, 1000, 1500, 2000, 2500, 3000]
  active_features: ["mean", "std", "peak", "t1_artery"]
  data_noise_components: ["thermal"]
  noise_type: "gaussian"
  normalization_mode: "per_curve"

simulation:
  T1_artery: 1850.0
  T_tau: 1800.0
  alpha_PCASL: 0.85
  alpha_VSASL: 0.56

noise_config:
  snr_range: [5, 30]
EOF

echo "Config created at /tmp/test_mae_config.yaml"
echo ""
echo "Running quick training test (2 epochs)..."
echo ""

cd "$(dirname "$0")/.."

python main.py /tmp/test_mae_config.yaml \
    --stage 2 \
    --output-dir /tmp/test_mae_loss_output \
    --no-wandb

echo ""
echo "=============================================="
echo "Test Complete!"
echo "Check /tmp/test_mae_loss_output for results"
echo "=============================================="

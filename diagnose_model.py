#!/usr/bin/env python3
"""
Quick diagnostic script to check model predictions without full validation.

Usage:
    python diagnose_model.py <run_dir>

This script will:
1. Load a trained model
2. Generate a few test signals with known ground truth
3. Print raw model outputs before and after denormalization
4. Diagnose common issues (mean collapse, scale problems, etc.)
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_model.py <run_dir>")
        print("Example: python diagnose_model.py results/spatial_ablation_jobs/01_Baseline")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    print(f"\n{'='*60}")
    print(f"DIAGNOSING MODEL: {run_dir.name}")
    print(f"{'='*60}\n")

    # Check required files
    config_path = run_dir / 'research_config.json'
    norm_stats_path = run_dir / 'norm_stats.json'
    models_dir = run_dir / 'trained_models'

    if not config_path.exists():
        print(f"ERROR: research_config.json not found in {run_dir}")
        sys.exit(1)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print("--- Configuration ---")
    print(f"  Encoder Type: {config.get('encoder_type', 'unknown')}")
    print(f"  Model Class: {config.get('model_class_name', 'DisentangledASLNet')}")
    print(f"  Active Features: {config.get('active_features', [])}")
    print(f"  Loss Mode: {config.get('loss_mode', 'not specified (legacy NLL)')}")
    print(f"  MAE Weight: {config.get('mae_weight', 'not specified')}")
    print(f"  NLL Weight: {config.get('nll_weight', 'not specified')}")
    print()

    # Load norm stats
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        print("--- Normalization Stats ---")
        print(f"  CBF: mean={norm_stats.get('y_mean_cbf', 'N/A'):.2f}, std={norm_stats.get('y_std_cbf', 'N/A'):.2f}")
        print(f"  ATT: mean={norm_stats.get('y_mean_att', 'N/A'):.2f}, std={norm_stats.get('y_std_att', 'N/A'):.2f}")
        print()
    else:
        print("WARNING: norm_stats.json not found")
        norm_stats = None

    # Check for trained models
    if not models_dir.exists():
        print(f"ERROR: trained_models/ directory not found")
        sys.exit(1)

    model_files = sorted(models_dir.glob('ensemble_model_*.pt'))
    print(f"--- Found {len(model_files)} trained models ---")

    if not model_files:
        print("ERROR: No model files found")
        sys.exit(1)

    # Detect model type from checkpoint
    first_state = torch.load(model_files[0], map_location='cpu')
    sd = first_state['model_state_dict'] if 'model_state_dict' in first_state else first_state
    keys_str = ' '.join(sd.keys())

    is_spatial = 'encoder1.double_conv' in keys_str or 'encoder1.' in keys_str

    print(f"  Model Type: {'Spatial (U-Net)' if is_spatial else 'Voxel (DisentangledASLNet)'}")
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Load model
    if is_spatial:
        from spatial_asl_network import SpatialASLNet
        plds = config.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000])
        model = SpatialASLNet(n_plds=len(plds))
    else:
        from enhanced_asl_network import DisentangledASLNet
        # Build model config from saved config
        plds = config.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000])
        hidden_sizes = config.get('hidden_sizes', [128, 64, 32])
        encoder_type = config.get('encoder_type', 'physics_processor')

        # Detect scalar features from checkpoint
        if encoder_type.lower() == 'mlp_only':
            num_scalar_features = sd['encoder.encoder_mlp.0.weight'].shape[1] - (len(plds) * 2)
        else:
            num_scalar_features = sd['encoder.pcasl_film.generator.0.weight'].shape[1]

        input_size = len(plds) * 2 + num_scalar_features

        model = DisentangledASLNet(
            mode='regression',
            input_size=input_size,
            n_plds=len(plds),
            num_scalar_features=num_scalar_features,
            hidden_sizes=hidden_sizes,
            encoder_type=encoder_type
        )

    # Load weights
    state_dict = torch.load(model_files[0], map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")

    # Generate test data
    print("--- Generating Test Data ---")

    if is_spatial:
        # Generate synthetic 2D phantom
        size = 64
        test_cbf = np.full((size, size), 60.0, dtype=np.float32)
        test_att = np.full((size, size), 1500.0, dtype=np.float32)

        from enhanced_simulation import RealisticASLSimulator
        from asl_simulation import ASLParameters

        params = ASLParameters()
        simulator = RealisticASLSimulator(params=params)
        plds_arr = np.array(plds)

        signals = np.zeros((len(plds) * 2, size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                pcasl = simulator._generate_pcasl_signal(plds_arr, test_att[i,j], test_cbf[i,j], 1850, 1800, 0.85)
                vsasl = simulator._generate_vsasl_signal(plds_arr, test_att[i,j], test_cbf[i,j], 1850, 0.56)
                signals[:len(plds), i, j] = pcasl
                signals[len(plds):, i, j] = vsasl

        # Add noise and scale
        signals_noisy = signals + 0.001 * np.random.randn(*signals.shape).astype(np.float32)
        signals_scaled = signals_noisy * 100.0

        input_tensor = torch.from_numpy(signals_scaled[np.newaxis, ...]).to(device)

        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Input stats: mean={input_tensor.mean().item():.4f}, std={input_tensor.std().item():.4f}")
        print(f"  Ground Truth: CBF=60.0, ATT=1500.0")
        print()

        # Run inference
        with torch.no_grad():
            cbf_pred, att_pred, log_var_cbf, log_var_att = model(input_tensor)

        print("--- Raw Model Outputs ---")
        print(f"  CBF prediction: mean={cbf_pred.mean().item():.2f}, std={cbf_pred.std().item():.4f}")
        print(f"  ATT prediction: mean={att_pred.mean().item():.2f}, std={att_pred.std().item():.4f}")
        print(f"  CBF log_var: mean={log_var_cbf.mean().item():.4f}")
        print(f"  ATT log_var: mean={log_var_att.mean().item():.4f}")
        print()

        # Check for issues
        print("--- Diagnostic Analysis ---")

        cbf_mean = cbf_pred.mean().item()
        att_mean = att_pred.mean().item()

        if cbf_pred.std().item() < 1.0:
            print("  [WARNING] CBF predictions have very low variance - possible mean collapse!")

        if att_pred.std().item() < 10.0:
            print("  [WARNING] ATT predictions have very low variance - possible mean collapse!")

        cbf_error = abs(cbf_mean - 60.0)
        att_error = abs(att_mean - 1500.0)

        print(f"  CBF Error: {cbf_error:.2f} ml/100g/min (expected: 60.0)")
        print(f"  ATT Error: {att_error:.2f} ms (expected: 1500.0)")

        if cbf_error > 20:
            print("  [PROBLEM] Large CBF error - model not predicting correctly")
        if att_error > 500:
            print("  [PROBLEM] Large ATT error - model not predicting correctly")

    else:
        # Voxel-wise test
        print("  Voxel-wise model diagnostic not yet implemented")
        print("  Use validate.py for full voxel-wise validation")

    print()
    print("="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

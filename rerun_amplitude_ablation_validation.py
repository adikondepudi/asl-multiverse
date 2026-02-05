#!/usr/bin/env python3
"""
Rerun validation and amplitude sensitivity tests for amplitude_ablation_v1 experiments.

Fixes two bugs:
1. validate.py now supports AmplitudeAwareSpatialASLNet (previously only SpatialASLNet)
2. Amplitude sensitivity test now uses correct filename (ensemble_model_*.pt, not spatial_model_*.pt)
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import yaml
import torch
import glob as glob_module

# Change to project root
os.chdir(Path(__file__).parent)

ABLATION_DIR = Path("amplitude_ablation_v1")
EXPERIMENTS = [
    "02_AmpAware_Full",
    "03_AmpAware_OutputMod_Only",
    "04_AmpAware_FiLM_Only",
    "05_AmpAware_Bottleneck_Only",
    "06_AmpAware_Physics_0p1",
    "07_AmpAware_Physics_0p3",
    "08_AmpAware_DomainRand",
    "09_AmpAware_Optimized",
]


def run_amplitude_sensitivity_test(exp_dir: Path):
    """Run amplitude sensitivity test with correct filename pattern."""
    print(f"\n{'='*60}")
    print(f"AMPLITUDE SENSITIVITY TEST: {exp_dir.name}")
    print(f"{'='*60}")

    # Load config
    config_path = exp_dir / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    training_config = cfg.get('training', {})
    model_class = training_config.get('model_class_name', 'SpatialASLNet')
    print(f"Model class: {model_class}")

    # Import the correct model class
    if model_class == 'AmplitudeAwareSpatialASLNet':
        from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet

        # Load checkpoint to check if it has FiLM keys
        # Due to a training bug, all models were trained with full architecture
        model_files = sorted(glob_module.glob(str(exp_dir / 'trained_models' / 'ensemble_model_*.pt')))
        if model_files:
            temp_state = torch.load(model_files[0], map_location='cpu', weights_only=False)
            if 'model_state_dict' in temp_state:
                temp_sd = temp_state['model_state_dict']
            else:
                temp_sd = temp_state
            has_film_keys = any('film' in k for k in temp_sd.keys())
        else:
            has_film_keys = True  # Default to full architecture

        if has_film_keys:
            # Use full architecture (how models were actually trained)
            model = AmplitudeAwareSpatialASLNet(
                n_plds=6,
                features=training_config.get('hidden_sizes', [32, 64, 128, 256]),
                use_film_at_bottleneck=True,
                use_film_at_decoder=True,
                use_amplitude_output_modulation=True,
            )
        else:
            model = AmplitudeAwareSpatialASLNet(
                n_plds=6,
                features=training_config.get('hidden_sizes', [32, 64, 128, 256]),
                use_film_at_bottleneck=training_config.get('use_film_at_bottleneck', True),
                use_film_at_decoder=training_config.get('use_film_at_decoder', True),
                use_amplitude_output_modulation=training_config.get('use_amplitude_output_modulation', True),
            )
    else:
        from spatial_asl_network import SpatialASLNet
        model = SpatialASLNet(n_plds=6)

    # Load trained weights - FIX: use correct filename pattern
    model_files = sorted(glob_module.glob(str(exp_dir / 'trained_models' / 'ensemble_model_*.pt')))

    if model_files:
        state_dict = torch.load(model_files[0], map_location='cpu')
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded: {model_files[0]}")
    else:
        print("ERROR: No trained model found!")
        return None

    model.eval()

    # Test amplitude sensitivity
    torch.manual_seed(42)
    base_input = torch.randn(4, 12, 64, 64) * 0.1

    scales = [0.1, 1.0, 10.0]
    results = {}

    with torch.no_grad():
        for scale in scales:
            scaled_input = base_input * scale
            output = model(scaled_input)
            cbf = output[0] if isinstance(output, tuple) else output[:, 0:1]
            cbf_mean = cbf.mean().item()
            results[f'scale_{scale}'] = cbf_mean
            print(f"  Scale {scale:5.1f}x -> CBF mean: {cbf_mean:10.4f}")

    # Compute sensitivity ratio
    cbf_01 = abs(results['scale_0.1'])
    cbf_10 = abs(results['scale_10.0'])
    ratio = cbf_10 / max(cbf_01, 1e-9)

    print(f"\n  Amplitude sensitivity ratio (10x/0.1x): {ratio:.2f}")
    print(f"  Is amplitude sensitive: {ratio > 5.0}")

    # Save results
    output_file = exp_dir / 'amplitude_sensitivity.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model_class': model_class,
            'scales': scales,
            'cbf_predictions': results,
            'sensitivity_ratio': ratio,
            'is_sensitive': ratio > 5.0,
            'used_trained_model': True  # Flag to indicate this used actual trained weights
        }, f, indent=2)
    print(f"Saved to {output_file}")

    return ratio


def run_validation(exp_dir: Path):
    """Run validation for an experiment."""
    print(f"\n{'='*60}")
    print(f"VALIDATION: {exp_dir.name}")
    print(f"{'='*60}")

    output_dir = exp_dir / "validation_results"

    cmd = [
        sys.executable, "validate.py",
        "--run_dir", str(exp_dir),
        "--output_dir", str(output_dir)
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"SUCCESS: Validation completed for {exp_dir.name}")
        return True
    else:
        print(f"FAILED: Validation failed for {exp_dir.name}")
        return False


def main():
    print("=" * 70)
    print("RERUNNING AMPLITUDE ABLATION VALIDATION")
    print("=" * 70)
    print(f"\nExperiments to process: {len(EXPERIMENTS)}")
    print("Experiments:", ", ".join(EXPERIMENTS))

    # Track results
    amp_results = {}
    val_results = {}

    for exp_name in EXPERIMENTS:
        exp_dir = ABLATION_DIR / exp_name

        if not exp_dir.exists():
            print(f"\nWARNING: {exp_dir} does not exist, skipping")
            continue

        # Run amplitude sensitivity test first (quick)
        ratio = run_amplitude_sensitivity_test(exp_dir)
        if ratio is not None:
            amp_results[exp_name] = ratio

        # Run full validation
        success = run_validation(exp_dir)
        val_results[exp_name] = success

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nAmplitude Sensitivity Results:")
    print("-" * 50)
    for exp, ratio in amp_results.items():
        sensitive = "YES" if ratio > 5.0 else "NO"
        print(f"  {exp}: ratio={ratio:.2f}, sensitive={sensitive}")

    print("\nValidation Results:")
    print("-" * 50)
    for exp, success in val_results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {exp}: {status}")

    # Count successes
    n_success = sum(val_results.values())
    n_total = len(val_results)
    print(f"\nTotal: {n_success}/{n_total} validations succeeded")


if __name__ == "__main__":
    main()

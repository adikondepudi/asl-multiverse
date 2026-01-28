#!/usr/bin/env python3
"""
Quick Pipeline Test Script
==========================
Tests the entire ASL training pipeline from start to finish in ~1 hour.

Usage:
    python test_pipeline.py [--mode spatial|voxel] [--quick]

This script will:
1. Generate a small test dataset (~1000 samples for spatial, ~10k for voxel)
2. Train a small model for 10-20 epochs
3. Run validation
4. Report success/failure

Use --quick for a faster ~15 minute smoke test.
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def log(msg, color=None):
    """Print colored log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if color:
        print(f"{color}[{timestamp}] {msg}{RESET}")
    else:
        print(f"[{timestamp}] {msg}")


def run_command(cmd, description, timeout=3600):
    """Run a command and return success status."""
    log(f"Running: {description}", BLUE)
    log(f"  Command: {' '.join(cmd)}", YELLOW)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            log(f"  SUCCESS ({elapsed:.1f}s)", GREEN)
            return True, result.stdout, elapsed
        else:
            log(f"  FAILED (exit code {result.returncode})", RED)
            log(f"  STDERR: {result.stderr[:500]}", RED)
            return False, result.stderr, elapsed
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s", RED)
        return False, "Timeout", timeout
    except Exception as e:
        log(f"  ERROR: {e}", RED)
        return False, str(e), 0


def create_test_config(output_dir: Path, mode: str, quick: bool) -> Path:
    """Create a minimal config for testing."""

    if quick:
        # Smoke test: ~15 minutes
        n_epochs = 5
        n_ensembles = 1
        batch_size = 64
        num_samples = 500 if mode == 'spatial' else 5000
        patience = 3
    else:
        # Full test: ~1 hour
        n_epochs = 15
        n_ensembles = 2
        batch_size = 128 if mode == 'spatial' else 256
        num_samples = 2000 if mode == 'spatial' else 50000
        patience = 5

    if mode == 'spatial':
        config = {
            'training': {
                'model_class_name': 'SpatialASLNet',
                'dropout_rate': 0.1,
                'weight_decay': 0.0001,
                'learning_rate': 0.0001,
                'log_var_cbf_min': -3.0,
                'log_var_cbf_max': 7.0,
                'log_var_att_min': -3.0,
                'log_var_att_max': 10.0,  # Reduced from 14 for stability
                'batch_size': batch_size,
                'n_ensembles': n_ensembles,
                'n_epochs': n_epochs,
                'validation_steps_per_epoch': 10,
                'early_stopping_patience': patience,
                'early_stopping_min_delta': 0.0,
                'norm_type': 'batch',
                'hidden_sizes': [32, 64, 128, 256],  # Smaller U-Net
            },
            'data': {
                'use_offline_dataset': True,
                'offline_dataset_path': str(output_dir / 'test_data'),
                'num_samples_to_load': num_samples,
                'pld_values': [500, 1000, 1500, 2000, 2500, 3000],
                'active_features': ['mean', 'std'],
                'data_noise_components': ['thermal'],
                'noise_type': 'gaussian',
                'normalization_mode': 'per_curve',
            },
            'simulation': {
                'T1_artery': 1850.0,
                'T_tau': 1800.0,
                'T2_factor': 1.0,
                'alpha_BS1': 1.0,
                'alpha_PCASL': 0.85,
                'alpha_VSASL': 0.56,
            },
            'noise_config': {
                'snr_range': [3.0, 15.0],
            },
            'wandb': {
                'wandb_project': 'asl-pipeline-test',
                'wandb_entity': None,
            }
        }
    else:  # voxel mode
        config = {
            'training': {
                'model_class_name': 'DisentangledASLNet',
                'encoder_type': 'physics_processor',
                'dropout_rate': 0.1,
                'weight_decay': 0.00001,
                'learning_rate': 0.001,
                'log_var_cbf_min': -3.0,
                'log_var_cbf_max': 7.0,
                'log_var_att_min': -3.0,
                'log_var_att_max': 10.0,
                'batch_size': batch_size,
                'n_ensembles': n_ensembles,
                'n_epochs': n_epochs,
                'steps_per_epoch': 200,
                'validation_steps_per_epoch': 20,
                'early_stopping_patience': patience,
                'early_stopping_min_delta': 0.0,
                'norm_type': 'batch',
                'hidden_sizes': [128, 64, 32],
                'transformer_d_model_focused': 32,
                'transformer_nhead_model': 4,
            },
            'data': {
                'use_offline_dataset': True,
                'offline_dataset_path': str(output_dir / 'test_data'),
                'num_samples_to_load': num_samples,
                'pld_values': [500, 1000, 1500, 2000, 2500, 3000],
                'active_features': ['mean', 'std', 'peak', 't1_artery'],
                'data_noise_components': ['thermal'],
                'noise_type': 'gaussian',
                'normalization_mode': 'per_curve',
            },
            'simulation': {
                'T1_artery': 1850.0,
                'T_tau': 1800.0,
                'T2_factor': 1.0,
                'alpha_BS1': 1.0,
                'alpha_PCASL': 0.85,
                'alpha_VSASL': 0.56,
            },
            'noise_config': {
                'snr_range': [3.0, 15.0],
            },
            'wandb': {
                'wandb_project': 'asl-pipeline-test',
                'wandb_entity': None,
            }
        }

    # Write config as YAML
    config_path = output_dir / 'test_config.yaml'

    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path, num_samples


def main():
    parser = argparse.ArgumentParser(description="Quick pipeline test for ASL training")
    parser.add_argument('--mode', choices=['spatial', 'voxel'], default='spatial',
                        help='Training mode: spatial (U-Net) or voxel (1D) (default: spatial)')
    parser.add_argument('--quick', action='store_true',
                        help='Run a faster ~15 minute smoke test instead of full ~1 hour test')
    parser.add_argument('--keep-data', action='store_true',
                        help='Keep generated test data after completion')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: test_run_YYYYMMDD_HHMMSS)')
    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f'test_run_{args.mode}_{timestamp}')

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / 'test_data'
    data_dir.mkdir(exist_ok=True)

    log("=" * 60)
    log(f"ASL Pipeline Test - {args.mode.upper()} Mode", BOLD)
    log(f"Quick mode: {args.quick}")
    log(f"Output: {output_dir}")
    log("=" * 60)

    results = {}
    total_start = time.time()

    try:
        # Step 1: Create config
        log("\n[1/4] Creating test configuration...", BOLD)
        config_path, num_samples = create_test_config(output_dir, args.mode, args.quick)
        log(f"  Config saved to: {config_path}", GREEN)
        log(f"  Training samples: {num_samples}", GREEN)
        results['config'] = True

        # Step 2: Generate test data
        log(f"\n[2/4] Generating {num_samples} test samples...", BOLD)

        if args.mode == 'spatial':
            # Spatial mode: smaller images for faster testing
            chunk_size = min(100, num_samples // 4)
            gen_cmd = [
                sys.executable, 'generate_clean_library.py',
                str(data_dir),
                '--spatial',
                '--image-size', '32',  # Small images for speed
                '--total_samples', str(num_samples),
                '--spatial-chunk-size', str(chunk_size)
            ]
        else:
            chunk_size = min(5000, num_samples // 4)
            gen_cmd = [
                sys.executable, 'generate_clean_library.py',
                str(data_dir),
                '--total_samples', str(num_samples),
                '--chunk_size', str(chunk_size)
            ]

        success, output, elapsed = run_command(gen_cmd, "Data generation", timeout=600)
        results['data_generation'] = success
        results['data_generation_time'] = elapsed

        if not success:
            log("Data generation failed! Aborting.", RED)
            return 1

        # Verify data was created
        if args.mode == 'spatial':
            data_files = list(data_dir.glob('spatial_chunk_*.npz'))
        else:
            data_files = list(data_dir.glob('dataset_chunk_*.npz'))

        log(f"  Generated {len(data_files)} data chunks", GREEN)

        # Step 3: Train model
        log(f"\n[3/4] Training model...", BOLD)

        train_cmd = [
            sys.executable, 'main.py',
            str(config_path),
            '--stage', '2',
            '--output-dir', str(output_dir / 'trained_models'),
            '--run-name', f'test_{args.mode}_{timestamp}'
        ]

        # Allow more time for training
        train_timeout = 1800 if args.quick else 3600
        success, output, elapsed = run_command(train_cmd, "Model training", timeout=train_timeout)
        results['training'] = success
        results['training_time'] = elapsed

        if not success:
            log("Training failed!", RED)
            # Don't abort - try to report what we have

        # Check if models were saved
        model_dir = output_dir / 'trained_models' / 'trained_models'
        if model_dir.exists():
            model_files = list(model_dir.glob('ensemble_model_*.pt'))
            log(f"  Saved {len(model_files)} model checkpoint(s)", GREEN)
            results['models_saved'] = len(model_files)
        else:
            log(f"  Warning: No model directory found", YELLOW)
            results['models_saved'] = 0

        # Step 4: Validate (only for voxel mode currently - spatial needs different validation)
        log(f"\n[4/4] Running validation...", BOLD)

        if args.mode == 'voxel' and results.get('models_saved', 0) > 0:
            val_cmd = [
                sys.executable, 'validate.py',
                '--run_dir', str(output_dir / 'trained_models'),
                '--output_dir', str(output_dir / 'validation_results')
            ]
            success, output, elapsed = run_command(val_cmd, "Validation", timeout=600)
            results['validation'] = success
            results['validation_time'] = elapsed
        elif args.mode == 'spatial':
            # For spatial mode, check if we have a spatial validation script
            spatial_val_script = Path(__file__).parent / 'validate_spatial.py'
            if spatial_val_script.exists() and results.get('models_saved', 0) > 0:
                val_cmd = [
                    sys.executable, 'validate_spatial.py',
                    '--run_dir', str(output_dir / 'trained_models'),
                    '--data_dir', str(data_dir),
                    '--output_dir', str(output_dir / 'validation_results')
                ]
                success, output, elapsed = run_command(val_cmd, "Spatial Validation", timeout=600)
                results['validation'] = success
            else:
                log("  Spatial validation skipped (script not found or no models)", YELLOW)
                results['validation'] = None
        else:
            log("  Validation skipped (no models to validate)", YELLOW)
            results['validation'] = None

    except KeyboardInterrupt:
        log("\n\nTest interrupted by user!", YELLOW)
        results['interrupted'] = True
    except Exception as e:
        log(f"\n\nUnexpected error: {e}", RED)
        import traceback
        traceback.print_exc()
        results['error'] = str(e)

    # Summary
    total_elapsed = time.time() - total_start
    log("\n" + "=" * 60)
    log("TEST SUMMARY", BOLD)
    log("=" * 60)

    passed = 0
    failed = 0
    for step, status in results.items():
        if '_time' in step:
            continue
        if status is True:
            log(f"  {step}: PASSED", GREEN)
            passed += 1
        elif status is False:
            log(f"  {step}: FAILED", RED)
            failed += 1
        elif status is None:
            log(f"  {step}: SKIPPED", YELLOW)
        elif isinstance(status, int):
            log(f"  {step}: {status}", BLUE)

    log(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    log(f"Results: {passed} passed, {failed} failed")

    # Save results
    results['total_time'] = total_elapsed
    results['mode'] = args.mode
    results['quick'] = args.quick

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Cleanup
    if not args.keep_data and failed == 0:
        log(f"\nCleaning up test data...", YELLOW)
        shutil.rmtree(data_dir, ignore_errors=True)

    if failed == 0:
        log("\n" + "=" * 60)
        log("ALL TESTS PASSED!", GREEN + BOLD)
        log("=" * 60)
        return 0
    else:
        log("\n" + "=" * 60)
        log(f"TESTS FAILED: {failed} failures", RED + BOLD)
        log("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())

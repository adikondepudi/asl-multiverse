#!/usr/bin/env python3
"""
Setup script for experiments with fixed loss configuration.

This creates a new set of experiments to test the MAE-based loss formulation
that forces the model to predict accurately instead of hedging with uncertainty.

Experiments:
1. Voxel-wise: MAE-only loss (baseline)
2. Voxel-wise: MAE + NLL balanced
3. Voxel-wise: MSE-only loss (comparison)
4. Spatial: L1 loss with ATT scaling
5. Spatial: L1 loss + DC loss

Usage:
    python setup_fixed_loss_experiments.py
    # Then: bash submit_all_fixed.sh (if on HPC) or run locally
"""

import os
import yaml
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("results/fixed_loss_experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base configurations
VOXEL_BASE = {
    'training': {
        'learning_rate': 0.001,
        'hidden_sizes': [128, 64, 32],
        'dropout_rate': 0.1,
        'n_ensembles': 3,
        'training_epochs': 50,
        'batch_size': 512,
        'early_stopping_patience': 15,
        'encoder_type': 'physics_processor',
        'transformer_d_model_focused': 32,
        'transformer_nhead_model': 4,
        'loss_weight_cbf': 1.0,
        'loss_weight_att': 1.0,
    },
    'data': {
        'dataset_path': 'asl_clean_library_10M',
        'pld_values': [500, 1000, 1500, 2000, 2500, 3000],
        'active_features': ['mean', 'std', 'peak', 't1_artery'],
        'data_noise_components': ['thermal'],
        'noise_type': 'gaussian',
        'normalization_mode': 'per_curve',
    },
    'simulation': {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
    },
    'noise_config': {
        'snr_range': [5, 30],
    }
}

SPATIAL_BASE = {
    'training': {
        'model_class_name': 'SpatialASLNet',
        'learning_rate': 0.001,
        'n_ensembles': 3,
        'training_epochs': 50,
        'batch_size': 16,
        'early_stopping_patience': 15,
        'features': [32, 64, 128, 256],
    },
    'data': {
        'dataset_path': 'asl_spatial_dataset_100k',
        'pld_values': [500, 1000, 1500, 2000, 2500, 3000],
        'noise_type': 'gaussian',
        'normalization_mode': 'per_curve',
        'data_noise_components': ['thermal'],
    },
    'simulation': {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
    },
    'noise_config': {
        'snr_range': [5, 30],
    }
}

# Experiment definitions
EXPERIMENTS = [
    # Voxel-wise experiments
    {
        'name': '01_Voxel_MAE_Only',
        'type': 'voxel',
        'changes': {
            'training': {
                'loss_mode': 'mae_only',
                'mae_weight': 1.0,
                'nll_weight': 0.0,
            }
        },
        'description': 'Voxel-wise with pure MAE loss (no uncertainty)'
    },
    {
        'name': '02_Voxel_MAE_NLL',
        'type': 'voxel',
        'changes': {
            'training': {
                'loss_mode': 'mae_nll',
                'mae_weight': 1.0,
                'nll_weight': 0.1,
            }
        },
        'description': 'Voxel-wise with MAE primary + NLL secondary'
    },
    {
        'name': '03_Voxel_MSE_Only',
        'type': 'voxel',
        'changes': {
            'training': {
                'loss_mode': 'mse_only',
                'mae_weight': 1.0,  # Actually used as MSE weight in this mode
                'nll_weight': 0.0,
            }
        },
        'description': 'Voxel-wise with pure MSE loss'
    },
    {
        'name': '04_Voxel_NLL_Only_Baseline',
        'type': 'voxel',
        'changes': {
            'training': {
                'loss_mode': 'nll_only',
                'mae_weight': 0.0,
                'nll_weight': 1.0,
            }
        },
        'description': 'Voxel-wise with pure NLL loss (legacy baseline for comparison)'
    },
    # Spatial experiments
    {
        'name': '05_Spatial_L1_Balanced',
        'type': 'spatial',
        'changes': {
            'training': {
                'loss_type': 'l1',
                'att_scale': 0.033,
                'cbf_weight': 1.0,
                'att_weight': 1.0,
                'dc_weight': 0.0,
            }
        },
        'description': 'Spatial with L1 loss and ATT scaling'
    },
    {
        'name': '06_Spatial_L1_DC',
        'type': 'spatial',
        'changes': {
            'training': {
                'loss_type': 'l1',
                'att_scale': 0.033,
                'cbf_weight': 1.0,
                'att_weight': 1.0,
                'dc_weight': 0.001,
            }
        },
        'description': 'Spatial with L1 loss + data consistency'
    },
    {
        'name': '07_Spatial_L2',
        'type': 'spatial',
        'changes': {
            'training': {
                'loss_type': 'l2',
                'att_scale': 0.001,  # Smaller scale for L2 due to squared term
                'cbf_weight': 1.0,
                'att_weight': 1.0,
                'dc_weight': 0.0,
            }
        },
        'description': 'Spatial with L2 loss'
    },
    {
        'name': '08_Spatial_Huber',
        'type': 'spatial',
        'changes': {
            'training': {
                'loss_type': 'huber',
                'att_scale': 0.033,
                'cbf_weight': 1.0,
                'att_weight': 1.0,
                'dc_weight': 0.0,
            }
        },
        'description': 'Spatial with Huber loss (robust to outliers)'
    },
]


def deep_merge(base, changes):
    """Recursively merge changes into base config."""
    result = base.copy()
    for key, value in changes.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main():
    print("=" * 60)
    print("Setting up Fixed Loss Experiments")
    print("=" * 60)
    print()

    all_dirs = []

    for exp in EXPERIMENTS:
        exp_dir = OUTPUT_DIR / exp['name']
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Select base config
        base = VOXEL_BASE.copy() if exp['type'] == 'voxel' else SPATIAL_BASE.copy()

        # Deep merge changes
        config = {}
        for section in base:
            config[section] = base[section].copy()

        if 'changes' in exp:
            config = deep_merge(config, exp['changes'])

        # Write config
        config_path = exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Write description
        desc_path = exp_dir / 'description.txt'
        with open(desc_path, 'w') as f:
            f.write(f"Experiment: {exp['name']}\n")
            f.write(f"Type: {exp['type']}\n")
            f.write(f"Description: {exp['description']}\n")

        print(f"Created: {exp['name']}")
        print(f"  Type: {exp['type']}")
        print(f"  {exp['description']}")
        print()

        all_dirs.append(str(exp_dir))

    # Write a simple run script
    run_script = OUTPUT_DIR / 'run_all.sh'
    with open(run_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Run all fixed loss experiments\n")
        f.write("set -e\n\n")

        for exp in EXPERIMENTS:
            exp_dir = OUTPUT_DIR / exp['name']
            stage = 2  # All regression experiments

            f.write(f"echo '=== Running {exp['name']} ==='\n")
            f.write(f"python main.py {exp_dir}/config.yaml --stage {stage} "
                   f"--output-dir {exp_dir} --no-wandb\n")
            f.write(f"python validate.py --run_dir {exp_dir} --output_dir {exp_dir}/validation_results\n")
            f.write("\n")

    os.chmod(run_script, 0o755)

    print("=" * 60)
    print(f"Created {len(EXPERIMENTS)} experiments in {OUTPUT_DIR}")
    print(f"Run script: {run_script}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# FILE: setup_optimization_grid.py
"""
Optimization Ablation Study for Final Model Selection (COMPACT VERSION)
========================================================================

Baseline: 10_Rician_Complex_Full (best from spatial_ablation_v2)
- noise_type: rician
- normalization_mode: global_scale
- dc_weight: 0.0001
- noise_components: [thermal, physio, drift]
- hidden_sizes: [32, 64, 128, 256]
- n_epochs: 100 (increased from 50)

10 experiments testing key improvements discovered from deep analysis.
"""
import os
import yaml
import copy
from pathlib import Path

# =========================================================
# BASELINE CONFIG (10_Rician_Complex_Full + 100 epochs)
# =========================================================
BASELINE_CONFIG = {
    "training": {
        "model_class_name": "SpatialASLNet",
        "dropout_rate": 0.1,
        "weight_decay": 0.0001,
        "learning_rate": 0.0001,

        # Loss configuration
        "loss_type": "l1",
        "att_scale": 0.033,
        "cbf_weight": 1.0,
        "att_weight": 1.0,
        "loss_weight_cbf": 1.0,
        "loss_weight_att": 1.0,

        # Uncertainty bounds
        "log_var_cbf_min": -3.0,
        "log_var_cbf_max": 7.0,
        "log_var_att_min": -3.0,
        "log_var_att_max": 10.0,

        # Training params
        "batch_size": 32,
        "n_ensembles": 3,
        "n_epochs": 100,  # Increased from 50 - models still improving
        "validation_steps_per_epoch": 25,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0,
        "norm_type": "batch",
        "hidden_sizes": [32, 64, 128, 256],

        # DC and variance
        "dc_weight": 0.0001,
        "variance_weight": 0.1,
    },

    "data": {
        "use_offline_dataset": True,
        "offline_dataset_path": "asl_spatial_dataset_100k",
        "num_samples_to_load": 100000,
        "pld_values": [500, 1000, 1500, 2000, 2500, 3000],
        "global_scale_factor": 10.0,
        "noise_type": "rician",
        "normalization_mode": "global_scale",
        "data_noise_components": ["thermal", "physio", "drift"],
    },

    "simulation": {
        "T1_artery": 1850.0,
        "T_tau": 1800.0,
        "T2_factor": 1.0,
        "alpha_BS1": 1.0,
        "alpha_PCASL": 0.85,
        "alpha_VSASL": 0.56,
    },

    "noise_config": {
        "snr_range": [3.0, 15.0],
        "physio_amp_range": [0.05, 0.15],
        "physio_freq_range": [0.5, 2.0],
        "drift_range": [-0.02, 0.02],
        "spike_probability": 0.05,
        "spike_magnitude_range": [2.0, 5.0],
    },

    "wandb": {
        "wandb_project": "asl-optimization-ablation",
        "wandb_entity": "adikondepudi",
    },
}

# =========================================================
# COMPACT EXPERIMENT DEFINITIONS (10 experiments)
# =========================================================
EXPERIMENTS = [
    # === CONTROL ===
    {
        "name": "00_Baseline",
        "hypothesis": "Control: 100 epochs baseline",
        "changes": {},
    },

    # === SINGLE-FACTOR TESTS ===
    {
        "name": "01_Capacity_Large",
        "hypothesis": "Does 2x model capacity improve accuracy?",
        "changes": {
            "training.hidden_sizes": [64, 128, 256, 512],
        },
    },
    {
        "name": "02_LR_2x",
        "hypothesis": "Does 2x learning rate improve convergence?",
        "changes": {
            "training.learning_rate": 0.0002,
        },
    },
    {
        "name": "03_CBF_Focus",
        "hypothesis": "Does CBF-focused loss (weight+variance) reduce CBF error?",
        "changes": {
            "training.cbf_weight": 1.3,
            "training.att_weight": 0.8,
            "training.variance_weight": 0.05,
        },
    },
    {
        "name": "04_SNR_Wide",
        "hypothesis": "Does wider SNR [2,25] improve robustness?",
        "changes": {
            "noise_config.snr_range": [2.0, 25.0],
        },
    },
    {
        "name": "05_DomainRand",
        "hypothesis": "Does physics randomization improve generalization?",
        "changes": {
            "simulation.domain_randomization": {
                "enabled": True,
                "T1_artery_range": [1550.0, 2150.0],
                "alpha_PCASL_range": [0.75, 0.95],
                "alpha_VSASL_range": [0.40, 0.70],
            },
        },
    },
    {
        "name": "06_No_DC",
        "hypothesis": "Is DC loss actually helping? (ablation)",
        "changes": {
            "training.dc_weight": 0.0,
        },
    },

    # === STRATEGIC COMBINATIONS ===
    {
        "name": "07_Training_Boost",
        "hypothesis": "Capacity + LR: training-focused improvements",
        "changes": {
            "training.hidden_sizes": [64, 128, 256, 512],
            "training.learning_rate": 0.0002,
        },
    },
    {
        "name": "08_Physics_Robust",
        "hypothesis": "SNR + DomainRand: physics-focused robustness",
        "changes": {
            "noise_config.snr_range": [2.0, 25.0],
            "simulation.domain_randomization": {
                "enabled": True,
                "T1_artery_range": [1550.0, 2150.0],
                "alpha_PCASL_range": [0.75, 0.95],
                "alpha_VSASL_range": [0.40, 0.70],
            },
        },
    },
    {
        "name": "09_Full_Optimized",
        "hypothesis": "All promising changes combined",
        "changes": {
            "training.hidden_sizes": [64, 128, 256, 512],
            "training.learning_rate": 0.0002,
            "training.cbf_weight": 1.3,
            "training.att_weight": 0.8,
            "training.variance_weight": 0.05,
            "noise_config.snr_range": [2.0, 25.0],
            "simulation.domain_randomization": {
                "enabled": True,
                "T1_artery_range": [1550.0, 2150.0],
                "alpha_PCASL_range": [0.75, 0.95],
                "alpha_VSASL_range": [0.40, 0.70],
            },
        },
    },
]


def apply_changes(config: dict, changes: dict) -> dict:
    """Apply nested changes to config using dot notation keys."""
    config = copy.deepcopy(config)

    for key, value in changes.items():
        parts = key.split('.')
        target = config

        # Navigate to parent
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Set value
        final_key = parts[-1]
        if isinstance(value, dict) and final_key in target and isinstance(target[final_key], dict):
            target[final_key].update(value)
        else:
            target[final_key] = value

    return config


def generate_slurm_script(job_name: str, run_dir: str, config_path: str) -> str:
    """Creates a SLURM script. 100 epochs ≈ 8 hours."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output={run_dir}/slurm.out
#SBATCH --error={run_dir}/slurm.err

source ~/.bashrc
conda activate asl_multiverse

cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "EXPERIMENT: {job_name}"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "============================================"

echo ""
echo "--- TRAINING (100 epochs) ---"
python main.py {config_path} --stage 2 --output-dir {run_dir}

echo ""
echo "--- VALIDATION ---"
python validate.py --run_dir {run_dir} --output_dir {run_dir}/validation_results

echo ""
echo "--- COMPLETE ---"
echo "Finished: $(date)"
"""
    return script


def main():
    base_dir = Path("optimization_ablation_v1")
    base_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("OPTIMIZATION ABLATION STUDY (COMPACT)")
    print("=" * 60)
    print(f"\nGenerating {len(EXPERIMENTS)} experiments:\n")

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)

        # Apply changes to baseline
        config = apply_changes(BASELINE_CONFIG, exp["changes"])
        config["_experiment"] = {
            "name": exp_name,
            "hypothesis": exp["hypothesis"],
            "changes": exp["changes"],
        }

        # Write config
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Write SLURM script
        slurm_path = exp_dir / "run.slurm"
        with open(slurm_path, 'w') as f:
            f.write(generate_slurm_script(exp_name, str(exp_dir), str(config_path)))

        # Print summary
        changes_str = ", ".join(f"{k.split('.')[-1]}" for k in exp["changes"].keys()) if exp["changes"] else "none"
        print(f"  {exp_name}")
        print(f"    → {exp['hypothesis']}")
        print(f"    → Changes: {changes_str}")
        print()

    print("=" * 60)
    print("Files generated in: optimization_ablation_v1/")
    print("=" * 60)


if __name__ == "__main__":
    main()

# FILE: setup_optimization_grid.py
"""
Optimization Ablation Study - 10 Experiments
Baseline: 10_Rician_Complex_Full + 100 epochs
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
        "loss_type": "l1",
        "att_scale": 0.033,
        "cbf_weight": 1.0,
        "att_weight": 1.0,
        "loss_weight_cbf": 1.0,
        "loss_weight_att": 1.0,
        "log_var_cbf_min": -3.0,
        "log_var_cbf_max": 7.0,
        "log_var_att_min": -3.0,
        "log_var_att_max": 10.0,
        "batch_size": 32,
        "n_ensembles": 3,
        "n_epochs": 100,
        "validation_steps_per_epoch": 25,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0,
        "norm_type": "batch",
        "hidden_sizes": [32, 64, 128, 256],
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
        "T1_artery": 1650.0,  # 3T consensus (Alsop 2015)
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
# 10 EXPERIMENTS
# =========================================================
EXPERIMENTS = [
    {
        "name": "00_Baseline",
        "hypothesis": "Control: 100 epochs baseline",
        "changes": {},
    },
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
        "hypothesis": "Does CBF-focused loss reduce CBF error?",
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
    {
        "name": "07_Training_Boost",
        "hypothesis": "Capacity + LR combined",
        "changes": {
            "training.hidden_sizes": [64, 128, 256, 512],
            "training.learning_rate": 0.0002,
        },
    },
    {
        "name": "08_Physics_Robust",
        "hypothesis": "SNR + DomainRand combined",
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
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        final_key = parts[-1]
        if isinstance(value, dict) and final_key in target and isinstance(target[final_key], dict):
            target[final_key].update(value)
        else:
            target[final_key] = value
    return config


def generate_slurm_script(job_name: str, run_dir: str, config_path: str) -> str:
    """Creates a SLURM script matching existing format."""
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
echo "--- STAGE 2: SPATIAL REGRESSION TRAINING ---"
python main.py {config_path} --stage 2 --output-dir {run_dir}

echo ""
echo "--- VALIDATION ---"
python validate.py --run_dir {run_dir} --output_dir {run_dir}/validation_results

echo ""
echo "--- JOB COMPLETE ---"
echo "Finished: $(date)"
"""
    return script


def main():
    base_dir = Path("optimization_ablation_v1")
    base_dir.mkdir(exist_ok=True)

    submit_script_lines = [
        "#!/bin/bash",
        "# Optimization Ablation Study - 10 Experiments",
        "# Generated by setup_optimization_grid.py",
        "",
        "echo '============================================'",
        f"echo 'Optimization Ablation: {len(EXPERIMENTS)} Experiments'",
        "echo '============================================'",
        "",
    ]

    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp["name"]
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)

        # Build config
        config = apply_changes(BASELINE_CONFIG, exp["changes"])
        config["_experiment"] = {
            "name": exp_name,
            "hypothesis": exp["hypothesis"],
        }

        # Write config
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Write SLURM script
        slurm_path = exp_dir / "run.slurm"
        with open(slurm_path, 'w') as f:
            f.write(generate_slurm_script(exp_name, str(exp_dir), str(config_path)))

        # Add to submit script
        submit_script_lines.append(f"# Experiment {i}: {exp_name}")
        submit_script_lines.append(f"# {exp['hypothesis']}")
        submit_script_lines.append(f"sbatch {slurm_path}")
        submit_script_lines.append(f'echo "Submitted {exp_name}"')
        submit_script_lines.append("")

    submit_script_lines.extend([
        "echo ''",
        "echo 'All jobs submitted. Monitor with: squeue -u $USER'",
    ])

    # Write submit_all.sh
    with open("submit_all.sh", "w") as f:
        f.write("\n".join(submit_script_lines))

    # Print summary
    print("=" * 60)
    print("OPTIMIZATION ABLATION STUDY")
    print("=" * 60)
    print(f"\nGenerated {len(EXPERIMENTS)} experiments in '{base_dir}/':\n")

    for exp in EXPERIMENTS:
        print(f"  {exp['name']}: {exp['hypothesis']}")

    print()
    print("=" * 60)
    print("WORKFLOW:")
    print("  1. sbatch generate_spatial_data.sh")
    print("  2. python setup_optimization_grid.py")
    print("  3. bash submit_all.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()

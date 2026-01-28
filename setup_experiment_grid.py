# FILE: setup_experiment_grid.py
"""
Spatial ASL U-Net Ablation Study
10 experiments testing specific hypotheses for SpatialASLNet architecture
"""
import os
import yaml
import copy
from pathlib import Path

# =========================================================
# SPATIAL MODEL EXPERIMENT DEFINITIONS
# =========================================================
# Parameters that matter for SpatialASLNet (U-Net):
#   - hidden_sizes (features): U-Net encoder/decoder channel sizes
#   - dc_weight: Data consistency loss weight (physics regularization)
#   - noise_type: gaussian vs rician (MRI physics fidelity)
#   - normalization_mode: per_curve vs global_scale
#   - noise_components: thermal only vs complex (physio, drift)
# =========================================================

EXPERIMENTS = [
    # === BASELINE ===
    {
        "name": "01_Baseline",
        "hypothesis": "Standard U-Net baseline with default settings",
        "hidden_sizes": [32, 64, 128, 256],
        "dc_weight": 0.0,
        "noise_type": "gaussian",
        "normalization_mode": "per_curve",
        "noise_components": ["thermal"],
    },

    # === CAPACITY ABLATION ===
    {
        "name": "02_Capacity_Small",
        "hypothesis": "Can a smaller U-Net achieve similar accuracy with faster inference?",
        "hidden_sizes": [16, 32, 64, 128],
        "dc_weight": 0.0,
        "noise_type": "gaussian",
        "normalization_mode": "per_curve",
        "noise_components": ["thermal"],
    },
    {
        "name": "03_Capacity_Large",
        "hypothesis": "Does increased capacity improve accuracy or cause overfitting?",
        "hidden_sizes": [64, 128, 256, 512],
        "dc_weight": 0.0,
        "noise_type": "gaussian",
        "normalization_mode": "per_curve",
        "noise_components": ["thermal"],
    },

    # === DATA CONSISTENCY LOSS ABLATION ===
    {
        "name": "04_DC_Loss_Light",
        "hypothesis": "Does light physics regularization improve generalization?",
        "hidden_sizes": [32, 64, 128, 256],
        "dc_weight": 0.0001,
        "noise_type": "gaussian",
        "normalization_mode": "per_curve",
        "noise_components": ["thermal"],
    },
    {
        "name": "05_DC_Loss_Moderate",
        "hypothesis": "Is stronger DC loss beneficial or does it fight supervised loss?",
        "hidden_sizes": [32, 64, 128, 256],
        "dc_weight": 0.001,
        "noise_type": "gaussian",
        "normalization_mode": "per_curve",
        "noise_components": ["thermal"],
    },

    # === NOISE MODEL FIDELITY ===
    {
        "name": "06_Rician_Noise",
        "hypothesis": "Does Rician noise (correct MRI physics) improve real-world transfer?",
        "hidden_sizes": [32, 64, 128, 256],
        "dc_weight": 0.0,
        "noise_type": "rician",
        "normalization_mode": "global_scale",  # Rician works better with global_scale
        "noise_components": ["thermal"],
    },
    {
        "name": "07_Global_Scale_Gaussian",
        "hypothesis": "Does preserving signal magnitude help even with Gaussian noise?",
        "hidden_sizes": [32, 64, 128, 256],
        "dc_weight": 0.0,
        "noise_type": "gaussian",
        "normalization_mode": "global_scale",
        "noise_components": ["thermal"],
    },

    # === NOISE COMPLEXITY ===
    {
        "name": "08_Complex_Noise",
        "hypothesis": "How does the model handle realistic clinical noise patterns?",
        "hidden_sizes": [32, 64, 128, 256],
        "dc_weight": 0.0,
        "noise_type": "gaussian",
        "normalization_mode": "per_curve",
        "noise_components": ["thermal", "physio", "drift"],
    },
    {
        "name": "09_Complex_Noise_DC",
        "hypothesis": "Does DC loss help regularize against complex noise?",
        "hidden_sizes": [32, 64, 128, 256],
        "dc_weight": 0.0001,
        "noise_type": "gaussian",
        "normalization_mode": "per_curve",
        "noise_components": ["thermal", "physio", "drift"],
    },

    # === FULL REALISTIC SCENARIO ===
    {
        "name": "10_Rician_Complex_Full",
        "hypothesis": "Best realistic setting: Rician noise + complex artifacts + DC regularization",
        "hidden_sizes": [32, 64, 128, 256],
        "dc_weight": 0.0001,
        "noise_type": "rician",
        "normalization_mode": "global_scale",
        "noise_components": ["thermal", "physio", "drift"],
    },
]

# =========================================================
# BASE CONFIG (Shared across all spatial experiments)
# =========================================================
BASE_CONFIG = {
    "training": {
        "model_class_name": "SpatialASLNet",
        "dropout_rate": 0.1,
        "weight_decay": 0.0001,
        "learning_rate": 0.0001,

        # Loss configuration
        "loss_weight_cbf": 1.0,
        "loss_weight_att": 1.0,

        # Uncertainty bounds (for future heteroscedastic uncertainty)
        "log_var_cbf_min": -3.0,
        "log_var_cbf_max": 7.0,
        "log_var_att_min": -3.0,
        "log_var_att_max": 10.0,

        "batch_size": 32,
        "n_ensembles": 3,          # 3 runs for statistical significance
        "n_epochs": 50,            # Spatial converges faster than voxel
        "validation_steps_per_epoch": 25,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0,
        "norm_type": "batch",
    },

    "data": {
        "use_offline_dataset": True,
        "offline_dataset_path": "asl_spatial_dataset_100k",
        "num_samples_to_load": 100000,
        "pld_values": [500, 1000, 1500, 2000, 2500, 3000],
        "global_scale_factor": 10.0,  # Used when normalization_mode=global_scale
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
        "wandb_project": "asl-spatial-ablation",
        "wandb_entity": "adikondepudi",
    },
}


def generate_slurm_script(job_name: str, run_dir: str, config_path: str) -> str:
    """Creates a SLURM script for spatial model training + validation."""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
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
python validate.py {run_dir} --output-dir {run_dir}/validation_results

echo ""
echo "--- JOB COMPLETE ---"
echo "Finished: $(date)"
"""
    return script


def main():
    base_dir = Path("spatial_ablation_jobs")
    base_dir.mkdir(exist_ok=True)

    submit_script_lines = [
        "#!/bin/bash",
        "# Spatial ASL U-Net Ablation Study",
        "# Generated by setup_experiment_grid.py",
        "",
        "echo '============================================'",
        "echo 'ASL Spatial Ablation: 10 Experiments'",
        "echo '============================================'",
        "",
    ]

    for i, exp in enumerate(EXPERIMENTS):
        # 1. Create Experiment Directory
        exp_name = exp["name"]
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)

        # 2. Build Config (merge base + experiment specifics)
        config = copy.deepcopy(BASE_CONFIG)

        # Apply experiment-specific settings
        config["training"]["hidden_sizes"] = exp["hidden_sizes"]
        config["training"]["dc_weight"] = exp["dc_weight"]
        config["data"]["noise_type"] = exp["noise_type"]
        config["data"]["normalization_mode"] = exp["normalization_mode"]
        config["data"]["data_noise_components"] = exp["noise_components"]

        # Store hypothesis for reference
        config["_experiment"] = {
            "name": exp_name,
            "hypothesis": exp["hypothesis"],
        }

        # Write config
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # 3. Write SLURM Script
        slurm_path = exp_dir / "run.slurm"
        with open(slurm_path, 'w') as f:
            f.write(generate_slurm_script(exp_name, str(exp_dir), str(config_path)))

        # 4. Add to Master Submit Script
        submit_script_lines.append(f"# Experiment {i+1}: {exp_name}")
        submit_script_lines.append(f"# Hypothesis: {exp['hypothesis']}")
        submit_script_lines.append(f"JOB_{i}=$(sbatch --parsable {slurm_path})")
        submit_script_lines.append(f'echo "Submitted {exp_name} as Job ID: $JOB_{i}"')
        submit_script_lines.append("")

    # 5. Add aggregation job with dependency
    num_jobs = len(EXPERIMENTS)
    job_vars = ":".join([f"$JOB_{i}" for i in range(num_jobs)])
    submit_script_lines.extend([
        "# Wait for all experiments, then aggregate results",
        f'DEPENDENCY="{job_vars}"',
        "",
        "# Create aggregation script if it doesn't exist",
        "if [ -f aggregate_results.slurm ]; then",
        '    sbatch --dependency=afterany:${DEPENDENCY} aggregate_results.slurm',
        '    echo "Aggregator job submitted with dependency on all experiments"',
        "else",
        '    echo "No aggregate_results.slurm found - skipping aggregation"',
        "fi",
        "",
        "echo ''",
        "echo 'All jobs submitted. Monitor with: squeue -u $USER'",
    ])

    # Write Master Script
    with open("submit_all.sh", "w") as f:
        f.write("\n".join(submit_script_lines))

    # Print summary
    print("=" * 60)
    print("SPATIAL ASL ABLATION STUDY")
    print("=" * 60)
    print(f"\nGenerated {len(EXPERIMENTS)} experiments in '{base_dir}/':\n")

    for i, exp in enumerate(EXPERIMENTS):
        print(f"  {exp['name']}")
        print(f"    → {exp['hypothesis']}")
        print(f"    → U-Net: {exp['hidden_sizes']}, DC: {exp['dc_weight']}, "
              f"Noise: {exp['noise_type']}/{exp['noise_components']}")
        print()

    print("=" * 60)
    print("NEXT STEPS:")
    print("  1. Generate data:  sbatch generate_spatial_data.sh")
    print("  2. Run experiments: bash submit_all.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()

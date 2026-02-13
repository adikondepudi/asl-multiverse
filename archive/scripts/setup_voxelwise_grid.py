#!/usr/bin/env python
"""
Voxel-wise ASL Network Experiment Grid
Tests key hypotheses for beating Least Squares fitting

Run: python setup_voxelwise_grid.py
Then: bash submit_voxelwise_all.sh
"""
import os
import yaml
import copy
from pathlib import Path

# =========================================================
# EXPERIMENT DEFINITIONS
# Focus on factors most likely to beat Least Squares:
#   1. Loss function (MAE vs NLL)
#   2. Feature selection (what info helps most)
#   3. Noise robustness (training on diverse noise)
#   4. Architecture (Conv1D encoder vs MLP-only)
# =========================================================

EXPERIMENTS = [
    # === LOSS FUNCTION COMPARISON ===
    {
        "name": "01_MAE_Loss",
        "hypothesis": "Pure MAE loss forces accurate predictions - best chance to beat LS",
        "loss_mode": "mae_only",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_components": ["thermal"],
        "encoder_type": "physics_processor",
    },
    {
        "name": "02_MAE_NLL_Balanced",
        "hypothesis": "MAE+NLL provides accuracy with uncertainty quantification",
        "loss_mode": "mae_nll",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_components": ["thermal"],
        "encoder_type": "physics_processor",
    },

    # === FEATURE ABLATION ===
    {
        "name": "03_Features_Minimal",
        "hypothesis": "Can we beat LS with just mean/std features?",
        "loss_mode": "mae_only",
        "active_features": ["mean", "std"],
        "noise_components": ["thermal"],
        "encoder_type": "physics_processor",
    },
    {
        "name": "04_Features_Full",
        "hypothesis": "Do additional features (ttp, com) improve over peak?",
        "loss_mode": "mae_only",
        "active_features": ["mean", "std", "peak", "ttp", "com", "t1_artery"],
        "noise_components": ["thermal"],
        "encoder_type": "physics_processor",
    },

    # === NOISE ROBUSTNESS ===
    {
        "name": "05_Robust_Training",
        "hypothesis": "Training on complex noise improves generalization",
        "loss_mode": "mae_only",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_components": ["thermal", "physio", "drift"],
        "encoder_type": "physics_processor",
    },
    {
        "name": "06_Wide_SNR_Range",
        "hypothesis": "Wider SNR range during training improves low-SNR performance",
        "loss_mode": "mae_only",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_components": ["thermal"],
        "encoder_type": "physics_processor",
        "snr_range": [1.0, 25.0],  # Wider range than default [3, 15]
    },

    # === ARCHITECTURE COMPARISON ===
    {
        "name": "07_MLP_Only",
        "hypothesis": "Is Conv1D encoder necessary or is MLP sufficient?",
        "loss_mode": "mae_only",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_components": ["thermal"],
        "encoder_type": "mlp_only",
    },
    {
        "name": "08_Larger_Network",
        "hypothesis": "Does more capacity help or hurt?",
        "loss_mode": "mae_only",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_components": ["thermal"],
        "encoder_type": "physics_processor",
        "hidden_sizes": [512, 256, 128],
    },

    # === BEST COMBINED SETTINGS ===
    {
        "name": "09_Best_Combo_A",
        "hypothesis": "MAE + Full Features + Robust Noise",
        "loss_mode": "mae_only",
        "active_features": ["mean", "std", "peak", "ttp", "com", "t1_artery"],
        "noise_components": ["thermal", "physio", "drift"],
        "encoder_type": "physics_processor",
    },
    {
        "name": "10_Best_Combo_B",
        "hypothesis": "MAE + Core Features + Wide SNR + Larger Network",
        "loss_mode": "mae_only",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_components": ["thermal", "physio"],
        "encoder_type": "physics_processor",
        "hidden_sizes": [512, 256, 128],
        "snr_range": [1.0, 25.0],
    },
]

# =========================================================
# BASE CONFIG
# =========================================================
BASE_CONFIG = {
    "training": {
        "model_class_name": "DisentangledASLNet",
        "encoder_type": "physics_processor",
        "hidden_sizes": [256, 128, 64],
        "dropout_rate": 0.1,
        "weight_decay": 0.0001,
        "learning_rate": 0.0001,
        "norm_type": "batch",

        # Loss config
        "loss_mode": "mae_only",
        "mae_weight": 1.0,
        "nll_weight": 0.1,

        # Uncertainty bounds
        "log_var_cbf_min": -5.0,
        "log_var_cbf_max": 7.0,
        "log_var_att_min": -5.0,
        "log_var_att_max": 10.0,

        # Training params
        "batch_size": 4096,
        "n_ensembles": 3,
        "n_epochs": 100,
        "validation_steps_per_epoch": 25,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0,
    },

    "data": {
        "use_offline_dataset": True,
        "offline_dataset_path": "asl_clean_library_v1",
        "num_samples_to_load": 5000000,
        "pld_values": [500, 1000, 1500, 2000, 2500, 3000],
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "data_noise_components": ["thermal"],
        "noise_type": "gaussian",
        "normalization_mode": "per_curve",
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
        "physio_amp_range": [0.03, 0.10],
        "physio_freq_range": [0.5, 1.5],
        "drift_range": [-0.015, 0.015],
        "spike_probability": 0.0,
    },

    "wandb": {
        "wandb_project": "asl-beat-ls-grid",
        "wandb_entity": "adikondepudi",
    },
}


def generate_slurm_script(job_name: str, run_dir: str, config_path: str) -> str:
    """Creates a SLURM script for voxel-wise training + validation."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
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
echo "--- STAGE 2: VOXEL-WISE REGRESSION TRAINING ---"
python main.py {config_path} --stage 2 --output-dir {run_dir}

echo ""
echo "--- VALIDATION ---"
python validate.py --run_dir {run_dir} --output_dir {run_dir}/validation_results

echo ""
echo "--- JOB COMPLETE ---"
echo "Finished: $(date)"
"""


def main():
    base_dir = Path("voxelwise_beat_ls_grid")
    base_dir.mkdir(exist_ok=True)

    submit_lines = [
        "#!/bin/bash",
        "# Voxel-wise ASL Experiment Grid: Beat Least Squares",
        "# Generated by setup_voxelwise_grid.py",
        "",
        "echo '============================================'",
        "echo 'ASL Voxel-wise Grid: Beat Least Squares'",
        "echo '============================================'",
        "",
    ]

    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp["name"]
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)

        # Build config
        config = copy.deepcopy(BASE_CONFIG)

        # Apply experiment-specific settings
        config["training"]["loss_mode"] = exp["loss_mode"]
        config["training"]["encoder_type"] = exp["encoder_type"]
        config["data"]["active_features"] = exp["active_features"]
        config["data"]["data_noise_components"] = exp["noise_components"]

        # Optional overrides
        if "hidden_sizes" in exp:
            config["training"]["hidden_sizes"] = exp["hidden_sizes"]
        if "snr_range" in exp:
            config["noise_config"]["snr_range"] = exp["snr_range"]

        # Store hypothesis
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
        submit_lines.append(f"# Experiment {i+1}: {exp_name}")
        submit_lines.append(f"# {exp['hypothesis']}")
        submit_lines.append(f"JOB_{i}=$(sbatch --parsable {slurm_path})")
        submit_lines.append(f'echo "Submitted {exp_name}: Job $JOB_{i}"')
        submit_lines.append("")

    submit_lines.extend([
        "echo ''",
        "echo 'All jobs submitted. Monitor with: squeue -u $USER'",
        f"echo 'Results will be in: {base_dir}/'",
    ])

    # Write submit script
    with open("submit_voxelwise_all.sh", "w") as f:
        f.write("\n".join(submit_lines))
    os.chmod("submit_voxelwise_all.sh", 0o755)

    # Print summary
    print("=" * 60)
    print("VOXEL-WISE ASL EXPERIMENT GRID: BEAT LEAST SQUARES")
    print("=" * 60)
    print(f"\nGenerated {len(EXPERIMENTS)} experiments in '{base_dir}/':\n")

    for exp in EXPERIMENTS:
        print(f"  {exp['name']}")
        print(f"    → {exp['hypothesis']}")
        print(f"    → Loss: {exp['loss_mode']}, Features: {len(exp['active_features'])}, "
              f"Noise: {exp['noise_components']}")
        print()

    print("=" * 60)
    print("NEXT STEPS:")
    print("  1. Ensure data exists: asl_clean_library_v1/")
    print("  2. Run experiments:    bash submit_voxelwise_all.sh")
    print("  3. Compare results:    python compare_ablation_results.py voxelwise_beat_ls_grid")
    print("=" * 60)


if __name__ == "__main__":
    main()

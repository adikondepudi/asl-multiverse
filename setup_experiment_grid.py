# FILE: setup_experiment_grid.py
import os
import yaml
import itertools
from pathlib import Path

# =========================================================
# 1. DEFINE THE MULTIVERSE (The Ablation Grid)
# =========================================================
GRID = {
    "experiment_name": ["ablation_study_v1"],
    
    # A. Feature Engineering Ablation
    "active_features": [
        ["mean", "std"],                           # Baseline
        ["mean", "std", "t1_artery"],              # Physics-Informed
        ["mean", "std", "peak", "t1_artery"],      # High-Dimension
    ],
    
    # B. Data Robustness Ablation (Noise Types)
    "training_noise_type": [
        ["thermal"],                               # Clean training
        ["thermal", "physio", "drift"]             # Robust training
    ],
    
    # C. Hyperparameter Ablation
    "hidden_sizes": [
        [128, 64, 32],
        [256, 128, 64]
    ]
}

# Base Config (Defaults)
BASE_CONFIG = {
    "training": {
        "learning_rate": 0.0001,
        "n_epochs": 100,
        "batch_size": 4096,
        "n_ensembles": 3
    },
    "simulation": {
        "pld_values": [500, 1000, 1500, 2000, 2500, 3000]
    },
    "wandb": {"wandb_project": "asl-ablation-study"}
}

def generate_slurm_script(job_name, run_dir, config_name):
    """Creates a self-validating SLURM script that runs both training stages."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output={run_dir}/slurm.out
#SBATCH --error={run_dir}/slurm.err

source ~/.bashrc
conda activate asl_multiverse

cd $SLURM_SUBMIT_DIR

echo "--- 1. STAGE 1: DENOISING PRE-TRAINING ---"
python main.py {config_name} --stage 1 --output-dir {run_dir}

echo "--- 2. STAGE 2: REGRESSION TRAINING ---"
python main.py {config_name} --stage 2 --output-dir {run_dir} --load-weights-from {run_dir}

echo "--- 3. AUTO-VALIDATION (NN vs LS) ---"
python validate.py --run_dir {run_dir} --output_dir {run_dir}/validation_results

echo "--- JOB COMPLETE ---"
"""

def main():
    # Generate Combinations
    keys, values = zip(*GRID.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    base_dir = Path("hpc_ablation_jobs")
    base_dir.mkdir(exist_ok=True)
    
    submit_script_lines = ["#!/bin/bash", "echo 'Launching Ablation Array...'"]
    
    for i, exp in enumerate(experiments):
        # 1. Create Unique Directory
        exp_id = f"exp{i:03d}_feats{len(exp['active_features'])}_noise{len(exp['training_noise_type'])}"
        exp_dir = base_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        # 2. Write Config YAML
        # (Merge base config with experiment specifics)
        current_config = BASE_CONFIG.copy()
        current_config['active_features'] = exp['active_features']
        current_config['data_noise_components'] = exp['training_noise_type']
        current_config['training']['hidden_sizes'] = exp['hidden_sizes']
        
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f)
            
        # 3. Write SLURM Script
        slurm_path = exp_dir / "run.slurm"
        with open(slurm_path, 'w') as f:
            f.write(generate_slurm_script(exp_id, str(exp_dir), str(config_path)))
            
        # 4. Add to Master Submit Script (use double quotes for variable expansion)
        submit_script_lines.append(f"JOB_{i}=$(sbatch --parsable {slurm_path})")
        submit_script_lines.append(f'echo "Submitted {exp_id} as Job ID: $JOB_{i}"')

    # 5. Build dependency string from captured job IDs
    num_jobs = len(experiments)
    job_vars = ":".join([f"$JOB_{i}" for i in range(num_jobs)])
    submit_script_lines.append("\n# Launch Aggregator with Dependency on all jobs")
    submit_script_lines.append(f'DEPENDENCY="{job_vars}"')
    submit_script_lines.append('sbatch --dependency=afterany:${DEPENDENCY} aggregate_results.slurm')
    submit_script_lines.append('echo "Aggregator job submitted with dependency on all experiments"')
    
    # Write Master Script
    with open("submit_all.sh", "w") as f:
        f.write("\n".join(submit_script_lines))
        
    print(f"Generated {len(experiments)} experiment configurations.")
    print("Run: 'bash submit_all.sh' to launch everything.")

if __name__ == "__main__":
    main()

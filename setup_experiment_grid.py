# FILE: setup_experiment_grid.py
"""
Targeted Ablation Study for ASL Neural Network
10 experiments testing specific scientific hypotheses
"""
import os
import yaml
import itertools
from pathlib import Path

# =========================================================
# EXPLICIT EXPERIMENT DEFINITIONS (Not Factorial)
# =========================================================
EXPERIMENTS = [
    # ID 01: Baseline
    {
        "name": "01_Baseline_Naive",
        "hypothesis": "How well can we do with just basic stats?",
        "active_features": ["mean", "std"],
        "noise_type": ["thermal"],
        "encoder_type": "physics_processor",  # Standard (with Conv1D)
        "hidden_sizes": [256, 128, 64],       # Standard size
    },
    # ID 02: Feature Ablation - Peak
    {
        "name": "02_Feature_Peak",
        "hypothesis": "Does adding Peak Height fix the bias?",
        "active_features": ["mean", "std", "peak"],
        "noise_type": ["thermal"],
        "encoder_type": "physics_processor",
        "hidden_sizes": [256, 128, 64],
    },
    # ID 03: Feature Ablation - Full (Best Model Candidate)
    {
        "name": "03_Feature_Full",
        "hypothesis": "Does T1 help biological variance?",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_type": ["thermal"],
        "encoder_type": "physics_processor",
        "hidden_sizes": [256, 128, 64],
    },
    # ID 04: Architecture - MLP Only (No Conv1D)
    {
        "name": "04_Arch_NoConv",
        "hypothesis": "Do we strictly need the Conv1D, or are scalars enough?",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_type": ["thermal"],
        "encoder_type": "mlp_only",  # NEW: MLP without Conv1D
        "hidden_sizes": [256, 128, 64],
    },
    # ID 05: Size Ablation - Small
    {
        "name": "05_Size_Small",
        "hypothesis": "Can we make it faster?",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_type": ["thermal"],
        "encoder_type": "physics_processor",
        "hidden_sizes": [128, 64, 32],  # Small
    },
    # ID 06: Size Ablation - Large
    {
        "name": "06_Size_Large",
        "hypothesis": "Are we underfitting?",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_type": ["thermal"],
        "encoder_type": "physics_processor",
        "hidden_sizes": [512, 256, 128],  # Large
    },
    # ID 07: Robustness - Full Features
    {
        "name": "07_Robust_Full",
        "hypothesis": "Does NN beat LS on realistic noise?",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_type": ["thermal", "physio", "drift"],  # Complex
        "encoder_type": "physics_processor",
        "hidden_sizes": [256, 128, 64],
    },
    # ID 08: Robustness - MLP Only
    {
        "name": "08_Robust_NoConv",
        "hypothesis": "Does the Conv1D layer help filter complex noise?",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_type": ["thermal", "physio", "drift"],
        "encoder_type": "mlp_only",
        "hidden_sizes": [256, 128, 64],
    },
    # ID 09: Robustness - Peak Only
    {
        "name": "09_Robust_Peak",
        "hypothesis": "Does Peak height help even more when noise is messy?",
        "active_features": ["mean", "std", "peak"],
        "noise_type": ["thermal", "physio", "drift"],
        "encoder_type": "physics_processor",
        "hidden_sizes": [256, 128, 64],
    },
    # ID 10: Robustness - Small Model
    {
        "name": "10_Robust_Small",
        "hypothesis": "Can a small model handle complex noise?",
        "active_features": ["mean", "std", "peak", "t1_artery"],
        "noise_type": ["thermal", "physio", "drift"],
        "encoder_type": "physics_processor",
        "hidden_sizes": [128, 64, 32],
    },
]

# Base Config (shared across all experiments)
BASE_CONFIG = {
    "training": {
        "model_class_name": "DisentangledASLNet",
        "dropout_rate": 0.1,
        "weight_decay": 0.0001,
        "learning_rate": 0.0003,
        "batch_size": 4096,
        "n_ensembles": 1,
        "n_epochs": 50,
        "validation_steps_per_epoch": 25,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "norm_type": "batch",
        "transformer_d_model_focused": 32,
        "transformer_nhead_model": 4,
        "transformer_nlayers_model": 2
    },
    "data": {
        "use_offline_dataset": True,
        "offline_dataset_path": "asl_offline_dataset_10M_baseline_v1",
        "num_samples_to_load": 2000000,
        "pld_values": [500, 1000, 1500, 2000, 2500, 3000]
    },
    "simulation": {
        "T1_artery": 1850.0,
        "T_tau": 1800.0,
        "T2_factor": 1.0,
        "alpha_BS1": 1.0,
        "alpha_PCASL": 0.85,
        "alpha_VSASL": 0.56,
        "pld_values": [500, 1000, 1500, 2000, 2500, 3000]
    },
    "wandb": {
        "wandb_project": "asl-ablation-study",
        "wandb_entity": "adikondepudi"
    }
}

def generate_slurm_script(job_name, run_dir, config_name):
    """Creates a SLURM script that runs both training stages."""
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

echo "=== EXPERIMENT: {job_name} ==="
echo "--- 1. STAGE 1: DENOISING PRE-TRAINING ---"
python main.py {config_name} --stage 1 --output-dir {run_dir}

echo "--- 2. STAGE 2: REGRESSION TRAINING ---"
python main.py {config_name} --stage 2 --output-dir {run_dir} --load-weights-from {run_dir}

echo "--- 3. AUTO-VALIDATION (NN vs LS) ---"
python validate.py --run_dir {run_dir} --output_dir {run_dir}/validation_results

echo "--- JOB COMPLETE ---"
"""

def main():
    base_dir = Path("hpc_ablation_jobs")
    base_dir.mkdir(exist_ok=True)
    
    submit_script_lines = [
        "#!/bin/bash",
        "echo '============================================'",
        "echo 'ASL Ablation Study: 10 Targeted Experiments'",
        "echo '============================================'",
        ""
    ]
    
    for i, exp in enumerate(EXPERIMENTS):
        # 1. Create Experiment Directory
        exp_name = exp["name"]
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # 2. Build Config (merge base + experiment specifics)
        import copy
        current_config = copy.deepcopy(BASE_CONFIG)
        current_config['active_features'] = exp['active_features']
        current_config['data_noise_components'] = exp['noise_type']
        current_config['training']['encoder_type'] = exp['encoder_type']
        current_config['training']['hidden_sizes'] = exp['hidden_sizes']
        current_config['experiment_hypothesis'] = exp['hypothesis']
        
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)
            
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

    # 5. Build dependency string from captured job IDs
    num_jobs = len(EXPERIMENTS)
    job_vars = ":".join([f"$JOB_{i}" for i in range(num_jobs)])
    submit_script_lines.append("# Launch Aggregator with Dependency on all jobs")
    submit_script_lines.append(f'DEPENDENCY="{job_vars}"')
    submit_script_lines.append('sbatch --dependency=afterany:${DEPENDENCY} aggregate_results.slurm')
    submit_script_lines.append('echo "Aggregator job submitted with dependency on all experiments"')
    
    # Write Master Script
    with open("submit_all.sh", "w") as f:
        f.write("\n".join(submit_script_lines))
        
    print(f"Generated {len(EXPERIMENTS)} targeted experiment configurations:")
    for exp in EXPERIMENTS:
        print(f"  - {exp['name']}: {exp['hypothesis']}")
    print("\nRun: 'bash submit_all.sh' to launch everything.")

if __name__ == "__main__":
    main()

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
# CRITICAL FIXES APPLIED: Match production hyperparameters for valid ablation comparisons
BASE_CONFIG = {
    "training": {
        "model_class_name": "DisentangledASLNet",
        "dropout_rate": 0.1,
        "weight_decay": 0.0001,
        
        # --- FIX 1: LOWER LEARNING RATE ---
        # Was 0.0003. Changed to 5e-5 to match production fine-tuning stability.
        # High LR during fine-tuning causes catastrophic forgetting of Stage 1 encoder.
        "learning_rate": 0.0001,
        
        # --- FIX 2: ADD MSE WEIGHT ---
        # Crucial for convergence. Forces model to predict accurate means,
        # not just "safely vague" predictions with high variance.
        "mse_weight": 50.0,
        
        # --- FIX 3: ADD LOG-VAR REGULARIZATION ---
        # Prevents uncertainty explosion/collapse during training.
        "log_var_reg_lambda": 0.05,
        
        # --- FIX 4: ADD LOG-VAR BOUNDS (match production) ---
        "log_var_cbf_min": -5.0,
        "log_var_cbf_max": 5.0,
        "log_var_att_min": -5.0,
        "log_var_att_max": 14.0,
        
        "batch_size": 16,   # Reduced further for safety on T4
        
        # --- FIX 5: INCREASE ENSEMBLES (Statistical Significance) ---
        # Was 1. Increased to 3 for statistically bulletproof comparisons.
        # A single run might get lucky/unlucky with random seed.
        "n_ensembles": 3,
        
        # --- FIX 6: INCREASE EPOCHS (Deep Convergence) ---
        # Was 50. With lower LR, model learns slower but better.
        "n_epochs": 100,
        
        "validation_steps_per_epoch": 25,
        
        # --- FIX 7: INCREASE PATIENCE (Overcome Plateaus) ---
        # Was 10. Give model time to overcome learning plateaus.
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.0,
        
        "norm_type": "batch",
        "transformer_d_model_focused": 32,
        "transformer_nhead_model": 4,
        "transformer_nlayers_model": 2
    },
    
    # --- FIX 8: EXPLICIT FINE-TUNING CONFIG ---
    "fine_tuning": {
        "enabled": True,
        "encoder_lr_factor": 20.0  # Encoder LR = learning_rate / 20
    },
    
    "data": {
        "use_offline_dataset": True,
        "offline_dataset_path": "asl_spatial_dataset_100k",
        
        # Using 100k for Spatial (Fits easily with Lazy Loading)
        # Was 5k which caused OOM in 1D mode, for Spatial we simply iterate dataset
        "num_samples_to_load": 100000,
        
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
    # Configurable noise parameters for ablation studies
    "noise_config": {
        # --- FIX 10: WIDEN SNR RANGE (Stress Testing) ---
        # Was [1.5, 10.0]. Clinical data has "hero" scans (SNR 20+) and disasters.
        # Wider range helps model interpolate better.
        "snr_range": [1.0, 15.0],
        
        "physio_amp_range": [0.05, 0.15],
        "physio_freq_range": [0.5, 2.0],
        "drift_range": [-0.02, 0.02],
        "spike_probability": 0.05,
        "spike_magnitude_range": [2.0, 5.0]
    },
    "wandb": {
        "wandb_project": "asl-ablation-study",
        "wandb_entity": "adikondepudi"
    }
}

def generate_slurm_script(job_name, run_dir, config_name, run_invivo=False):
    """Creates a SLURM script that runs both training stages AND validation."""
    
    # SpatialASLNet does not support Stage 1 (Denoising/Reconstruction)
    # We skip directly to Stage 2 (Regression)
    stage_1_cmd = ""
    stage_2_load_arg = f"--load-weights-from {run_dir}"
    
    # Check config for model class (hacky parse since we don't have the dict here easily)
    # But we know we are forcing SpatialASLNet now.
    # If we are strictly spatial, we skip stage 1.
    stage_1_cmd = f"# Skipping Stage 1 for SpatialASLNet (Regression Only)\n"
    stage_2_load_arg = "" # Start from scratch

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output={run_dir}/slurm.out
#SBATCH --error={run_dir}/slurm.err

source ~/.bashrc
conda activate asl_multiverse

cd $SLURM_SUBMIT_DIR

echo "=== EXPERIMENT: {job_name} ==="
echo "--- 1. STAGE 1: SKIPPED (Spatial) ---"
{stage_1_cmd}

echo "--- 2. STAGE 2: REGRESSION TRAINING ---"
python main.py {config_name} --stage 2 --output-dir {run_dir} {stage_2_load_arg}

echo "--- 3. AUTO-VALIDATION (NN vs LS) ---"
python validate.py --run_dir {run_dir} --output_dir {run_dir}/validation_results
"""
    if run_invivo:
        script += f"""
echo "--- 4. IN-VIVO PREDICTION ---"
# Assumes standard raw data location or passed via env var
RAW_DATA=${{RAW_DATA:-"Multiverse"}}
python predict_on_invivo.py processed_npy_cache {run_dir} {run_dir}/invivo_maps

echo "--- 5. IN-VIVO EVALUATION ---"
python run_all_evaluations.py {run_dir}/invivo_maps processed_npy_cache {run_dir}/invivo_validation {run_dir} {run_dir}/invivo_metrics
"""
    script += """
echo "--- JOB COMPLETE ---"
"""
    return script

def load_base_config():
    """Load base config from external YAML file."""
    import copy
    config_path = Path("config/base_template.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Fallback to inline config if YAML doesn't exist
        return BASE_CONFIG

def main():
    import copy
    base_dir = Path("hpc_ablation_jobs")
    base_dir.mkdir(exist_ok=True)
    
    # Load Base Template
    base_config = load_base_config()
    
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
        current_config = copy.deepcopy(base_config)
        current_config['training']['model_class_name'] = "SpatialASLNet" # Force Spatial
        current_config['active_features'] = exp['active_features']
        current_config['data_noise_components'] = exp['noise_type']
        current_config['training']['encoder_type'] = exp['encoder_type']
        current_config['training']['hidden_sizes'] = exp['hidden_sizes']
        current_config['experiment_hypothesis'] = exp['hypothesis']
        
        # Update offline dataset path to spatial dataset (generated by generate_spatial_data.sh)
        current_config['data']['offline_dataset_path'] = 'asl_spatial_dataset_100k'
        
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)
            
        # 3. Write SLURM Script (with optional in-vivo)
        run_invivo = exp.get('run_invivo', False)
        slurm_path = exp_dir / "run.slurm"
        with open(slurm_path, 'w') as f:
            f.write(generate_slurm_script(exp_name, str(exp_dir), str(config_path), run_invivo))
            
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

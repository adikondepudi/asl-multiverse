#!/usr/bin/env python3
"""
Amplitude-Aware Architecture Ablation Study - 10 Experiments

Tests the key components of the AmplitudeAwareSpatialASLNet to understand:
1. Does the amplitude-aware architecture fix CBF estimation?
2. Which components (FiLM, output modulation) are most important?
3. Does physics loss help or hurt?
4. What's the optimal configuration?

Key comparisons:
- SpatialASLNet (baseline) vs AmplitudeAwareSpatialASLNet
- FiLM conditioning: bottleneck only vs decoder vs both vs none
- Output modulation: with vs without
- Physics loss: 0 vs 0.1 vs 0.3
"""

import os
import yaml
import copy
from pathlib import Path

# =========================================================
# BASELINE CONFIG (Small dataset for quick ablation)
# =========================================================
BASELINE_CONFIG = {
    "training": {
        "model_class_name": "SpatialASLNet",  # Will be changed per experiment
        "hidden_sizes": [32, 64, 128, 256],
        "dropout_rate": 0.1,
        "norm_type": "group",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "n_epochs": 50,  # Reduced for ablation
        "n_ensembles": 3,
        "validation_steps_per_epoch": 25,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0,
        # Loss config
        "loss_type": "l1",
        "att_scale": 0.033,
        "cbf_weight": 1.0,
        "att_weight": 1.0,
        "dc_weight": 0.0,
        "variance_weight": 0.1,
        # Uncertainty bounds
        "log_var_cbf_min": -5.0,
        "log_var_cbf_max": 10.0,
        "log_var_att_min": -5.0,
        "log_var_att_max": 14.0,
        # Amplitude-aware settings (used when model is AmplitudeAwareSpatialASLNet)
        "use_film_at_bottleneck": True,
        "use_film_at_decoder": True,
        "use_amplitude_output_modulation": True,
        # Performance
        "use_amp": True,
        "use_tf32": True,
        "use_compile": False,
        "cudnn_benchmark": True,
        "num_workers": 4,
        "pin_memory": True,
    },
    "data": {
        "use_offline_dataset": True,
        "offline_dataset_path": "asl_spatial_dataset_ablation",  # Smaller dataset
        "num_samples_to_load": 20000,  # 20k for quick ablation
        "pld_values": [500, 1000, 1500, 2000, 2500, 3000],
        "normalization_mode": "global_scale",
        "global_scale_factor": 10.0,
        "noise_type": "rician",
        "data_noise_components": ["thermal", "physio", "drift"],
    },
    "noise_config": {
        "snr_range": [2.0, 25.0],
        "physio_amp_range": [0.03, 0.15],
        "physio_freq_range": [0.5, 2.0],
        "drift_range": [-0.03, 0.03],
        "spike_probability": 0.02,
        "spike_magnitude_range": [2.0, 5.0],
    },
    "simulation": {
        "T1_artery": 1850.0,
        "T_tau": 1800.0,
        "alpha_PCASL": 0.85,
        "alpha_VSASL": 0.56,
        "T2_factor": 1.0,
        "alpha_BS1": 1.0,
    },
    "wandb": {
        "wandb_project": "asl-amplitude-ablation",
        "wandb_entity": "adikondepudi",
    },
}

# =========================================================
# 10 EXPERIMENTS
# =========================================================
EXPERIMENTS = [
    # --- BASELINES ---
    {
        "name": "00_Baseline_SpatialASL",
        "hypothesis": "Control: Standard SpatialASLNet (shows amplitude invariance problem)",
        "changes": {
            "training.model_class_name": "SpatialASLNet",
        },
    },
    {
        "name": "01_PerCurve_Norm",
        "hypothesis": "Confirm: per_curve normalization destroys CBF info (should be worse)",
        "changes": {
            "training.model_class_name": "SpatialASLNet",
            "data.normalization_mode": "per_curve",
        },
    },

    # --- AMPLITUDE-AWARE COMPONENTS ---
    {
        "name": "02_AmpAware_Full",
        "hypothesis": "Full AmplitudeAwareSpatialASLNet with all components",
        "changes": {
            "training.model_class_name": "AmplitudeAwareSpatialASLNet",
            "training.use_film_at_bottleneck": True,
            "training.use_film_at_decoder": True,
            "training.use_amplitude_output_modulation": True,
        },
    },
    {
        "name": "03_AmpAware_OutputMod_Only",
        "hypothesis": "Amplitude output modulation only (no FiLM) - is direct amplitude enough?",
        "changes": {
            "training.model_class_name": "AmplitudeAwareSpatialASLNet",
            "training.use_film_at_bottleneck": False,
            "training.use_film_at_decoder": False,
            "training.use_amplitude_output_modulation": True,
        },
    },
    {
        "name": "04_AmpAware_FiLM_Only",
        "hypothesis": "FiLM conditioning only (no output modulation) - is FiLM alone enough?",
        "changes": {
            "training.model_class_name": "AmplitudeAwareSpatialASLNet",
            "training.use_film_at_bottleneck": True,
            "training.use_film_at_decoder": True,
            "training.use_amplitude_output_modulation": False,
        },
    },
    {
        "name": "05_AmpAware_Bottleneck_Only",
        "hypothesis": "FiLM at bottleneck only (minimal modification)",
        "changes": {
            "training.model_class_name": "AmplitudeAwareSpatialASLNet",
            "training.use_film_at_bottleneck": True,
            "training.use_film_at_decoder": False,
            "training.use_amplitude_output_modulation": False,
        },
    },

    # --- PHYSICS LOSS ---
    {
        "name": "06_AmpAware_Physics_0p1",
        "hypothesis": "Full AmplitudeAware + physics loss (dc_weight=0.1)",
        "changes": {
            "training.model_class_name": "AmplitudeAwareSpatialASLNet",
            "training.use_film_at_bottleneck": True,
            "training.use_film_at_decoder": True,
            "training.use_amplitude_output_modulation": True,
            "training.dc_weight": 0.1,
        },
    },
    {
        "name": "07_AmpAware_Physics_0p3",
        "hypothesis": "Full AmplitudeAware + stronger physics loss (dc_weight=0.3)",
        "changes": {
            "training.model_class_name": "AmplitudeAwareSpatialASLNet",
            "training.use_film_at_bottleneck": True,
            "training.use_film_at_decoder": True,
            "training.use_amplitude_output_modulation": True,
            "training.dc_weight": 0.3,
        },
    },

    # --- DOMAIN RANDOMIZATION ---
    {
        "name": "08_AmpAware_DomainRand",
        "hypothesis": "Full AmplitudeAware + domain randomization for generalization",
        "changes": {
            "training.model_class_name": "AmplitudeAwareSpatialASLNet",
            "training.use_film_at_bottleneck": True,
            "training.use_film_at_decoder": True,
            "training.use_amplitude_output_modulation": True,
            "simulation.domain_randomization": {
                "enabled": True,
                "T1_artery_range": [1550.0, 2150.0],
                "alpha_PCASL_range": [0.75, 0.95],
                "alpha_VSASL_range": [0.40, 0.70],
                "T_tau_perturb": 0.10,
            },
        },
    },

    # --- OPTIMIZED COMBINATION ---
    {
        "name": "09_AmpAware_Optimized",
        "hypothesis": "Best combination: Full AmplitudeAware + DomainRand (no DC based on prior ablation)",
        "changes": {
            "training.model_class_name": "AmplitudeAwareSpatialASLNet",
            "training.use_film_at_bottleneck": True,
            "training.use_film_at_decoder": True,
            "training.use_amplitude_output_modulation": True,
            "training.dc_weight": 0.0,  # Prior ablation showed DC hurts
            "simulation.domain_randomization": {
                "enabled": True,
                "T1_artery_range": [1550.0, 2150.0],
                "alpha_PCASL_range": [0.75, 0.95],
                "alpha_VSASL_range": [0.40, 0.70],
                "T_tau_perturb": 0.10,
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
    """Creates a SLURM script for the experiment."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name[:8]}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output={run_dir}/slurm_%j.out
#SBATCH --error={run_dir}/slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "EXPERIMENT: {job_name}"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# --- TRAINING ---
echo ""
echo "--- STAGE 2: SPATIAL TRAINING ---"
python main.py {config_path} --stage 2 --output-dir {run_dir}

# --- VALIDATION ---
echo ""
echo "--- VALIDATION ---"
python validate.py --run_dir {run_dir} --output_dir {run_dir}/validation_results

# --- AMPLITUDE SENSITIVITY TEST ---
echo ""
echo "--- AMPLITUDE SENSITIVITY TEST ---"
python -c "
import torch
import json
import sys
sys.path.insert(0, '.')

# Load config to get model class
import yaml
with open('{config_path}') as f:
    cfg = yaml.safe_load(f)

model_class = cfg['training']['model_class_name']
print(f'Testing model: {{model_class}}')

if model_class == 'AmplitudeAwareSpatialASLNet':
    from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
    model = AmplitudeAwareSpatialASLNet(
        in_channels=12,
        hidden_sizes=cfg['training']['hidden_sizes'],
        use_film_at_bottleneck=cfg['training'].get('use_film_at_bottleneck', True),
        use_film_at_decoder=cfg['training'].get('use_film_at_decoder', True),
        use_amplitude_output_modulation=cfg['training'].get('use_amplitude_output_modulation', True),
    )
else:
    from spatial_asl_network import SpatialASLNet
    model = SpatialASLNet(n_plds=6, features=cfg['training']['hidden_sizes'])

# Load trained weights
import glob
model_files = sorted(glob.glob('{run_dir}/trained_models/spatial_model_*.pt'))
if model_files:
    state_dict = torch.load(model_files[0], map_location='cpu')
    model.load_state_dict(state_dict)
    print(f'Loaded: {{model_files[0]}}')
else:
    print('WARNING: No trained model found, testing untrained')

model.eval()

# Test amplitude sensitivity
torch.manual_seed(42)
base_input = torch.randn(4, 12, 64, 64) * 0.1

scales = [0.1, 1.0, 10.0]
results = {{}}

with torch.no_grad():
    for scale in scales:
        scaled_input = base_input * scale
        output = model(scaled_input)
        cbf = output[0] if isinstance(output, tuple) else output[:, 0:1]
        cbf_mean = cbf.mean().item()
        results[f'scale_{{scale}}'] = cbf_mean
        print(f'  Scale {{scale:5.1f}}x -> CBF mean: {{cbf_mean:10.4f}}')

# Compute sensitivity ratio
cbf_01 = abs(results['scale_0.1'])
cbf_10 = abs(results['scale_10.0'])
ratio = cbf_10 / max(cbf_01, 1e-9)

print(f'')
print(f'  Amplitude sensitivity ratio (10x/0.1x): {{ratio:.2f}}')
print(f'  Is amplitude sensitive: {{ratio > 5.0}}')

# Save results
with open('{run_dir}/amplitude_sensitivity.json', 'w') as f:
    json.dump({{
        'model_class': model_class,
        'scales': scales,
        'cbf_predictions': results,
        'sensitivity_ratio': ratio,
        'is_sensitive': ratio > 5.0
    }}, f, indent=2)
print(f'Saved to {run_dir}/amplitude_sensitivity.json')
"

echo ""
echo "============================================"
echo "EXPERIMENT COMPLETE: {job_name}"
echo "Finished: $(date)"
echo "============================================"
"""
    return script


def main():
    base_dir = Path("amplitude_ablation_v1")
    base_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("AMPLITUDE-AWARE ARCHITECTURE ABLATION STUDY")
    print("=" * 70)
    print(f"\nGenerating {len(EXPERIMENTS)} experiments in '{base_dir}/':\n")

    for exp in EXPERIMENTS:
        print(f"  {exp['name']}")
        print(f"    Hypothesis: {exp['hypothesis']}")
        print()

    # Generate experiments
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

    # Write master orchestrator script
    orchestrator = f"""#!/bin/bash
#SBATCH --job-name=amp-ablation
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/amp_ablation_orchestrator_%j.out
#SBATCH --error=slurm_logs/amp_ablation_orchestrator_%j.err

# =============================================================================
# AMPLITUDE-AWARE ABLATION - MASTER ORCHESTRATOR ({len(EXPERIMENTS)} experiments)
# =============================================================================

set -e

echo "============================================"
echo "AMPLITUDE-AWARE ABLATION ({len(EXPERIMENTS)} experiments)"
echo "Started: $(date)"
echo "============================================"

mkdir -p slurm_logs
cd $SLURM_SUBMIT_DIR

# --- Load Environment ---
echo ""
echo "[1/4] Loading environment..."
source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

# =============================================================================
# STEP 1: Data Generation (20k samples for quick ablation)
# =============================================================================
echo ""
echo "[2/4] Submitting data generation (20k samples)..."

cat > slurm_logs/gen_ablation_data.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=amp-datagen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/amp_datagen_%j.out
#SBATCH --error=slurm_logs/amp_datagen_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "Generating spatial dataset for ablation (20k samples)..."
rm -rf asl_spatial_dataset_ablation
python generate_clean_library.py asl_spatial_dataset_ablation \\
    --spatial --total_samples 20000 --spatial-chunk-size 500 --image-size 64

echo "Done: $(date)"
EOF

DATA_JOB=$(sbatch --parsable slurm_logs/gen_ablation_data.slurm)
echo "  Data generation job: $DATA_JOB"

# =============================================================================
# STEP 2: Submit Training Jobs
# =============================================================================
echo ""
echo "[3/4] Submitting {len(EXPERIMENTS)} training experiments..."
echo "  (All depend on data job $DATA_JOB)"
echo ""

JOB_IDS=""
"""

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        orchestrator += f"""
# {exp_name}: {exp['hypothesis']}
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB {base_dir}/{exp_name}/run.slurm)
echo "  {exp_name} -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"
"""

    orchestrator += f"""
# =============================================================================
# STEP 3: Aggregation Job
# =============================================================================
echo ""
echo "[4/4] Submitting aggregation job..."

cat > {base_dir}/aggregate.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=amp-aggregate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output={base_dir}/aggregate_%j.out
#SBATCH --error={base_dir}/aggregate_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

python compare_amplitude_ablation.py --results_dir {base_dir}
EOF

AGG_JOB=$(sbatch --parsable --dependency=afterany:$JOB_IDS {base_dir}/aggregate.slurm)
echo "  Aggregation -> Job $AGG_JOB"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================"
echo "ALL JOBS SUBMITTED"
echo "============================================"
echo "Data generation: $DATA_JOB"
echo "Training jobs:   {len(EXPERIMENTS)} experiments"
echo "Aggregation:     $AGG_JOB"
echo ""
echo "Monitor: squeue -u \\$USER"
echo "Results: {base_dir}/amplitude_ablation_summary.csv"
echo "============================================"
"""

    with open("run_amplitude_ablation.sh", "w") as f:
        f.write(orchestrator)

    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"\nGenerated:")
    print(f"  - {len(EXPERIMENTS)} experiment configs in '{base_dir}/'")
    print(f"  - Orchestrator script: run_amplitude_ablation.sh")
    print()
    print("To run on HPC:")
    print("  sbatch run_amplitude_ablation.sh")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

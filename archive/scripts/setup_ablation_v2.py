#!/usr/bin/env python3
"""
Ablation Study V2 (Exp 10-20): Closing the Domain Gap & ATT Improvements

Round 1 (Exp 00-09) established that amplitude awareness is essential for CBF.
Round 2 focuses on the remaining weaknesses:
  1. Simulation-to-real domain gap (in-vivo CBF bias = +27 ml/100g/min)
  2. ATT accuracy (ATT MAE still 18-20 ms in simulation)
  3. Training regime (data scale, model capacity, ensemble size)

All experiments use AmplitudeAwareSpatialASLNet with full amplitude awareness
(the clear winner from round 1). The baseline for comparison is Exp 09.
"""

import os
import yaml
import copy
from pathlib import Path

# =========================================================
# BASELINE CONFIG (Exp 09 settings = current best)
# =========================================================
BASELINE_CONFIG = {
    "training": {
        "model_class_name": "AmplitudeAwareSpatialASLNet",
        "hidden_sizes": [32, 64, 128, 256],
        "dropout_rate": 0.1,
        "norm_type": "group",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "n_epochs": 50,
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
        # Amplitude-aware (locked from round 1)
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
        "offline_dataset_path": "asl_spatial_dataset_ablation_v2",
        "num_samples_to_load": 20000,
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
        "T1_artery": 1650.0,  # 3T consensus (Alsop 2015)
        "T_tau": 1800.0,
        "alpha_PCASL": 0.85,
        "alpha_VSASL": 0.56,
        "T2_factor": 1.0,
        "alpha_BS1": 1.0,
        "domain_randomization": {
            "enabled": True,
            "T1_artery_range": [1550.0, 2150.0],
            "alpha_PCASL_range": [0.75, 0.95],
            "alpha_VSASL_range": [0.40, 0.70],
            "alpha_BS1_range": [0.85, 1.0],
            "T_tau_perturb": 0.10,
        },
    },
    "wandb": {
        "wandb_project": "asl-ablation-v2",
        "wandb_entity": "adikondepudi",
    },
}

# =========================================================
# 11 EXPERIMENTS (Exp 10-20)
# =========================================================
EXPERIMENTS = [
    # --- THEME 1: CLOSING THE DOMAIN GAP ---
    {
        "name": "10_ExtendedDomainRand",
        "hypothesis": "Wider domain randomization ranges to better cover in-vivo variability (especially background suppression)",
        "changes": {
            "simulation.domain_randomization": {
                "enabled": True,
                "T1_artery_range": [1400.0, 2300.0],
                "alpha_PCASL_range": [0.65, 0.98],
                "alpha_VSASL_range": [0.30, 0.75],
                "alpha_BS1_range": [0.70, 1.0],
                "T_tau_perturb": 0.15,
            },
        },
    },
    {
        "name": "11_MoreData_50k",
        "hypothesis": "2.5x more training data (50k vs 20k) reduces overfitting to synthetic distribution",
        "changes": {
            "data.num_samples_to_load": 50000,
        },
    },
    {
        "name": "12_MoreData_100k",
        "hypothesis": "5x more training data (100k vs 20k) further reduces overfitting",
        "changes": {
            "data.num_samples_to_load": 100000,
        },
    },
    {
        "name": "13_AggressiveNoise",
        "hypothesis": "More realistic noise augmentation (motion, spikes, wider SNR) improves in-vivo robustness",
        "changes": {
            "data.data_noise_components": ["thermal", "physio", "drift", "spikes"],
            "noise_config": {
                "snr_range": [1.0, 30.0],
                "physio_amp_range": [0.03, 0.20],
                "physio_freq_range": [0.5, 2.0],
                "drift_range": [-0.05, 0.05],
                "spike_probability": 0.05,
                "spike_magnitude_range": [2.0, 8.0],
                "spatial_noise_sigma": 1.2,
                "motion_probability": 0.3,
                "motion_shift_range": [1, 4],
                "motion_rotate_range": [-3, 3],
            },
        },
    },

    # --- THEME 2: ATT IMPROVEMENTS ---
    {
        "name": "14_ATT_Rebalanced",
        "hypothesis": "Current att_scale=0.033 massively underweights ATT loss; rebalancing to 1.0 with att_weight=2.0 should improve ATT",
        "changes": {
            "training.att_scale": 1.0,
            "training.att_weight": 2.0,
        },
    },
    {
        "name": "15_HuberLoss",
        "hypothesis": "Huber loss is more robust to outliers than L1, may help ATT in difficult voxels",
        "changes": {
            "training.loss_type": "huber",
        },
    },
    {
        "name": "16_L2Loss",
        "hypothesis": "L2 loss penalizes large errors more heavily, may reduce worst-case ATT errors",
        "changes": {
            "training.loss_type": "l2",
        },
    },

    # --- THEME 3: TRAINING REGIME ---
    {
        "name": "17_LargerModel",
        "hypothesis": "Doubling model capacity [64,128,256,512] may capture more complex spatial patterns",
        "changes": {
            "training.hidden_sizes": [64, 128, 256, 512],
        },
    },
    {
        "name": "18_LongerTraining",
        "hypothesis": "100 epochs (vs 50) with higher patience may find better minima, especially for ATT",
        "changes": {
            "training.n_epochs": 100,
            "training.early_stopping_patience": 25,
        },
    },
    {
        "name": "19_Ensemble5",
        "hypothesis": "5-member ensemble (vs 3) improves prediction quality via better averaging",
        "changes": {
            "training.n_ensembles": 5,
        },
    },

    # --- BEST COMBINATION ---
    {
        "name": "20_BestCombo",
        "hypothesis": "Combine extended domain rand + 100k data + aggressive noise + ATT rebalancing + longer training + 5 ensembles",
        "changes": {
            "data.num_samples_to_load": 100000,
            "data.data_noise_components": ["thermal", "physio", "drift", "spikes"],
            "training.n_epochs": 100,
            "training.early_stopping_patience": 25,
            "training.n_ensembles": 5,
            "training.att_scale": 1.0,
            "training.att_weight": 2.0,
            "noise_config": {
                "snr_range": [1.0, 30.0],
                "physio_amp_range": [0.03, 0.20],
                "physio_freq_range": [0.5, 2.0],
                "drift_range": [-0.05, 0.05],
                "spike_probability": 0.05,
                "spike_magnitude_range": [2.0, 8.0],
                "spatial_noise_sigma": 1.2,
                "motion_probability": 0.3,
                "motion_shift_range": [1, 4],
                "motion_rotate_range": [-3, 3],
            },
            "simulation.domain_randomization": {
                "enabled": True,
                "T1_artery_range": [1400.0, 2300.0],
                "alpha_PCASL_range": [0.65, 0.98],
                "alpha_VSASL_range": [0.30, 0.75],
                "alpha_BS1_range": [0.70, 1.0],
                "T_tau_perturb": 0.15,
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


def generate_slurm_script(job_name: str, run_dir: str, config_path: str,
                          time_limit: str = "6:00:00", mem: str = "64G") -> str:
    """Creates a SLURM script for the experiment."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name[:8]}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=8
#SBATCH --time={time_limit}
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
import yaml
sys.path.insert(0, '.')

with open('{config_path}') as f:
    cfg = yaml.safe_load(f)

model_class = cfg['training']['model_class_name']
print(f'Testing model: {{model_class}}')

from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
model = AmplitudeAwareSpatialASLNet(
    n_plds=len(cfg['data']['pld_values']),
    features=cfg['training']['hidden_sizes'],
    use_film_at_bottleneck=cfg['training'].get('use_film_at_bottleneck', True),
    use_film_at_decoder=cfg['training'].get('use_film_at_decoder', True),
    use_amplitude_output_modulation=cfg['training'].get('use_amplitude_output_modulation', True),
)

import glob
model_files = sorted(glob.glob('{run_dir}/trained_models/ensemble_model_*.pt'))
if model_files:
    state_dict = torch.load(model_files[0], map_location='cpu')
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    print(f'Loaded: {{model_files[0]}}')
else:
    print('WARNING: No trained model found, testing untrained')

model.eval()

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
        print(f'  Scale {{scale:5.1f}}x -> CBF mean: {{cbf_mean:12.7f}}')

cbf_01 = abs(results['scale_0.1'])
cbf_10 = abs(results['scale_10.0'])
ratio = cbf_10 / max(cbf_01, 1e-9)

print(f'')
print(f'  Amplitude sensitivity ratio (10x/0.1x): {{ratio:.2f}}')
print(f'  Is amplitude sensitive: {{ratio > 5.0}}')

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
    base_dir = Path("amplitude_ablation_v2")
    base_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY V2: DOMAIN GAP & ATT IMPROVEMENTS (Exp 10-20)")
    print("=" * 70)
    print(f"\nGenerating {len(EXPERIMENTS)} experiments in '{base_dir}/':\n")

    for exp in EXPERIMENTS:
        print(f"  {exp['name']}")
        print(f"    {exp['hypothesis']}")
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

        # Determine resource needs
        n_samples = config["data"].get("num_samples_to_load", 20000)
        n_epochs = config["training"].get("n_epochs", 50)
        n_ensembles = config["training"].get("n_ensembles", 3)
        features = config["training"].get("hidden_sizes", [32, 64, 128, 256])
        max_feat = max(features)

        # Scale time limit based on workload
        base_hours = 6
        if n_samples > 50000:
            base_hours = max(base_hours, 12)
        if n_epochs > 50:
            base_hours = max(base_hours, 10)
        if n_ensembles > 3:
            base_hours = int(base_hours * n_ensembles / 3)
        if max_feat > 256:
            base_hours = int(base_hours * 1.5)
        time_limit = f"{base_hours}:00:00"

        mem = "128G" if n_samples > 50000 else "64G"

        # Write SLURM script
        slurm_path = exp_dir / "run.slurm"
        with open(slurm_path, 'w') as f:
            f.write(generate_slurm_script(exp_name, str(exp_dir), str(config_path),
                                          time_limit=time_limit, mem=mem))

    # Write master orchestrator script
    orchestrator = f"""#!/bin/bash
#SBATCH --job-name=abl-v2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/ablation_v2_orchestrator_%j.out
#SBATCH --error=slurm_logs/ablation_v2_orchestrator_%j.err

# =============================================================================
# ABLATION STUDY V2: DOMAIN GAP & ATT IMPROVEMENTS ({len(EXPERIMENTS)} experiments)
#
# Baseline: Exp 09 (AmplitudeAware Optimized)
# Focus: Domain gap, ATT accuracy, training regime
# =============================================================================

set -e

echo "============================================"
echo "ABLATION V2 ({len(EXPERIMENTS)} experiments)"
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
# STEP 1: Data Generation (100k samples - largest needed)
# Some experiments use 20k/50k subsets via num_samples_to_load
# =============================================================================
echo ""
echo "[2/4] Submitting data generation (100k samples)..."

cat > slurm_logs/gen_ablation_v2_data.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=v2-dgen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/v2_datagen_%j.out
#SBATCH --error=slurm_logs/v2_datagen_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "Generating spatial dataset for ablation v2 (100k samples)..."
rm -rf asl_spatial_dataset_ablation_v2
python generate_clean_library.py asl_spatial_dataset_ablation_v2 \\
    --spatial --total_samples 100000 --spatial-chunk-size 500 --image-size 64

echo "Done: $(date)"
ls -lh asl_spatial_dataset_ablation_v2/ | head -5
echo "Total chunks: $(ls asl_spatial_dataset_ablation_v2/spatial_chunk_*.npz 2>/dev/null | wc -l)"
EOF

DATA_JOB=$(sbatch --parsable slurm_logs/gen_ablation_v2_data.slurm)
echo "  Data generation job: $DATA_JOB"

# =============================================================================
# STEP 2: Submit Training Jobs (all depend on data generation)
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
# {exp_name}: {exp['hypothesis'][:70]}
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

cat > {base_dir}/aggregate.slurm << 'AGGEOF'
#!/bin/bash
#SBATCH --job-name=v2-agg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output={base_dir}/aggregate_%j.out
#SBATCH --error={base_dir}/aggregate_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "AGGREGATING RESULTS"
echo "============================================"

python -c "
import json, csv, os
from pathlib import Path

base = Path('{base_dir}')
experiments = sorted([d for d in base.iterdir() if d.is_dir() and d.name[0].isdigit()])

rows = []
for exp_dir in experiments:
    row = {{'name': exp_dir.name}}

    # Load amplitude sensitivity
    amp_file = exp_dir / 'amplitude_sensitivity.json'
    if amp_file.exists():
        with open(amp_file) as f:
            amp = json.load(f)
        row['amp_ratio'] = amp.get('sensitivity_ratio', None)
        row['is_amp_sensitive'] = amp.get('is_sensitive', None)

    # Load validation metrics
    val_file = exp_dir / 'validation_results' / 'llm_analysis_report.json'
    if val_file.exists():
        with open(val_file) as f:
            val = json.load(f)
        for scenario, metrics in val.items():
            if 'CBF' in metrics:
                row['cbf_mae'] = metrics['CBF']['Neural_Net']['MAE']
                row['cbf_bias'] = metrics['CBF']['Neural_Net']['Bias']
                row['cbf_win_rate'] = metrics['CBF']['NN_vs_LS_Win_Rate']
                row['att_mae'] = metrics['ATT']['Neural_Net']['MAE']
                row['att_bias'] = metrics['ATT']['Neural_Net']['Bias']
                row['att_win_rate'] = metrics['ATT']['NN_vs_LS_Win_Rate']
                break

    rows.append(row)

# Write CSV
csv_path = base / 'ablation_v2_summary.csv'
fields = ['name', 'amp_ratio', 'is_amp_sensitive', 'cbf_mae', 'cbf_bias', 'cbf_win_rate', 'att_mae', 'att_bias', 'att_win_rate']
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f'Saved: {{csv_path}}')

# Print summary
print()
print('=' * 80)
print('ABLATION V2 RESULTS SUMMARY')
print('=' * 80)
print('%-30s %10s %10s %10s %10s' % ('Experiment', 'CBF MAE', 'ATT MAE', 'CBF Win%', 'ATT Win%'))
print('-' * 80)
for r in rows:
    cbf = '%10.2f' % r['cbf_mae'] if r.get('cbf_mae') else '%10s' % 'N/A'
    att = '%10.2f' % r['att_mae'] if r.get('att_mae') else '%10s' % 'N/A'
    cw = '%9.1f%%' % (r['cbf_win_rate']*100) if r.get('cbf_win_rate') else '%10s' % 'N/A'
    aw = '%9.1f%%' % (r['att_win_rate']*100) if r.get('att_win_rate') else '%10s' % 'N/A'
    print('%-30s %s %s %s %s' % (r['name'], cbf, att, cw, aw))
print()
print('Exp 09 reference: CBF MAE=0.49, ATT MAE=18.7, CBF Win=97.5%, ATT Win=96.8%')
"

echo "Done: $(date)"
AGGEOF

AGG_JOB=$(sbatch --parsable --dependency=afterany:$JOB_IDS {base_dir}/aggregate.slurm)
echo "  Aggregation -> Job $AGG_JOB"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================"
echo "ALL JOBS SUBMITTED"
echo "============================================"
echo "Data generation:  $DATA_JOB (100k samples, ~2-4 hours)"
echo "Training jobs:    {len(EXPERIMENTS)} experiments"
echo "Aggregation:      $AGG_JOB"
echo ""
echo "Monitor: squeue -u \\$USER"
echo "Results: {base_dir}/ablation_v2_summary.csv"
echo "============================================"
"""

    orchestrator_path = Path("run_ablation_v2.sh")
    with open(orchestrator_path, "w") as f:
        f.write(orchestrator)

    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"\nGenerated:")
    print(f"  - {len(EXPERIMENTS)} experiment configs in '{base_dir}/'")
    print(f"  - Orchestrator script: {orchestrator_path}")
    print()
    print("To run on HPC:")
    print(f"  sbatch {orchestrator_path}")
    print()
    print("Or to run setup only (no submission):")
    print(f"  python {Path(__file__).name}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

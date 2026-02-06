#!/bin/bash
#SBATCH --job-name=abl-v2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/ablation_v2_orchestrator_%j.out
#SBATCH --error=slurm_logs/ablation_v2_orchestrator_%j.err

# =============================================================================
# ABLATION STUDY V2: DOMAIN GAP & ATT IMPROVEMENTS (11 experiments)
#
# Baseline: Exp 09 (AmplitudeAware Optimized)
# Focus: Domain gap, ATT accuracy, training regime
# =============================================================================

set -e

echo "============================================"
echo "ABLATION V2 (11 experiments)"
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
python generate_clean_library.py asl_spatial_dataset_ablation_v2 \
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
echo "[3/4] Submitting 11 training experiments..."
echo "  (All depend on data job $DATA_JOB)"
echo ""

JOB_IDS=""

# 10_ExtendedDomainRand: Wider domain randomization ranges to better cover in-vivo variability 
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/10_ExtendedDomainRand/run.slurm)
echo "  10_ExtendedDomainRand -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 11_MoreData_50k: 2.5x more training data (50k vs 20k) reduces overfitting to synthetic 
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/11_MoreData_50k/run.slurm)
echo "  11_MoreData_50k -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 12_MoreData_100k: 5x more training data (100k vs 20k) further reduces overfitting
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/12_MoreData_100k/run.slurm)
echo "  12_MoreData_100k -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 13_AggressiveNoise: More realistic noise augmentation (motion, spikes, wider SNR) improves
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/13_AggressiveNoise/run.slurm)
echo "  13_AggressiveNoise -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 14_ATT_Rebalanced: Current att_scale=0.033 massively underweights ATT loss; rebalancing t
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/14_ATT_Rebalanced/run.slurm)
echo "  14_ATT_Rebalanced -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 15_HuberLoss: Huber loss is more robust to outliers than L1, may help ATT in difficu
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/15_HuberLoss/run.slurm)
echo "  15_HuberLoss -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 16_L2Loss: L2 loss penalizes large errors more heavily, may reduce worst-case ATT
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/16_L2Loss/run.slurm)
echo "  16_L2Loss -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 17_LargerModel: Doubling model capacity [64,128,256,512] may capture more complex spat
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/17_LargerModel/run.slurm)
echo "  17_LargerModel -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 18_LongerTraining: 100 epochs (vs 50) with higher patience may find better minima, especi
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/18_LongerTraining/run.slurm)
echo "  18_LongerTraining -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 19_Ensemble5: 5-member ensemble (vs 3) improves prediction quality via better averag
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/19_Ensemble5/run.slurm)
echo "  19_Ensemble5 -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# 20_BestCombo: Combine extended domain rand + 100k data + aggressive noise + ATT reba
JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB amplitude_ablation_v2/20_BestCombo/run.slurm)
echo "  20_BestCombo -> Job $JOB_ID"
[ -z "$JOB_IDS" ] && JOB_IDS="$JOB_ID" || JOB_IDS="$JOB_IDS:$JOB_ID"

# =============================================================================
# STEP 3: Aggregation Job
# =============================================================================
echo ""
echo "[4/4] Submitting aggregation job..."

cat > amplitude_ablation_v2/aggregate.slurm << 'AGGEOF'
#!/bin/bash
#SBATCH --job-name=v2-agg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=amplitude_ablation_v2/aggregate_%j.out
#SBATCH --error=amplitude_ablation_v2/aggregate_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "AGGREGATING RESULTS"
echo "============================================"

python -c "
import json, csv, os
from pathlib import Path

base = Path('amplitude_ablation_v2')
experiments = sorted([d for d in base.iterdir() if d.is_dir() and d.name[0].isdigit()])

rows = []
for exp_dir in experiments:
    row = {'name': exp_dir.name}

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
print(f'Saved: {csv_path}')

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

AGG_JOB=$(sbatch --parsable --dependency=afterany:$JOB_IDS amplitude_ablation_v2/aggregate.slurm)
echo "  Aggregation -> Job $AGG_JOB"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================"
echo "ALL JOBS SUBMITTED"
echo "============================================"
echo "Data generation:  $DATA_JOB (100k samples, ~2-4 hours)"
echo "Training jobs:    11 experiments"
echo "Aggregation:      $AGG_JOB"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: amplitude_ablation_v2/ablation_v2_summary.csv"
echo "============================================"

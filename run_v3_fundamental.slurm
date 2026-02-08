#!/bin/bash
#SBATCH --job-name=v3_fundamental
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=v3_experiments/slurm_%j.out
#SBATCH --error=v3_experiments/slurm_%j.err

# =============================================================================
# V3 FUNDAMENTAL LIMIT TEST
# =============================================================================
# One script. Three experiments. Answers: "Is the NN fundamentally limited?"
#
# Exp 1 (v3_bugfix):   All bugs fixed. Does correcting T1/att_scale/domain-rand fix CBF?
# Exp 2 (v3_capacity): 2x bigger model, 2x longer training. Is it underfitting?
# Exp 3 (v3_oracle):   No domain rand. NN and LS see identical physics. Kill shot.
#
# INTERPRETATION:
#   Exp 1 CBF win rate >50%  → Bugs were the problem. Ship it.
#   Exp 1 CBF win rate <30%  → Bugs aren't enough. Check Exp 2 and 3.
#   Exp 2 >> Exp 1           → Underfitting. Need more capacity.
#   Exp 2 ≈ Exp 1            → Not a capacity issue.
#   Exp 3 CBF win rate >50%  → NN CAN beat LS with matched physics. Domain rand hurts.
#   Exp 3 CBF win rate <30%  → NN is fundamentally limited for CBF. Pivot paper.
#
# Runtime: ~24-36 hours total (data gen ~2h, 3 trainings ~6-10h each, validation ~1h)
# =============================================================================

set -e  # Exit on any error

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse
cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "V3 FUNDAMENTAL LIMIT TEST"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

mkdir -p v3_experiments

# =============================================================================
# PHASE 1: GENERATE TRAINING DATA (~2 hours)
# =============================================================================
# Two datasets: one WITH domain randomization (Exp 1 & 2), one WITHOUT (Exp 3)

echo ""
echo "========== PHASE 1: DATA GENERATION =========="
echo "Started: $(date)"

# Dataset for Exp 1 & 2: domain randomization ON (alpha_BS1, T1_artery variation)
if [ ! -d "v3_spatial_dataset" ] || [ $(ls v3_spatial_dataset/spatial_chunk_*.npz 2>/dev/null | wc -l) -lt 10 ]; then
    echo "Generating v3_spatial_dataset (WITH domain randomization)..."
    python generate_clean_library.py v3_spatial_dataset \
        --spatial --total_samples 100000 --spatial-chunk-size 500 \
        --config config/v3_bugfix.yaml
    echo "Dataset 1 done: $(date)"
else
    echo "v3_spatial_dataset already exists, skipping."
fi

# Dataset for Exp 3: domain randomization OFF (fixed physics = oracle conditions)
if [ ! -d "v3_spatial_dataset_oracle" ] || [ $(ls v3_spatial_dataset_oracle/spatial_chunk_*.npz 2>/dev/null | wc -l) -lt 10 ]; then
    echo "Generating v3_spatial_dataset_oracle (NO domain randomization)..."
    python generate_clean_library.py v3_spatial_dataset_oracle \
        --spatial --total_samples 100000 --spatial-chunk-size 500
    echo "Dataset 2 done: $(date)"
else
    echo "v3_spatial_dataset_oracle already exists, skipping."
fi

echo "PHASE 1 COMPLETE: $(date)"

# =============================================================================
# PHASE 2: TRAIN THREE MODELS (sequential, ~6-10h each)
# =============================================================================

echo ""
echo "========== PHASE 2: TRAINING =========="

# --- Exp 1: All Bugs Fixed ---
echo ""
echo "--- EXP 1: BUGFIX BASELINE (SpatialASLNet, corrected params) ---"
echo "Started: $(date)"
mkdir -p v3_experiments/01_bugfix
python main.py config/v3_bugfix.yaml \
    --stage 2 \
    --output-dir v3_experiments/01_bugfix \
    --run-name v3_01_bugfix
echo "Exp 1 training done: $(date)"

# --- Exp 2: Bigger + Longer ---
echo ""
echo "--- EXP 2: CAPACITY TEST (larger model, 200 epochs) ---"
echo "Started: $(date)"
mkdir -p v3_experiments/02_capacity
python main.py config/v3_capacity.yaml \
    --stage 2 \
    --output-dir v3_experiments/02_capacity \
    --run-name v3_02_capacity
echo "Exp 2 training done: $(date)"

# --- Exp 3: Oracle ---
echo ""
echo "--- EXP 3: ORACLE (no domain rand, fixed physics) ---"
echo "Started: $(date)"
mkdir -p v3_experiments/03_oracle
python main.py config/v3_oracle.yaml \
    --stage 2 \
    --output-dir v3_experiments/03_oracle \
    --run-name v3_03_oracle
echo "Exp 3 training done: $(date)"

echo "PHASE 2 COMPLETE: $(date)"

# =============================================================================
# PHASE 3: CORRECTED VALIDATION (all 3 models, ~1h total)
# =============================================================================

echo ""
echo "========== PHASE 3: CORRECTED VALIDATION =========="

for exp in 01_bugfix 02_capacity 03_oracle; do
    echo ""
    echo "--- Validating $exp ---"
    echo "Started: $(date)"
    python run_corrected_validation.py \
        --run_dir v3_experiments/$exp \
        --output_dir v3_experiments/$exp/corrected_validation
    echo "$exp validation done: $(date)"
done

echo "PHASE 3 COMPLETE: $(date)"

# =============================================================================
# PHASE 4: SUMMARY
# =============================================================================

echo ""
echo "============================================"
echo "V3 FUNDAMENTAL LIMIT TEST COMPLETE"
echo "Finished: $(date)"
echo "============================================"
echo ""
echo "RESULTS LOCATION:"
echo "  Exp 1 (bugfix):   v3_experiments/01_bugfix/corrected_validation/"
echo "  Exp 2 (capacity): v3_experiments/02_capacity/corrected_validation/"
echo "  Exp 3 (oracle):   v3_experiments/03_oracle/corrected_validation/"
echo ""

# Print summary table from JSONs
python -c "
import json, os, sys

exps = [
    ('01_bugfix',   'Bugfix Baseline'),
    ('02_capacity', 'Larger+Longer'),
    ('03_oracle',   'Oracle (no DR)'),
]

print('='*80)
print(f'{\"Experiment\":<22} {\"CBF MAE\":>8} {\"CBF Win%\":>9} {\"ATT MAE\":>8} {\"ATT Win%\":>9} {\"Verdict\":>12}')
print('-'*80)

for exp_dir, exp_name in exps:
    json_path = f'v3_experiments/{exp_dir}/corrected_validation/corrected_baseline_comparison.json'
    if not os.path.exists(json_path):
        print(f'{exp_name:<22} --- results not found ---')
        continue
    with open(json_path) as f:
        data = json.load(f)
    # Find SNR=10 results (primary comparison point)
    snr10 = None
    for snr_result in data.get('multi_snr_results', []):
        if snr_result.get('snr') == 10:
            snr10 = snr_result
            break
    if snr10 is None and 'snr_10' in data:
        snr10 = data['snr_10']
    if snr10 is None:
        # Try top-level
        snr10 = data

    cbf_mae = snr10.get('nn_cbf_mae', snr10.get('cbf_mae_nn', '?'))
    cbf_win = snr10.get('cbf_win_rate', snr10.get('nn_cbf_win_rate', '?'))
    att_mae = snr10.get('nn_att_mae', snr10.get('att_mae_nn', '?'))
    att_win = snr10.get('att_win_rate', snr10.get('nn_att_win_rate', '?'))

    if isinstance(cbf_win, (int, float)):
        verdict = 'PASS' if cbf_win > 50 else ('MARGINAL' if cbf_win > 35 else 'FAIL')
        cbf_win_str = f'{cbf_win:.1f}%'
    else:
        verdict = '?'
        cbf_win_str = str(cbf_win)

    cbf_mae_str = f'{cbf_mae:.2f}' if isinstance(cbf_mae, (int, float)) else str(cbf_mae)
    att_mae_str = f'{att_mae:.1f}' if isinstance(att_mae, (int, float)) else str(att_mae)
    att_win_str = f'{att_win:.1f}%' if isinstance(att_win, (int, float)) else str(att_win)

    print(f'{exp_name:<22} {cbf_mae_str:>8} {cbf_win_str:>9} {att_mae_str:>8} {att_win_str:>9} {verdict:>12}')

print('='*80)
print()
print('DECISION:')
print('  CBF Win% > 50%  at SNR=10 → NN has a real advantage')
print('  CBF Win% 35-50% at SNR=10 → Marginal, check low-SNR')
print('  CBF Win% < 35%  at SNR=10 → NN fundamentally limited for CBF')
print()
print('  If Exp 3 (Oracle) > 50% but Exp 1 < 50% → Domain rand hurts')
print('  If Exp 2 >> Exp 1 → Underfitting, need more capacity')
print('  If Exp 3 < 35%   → Pivot to ATT-only or speed story')
" 2>/dev/null || echo "(Summary script failed - check JSONs manually)"

echo ""
echo "Done. $(date)"

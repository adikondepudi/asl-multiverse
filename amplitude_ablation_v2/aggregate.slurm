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

#!/bin/bash
#SBATCH --job-name=17_Large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=amplitude_ablation_v2/17_LargerModel/slurm_%j.out
#SBATCH --error=amplitude_ablation_v2/17_LargerModel/slurm_%j.err

source /cm/shared/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate asl_multiverse

cd $SLURM_SUBMIT_DIR

echo "============================================"
echo "EXPERIMENT: 17_LargerModel"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# --- TRAINING ---
echo ""
echo "--- STAGE 2: SPATIAL TRAINING ---"
python main.py amplitude_ablation_v2/17_LargerModel/config.yaml --stage 2 --output-dir amplitude_ablation_v2/17_LargerModel

# --- VALIDATION ---
echo ""
echo "--- VALIDATION ---"
python validate.py --run_dir amplitude_ablation_v2/17_LargerModel --output_dir amplitude_ablation_v2/17_LargerModel/validation_results

# --- AMPLITUDE SENSITIVITY TEST ---
echo ""
echo "--- AMPLITUDE SENSITIVITY TEST ---"
python -c "
import torch
import json
import sys
import yaml
sys.path.insert(0, '.')

with open('amplitude_ablation_v2/17_LargerModel/config.yaml') as f:
    cfg = yaml.safe_load(f)

model_class = cfg['training']['model_class_name']
print(f'Testing model: {model_class}')

from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
model = AmplitudeAwareSpatialASLNet(
    n_plds=len(cfg['data']['pld_values']),
    features=cfg['training']['hidden_sizes'],
    use_film_at_bottleneck=cfg['training'].get('use_film_at_bottleneck', True),
    use_film_at_decoder=cfg['training'].get('use_film_at_decoder', True),
    use_amplitude_output_modulation=cfg['training'].get('use_amplitude_output_modulation', True),
)

import glob
model_files = sorted(glob.glob('amplitude_ablation_v2/17_LargerModel/trained_models/ensemble_model_*.pt'))
if model_files:
    state_dict = torch.load(model_files[0], map_location='cpu')
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    print(f'Loaded: {model_files[0]}')
else:
    print('WARNING: No trained model found, testing untrained')

model.eval()

torch.manual_seed(42)
base_input = torch.randn(4, 12, 64, 64) * 0.1

scales = [0.1, 1.0, 10.0]
results = {}

with torch.no_grad():
    for scale in scales:
        scaled_input = base_input * scale
        output = model(scaled_input)
        cbf = output[0] if isinstance(output, tuple) else output[:, 0:1]
        cbf_mean = cbf.mean().item()
        results[f'scale_{scale}'] = cbf_mean
        print(f'  Scale {scale:5.1f}x -> CBF mean: {cbf_mean:12.7f}')

cbf_01 = abs(results['scale_0.1'])
cbf_10 = abs(results['scale_10.0'])
ratio = cbf_10 / max(cbf_01, 1e-9)

print(f'')
print(f'  Amplitude sensitivity ratio (10x/0.1x): {ratio:.2f}')
print(f'  Is amplitude sensitive: {ratio > 5.0}')

with open('amplitude_ablation_v2/17_LargerModel/amplitude_sensitivity.json', 'w') as f:
    json.dump({
        'model_class': model_class,
        'scales': scales,
        'cbf_predictions': results,
        'sensitivity_ratio': ratio,
        'is_sensitive': ratio > 5.0
    }, f, indent=2)
print(f'Saved to amplitude_ablation_v2/17_LargerModel/amplitude_sensitivity.json')
"

echo ""
echo "============================================"
echo "EXPERIMENT COMPLETE: 17_LargerModel"
echo "Finished: $(date)"
echo "============================================"

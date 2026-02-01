# Production Model Training Guide

## Quick Start

```bash
# Full pipeline (data generation + training + validation)
sbatch run_production_pipeline.sh

# If you already have data, skip generation
sbatch run_production_pipeline.sh --train-only
```

**Estimated time**: ~56 hours total (8h data + 48h training)

---

## Overview

This guide explains how to train a production-quality ASL parameter estimation model that beats least-squares fitting by **4-5x on CBF** and **15-17x on ATT**.

### What You'll Get

After training, you'll have:
- **5-model ensemble** for robust predictions
- **Win rate >80%** vs least-squares on CBF
- **Win rate >95%** vs least-squares on ATT
- Models saved in `production_model_v1/trained_models/`

---

## Pipeline Components

### 1. Data Generation (`generate_production_data.sh`)

Generates 500,000 synthetic 64×64 spatial phantoms with:
- Realistic tissue segmentation (GM, WM, CSF)
- Pathological variations (stroke, tumor, elderly)
- CBF range: 0-150 ml/100g/min
- ATT range: 500-3000 ms

```bash
# Run standalone
sbatch generate_production_data.sh

# Output: asl_spatial_dataset_500k/
```

**Time**: ~8 hours on 32 CPUs

### 2. Model Training (`train_production.sh`)

Trains a 5-model SpatialASLNet ensemble with:
- U-Net architecture (4-level encoder-decoder)
- L1 loss (no physics constraint)
- Global scale normalization
- Rician noise augmentation
- Domain randomization

```bash
# Run standalone (requires dataset)
sbatch train_production.sh

# Output: production_model_v1/trained_models/
```

**Time**: ~48 hours on A100 GPU

### 3. Validation (automatic)

Runs comprehensive validation comparing NN to least-squares:
- 50 test phantoms at SNR=10
- Metrics: MAE, RMSE, Bias, R², Win Rate

**Output**: `production_model_v1/validation_results/llm_analysis_report.json`

---

## Configuration Files

### `config/production_v1.yaml` (Recommended)

Full production config with all optimizations:
- 500k samples, 200 epochs, 5 ensembles
- Best settings from ablation studies

### `config/production_quick.yaml` (For Testing)

Reduced scale for quick iteration:
- 100k samples, 50 epochs, 3 ensembles
- ~4-6 hours instead of ~48 hours

### `config/production_dual_encoder.yaml` (Experimental)

Alternative Y-Net architecture:
- Separate PCASL/VSASL encoder streams
- Theoretical: better modality-specific features
- Unproven - use for comparison only

---

## Critical Settings (DO NOT CHANGE)

These settings are **essential** for beating least-squares:

```yaml
data:
  normalization_mode: global_scale    # CRITICAL - preserves CBF information
  global_scale_factor: 10.0
  noise_type: rician                  # MRI-correct physics

training:
  dc_weight: 0.0                      # NO physics constraint (hurts performance)
  loss_type: l1                       # MAE loss
```

### Why These Matter

1. **`normalization_mode: global_scale`**
   - Z-score normalization DESTROYS CBF information (mathematically proven)
   - Global scale preserves signal amplitude (proportional to CBF)

2. **`dc_weight: 0.0`**
   - Physics-informed loss actually HURTS performance
   - Removing it gave 35% improvement in ablation studies

3. **`noise_type: rician`**
   - Rician noise is the correct model for MRI magnitude images
   - Gaussian noise doesn't match real data characteristics

---

## Resource Requirements

### Data Generation
- **Partition**: CPU
- **CPUs**: 32
- **Memory**: 128 GB
- **Time**: 8 hours

### Training
- **Partition**: GPU
- **GPU**: 1x A100 (or V100)
- **CPUs**: 16
- **Memory**: 128 GB
- **Time**: 48 hours

### Total Storage
- Dataset: ~200 GB (500k samples)
- Models: ~500 MB (5 ensembles)
- Logs: ~1 GB

---

## Monitoring Progress

```bash
# Check job status
squeue -u $USER

# Watch training progress
tail -f slurm_logs/prod_train_*.out

# Check GPU utilization
ssh <gpu-node> nvidia-smi

# W&B dashboard
# https://wandb.ai/adikondepudi/asl-production-v1
```

---

## Expected Results

Based on ablation studies, the production model should achieve:

| Metric | CBF | ATT |
|--------|-----|-----|
| NN MAE | 4-5 | 20-25 |
| LS MAE | 22-23 | 380+ |
| Win Rate | 80-85% | 95-97% |
| R² | 0.90+ | 0.99+ |
| Improvement | 4-5× | 15-17× |

---

## Troubleshooting

### "Dataset not found"
```bash
# Check if data generation completed
ls -la asl_spatial_dataset_500k/
# Should have ~1000 spatial_chunk_*.npz files
```

### "CUDA out of memory"
```yaml
# Reduce batch size in config
training:
  batch_size: 16  # Down from 32
```

### "Training loss not decreasing"
- Check normalization mode is `global_scale`
- Verify dataset was generated correctly
- Check W&B for training curves

### "Low win rate on validation"
- Ensure `dc_weight: 0.0`
- Check that normalization mode matches training
- Verify ensemble averaging is working

---

## Advanced Usage

### Custom Training

```bash
# Train with custom config
python main.py config/my_config.yaml \
    --stage 2 \
    --output-dir my_experiment \
    --offline_dataset_path asl_spatial_dataset_500k
```

### Custom Validation

```bash
# Validate existing model
python validate.py \
    --run_dir production_model_v1 \
    --output_dir validation_results
```

### Using Trained Model

```python
import torch
from spatial_asl_network import SpatialASLNet
import json

# Load model
model = SpatialASLNet(n_plds=6)
state = torch.load('production_model_v1/trained_models/ensemble_model_0.pt')
model.load_state_dict(state['model_state_dict'])
model.eval()

# Load normalization stats
with open('production_model_v1/norm_stats.json') as f:
    norm_stats = json.load(f)

# Inference
with torch.no_grad():
    # input: (1, 12, H, W) - 6 PCASL + 6 VSASL PLDs
    cbf_norm, att_norm, _, _ = model(input_tensor)

    # Denormalize
    cbf = cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att = att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
```

---

## File Structure After Training

```
production_model_v1/
├── config.yaml                    # Copy of training config
├── norm_stats.json               # Normalization statistics
├── research_config.json          # Full config dump
├── trained_models/
│   ├── ensemble_model_0.pt       # Model 1 of 5
│   ├── ensemble_model_1.pt
│   ├── ensemble_model_2.pt
│   ├── ensemble_model_3.pt
│   └── ensemble_model_4.pt
├── validation_results/
│   ├── llm_analysis_report.json  # Machine-readable metrics
│   ├── llm_analysis_report.md    # Human-readable report
│   └── interactive_plot_data.json
├── slurm.out                     # Training logs
└── slurm.err                     # Error logs
```

---

## Next Steps After Training

1. **Validate on real data**: Test on clinical ASL acquisitions
2. **Multi-SNR testing**: Evaluate at SNR=3, 5, 10, 20
3. **Cross-validation**: Train multiple folds for uncertainty
4. **Deployment**: Export for clinical use

---

## References

- **Analysis Report**: `results/EXPERIMENT_ANALYSIS_REPORT.md`
- **Codebase Guide**: `CLAUDE.md`
- **Best Ablation Result**: `results/optimization_ablation_v1/06_No_DC/`

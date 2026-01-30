# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASL Multiverse is a neural network framework for Arterial Spin Labeling (ASL) MRI parameter estimation. It trains models to predict Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) from combined PCASL and VSASL signals.

**Goal**: Beat least-squares fitting methods in accuracy, robustness to noise, and computational speed.

## Critical Bug Fixes (2026-01-29)

### NORM_STATS_INDICES Bug (FIXED)
**Location**: `feature_registry.py`, lines 48-57

The `NORM_STATS_INDICES` dictionary had incorrect indices that didn't match the actual output order of `compute_feature_vector()`. This caused **wrong normalization statistics** to be applied during training, leading to models that output near-constant predictions.

**Old (BUGGY)**:
```python
NORM_STATS_INDICES = {
    'mean': [0, 2],  # WRONG - assumed interleaved layout
    'std': [1, 3],
    ...
}
```

**New (FIXED)**:
```python
NORM_STATS_INDICES = {
    'mean': [0, 1],  # CORRECT - grouped layout (mean_p, mean_v)
    'std': [2, 3],   # (std_p, std_v)
    ...
}
```

**Impact**: All models trained before this fix used incorrect feature normalization. **You must regenerate norm_stats and retrain models.**

### Other Fixes
- `validate.py`: Hardcoded physics parameters replaced with `self.params.*`
- `validate.py`: Fixed LS self-comparison bug (`abs(x-x)` → `abs(x-truth)`)
- `validate.py`: Fixed spatial validation dtype mismatch (float64 → float32)
- `validate.py`: Added proper ground truth tracking for LS comparison

### Spatial Model Mean Prediction Bug (FIXED - 2026-01-30)
**Location**: `spatial_asl_network.py`

The spatial U-Net model (`SpatialASLNet`) was learning to predict the dataset mean instead of learning the input-output mapping.

**Root Cause**: Mismatch between 1D and spatial training pipelines:
- **1D training**: Targets are **normalized** to z-scores before training. Model outputs unbounded predictions.
- **Spatial training (buggy)**: Targets were **raw** values (CBF: 20-100, ATT: 500-3000). Model had constrained outputs (`softplus*100`, `sigmoid*3000`) that started near the dataset mean and stayed there.

**Old (BUGGY)**:
```python
# SpatialASLNet.forward() - constrained outputs with initialization bias
cbf_map = F.softplus(out[:, 0:1, :, :]) * 100.0   # At init: ~69.3 (near mean!)
att_map = 3000.0 * torch.sigmoid(out[:, 1:2, :, :])  # At init: 1500 (near mean!)
```

**New (FIXED)**:
```python
# SpatialASLNet.forward() - unbounded normalized outputs
cbf_norm = out[:, 0:1, :, :]  # At init: ~0 (normalized mean)
att_norm = out[:, 1:2, :, :]  # Denormalize at inference
```

**Changes Made**:
1. `SpatialASLNet.forward()`: Now outputs **unbounded normalized predictions** (z-scores)
2. `MaskedSpatialLoss`: Now accepts `norm_stats` and **normalizes targets** before computing loss
3. `main.py`: Passes `norm_stats` to `MaskedSpatialLoss`
4. `validate.py`, `predict_on_invivo.py`, `diagnose_model.py`: **Denormalize** predictions before use

**Impact**: Spatial models trained before this fix produce constant/mean predictions. **Retrain spatial models with the fixed code.**

## Common Commands

### Data Generation
```bash
# Generate 1D voxel-wise clean signal library
python generate_clean_library.py <output_dir> --total_samples 10000000 --chunk_size 25000

# Generate 2D spatial data (phantoms)
python generate_clean_library.py <output_dir> --mode spatial --total_samples 100000 --size 64
```

### Training
```bash
# Stage 1: Self-supervised denoising pre-training (OPTIONAL)
python main.py config/v5_stage1_pretrain.yaml --stage 1 --output-dir ./results/stage1

# Stage 2: Supervised regression training
python main.py config/v5_stage2_MoE_finetune.yaml --stage 2 --output-dir ./results/stage2

# Stage 2 with pre-trained encoder from Stage 1
python main.py config/v5_stage2_MoE_finetune.yaml --stage 2 --output-dir ./results/stage2 --load-weights-from ./results/stage1
```

### Validation
```bash
# Run validation on a trained model directory
python validate.py <run_dir> --output-dir validation_results
```

### Dashboard
```bash
streamlit run asl_interactive_dashboard.py
```

## Architecture

### Two-Stage Training Pipeline
1. **Stage 1 (Denoising)**: Self-supervised pre-training where the encoder learns to reconstruct clean signal shapes from noisy inputs
2. **Stage 2 (Regression)**: Supervised training to predict CBF/ATT with uncertainty estimation

### Core Modules

- **`main.py`**: Entry point. Loads YAML config, creates dataloaders, instantiates trainer
- **`enhanced_asl_network.py`**: Neural network definitions
  - `DisentangledASLNet`: Main model class supporting denoising and regression modes
  - `PhysicsInformedASLProcessor`: Dual-stream Conv1D encoder with FiLM conditioning
  - `MLPOnlyEncoder`: Ablation control (no Conv1D)
  - `UncertaintyHead`: Outputs mean + bounded log_var for NLL loss
- **`spatial_asl_network.py`**: U-Net based `SpatialASLNet` for 2D spatial processing
  - `KineticModel`: Differentiable forward model for data consistency loss
- **`asl_trainer.py`**: `EnhancedASLTrainer` handles ensemble training, GPU-resident noise injection, online feature computation
- **`enhanced_simulation.py`**: `RealisticASLSimulator` generates physically accurate ASL signals with various noise models
- **`feature_registry.py`**: **Single source of truth** for feature dimensions, config validation, and default physics parameters
- **`noise_engine.py`**: Modular noise injection (thermal, physio, drift, spikes)
- **`utils.py`**: `ParallelStreamingStatsCalculator` for normalization stats, `process_signals_dynamic` for inference preprocessing

### Data Flow
1. Clean signals generated offline (`generate_clean_library.py`) → stored as `.npz` chunks
2. Training: signals loaded to GPU → noise injected dynamically → features computed on-the-fly
3. Signal processing: raw signals → per-curve normalization (shape vectors) + engineered scalar features → concatenated input

### Configuration
YAML configs in `config/` are flattened into `ResearchConfig` dataclass. Key sections:
- `training`: model architecture, learning rate, loss weights
- `data`: dataset path, PLDs, `active_features`, `data_noise_components`
- `simulation`: physics parameters (T1_artery, T_tau, alpha values)
- `noise_config`: SNR range, physio/drift/spike parameters

## Best Practices for Beating Least Squares

### 1. Use Constrained ATT Range
**Problem**: Default ATT range (500-4000ms) exceeds max PLD (3000ms). Signals with ATT > 3000ms have zero/minimal signal at all measured PLDs, making estimation ill-posed.

**Solution**: Constrain ATT range to match PLD coverage:
```yaml
simulation:
  # In enhanced_simulation.py PhysiologicalVariation:
  att_range: [500.0, 3000.0]  # Keep within PLD range
```

### 2. Use MAE Loss (Not Pure NLL)
Pure NLL loss allows the model to minimize loss by predicting high uncertainty instead of accurate values.

**Recommended**:
```yaml
training:
  loss_mode: "mae_only"  # Forces accurate predictions
  # OR
  loss_mode: "mae_nll"   # Balanced: accuracy + uncertainty
  mae_weight: 1.0
  nll_weight: 0.1
```

### 3. Sufficient Training Data
- Minimum: 1M samples for 1D models
- Recommended: 5-10M samples
- Ensure diverse parameter coverage (CBF, ATT, T1 variations)

### 4. Match Training and Validation Noise
Training uses `NoiseInjector` while validation uses different noise functions. Ensure consistency:
- Use same noise type (gaussian/rician)
- Use same noise components
- Use similar SNR ranges

### 5. Verify Normalization Statistics
After any code changes, regenerate norm_stats:
```python
# In main.py, stats are auto-calculated from training data
# Check that scalar_features_mean/std have correct dimensionality
```

## Key Concepts

### Feature System
Active features are configurable via `active_features` list: `['mean', 'std', 'peak', 'ttp', 'com', 't1_artery', 'z_coord']`. The `FeatureRegistry` validates configs and computes dimensions dynamically.

**Feature order in norm_stats**:
`[mean_p, mean_v, std_p, std_v, ttp_p, ttp_v, com_p, com_v, peak_p, peak_v, wsum_p, wsum_v]`

### Noise Components
Configurable via `data_noise_components`: `['thermal', 'physio', 'drift', 'spikes']`. Noise is applied during training, not pre-computed.

### Noise Type
Configurable via `noise_type` in data section:
- `'gaussian'` (default): Standard Gaussian additive noise
- `'rician'`: Rician noise - correct MRI physics for magnitude images

### Normalization Mode
Configurable via `normalization_mode` in data section:
- `'per_curve'` (default): Z-score normalize each curve individually. Creates SNR-invariant "shape vectors".
- `'global_scale'`: Multiply signals by `global_scale_factor` (default: 10.0). Preserves absolute magnitude.

### Physics Parameters
- PLDs: Post-labeling delays in ms (default: 500-3000 in 500ms steps)
- T1_artery: Arterial blood T1 (~1850ms)
- T_tau: Label duration (1800ms)
- alpha_PCASL/VSASL: Labeling efficiencies

### Output Targets
Models predict **normalized** CBF and ATT. Denormalization uses `norm_stats.json` saved during training:
```python
cbf_pred = cbf_norm * y_std_cbf + y_mean_cbf
att_pred = att_norm * y_std_att + y_mean_att
```

## Diagnostics

### Quick Checks
- **`diagnose_model.py <run_dir>`**: Quick check of model predictions
- **`validate.py <run_dir>`**: Full validation with LS comparison

### Signs of Training Failure
1. **Near-constant predictions**: Model outputs similar values regardless of input
   - Check: NN prediction std << true value std
   - Cause: Usually normalization bug or mode collapse from NLL loss

2. **Very high bias**: Predictions systematically offset from truth
   - Check: Mean error >> expected noise level
   - Cause: Denormalization bug or training data mismatch

3. **R² near zero or negative**: Model no better than predicting the mean
   - Cause: Features not informative or training didn't converge

### Debugging Steps
1. Check `norm_stats.json` - verify reasonable mean/std values
2. Run `diagnose_model.py` - see raw prediction distributions
3. Check training loss curves in wandb
4. Verify input dimensions match between training and inference

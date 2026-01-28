# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASL Multiverse is a neural network framework for Arterial Spin Labeling (ASL) MRI parameter estimation. It trains models to predict Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) from combined PCASL and VSASL signals, outperforming conventional least-squares fitting methods.

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
# Stage 1: Self-supervised denoising pre-training
python main.py config/v5_stage1_pretrain.yaml --stage 1 --output-dir ./results/stage1

# Stage 2: Supervised regression training (optionally load Stage 1 encoder)
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
2. **Stage 2 (Regression)**: Supervised training to predict CBF/ATT with uncertainty estimation (NLL loss)

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

## Key Concepts

### Feature System
Active features are configurable via `active_features` list: `['mean', 'std', 'peak', 'ttp', 'com', 't1_artery', 'z_coord']`. The `FeatureRegistry` validates configs and computes dimensions dynamically.

### Noise Components
Configurable via `data_noise_components`: `['thermal', 'physio', 'drift', 'spikes']`. Noise is applied during training, not pre-computed.

### Noise Type (NEW)
Configurable via `noise_type` in data section:
- `'gaussian'` (default): Standard Gaussian additive noise (legacy behavior)
- `'rician'`: Rician noise - correct MRI physics for magnitude images. Creates positive bias at low SNR matching real MRI data. Recommended for in-vivo applications.

### Normalization Mode (NEW)
Configurable via `normalization_mode` in data section:
- `'per_curve'` (default): Z-score normalize each curve individually. Creates SNR-invariant "shape vectors".
- `'global_scale'`: Multiply signals by `global_scale_factor` (default: 10.0). Preserves absolute magnitude information, similar to IVIM-NET's S(b)/S(b=0) approach. Use when signal magnitude carries information.

Example config for Rician + Global Scaling (recommended for in-vivo):
```yaml
data:
  noise_type: "rician"
  normalization_mode: "global_scale"
  global_scale_factor: 10.0
```

### Physics Parameters
- PLDs: Post-labeling delays in ms (default: 500-3000 in 500ms steps)
- T1_artery: Arterial blood T1 (~1850ms)
- T_tau: Label duration (1800ms)
- alpha_PCASL/VSASL: Labeling efficiencies

### Output Targets
Models predict normalized CBF and ATT. Denormalization uses `norm_stats.json` saved during training.

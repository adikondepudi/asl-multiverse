# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASL Multiverse is a neural network framework for Arterial Spin Labeling (ASL) MRI parameter estimation. It trains models to predict Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) from combined PCASL and VSASL signals.

**Goal**: Beat least-squares fitting methods in accuracy, robustness to noise, and computational speed.

## Major Architecture Update (2026-01-30)

This codebase has been significantly enhanced based on comprehensive methodological analysis. Key changes:

### 1. Normalization Fix: Z-Score Destroys CBF Information

**Problem**: Per-pixel temporal z-score normalization mathematically removes CBF information.

The signal model is: `x ≈ CBF · M0 · k(ATT, T1)`

After z-score: `z = (x - mean) / std`, the `CBF · M0` term **cancels out completely**.

**Solution**: Use `normalization_mode: global_scale` in config:
```yaml
data:
  normalization_mode: "global_scale"  # REQUIRED for CBF estimation
  global_scale_factor: 1.0
```

### 2. Dual-Encoder Y-Net Architecture

New `DualEncoderSpatialASLNet` processes PCASL and VSASL through separate encoder streams:

- **Stream A (PCASL)**: Extracts transit-dependent features (ATT-sensitive)
- **Stream B (VSASL)**: Extracts transit-independent perfusion features
- **Fusion**: At bottleneck layer with dual skip connections

This allows the network to learn modality-specific representations before fusion.

### 3. Physics-Informed Loss Function

Enhanced `KineticModel` with:
- Domain randomization support (variable T1, alpha, tau)
- `compute_physics_loss()` method for data consistency
- Combined loss: `L = L_supervised + λ_phys * L_physics`

### 4. Bias-Reduced Training

New `BiasReducedLoss` class addressing variance collapse:
- MAE + NLL balanced loss
- Variance matching penalty
- Probabilistic mode with KL regularization
- Cramér-Rao Bound estimation for theoretical minimum variance

### 5. Domain Randomization

Physics parameters are now sampled per-batch during training:
- Blood T1: 1550-2150 ms (hematocrit variations)
- α_PCASL: 0.75-0.95 (labeling efficiency)
- α_VSASL: 0.40-0.70 (VSS efficiency)
- τ (label duration): ±10% variation

### 6. Rician Noise Model

Proper MRI physics for magnitude images:
```python
S_noisy = sqrt((S + N_real)² + N_imag²)
```
Creates positive bias at low SNR matching real MRI data.

### 7. Comprehensive Validation Metrics

New `validation_metrics.py` with:
- Bland-Altman analysis (bias + limits of agreement)
- Intraclass Correlation Coefficient (ICC)
- Concordance Correlation Coefficient (CCC)
- Structural Similarity Index (SSIM)
- Win Rate vs baseline methods

## Common Commands

```bash
# Generate spatial training data
python generate_clean_library.py <output_dir> --spatial --total_samples 100000

# Train spatial model (standard U-Net)
python main.py config/spatial_mae_loss.yaml --stage 2 --output-dir ./results/run

# Train with dual-encoder architecture
# Edit config: model_class_name: "DualEncoderSpatialASLNet"
python main.py config/spatial_mae_loss.yaml --stage 2 --output-dir ./results/dual_encoder

# Validate model
python validate.py --run_dir <run_dir> --output_dir validation_results

# Quick diagnostic
python diagnose_model.py <run_dir>
```

## Architecture

### Key Files

| File | Purpose |
|------|---------|
| `spatial_asl_network.py` | SpatialASLNet, DualEncoderSpatialASLNet, MaskedSpatialLoss, BiasReducedLoss, KineticModel |
| `main.py` | Training entry point |
| `asl_trainer.py` | EnhancedASLTrainer with noise injection |
| `validate.py` | Validation with LS comparison |
| `validation_metrics.py` | Comprehensive metrics (Bland-Altman, ICC, CCC, SSIM) |
| `generate_clean_library.py` | Data generation (spatial and 1D) |
| `enhanced_simulation.py` | SpatialPhantomGenerator with domain randomization |
| `noise_engine.py` | NoiseInjector with Rician noise support |
| `feature_registry.py` | Feature dimensions, norm_stats indices |

### Model Classes

1. **SpatialASLNet**: Standard U-Net with 4-level encoder-decoder
   - Input: `(B, 2*N_plds, H, W)` - PCASL + VSASL concatenated
   - Output: Normalized (z-score) predictions

2. **DualEncoderSpatialASLNet**: Y-Net with separate PCASL/VSASL streams
   - Better sensor fusion for complementary modalities
   - Optional constrained output (Softplus/Sigmoid)

### Loss Functions

1. **MaskedSpatialLoss**: Standard supervised loss with variance penalty
2. **BiasReducedLoss**: Addresses variance collapse with NLL + variance matching

## Data Flow

1. `SpatialDataset` loads phantoms, applies M0 scaling (*100)
2. Trainer's `_process_batch_on_gpu`:
   - Applies Rician noise (if configured)
   - Applies normalization (`global_scale` or `per_curve`)
3. Model outputs NORMALIZED predictions
4. Loss computed in normalized space
5. At inference: denormalize using `norm_stats`

## Configuration

Key parameters in `config/spatial_mae_loss.yaml`:

```yaml
training:
  model_class_name: "SpatialASLNet"  # or "DualEncoderSpatialASLNet"
  loss_type: "l1"
  dc_weight: 0.1          # Physics-informed loss weight
  variance_weight: 0.1    # Anti-collapse penalty

data:
  noise_type: "rician"    # MRI-correct physics
  normalization_mode: "global_scale"  # CRITICAL for CBF!

simulation:
  domain_randomization:
    enabled: true
    T1_artery_range: [1550.0, 2150.0]
    alpha_PCASL_range: [0.75, 0.95]
    alpha_VSASL_range: [0.40, 0.70]
```

## Physics Parameters

- PLDs: [500, 1000, 1500, 2000, 2500, 3000] ms
- T1_artery: 1850 ms (default), 1550-2150 ms (randomized)
- T_tau (label duration): 1800 ms
- α_PCASL: 0.85 (default), 0.75-0.95 (randomized)
- α_VSASL: 0.56 (default), 0.40-0.70 (randomized)

## Troubleshooting

### CBF estimation fails but ATT works
**Cause**: Using `per_curve` normalization mode (z-score)
**Fix**: Set `normalization_mode: "global_scale"` in config

### Model predicts near-constant values (variance collapse)
**Cause**: L1/L2 loss encourages mean prediction at low SNR
**Fix**: Increase `variance_weight` or use `BiasReducedLoss`

### Train/eval mismatch (loss drops but validation fails)
**Cause**: BatchNorm running statistics
**Fix**: Already addressed - SpatialASLNet uses GroupNorm

### Systematic bias on validation
**Cause**: Domain gap from fixed physics parameters
**Fix**: Enable domain randomization in config

## References

Key literature for understanding this codebase:

1. **MULTIVERSE ASL**: Xu et al. (2025) - Joint PCASL/VSASL protocol
2. **Bias-Reduced NNs**: Mao et al. (2023) - Variance collapse solution
3. **IVIM-NET**: Kaandorp et al. (2021) - Physics-informed architecture
4. **General Kinetic Model**: Buxton et al. - PCASL signal equation
5. **ASL Consensus**: Alsop et al. (2015) - Standard implementation

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ASL Multiverse** is a neural network framework for Arterial Spin Labeling (ASL) MRI parameter estimation. It trains models to predict **Cerebral Blood Flow (CBF)** and **Arterial Transit Time (ATT)** from combined PCASL and VSASL signals.

**Primary Goal**: Beat least-squares fitting methods in:
- Accuracy (lower MAE, bias)
- Robustness to noise (especially low SNR)
- Computational speed

**Secondary Goals**:
- Quantify prediction uncertainty
- Generalize across acquisition parameters via domain randomization
- Support both voxel-wise (1D) and spatial (2D) estimation

---

## Key Research Findings (Feb 2025)

### Critical Discovery: Spatial Models Dramatically Outperform Voxel-Wise

| Metric | Spatial (2D) | Voxel-Wise (1D) | Improvement |
|--------|--------------|-----------------|-------------|
| CBF Win Rate vs LS | **84.2%** | 4.0% | 21× better |
| CBF MAE | 4.0 ml/100g/min | 18.0 ml/100g/min | 4.5× better |
| ATT Win Rate vs LS | **95.8%** | 36.0% | 2.7× better |
| ATT MAE | 21.8 ms | 48.6 ms | 2.2× better |

**Recommendation**: Always use spatial models for CBF estimation. Voxel-wise models suffer from variance collapse (predict ~constant values).

---

## Ablation Study Results

### Study 1: Amplitude Ablation (amplitude_ablation_v1/)

Tests whether AmplitudeAwareSpatialASLNet fixes CBF estimation by preserving amplitude information that GroupNorm destroys.

#### Amplitude Sensitivity Results

We scaled inputs by [0.1×, 1×, 10×] and measured if CBF predictions changed. Sensitive models should show ratio >> 1.

| Exp | Config | Sensitivity Ratio | Sensitive? |
|-----|--------|-------------------|------------|
| 00 | Baseline SpatialASL | **1.00** | NO |
| 01 | PerCurve Norm | **1.00** | NO |
| 02 | AmpAware Full (FiLM + OutputMod) | **257.95** | YES |
| 03 | OutputMod Only (no FiLM) | **98.53** | YES |
| 04 | FiLM Only (no OutputMod) | **1.15** | NO |
| 05 | Bottleneck FiLM Only | **0.78** | NO |
| 06 | Full + Physics (dc=0.1) | **92.51** | YES |
| 07 | Full + Physics (dc=0.3) | **113.91** | YES |
| 08 | Full + DomainRand | **169.28** | YES |
| 09 | Optimized | **88.90** | YES |

#### Critical Finding: Output Modulation is Essential

**Exp 03 vs 04 is the key comparison:**
- Exp 03 (Output Modulation ONLY): **98.53** - WORKS
- Exp 04 (FiLM ONLY): **1.15** - FAILS

**FiLM conditioning alone does NOT preserve amplitude.** The output modulation that directly scales CBF by extracted amplitude is the critical component.

#### Validation Results (SNR=10)

| Model | CBF MAE | CBF Win Rate | ATT MAE | ATT Win Rate |
|-------|---------|--------------|---------|--------------|
| Baseline SpatialASL | 4.01 | **84.2%** | 21.81 | **95.8%** |
| PerCurve Norm | 4.49 | 83.5% | 28.09 | 94.9% |

PerCurve normalization increases ATT bias from -0.43 to -12.6 ms.

---

### Study 2: HPC Ablation (hpc_ablation_jobs/)

Tests voxel-wise model configurations across 4 scenarios.

#### CBF Results: Fundamental Failure

| Exp | Config | Scenario C Win Rate | Scenario D Win Rate |
|-----|--------|---------------------|---------------------|
| 01 | Baseline | 1.2% | 4.0% |
| 03 | Feature Full | 1.0% | 2.4% |
| 05 | Size Small | 0.4% | 2.2% |
| 10 | Robust Small | 1.0% | 3.8% |

**All voxel-wise configs fail for CBF** (<5% win rate). The NN predicts nearly constant values (bias ≈ MAE ≈ 17-19 ml/100g/min).

#### ATT Results: More Promising

| Exp | Config | Scenario C Win Rate | Scenario D Win Rate |
|-----|--------|---------------------|---------------------|
| 01 | Baseline | **28.6%** | 33.2% |
| 05 | Size Small | 26.4% | **36.0%** |
| 08 | Robust NoConv | 23.8% | **35.0%** |
| 10 | Robust Small | 22.8% | **34.7%** |

Small models + robust training work best for ATT at low SNR.

#### Root Cause: Variance Collapse

Voxel-wise CBF bias analysis:
- NN Bias: +17 to +19 ml/100g/min (predicts near-constant mean)
- LS Bias: ~0 ml/100g/min

The NN gives up learning signal and predicts the prior distribution mean.

---

## Critical Design Decisions

### 1. Use Spatial Models for CBF (NOT Voxel-Wise)

```
Spatial:    CBF Win Rate = 84%,  MAE = 4.0 ml/100g/min
Voxel-wise: CBF Win Rate = 4%,   MAE = 18.0 ml/100g/min
```

Spatial context provides crucial denoising through local averaging and structural priors.

### 2. Normalization: Z-Score Destroys CBF

```
Signal Model: x ≈ CBF · M0 · k(ATT, T1)
Z-score:      z = (x - mean) / std
Result:       CBF · M0 cancels out completely!
```

**Always use `normalization_mode: "global_scale"`**

### 3. Output Modulation is Critical for Amplitude-Aware

FiLM alone (ratio=1.15) vs OutputMod alone (ratio=98.53):
```yaml
training:
  use_amplitude_output_modulation: true  # REQUIRED - this is the critical component
  use_film_at_bottleneck: true           # Supplementary, not sufficient alone
  use_film_at_decoder: true              # Supplementary, not sufficient alone
```

### 4. GroupNorm over BatchNorm

Use **GroupNorm** throughout spatial models:
- Train/eval consistency (no running statistics)
- BatchNorm with small batches causes mismatch

---

## Architecture Overview

### Spatial Models (2D) - USE FOR CBF

| Model | File | Performance |
|-------|------|-------------|
| **SpatialASLNet** | `spatial_asl_network.py` | 84% CBF win rate, 96% ATT win rate |
| **DualEncoderSpatialASLNet** | `spatial_asl_network.py` | Y-Net with separate PCASL/VSASL streams |
| **AmplitudeAwareSpatialASLNet** | `amplitude_aware_spatial_network.py` | Preserves amplitude via output modulation |

### Voxel-Wise Models (1D) - ATT ONLY

| Model | File | Performance |
|-------|------|-------------|
| **DisentangledASLNet** | `enhanced_asl_network.py` | <5% CBF win rate, 20-36% ATT win rate |

### Loss Functions

| Loss | Purpose |
|------|---------|
| **MaskedSpatialLoss** | Supervised loss with variance penalty |
| **BiasReducedLoss** | Addresses variance collapse (MAE + NLL) |
| **AmplitudeAwareLoss** | For amplitude-aware architecture |

---

## File Structure

```
asl-multiverse/
├── Core Models
│   ├── spatial_asl_network.py        # SpatialASLNet, DualEncoder (USE THESE)
│   ├── amplitude_aware_spatial_network.py  # AmplitudeAwareSpatialASLNet
│   └── enhanced_asl_network.py       # DisentangledASLNet (voxel-wise, ATT only)
│
├── Training & Validation
│   ├── main.py                       # Training entry point
│   ├── asl_trainer.py                # EnhancedASLTrainer
│   ├── validate.py                   # Validation with LS comparison
│   ├── validate_spatial.py           # Spatial validation
│   └── validation_metrics.py         # Bland-Altman, ICC, CCC, SSIM
│
├── Data Generation & Simulation
│   ├── generate_clean_library.py     # Data generation (spatial/1D)
│   ├── enhanced_simulation.py        # SpatialPhantomGenerator
│   ├── asl_simulation.py             # JIT-compiled signal generation
│   └── noise_engine.py               # NoiseInjector (Rician noise)
│
├── Configuration & Utilities
│   ├── feature_registry.py           # Feature dims, norm_stats indices
│   ├── utils.py                      # Normalization, helpers
│   └── multiverse_functions.py       # Least-squares baseline
│
├── Experiment Results
│   ├── amplitude_ablation_v1/        # 10 spatial experiments (COMPLETED)
│   │   └── Key finding: Output modulation critical (98× vs 1.15×)
│   ├── hpc_ablation_jobs/            # 10 voxel-wise experiments (COMPLETED)
│   │   └── Key finding: Voxel-wise fails for CBF (<5% win rate)
│   └── production_model_v1/          # Production models
│
└── Config Files
    └── config/
        ├── spatial_mae_loss.yaml     # Primary spatial config
        └── production_v1.yaml        # Production training
```

---

## Common Commands

```bash
# Generate spatial training data
python generate_clean_library.py <output_dir> --spatial --total_samples 100000

# Train spatial model (RECOMMENDED for CBF)
python main.py config/spatial_mae_loss.yaml --stage 2 --output-dir ./results/run

# Train amplitude-aware model
python main.py config/amplitude_aware_spatial.yaml --stage 2 --output-dir ./results/amp_aware

# Validate model
python validate.py --run_dir <run_dir> --output_dir validation_results

# Spatial validation
python validate_spatial.py <run_dir>

# Quick diagnostic
python diagnose_model.py <run_dir>
```

---

## Recommended Configuration

```yaml
training:
  model_class_name: "SpatialASLNet"  # Or "AmplitudeAwareSpatialASLNet"
  hidden_sizes: [32, 64, 128, 256]
  loss_type: "l1"
  dc_weight: 0.0           # Physics loss (ablation showed minimal benefit)
  variance_weight: 0.1     # Anti-collapse penalty
  learning_rate: 0.0001
  n_ensembles: 3
  batch_size: 32

  # For AmplitudeAwareSpatialASLNet ONLY:
  use_film_at_bottleneck: true
  use_film_at_decoder: true
  use_amplitude_output_modulation: true  # CRITICAL - must be true

data:
  normalization_mode: "global_scale"  # NEVER use per_curve for CBF
  global_scale_factor: 10.0
  noise_type: "rician"
  pld_values: [500, 1000, 1500, 2000, 2500, 3000]

simulation:
  T1_artery: 1850.0
  T_tau: 1800.0
  alpha_PCASL: 0.85
  alpha_VSASL: 0.56
  domain_randomization:
    enabled: true
    T1_artery_range: [1550.0, 2150.0]
    alpha_PCASL_range: [0.75, 0.95]
    alpha_VSASL_range: [0.40, 0.70]
    alpha_BS1_range: [0.85, 1.0]  # CRITICAL for real-world robustness

noise_config:
  snr_range: [2.0, 25.0]
  physio_amp_range: [0.03, 0.15]
```

---

## Physics Parameters

| Parameter | Default | Randomized Range | Unit |
|-----------|---------|------------------|------|
| PLDs | [500, 1000, 1500, 2000, 2500, 3000] | - | ms |
| T1_artery | 1850 | 1550-2150 | ms |
| T_tau (label duration) | 1800 | ±10% | ms |
| α_PCASL | 0.85 | 0.75-0.95 | - |
| α_VSASL | 0.56 | 0.40-0.70 | - |
| α_BS1 | 1.0 | 0.85-1.0 | - |

**Background Suppression (α_BS1)**: Critical for real-world robustness. In-vivo ASL uses background suppression pulses that attenuate the signal:
- PCASL: effective α = α_PCASL × (α_BS1)^4 (4 BS pulses)
- VSASL: effective α = α_VSASL × (α_BS1)^3 (3 BS pulses)
- α_BS1 = 1.0: No BS (synthetic data default)
- α_BS1 ≈ 0.85-0.95: Typical in-vivo BS

**Tissue Ranges** (SpatialPhantomGenerator):
- Gray matter: CBF 50-70 ml/100g/min, ATT 1000-1600 ms
- White matter: CBF 18-28 ml/100g/min, ATT 1200-1800 ms

---

## Troubleshooting

### CBF estimation fails but ATT works
**Cause**: Using `per_curve` normalization or voxel-wise model
**Fix**: Use `normalization_mode: "global_scale"` AND spatial models

### Model predicts near-constant CBF values
**Cause**: Variance collapse (common in voxel-wise models)
**Fix**: Use spatial models. If using voxel-wise, increase `variance_weight` or use `BiasReducedLoss`

### Voxel-wise CBF win rate <5%
**Cause**: Fundamental limitation - single voxels lack spatial context for denoising
**Fix**: Use spatial models (SpatialASLNet achieves 84% win rate)

### AmplitudeAware model not sensitive to amplitude
**Cause**: Output modulation disabled
**Fix**: Set `use_amplitude_output_modulation: true` (ablation proved this is essential, not FiLM)

### Train/eval mismatch
**Cause**: BatchNorm running statistics
**Fix**: Models already use GroupNorm

### Systematic bias on validation
**Cause**: Domain gap from fixed physics parameters
**Fix**: Enable domain randomization

---

## Validation Metrics

| Metric | Purpose | Good Value |
|--------|---------|------------|
| MAE | Mean absolute error | CBF <5, ATT <25 |
| Bias | Systematic error | Near 0 |
| Win Rate | % NN beats LS | >50% |
| R² | Variance explained | >0.9 |
| ICC | Reliability | >0.9 |
| CCC | Accuracy + precision | >0.9 |

---

## Key Innovations

### 1. Spatial > Voxel-Wise (21× Better CBF Win Rate)
- Spatial context enables local averaging for denoising
- Structural priors from tissue boundaries
- Neighboring voxels share physics parameters

### 2. Output Modulation for Amplitude (98× vs 1.15×)
- Extract amplitude BEFORE GroupNorm destroys it
- Direct scaling of CBF prediction by amplitude
- FiLM alone is insufficient

### 3. Domain Randomization
- Physics parameters sampled per-batch
- Prevents overfitting to fixed acquisition parameters
- Maintains amplitude sensitivity (169× with domain rand)

### 4. Rician Noise Model
```python
S_noisy = sqrt((S + N_real)² + N_imag²)
```
MRI-correct physics with positive bias at low SNR.

---

## Future Directions

1. **Complete validation for Exp 02-09** - Amplitude-aware models need validation runs
2. **Hybrid approach** - Spatial for CBF, potentially voxel-wise for ATT
3. **Larger spatial context** - Current 64×64 patches may be too small
4. **Multi-scale fusion** - Combine voxel features with spatial context

---

## References

1. **MULTIVERSE ASL**: Xu et al. (2025) - Joint PCASL/VSASL protocol
2. **Bias-Reduced NNs**: Mao et al. (2023) - Variance collapse solution
3. **IVIM-NET**: Kaandorp et al. (2021) - Physics-informed architecture
4. **General Kinetic Model**: Buxton et al. - PCASL signal equation
5. **ASL Consensus**: Alsop et al. (2015) - Standard implementation
6. **FiLM**: Perez et al. (2018) - Feature-wise Linear Modulation

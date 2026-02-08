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
| CBF Win Rate vs LS | **85.8%** | 4.0% | 21x better |
| CBF MAE | 3.47 ml/100g/min | 18.0 ml/100g/min | 5.2x better |
| ATT Win Rate vs LS | **96.1%** | 36.0% | 2.7x better |
| ATT MAE | 21.4 ms | 48.6 ms | 2.3x better |

**Errata (2026-02-08)**: Values corrected from original (84.2%/4.0/95.8%/21.8) to match `amplitude_ablation_v1/00_Baseline_SpatialASL/validation_results/llm_analysis_report.json`. Win rates are inflated due to broken LS baseline.

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
| 02 | AmpAware Full (FiLM + OutputMod) | **79.87** | YES |
| 03 | OutputMod Only (no FiLM) | **90.32** | YES |
| 04 | FiLM Only (no OutputMod) | **40.56** | YES |
| 05 | Bottleneck FiLM Only | **1.05** | NO |
| 06 | Full + Physics (dc=0.1) | **18.01** | YES |
| 07 | Full + Physics (dc=0.3) | **110.17** | YES |
| 08 | Full + DomainRand | **93.51** | YES |
| 09 | Optimized | **376.18** | YES |

#### Key Finding: Both FiLM and Output Modulation Preserve Amplitude

**Exp 03 vs 04 comparison (corrected values from JSON):**
- Exp 03 (Output Modulation ONLY): **90.32** - WORKS
- Exp 04 (FiLM ONLY): **40.56** - ALSO WORKS

Both mechanisms independently preserve amplitude sensitivity. Output modulation gives ~2x higher ratios in this config, but FiLM alone is also effective. The best results come from combining both (Exp 02: 79.87x, Exp 09: 376.18x).

**Note**: Original CLAUDE.md reported Exp 04 = 1.15, which was incorrect. The v1 experiments had a training bug where SLURM scripts searched for `spatial_model_*.pt` but models saved as `ensemble_model_*.pt`, causing the original sensitivity tests to run on untrained models. The `rerun_amplitude_ablation_validation.py` script fixed this by loading the correct checkpoints, producing the corrected values above.

#### Validation Results (SNR=10)

| Model | CBF MAE | CBF Win Rate | ATT MAE | ATT Win Rate |
|-------|---------|--------------|---------|--------------|
| Baseline SpatialASL | 3.47 | **85.8%** | 21.37 | **96.1%** |
| PerCurve Norm | 4.49 | 83.5% | 28.09 | 94.9% |

**Source**: `amplitude_ablation_v1/00_Baseline_SpatialASL/validation_results/llm_analysis_report.json`

PerCurve normalization increases ATT bias from -0.43 to -12.6 ms.

**WARNING**: These win rates are measured against a broken LS baseline (LS CBF MAE=23.1, LS ATT MAE=383.8). The LS fitter uses `alpha_BS1=1.0` and `T1_artery=1850`, producing catastrophically high errors. Expect win rates to drop significantly once LS is corrected.

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
Spatial:    CBF Win Rate = 86%,  MAE = 3.5 ml/100g/min
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

### 3. Both FiLM and Output Modulation Preserve Amplitude

FiLM alone (ratio=40.56) vs OutputMod alone (ratio=90.32) -- both work, best combined (376.18x for Exp 09):
```yaml
training:
  use_amplitude_output_modulation: true  # Effective alone (90x), best combined
  use_film_at_bottleneck: true           # Effective alone (41x), best combined
  use_film_at_decoder: true              # Effective alone (41x), best combined
```
**Note**: Exp 02 (Full FiLM + OutputMod) sensitivity = 79.87x, lower than Exp 03 (OutputMod only, 90.32x). The combined mechanism does not simply add. Exp 09 (Optimized) achieves 376.18x through additional config tuning.

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
| ~~AmplitudeAwareLoss~~ | **Dead code** — defined in `amplitude_aware_spatial_network.py` but never instantiated by `asl_trainer.py`. Training always uses `MaskedSpatialLoss`. |

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
│   │   └── Key finding: OutputMod (90x) and FiLM (41x) preserve amplitude
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
  T1_artery: 1650.0  # 3T consensus (Alsop 2015); was 1850 in older code
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
| T1_artery | 1650 (3T consensus) | 1550-2150 | ms |
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
**Cause**: Neither FiLM nor output modulation enabled
**Fix**: Enable `use_amplitude_output_modulation: true` and/or FiLM flags. Both independently preserve amplitude; Exp 09 (Optimized) gives best results (376x).

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

### 2. Amplitude-Aware Mechanisms (FiLM 41x + OutputMod 90x, Optimized 376x)
- Extract amplitude BEFORE GroupNorm destroys it
- Both FiLM and output modulation independently preserve amplitude
- Exp 09 (Optimized config) achieves highest ratio at 376x

### 3. Domain Randomization
- Physics parameters sampled per-batch
- Prevents overfitting to fixed acquisition parameters
- Maintains amplitude sensitivity (93.5× with domain rand)

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

## Known Issues & Errata

### V1 Training Bug (Resolved)
Original v1 SLURM scripts searched for `spatial_model_*.pt` but models saved as `ensemble_model_*.pt`. This caused the original amplitude sensitivity tests to run on untrained/randomly initialized models. The `rerun_amplitude_ablation_validation.py` script fixed this by loading the correct checkpoints, but CLAUDE.md was not updated until Feb 2026.

### LS Baseline Issues (In Progress)
- **In-vivo LS uses `alpha_BS1=1.0`** but real Philips data has background suppression (~0.93 per pulse). This causes LS to over-predict signal by ~34% (PCASL) / ~24% (VSASL), leading to catastrophically low CBF estimates (8-14 vs expected 40-60).
- **T1_artery was 1850ms** in code; ASL consensus (Alsop 2015) recommends **1650ms** at 3T.
- **ATT upper bound of 6000ms** allows optimization topology trap where solver pushes ATT to non-physiological values to compensate for amplitude mismatch.
- **Win rates against broken LS are inflated** -- expect ~97% to drop significantly once LS is corrected.

### V2 Experiment Failures (2026-02-08)
4 of 11 v2 experiments failed or are incomplete:
- **Exp 15 (HuberLoss)**: FAILED - data loading hang, no models produced
- **Exp 17 (LargerModel)**: FAILED - data loading hang, no models produced
- **Exp 18 (LongerTraining)**: PARTIAL - 89/100 epochs completed, models exist but no validation
- **Exp 20 (BestCombo)**: PARTIAL - 51/100 epochs completed, models exist but no validation

### V2 MAE Unit Reporting
V2 validation uses `validate.py` which denormalizes predictions (lines 746-747) before computing metrics. CBF MAE values (e.g., 0.44 ml/100g/min for Exp 14) are in physical units. Denormalization formula: `CBF_physical = CBF_normalized * y_std_cbf + y_mean_cbf` where `y_std_cbf ~ 23.08` and `y_mean_cbf ~ 59.98` (from norm_stats.json).

### Amplitude Sensitivity Value Discrepancies (2026-02-08)
CLAUDE.md previously reported values that do not match the current amplitude_sensitivity.json files on disk:
- Exp 02: was 257.95, actual JSON = **79.87** (3.2x overstatement)
- Exp 06: was 92.51, actual JSON = **18.01** (5.1x overstatement)
- Exp 07: was 113.91, actual JSON = **110.17** (3.4% off)

The `rerun_amplitude_ablation_validation.py` script adds a `used_trained_model: true` flag, but NO current JSON files have this flag, suggesting the rerun either was never executed or the files were overwritten. Values in this CLAUDE.md have been corrected to match the JSON files on disk as of 2026-02-08.

---

## Known Bugs (Severity-Rated)

### CRITICAL

**1. att_scale=0.033 Legacy Bug**
All 10 v1 experiments and 9 of 11 v2 experiments use `att_scale: 0.033` (legacy from unnormalized voxel-wise targets). With z-score normalized spatial targets, this should be 1.0. ATT loss is effectively weighted at only 3.3% of CBF loss, causing the model to under-optimize ATT predictions. Only v2 Exp 14 (ATT_Rebalanced) and Exp 20 (BestCombo) use the correct value of 1.0. Exp 14 with att_scale=1.0 achieves the best ATT MAE (15.35 ms vs 18-21 ms for others).
- **Files**: All `config.yaml` files in `amplitude_ablation_v1/*/` and most in `amplitude_ablation_v2/*/`
- **Fix**: Set `att_scale: 1.0` in all future experiments

**2. Rician Noise Implementation Status (RESOLVED in NoiseInjector, NOT in SpatialNoiseEngine training path)**
`NoiseInjector` (noise_engine.py L108-149) now correctly implements Control/Label pair Rician noise. However, `SpatialNoiseEngine.simulate_realistic_acquisition` (L524+) has a separate correct implementation that is NOT used in the standard training pipeline. The training pipeline uses `NoiseInjector`, which now has the correct implementation.
- **Status**: The CLAUDE.md line reference "L106-118 applies Rician noise directly to difference signals" is OUTDATED. The current code at those lines implements the correct approach.
- **Remaining issue**: `SpatialNoiseEngine.simulate_realistic_acquisition` is dead code for training purposes.

### HIGH

**3. Domain Randomization Silently Disabled When dc_weight=0.0**
Domain randomization parameters are passed to `KineticModel` (spatial_asl_network.py), which is only called during DC (physics consistency) loss computation. When `dc_weight=0.0` (the default in all experiments), the kinetic model forward pass is never called (guarded at spatial_asl_network.py:777: `if self.dc_weight > 0`), so domain randomization has NO effect on training. The data generation script (`generate_clean_library.py`) uses fixed `ASLParameters` without domain randomization.
- **Impact**: ALL experiments with dc_weight=0 trained on fixed physics parameters only
- **Files**: `asl_trainer.py:145-161`, `spatial_asl_network.py:777`
- **Fix**: Move domain randomization to data augmentation (apply during data loading or in the training loop directly on input signals)

**4. T1_artery=1850 in ALL Experiment Configs**
All v1 and v2 experiments use `T1_artery: 1850.0`. The ASL consensus paper (Alsop et al. 2015) recommends 1650ms at 3T. This introduces a systematic bias in signal generation and LS fitting.
- **Files**: All `config.yaml` files
- **Fix**: Use `T1_artery: 1650.0` in future experiments; note that existing results were internally consistent (both NN and LS used the same value)

### MEDIUM

**5. Ensemble Diversity: All Members Get Same Data Order**
All ensemble members train on the same data in the same order. Without different data shuffling, random augmentation, or dropout, ensemble members may converge to similar solutions, reducing the benefit of ensembling for uncertainty estimation.
- **Files**: `asl_trainer.py` training loop
- **Fix**: Use different random seeds for data shuffling per ensemble member

**6. log_var Hardcoded at -5.0 (Uncertainty is Placeholder)**
The spatial models output `log_var_cbf` and `log_var_att` channels, initialized with bias=-5.0 (spatial_asl_network.py:500). However, `MaskedSpatialLoss.forward()` only receives `pred_cbf` and `pred_att` (asl_trainer.py:468), never the log_var outputs. The log_var heads are never trained through gradient descent when using MaskedSpatialLoss. They would only be trained via `BiasReducedLoss` (which includes NLL), but that is not the default loss.
- **Impact**: Uncertainty estimates from the model are meaningless (always exp(-5) ~ 0.007)
- **Files**: `spatial_asl_network.py:498-500`, `asl_trainer.py:468`
- **Fix**: Either pass log_var to loss function for NLL training, or remove log_var heads to reduce confusion

### LOW

**7. AmplitudeAwareLoss Defined but Never Instantiated (Dead Code)**
`AmplitudeAwareLoss` is defined in `amplitude_aware_spatial_network.py` but `asl_trainer.py` always uses `MaskedSpatialLoss`. This class is unreachable dead code.
- **Files**: `amplitude_aware_spatial_network.py`
- **Fix**: Remove or document as experimental/unused

**8. ~~VSASL T_sat_vs Parameter Ignored~~ (RESOLVED)**
Previously reported as unused, but `_generate_vsasl_signal_jit` (asl_simulation.py:28-29) DOES use `T_sat_vs`: when `ATT > T_sat_vs`, it applies a saturation recovery factor `SIB = 1 - exp(-(ATT - T_sat_vs) / T1_artery)`. This correctly models VSASL behavior for delayed arrival.
- **Status**: Not a bug. Code is correct.

---

## References

1. **MULTIVERSE ASL**: Xu et al. (2025) - Joint PCASL/VSASL protocol
2. **Bias-Reduced NNs**: Mao et al. (2023) - Variance collapse solution
3. **IVIM-NET**: Kaandorp et al. (2021) - Physics-informed architecture
4. **General Kinetic Model**: Buxton et al. - PCASL signal equation
5. **ASL Consensus**: Alsop et al. (2015) - Standard implementation
6. **FiLM**: Perez et al. (2018) - Feature-wise Linear Modulation

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ASL Multiverse** is a neural network framework for Arterial Spin Labeling (ASL) MRI parameter estimation. It trains models to predict **Cerebral Blood Flow (CBF)** and **Arterial Transit Time (ATT)** from combined PCASL and VSASL signals.

**Primary Goal**: Match or modestly beat least-squares (LS) fitting methods in:
- Accuracy (lower MAE, bias) -- achieved only with enhanced architecture at low-moderate SNR
- Robustness to noise (especially low SNR) -- NN advantage strongest here
- Computational speed -- NN is orders of magnitude faster

**Secondary Goals**:
- Quantify prediction uncertainty (currently placeholder -- log_var not trained)
- Generalize across acquisition parameters via domain randomization
- Support both voxel-wise (1D) and spatial (2D) estimation

---

## Key Research Findings (Corrected Feb 2026)

### Win Rate Collapse After LS Correction

Original results used a **broken LS baseline** (`alpha_BS1=1.0`, `T1_artery=1850`, single-start optimizer, ATT bound 6000ms) producing LS CBF MAE=23.1 and LS ATT MAE=383.8. Corrected LS (`alpha_BS1=0.93`, `T1_artery=1650`, multi-start, tighter ATT bounds) dramatically changes the picture:

| Model | Old CBF Win Rate | Corrected CBF Win Rate (@SNR=10) | Change |
|-------|-----------------|----------------------------------|--------|
| Exp 00 (Baseline SpatialASLNet) | 85.8% | **20.8%** | COLLAPSED |
| Exp 14 (AmplitudeAware + ATT Rebal) | 97.8% | **57.8%** | Significant drop |
| Production v1 | ~97% | **2.9%** (SNR=3) | CATASTROPHIC |

### Corrected Win Rates by SNR -- Best Model (Exp 14)

| SNR | CBF Win Rate | ATT Win Rate |
|-----|-------------|-------------|
| 3 | 59.8% | 67.9% |
| 5 | 59.3% | 66.3% |
| 10 | 57.8% | 65.3% |
| 15 | 56.5% | 63.4% |
| 25 | 54.2% | 60.7% |

### Corrected Win Rates by SNR -- Baseline (Exp 00)

| SNR | CBF Win Rate | ATT Win Rate |
|-----|-------------|-------------|
| 3 | 40.3% | 69.8% |
| 5 | 33.8% | 66.1% |
| 10 | 20.8% | 55.0% |
| 15 | 14.4% | 46.5% |
| 25 | 10.1% | 36.0% |

**Key takeaways**:
- Baseline SpatialASLNet **loses** to corrected LS for CBF at all SNR >= 5
- Enhanced architecture (Exp 14) has modest but consistent CBF advantage (~55-60%)
- ATT advantage is more robust (60-68% for Exp 14)
- NN advantage is largest at low SNR and diminishes at high SNR

### Spatial vs Voxel-Wise (Still Valid)

| Metric | Spatial (2D) | Voxel-Wise (1D) |
|--------|--------------|-----------------|
| CBF MAE (Exp 00, broken LS) | 3.47 ml/100g/min | 18.0 ml/100g/min |
| ATT MAE | 21.4 ms | 48.6 ms |

Spatial models dramatically outperform voxel-wise. Voxel-wise models suffer from complete variance collapse (predict ~constant CBF values).

---

## Amplitude Awareness: Architecture Helps, Mechanism Unproven

### What was claimed
AmplitudeAwareSpatialASLNet "preserves amplitude information" via FiLM and output modulation, achieving sensitivity ratios up to 376x.

### What is actually true (amplitude_audit_report.md)

1. **Sensitivity test is INVALID**: All ratios (v1 and v2) computed using random Gaussian noise inputs, not ASL signals. The ratios are scientifically meaningless.

2. **Exp 10 disproves the causal link**: Sensitivity ratio = 0.36 (NOT sensitive), yet CBF MAE = 0.478 -- matching models with 90+ ratios. The "amplitude awareness" mechanism is not what drives accuracy.

3. **Super-linearity problem**: Amplitude-aware models have CBF linearity slope ~1.9 (should be 1.0). R^2 vs identity is NEGATIVE (-0.52 to -0.80). At CBF=150, models predict ~300 (hitting clamp).

4. **Baseline has complete variance collapse**: SpatialASLNet predicts ~55 for ALL CBF values (slope=0.026). The validation MAE of 3.47 looks acceptable only because the validation CBF distribution is narrow.

5. **Architecture capacity is the likely explanation**: The extra FiLM layers and output modulation pathways increase model capacity, enabling better fitting regardless of whether amplitude information is actually preserved.

### Practical Recommendation
Keep `AmplitudeAwareSpatialASLNet` for its performance (CBF MAE 0.80 vs 2.64 at SNR=10), but do NOT claim "amplitude awareness" as the mechanism.

---

## Architecture Overview

### Spatial Models (2D)

| Model | File | Notes |
|-------|------|-------|
| **SpatialASLNet** | `models/spatial_asl_network.py` | Baseline U-Net; suffers from variance collapse |
| **DualEncoderSpatialASLNet** | `models/spatial_asl_network.py` | Y-Net with separate PCASL/VSASL streams |
| **AmplitudeAwareSpatialASLNet** | `models/amplitude_aware_spatial_network.py` | Best performer; mechanism unproven |

### Voxel-Wise Models (1D) -- ATT ONLY

| Model | File | Notes |
|-------|------|-------|
| **DisentangledASLNet** | `models/enhanced_asl_network.py` | <5% CBF win rate, 20-36% ATT win rate |

### Loss Functions

| Loss | Purpose | Status |
|------|---------|--------|
| **MaskedSpatialLoss** | Supervised loss with variance penalty | Default, used in all experiments |
| **BiasReducedLoss** | MAE + NLL for variance collapse | Defined but not used by default |
| ~~AmplitudeAwareLoss~~ | Dead code | Never instantiated by trainer |

---

## File Structure

```
asl-multiverse/
├── main.py                          # Training entry point (stays at root)
├── CLAUDE.md, README.md, requirements.txt
│
├── models/                          # Neural network architectures
│   ├── spatial_asl_network.py       # SpatialASLNet, DualEncoder, KineticModel
│   ├── amplitude_aware_spatial_network.py  # AmplitudeAwareSpatialASLNet
│   └── enhanced_asl_network.py      # DisentangledASLNet (voxel-wise)
│
├── training/                        # Training loop and utilities
│   └── asl_trainer.py               # EnhancedASLTrainer, FastTensorDataLoader
│
├── validation/                      # Validation scripts and metrics
│   ├── validate.py                  # Validation with LS comparison
│   ├── validate_spatial.py          # Spatial model validation
│   └── validation_metrics.py        # Bland-Altman, ICC, CCC, SSIM
│
├── simulation/                      # Signal simulation and data generation
│   ├── asl_simulation.py            # JIT-compiled ASL signal generation
│   ├── enhanced_simulation.py       # SpatialPhantomGenerator, RealisticASLSimulator
│   ├── noise_engine.py              # NoiseInjector (Rician noise)
│   └── generate_clean_library.py    # Training data generation
│
├── utils/                           # Utilities and feature management
│   ├── helpers.py                   # Normalization, signal processing
│   └── feature_registry.py          # Feature dims, norm_stats indices
│
├── baselines/                       # Least-squares fitting methods
│   ├── multiverse_functions.py      # Combined PCASL+VSASL LS fitter
│   ├── pcasl_functions.py           # PCASL-only LS fitter
│   ├── vsasl_functions.py           # VSASL-only LS fitter
│   └── basil_baseline.py            # FSL BASIL wrapper
│
├── inference/                       # In-vivo prediction scripts
│   ├── predict_on_invivo.py         # Voxel-wise in-vivo inference
│   └── predict_spatial_invivo.py    # Spatial in-vivo inference
│
├── config/                          # YAML experiment configs
├── docs/                            # Key analysis documents
│   ├── decision_gates.md            # Publication decision framework
│   ├── amplitude_audit_report.md    # Amplitude sensitivity audit
│   └── publication_readiness_assessment.md
│
├── archive/                         # Dead scripts, old docs, shell scripts
│   ├── scripts/                     # Setup, debug, compare, analysis scripts
│   ├── shell/                       # .sh and .slurm files
│   ├── tests/                       # Old test files
│   └── docs/                        # Obsolete status reports
│
├── amplitude_ablation_v1/           # 10 spatial experiments (COMPLETED)
├── amplitude_ablation_v2/           # 11 experiments, 4 incomplete
├── hpc_ablation_jobs/               # 10 voxel-wise experiments (COMPLETED)
└── data/                            # In-vivo data (invivo_processed_npy/, invivo_validated/)
```

---

## Common Commands

```bash
# Generate spatial training data
python -m simulation.generate_clean_library <output_dir> --spatial --total_samples 100000

# Train spatial model
python main.py config/v5_baseline_spatial.yaml --stage 2 --output-dir ./results/run

# Train amplitude-aware model
python main.py config/v5_amplitude_aware.yaml --stage 2 --output-dir ./results/amp_aware

# Validate model
python -m validation.validate --run_dir <run_dir> --output_dir validation_results

# Spatial validation
python -m validation.validate_spatial <run_dir>
```

---

## Recommended Configuration

```yaml
training:
  model_class_name: "AmplitudeAwareSpatialASLNet"  # Best performer
  hidden_sizes: [32, 64, 128, 256]
  loss_type: "l1"
  dc_weight: 0.0           # Physics loss (ablation showed minimal benefit)
  variance_weight: 0.1     # Anti-collapse penalty
  learning_rate: 0.0001
  n_ensembles: 3
  batch_size: 32
  att_scale: 1.0            # CRITICAL: was 0.033 in v1 (bug)

  # AmplitudeAwareSpatialASLNet config:
  use_film_at_bottleneck: true
  use_film_at_decoder: true
  use_amplitude_output_modulation: true

data:
  normalization_mode: "global_scale"  # NEVER use per_curve for CBF
  global_scale_factor: 10.0
  noise_type: "rician"
  pld_values: [500, 1000, 1500, 2000, 2500, 3000]

simulation:
  T1_artery: 1650.0  # 3T consensus (Alsop 2015); all existing experiments used 1850
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

## Critical Design Decisions

### 1. Use Spatial Models (NOT Voxel-Wise)
Voxel-wise models catastrophically fail for CBF (<5% win rate, variance collapse). Spatial context provides crucial denoising.

### 2. Normalization: Z-Score Destroys CBF
```
Signal Model: x ~ CBF * M0 * k(ATT, T1)
Z-score:      z = (x - mean) / std  -->  CBF * M0 cancels out!
```
Always use `normalization_mode: "global_scale"`.

### 3. GroupNorm over BatchNorm
GroupNorm throughout spatial models for train/eval consistency.

### 4. att_scale Must Be 1.0
Legacy `att_scale: 0.033` causes ATT loss to be weighted at only 3.3% of CBF loss.

---

## Physics Parameters

| Parameter | Default | Randomized Range | Unit |
|-----------|---------|------------------|------|
| PLDs | [500, 1000, 1500, 2000, 2500, 3000] | - | ms |
| T1_artery | 1650 (3T consensus) | 1550-2150 | ms |
| T_tau (label duration) | 1800 | +/-10% | ms |
| alpha_PCASL | 0.85 | 0.75-0.95 | - |
| alpha_VSASL | 0.56 | 0.40-0.70 | - |
| alpha_BS1 | 1.0 | 0.85-1.0 | - |

**Background Suppression (alpha_BS1)**: Critical for real-world robustness.
- PCASL: effective alpha = alpha_PCASL * (alpha_BS1)^4
- VSASL: effective alpha = alpha_VSASL * (alpha_BS1)^3
- alpha_BS1 = 1.0: No BS (synthetic default). In-vivo: ~0.85-0.95.

**Tissue Ranges** (SpatialPhantomGenerator):
- Gray matter: CBF 50-70, ATT 1000-1600 ms
- White matter: CBF 18-28, ATT 1200-1800 ms

---

## Honest Assessment

### What we CAN claim:
1. Spatial models dramatically outperform voxel-wise for CBF (confirmed even with corrected LS)
2. AmplitudeAwareSpatialASLNet architecture provides measurable CBF improvement over baseline SpatialASLNet (MAE 0.80 vs 2.64 at SNR=10)
3. NN has modest but statistically significant advantage over corrected LS at low-moderate SNR (54-60% CBF, 61-68% ATT for best model)
4. NN is orders of magnitude faster than iterative LS
5. Standard SpatialASLNet suffers from complete CBF variance collapse

### What we CANNOT claim:
1. ~~NN dramatically outperforms LS (97% win rate)~~ -- artifact of broken LS baseline
2. ~~NN always better than LS~~ -- at high SNR (25+), LS approaches or beats NN for CBF
3. ~~"Amplitude awareness" is the mechanism~~ -- Exp 10 disproves causal link; sensitivity test was invalid
4. ~~Sensitivity ratios are meaningful~~ -- computed with random Gaussian noise, not ASL signals
5. ~~Production models are ready~~ -- production_v1 is catastrophically broken (CBF bias -17.7)
6. ~~CBF predictions are accurate at high values~~ -- super-linearity (slope ~1.9) at CBF >80

---

## Known Bugs (Severity-Rated)

### CRITICAL

**1. att_scale=0.033 Legacy Bug**
All 10 v1 experiments and 9 of 11 v2 experiments use `att_scale: 0.033`. With z-score normalized spatial targets, this should be 1.0. Only Exp 14 and Exp 20 use the correct value.
- **Fix**: Set `att_scale: 1.0` in all future experiments

**2. Super-Linearity in AmplitudeAware Models**
CBF linearity slope ~1.9 (should be 1.0). At CBF=100, model predicts ~133. At CBF=150, hits 300 clamp. R^2 vs identity is negative. Root cause: narrow training CBF range [20-70] + unbounded z-score output.
- **Fix**: Expand training CBF range; add bounded output activation

### HIGH

**3. Domain Randomization Silently Disabled When dc_weight=0.0**
Domain randomization parameters are passed to `KineticModel` which is only called during DC loss. When `dc_weight=0.0` (default), domain randomization has NO effect. Data generation uses fixed physics.
- **Files**: `training/asl_trainer.py`, `models/spatial_asl_network.py`
- **Fix**: Move domain randomization to data augmentation

**4. T1_artery=1850 in ALL Experiment Configs**
ASL consensus (Alsop 2015) recommends 1650ms at 3T. All experiments used 1850.
- **Fix**: Use 1650 in future; existing results are internally consistent

**5. Baseline SpatialASLNet Variance Collapse**
Predicts ~55 for ALL CBF values (slope=0.026). The 3.47 MAE appears acceptable only because validation CBF distribution is narrow (centered ~55-60).
- **Fix**: Use AmplitudeAwareSpatialASLNet or investigate capacity-matched alternatives

### MEDIUM

**6. Ensemble Diversity: All Members Get Same Data Order**
Ensemble members may converge to similar solutions without different shuffling.
- **Fix**: Use different random seeds per member

**7. log_var Hardcoded at -5.0 (Uncertainty is Placeholder)**
Spatial models output `log_var_cbf`/`log_var_att` but these are never trained (MaskedSpatialLoss doesn't use them). Uncertainty estimates are meaningless.
- **Fix**: Pass log_var to loss for NLL training, or remove log_var heads

**8. Production v1 Models Are Broken**
CBF win rate 2.9%, bias -17.7 ml/100g/min. Do NOT use for inference.

### LOW

**9. AmplitudeAwareLoss (Dead Code)**
Defined in `models/amplitude_aware_spatial_network.py` but never instantiated.

**10. Rician Noise: SpatialNoiseEngine.simulate_realistic_acquisition Is Dead Code**
Training pipeline uses `NoiseInjector` (correct implementation). The SpatialNoiseEngine path is unused.

---

## Ablation Study Results

### Study 1: Amplitude Ablation (amplitude_ablation_v1/)

**WARNING**: All win rates below measured against broken LS baseline. See corrected results above.

| Exp | Config | CBF MAE | CBF Win Rate (broken LS) | ATT MAE |
|-----|--------|---------|--------------------------|---------|
| 00 | Baseline SpatialASL | 3.47 | 85.8% | 21.37 |
| 02 | AmpAware Full | 0.46 | 97.7% | 20.06 |
| 03 | OutputMod Only | 0.50 | 97.6% | 23.31 |
| 06 | Physics dc=0.1 | 0.51 | 97.5% | 19.21 |
| 07 | Physics dc=0.3 | 0.53 | 97.5% | 21.65 |
| 08 | DomainRand | 0.46 | 97.8% | 18.62 |
| 09 | Optimized | 0.49 | 97.5% | 18.68 |
| 14 (v2) | ATT Rebalanced | 0.44 | 97.8% | **15.35** |

All amplitude-aware models achieve CBF MAE 0.44-0.53 vs baseline 3.47 -- a 7x improvement. But within the group, sensitivity ratio has zero correlation with accuracy.

### Study 2: HPC Ablation (hpc_ablation_jobs/) -- Voxel-Wise

All voxel-wise configs fail for CBF (<5% win rate). Best ATT win rate: 36% (Exp 05, small model).

---

## Validation Metrics

| Metric | Purpose | Good Value |
|--------|---------|------------|
| MAE | Mean absolute error | CBF <1.0 (AmplAware), ATT <20 |
| Bias | Systematic error | Near 0 |
| Win Rate | % NN beats corrected LS | >55% |
| R^2 | Variance explained | >0.9 |

---

## Future Directions

1. **Capacity-matched ablation**: Test SpatialASLNet with same parameter count as AmplitudeAware to determine if improvement is from FiLM/OutputMod or just extra capacity
2. **Fix super-linearity**: Expand training CBF range to [10, 150+], add bounded output activation
3. **In-vivo validation**: Configure FSL/BASIL, obtain in-vivo comparison data
4. **Retrain with corrected parameters**: att_scale=1.0, T1_artery=1650, functional domain randomization
5. **Larger spatial context**: Current 64x64 patches may be too small

---

## References

1. **MULTIVERSE ASL**: Xu et al. (2025) - Joint PCASL/VSASL protocol
2. **Bias-Reduced NNs**: Mao et al. (2023) - Variance collapse solution
3. **IVIM-NET**: Kaandorp et al. (2021) - Physics-informed architecture
4. **General Kinetic Model**: Buxton et al. - PCASL signal equation
5. **ASL Consensus**: Alsop et al. (2015) - Standard implementation
6. **FiLM**: Perez et al. (2018) - Feature-wise Linear Modulation

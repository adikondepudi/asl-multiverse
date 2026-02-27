# CBF Bias Reduction Progression

**Purpose**: Step-by-step documentation of how each architectural and methodological change reduced CBF estimation bias. Prepared for PI review (Feng Xu, Feb 2026).

---

## Overview

Starting from a voxel-wise MLP with per-curve normalization (CBF win rate <5%), we systematically improved CBF estimation through four stages:

| Stage | Key Change | CBF Win Rate vs LS | Primary Mechanism |
|-------|-----------|-------------------|-------------------|
| 0 | Voxel-wise + z-score | <5% | Baseline (broken) |
| 1 | Global-scale normalization | ~10-20% | Preserve CBF amplitude |
| 2 | Spatial U-Net (SpatialASLNet) | ~20-40% | Spatial denoising |
| 3 | AmplitudeAware architecture | ~55-60% | Increased model capacity |

Additionally, a physics correction (constant SIB) eliminated a bias dip at ATT~2100ms in the least-squares baseline.

---

## Stage 0: Voxel-Wise Model with Per-Curve Normalization

**Architecture**: DisentangledASLNet (1D MLP), trained on individual voxel signals.

**Problem**: Per-pixel temporal z-score normalization mathematically destroys CBF information.

The ASL signal model is:

```
Signal(PLD) ~ CBF * M0 * k(ATT, T1, PLD)
```

where `k(ATT, T1, PLD)` is a shape function depending on transit time and T1. Z-score normalization computes:

```
z = (x - mean(x)) / std(x)
```

This divides out the amplitude `CBF * M0`, leaving only the *shape* of the curve — which encodes ATT but not CBF. The network literally cannot learn CBF from z-scored inputs.

**Result**:
- CBF win rate vs LS: <5% at all SNR levels
- Complete variance collapse: model predicts ~constant CBF for all inputs
- ATT estimation partially works (shape information preserved): 20-36% win rate

**Reference**: `hpc_ablation_jobs/` (10 voxel-wise experiments, all failed for CBF)

---

## Stage 1: Normalization Fix — Global Scale

**Change**: Replace `normalization_mode: "per_curve"` with `normalization_mode: "global_scale"`.

Instead of per-voxel z-scoring, we multiply all signals by a fixed constant:

```
x_normalized = x * M0_scale_factor * global_scale_factor
```

This preserves the proportionality `x ~ CBF`, so the network can learn CBF from signal amplitude.

**Why this matters**: The choice of normalization is not a hyperparameter — it is a mathematical constraint. Z-score normalization makes CBF estimation *impossible* regardless of network architecture.

**Configuration**:
```yaml
data:
  normalization_mode: "global_scale"
  global_scale_factor: 1.0   # or 10.0 for AmplitudeAware
```

---

## Stage 2: Spatial U-Net (SpatialASLNet)

**Change**: Replace voxel-wise MLP with a U-Net operating on 64×64 spatial patches.

**Architecture**: Standard encoder-decoder U-Net with GroupNorm, 4 resolution levels (`features: [32, 64, 128, 256]`), and skip connections. Input: 12 channels (6 PCASL PLDs + 6 VSASL PLDs). Output: 2 channels (CBF, ATT) as z-score normalized values.

**Why spatial context helps**:
1. **Denoising**: Neighboring voxels share similar CBF/ATT values (spatial smoothness prior). The U-Net implicitly learns this spatial correlation, effectively denoising the input.
2. **Multi-scale features**: The encoder hierarchy captures both local signal patterns and broader tissue boundaries.
3. **Skip connections**: Preserve high-frequency spatial detail for accurate boundary delineation.

**Result** (v4 retraining pending — values below from v1/v2 with broken LS):
- CBF MAE: 3.47 ml/100g/min (7× better than voxel-wise)
- ATT MAE: 21.4 ms
- **Caveat**: Baseline SpatialASLNet still exhibits variance collapse (predicts ~55 ml/100g/min for all CBF values, slope=0.026). The low MAE reflects a narrow validation CBF distribution centered ~55-60, not accurate estimation across CBF range.

**Reference**: `amplitude_ablation_v1/00_SpatialASLNet/`, `amplitude_ablation_v4/A_Baseline_SpatialASL/` (v4 retrained)

---

## Stage 3: Amplitude-Aware Architecture (AmplitudeAwareSpatialASLNet)

**Change**: Add FiLM conditioning and output modulation pathways to the U-Net.

**Architecture additions**:
1. **AmplitudeFeatureExtractor**: Extracts signal statistics (mean, std, peak) *before* GroupNorm destroys them
2. **FiLM conditioning** (Feature-wise Linear Modulation): Injects amplitude features into bottleneck and decoder layers via learned affine transforms: `γ * features + β`
3. **Output modulation**: Direct pathway from extracted amplitude to CBF output scaling

**Why it helps**: The additional FiLM layers and output modulation pathways substantially increase model capacity. The extra parameters allow the network to learn more complex mappings, particularly for CBF estimation where signal amplitude matters.

**Important caveat**: While this architecture was designed to "preserve amplitude information," a controlled audit (`docs/amplitude_audit_report.md`) found that the sensitivity test was invalid (computed on random noise, not ASL signals) and that Exp 10 (low sensitivity ratio) achieved comparable accuracy. The performance improvement likely comes from increased model capacity rather than a specific "amplitude awareness" mechanism.

**Result** (v4 retraining pending — values below from v2 corrected LS comparison):
- CBF MAE: 0.44-0.80 ml/100g/min (4-8× better than baseline SpatialASLNet)
- CBF win rate vs corrected LS: ~55-60% at low-moderate SNR
- ATT win rate vs corrected LS: ~61-68%
- Super-linearity at high CBF (slope ~1.9) remains an open issue

**Reference**: `amplitude_ablation_v4/B_AmplitudeAware/` (v4 retrained)

**Configuration**:
```yaml
training:
  model_class_name: AmplitudeAwareSpatialASLNet
  use_film_at_bottleneck: true
  use_film_at_decoder: true
  use_amplitude_output_modulation: true
```

---

## Physics Correction: Constant SIB (VSASL Saturation Recovery)

### Background

The VSASL signal model includes a Saturation-Inversion-Balance (SIB) factor that accounts for blood magnetization recovery during the saturation delay `T_sat`:

**Old formula (ATT-dependent — INCORRECT)**:
```
SIB = 1.0                              if ATT ≤ T_sat
SIB = 1.0 - exp(-(ATT - T_sat) / T1)  if ATT > T_sat
```

This created a **discontinuity at ATT = T_sat = 2000 ms**, where SIB jumped from 1.0 to ~0.0 and then slowly recovered. This discontinuity caused a visible dip in the LS CBF bias curve near ATT = 2100 ms.

**Corrected formula (constant — Qin et al. MRM 2022)**:
```
SIB = 1.0 - exp(-T_sat / T1)
```

The SIB factor is **constant** (~0.70 for T_sat=2000ms, T1=1650ms) because the non-selective saturation pulse zeroes *all* blood magnetization at once. After T_sat seconds of T1 recovery, blood reaches a fixed fraction of equilibrium — regardless of when it arrives at the imaging voxel (i.e., regardless of ATT).

### Verification

The corrected SIB was implemented in:
- `simulation/asl_simulation.py` — JIT-compiled signal generation
- `simulation/generate_clean_library.py` — spatial training data generation
- `baselines/multiverse_functions.py` — LS fitter
- `models/spatial_asl_network.py` — KineticModel (physics loss)

**v3 LS bias results confirm the fix** (from `bias_cov_results_v3/`):

| ATT (ms) | CBF Bias (SNR=10) | CBF Bias (SNR=5) |
|----------|-------------------|-------------------|
| 1900 | +0.023 | +0.048 |
| 2000 | +0.109 | +0.112 |
| 2100 | **+0.003** | +0.020 |
| 2200 | -0.013 | -0.022 |

The bias at ATT=2100ms is now 0.003 ml/100g/min — effectively flat. Maximum |CBF bias| across all ATT values at SNR=10 is 0.18 ml/100g/min.

### Impact on NN Models

Models trained on old (ATT-dependent SIB) data learned a signal model with the discontinuity baked in. The v4 retraining uses corrected constant SIB in both training data generation and evaluation, ensuring consistency between training and test physics.

---

## Summary of v4 Retraining

All v4 experiments use:
- **Constant SIB**: `SIB = 1.0 - exp(-T_sat_vs / T1_artery)` (T_sat_vs=2000ms)
- **T1_artery = 1650 ms** (3T consensus, Alsop 2015)
- **att_scale = 1.0** (corrected from legacy 0.033)
- **dc_weight = 0.0** (physics loss disabled)
- **Domain randomization** in training data generation (per-phantom parameter sampling)

### Training Data
Generated with `simulation.generate_clean_library` using `--spatial --domain-rand`:
```bash
python -m simulation.generate_clean_library asl_spatial_dataset_v4 \
    --spatial --total_samples 100000 --spatial-chunk-size 500 \
    --image-size 64 --domain-rand
```

### Experiments

| Exp | Config | Architecture | Epochs | Ensembles |
|-----|--------|-------------|--------|-----------|
| A | `config/v4_baseline_spatial.yaml` | SpatialASLNet | 50 | 3 |
| B | `config/v4_amplitude_aware.yaml` | AmplitudeAwareSpatialASLNet | 100 | 3 |

### Evaluation

Bias/CoV plots generated with:
```bash
python generate_bias_cov_plots_v2.py \
    --output-dir bias_cov_results_v4 \
    --snr 3.0 5.0 10.0 \
    --n-phantoms 10 --n-ls-realizations 1000 \
    --models "Baseline SpatialASLNet:amplitude_ablation_v4/A_Baseline_SpatialASL" \
             "AmplitudeAware:amplitude_ablation_v4/B_AmplitudeAware"
```

*Note: Bias/CoV plots will be added after v4 training completes.*

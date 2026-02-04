# Root Cause Analysis: CBF Estimation Failure in Spatial ASL Network

**Date**: 2026-02-04
**Analysis by**: Claude (Opus 4.5) with comprehensive codebase review

---

## Executive Summary

The production_model_v1 spatial neural network **cannot accurately estimate CBF** due to a fundamental architectural flaw: **GroupNorm in the first layer destroys signal amplitude information**, and CBF is encoded primarily in signal amplitude.

### Key Metrics

| Metric | Measured | Expected | Status |
|--------|----------|----------|--------|
| GM CBF | 38.9 ± 5.1 | 50-80 | **-35% bias** |
| WM CBF | 30.7 ± 2.7 | 20-25 | +23% bias |
| GM/WM Ratio | 1.27 | 2.5-3.0 | **Poor contrast** |
| GM ATT | 1477 ± 147 | 700-1200 | High bias |
| ATT R² (validation) | 0.993 | >0.95 | Good |
| CBF R² (validation) | 0.73 | >0.90 | Mediocre |

---

## The Physics of ASL Signal

### Signal Model
The ASL difference signal follows the General Kinetic Model (Buxton et al.):

```
ΔS(t) = 2α · M₀ · CBF · k(t, ATT, T₁, τ)
```

Where:
- **CBF**: Encoded as a **multiplicative amplitude factor**
- **ATT**: Encoded in the **temporal shape** k(t)
- **α**: Labeling efficiency (~0.85 for PCASL)
- **M₀**: Equilibrium magnetization
- **τ**: Label duration (1800 ms)
- **T₁**: Blood T1 relaxation (1850 ms at 3T)

### Information Encoding

| Parameter | Encoded In | Can Survive Normalization? |
|-----------|------------|---------------------------|
| **CBF** | Signal amplitude | NO - destroyed by any per-sample normalization |
| **ATT** | Temporal shape (when signal rises/falls) | YES - shape is preserved |
| **T₁** | Decay rate | Partially - affects shape |

---

## Root Cause Analysis

### The GroupNorm Problem

The `SpatialASLNet` uses GroupNorm after every convolution:

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ...),
            nn.GroupNorm(num_groups, out_channels),  # <- PROBLEM
            nn.ReLU(inplace=True),
            ...
        )
```

**GroupNorm normalizes each sample to zero-mean, unit-variance:**

```
x_norm = (x - μ) / σ   where μ, σ computed over (channels/groups, H, W)
```

### Experimental Proof

Testing the trained model with inputs at different scales:

| Input Scale | CBF Prediction | ATT Prediction |
|-------------|---------------|----------------|
| 0.01× | 10.6 ± 3.6 | 405 ± 383 |
| 1× | 10.6 ± 3.6 | 405 ± 383 |
| 100× | 10.6 ± 3.6 | 405 ± 383 |

**The model produces IDENTICAL outputs regardless of input amplitude!**

### Why Learnable γ/β Don't Help

GroupNorm has learnable scale (γ) and shift (β) parameters, but:
- These are **fixed constants**, not input-dependent
- They apply the same transformation regardless of input amplitude
- They cannot recover the original scale information

### Why global_scale Doesn't Help

The config specified `normalization_mode: global_scale`, but this only multiplies the input by a constant (10.0). GroupNorm then normalizes this away:

```
Input: S × 1000 (after M0 scaling × global_scale)
After Conv: W·S × 1000
After GroupNorm: (W·S × 1000 - μ) / σ ≈ same as (W·S × 1 - μ') / σ'
```

---

## Why Simulated Validation Looked Acceptable

The model achieved R²=0.73 for CBF on simulated data because it could exploit:

1. **Spatial Correlations**: GM/WM boundaries have different textures
2. **ATT-CBF Correlation**: In training data, regions with certain ATT values may correlate with certain CBF values
3. **Shape Variations**: Higher CBF slightly modifies curve shape (secondary effect)

But these indirect cues **don't generalize to real data** where:
- Pathology breaks spatial correlations
- ATT-CBF relationships vary by individual
- Only amplitude provides reliable CBF information

---

## Solution Architecture

### Recommended Approach: Amplitude-Aware U-Net

The solution is to **explicitly extract and preserve amplitude information** before GroupNorm normalization.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AMPLITUDE-AWARE SPATIAL ASL NET                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: (B, 12, H, W)  [PCASL + VSASL, 6 PLDs each]                │
│           │                                                         │
│           ├──────────────────────────────────────┐                  │
│           │                                      │                  │
│           ▼                                      ▼                  │
│  ┌─────────────────┐                  ┌─────────────────────┐      │
│  │ AMPLITUDE PATH  │                  │   SPATIAL PATH      │      │
│  │ (No GroupNorm)  │                  │   (Standard U-Net)  │      │
│  └────────┬────────┘                  └──────────┬──────────┘      │
│           │                                      │                  │
│           ▼                                      │                  │
│  ┌─────────────────┐                             │                  │
│  │ Global Pooling  │                             │                  │
│  │ Extract:        │                             │                  │
│  │ - mean(PCASL)   │                             │                  │
│  │ - std(PCASL)    │                             │                  │
│  │ - max(PCASL)    │                             │                  │
│  │ - mean(VSASL)   │                             │                  │
│  │ - std(VSASL)    │                             │                  │
│  │ - max(VSASL)    │                             │                  │
│  │ - PCASL/VSASL   │                             │                  │
│  │   ratio         │                             │                  │
│  └────────┬────────┘                             │                  │
│           │                                      │                  │
│           ▼                                      ▼                  │
│  ┌─────────────────┐                  ┌─────────────────────┐      │
│  │ Amplitude MLP   │                  │   Bottleneck        │      │
│  │ (scalar→vector) │ ──── FiLM ────▶ │   Features          │      │
│  └────────┬────────┘   Modulation    └──────────┬──────────┘      │
│           │                                      │                  │
│           │                                      ▼                  │
│           │                           ┌─────────────────────┐      │
│           │                           │     Decoder         │      │
│           │                           └──────────┬──────────┘      │
│           │                                      │                  │
│           └──────────────────────────────────────┤                  │
│                                                  ▼                  │
│                                       ┌─────────────────────┐      │
│                                       │  Output Heads       │      │
│                                       │  CBF: amp_scale *   │      │
│                                       │       spatial_pred  │      │
│                                       │  ATT: spatial_pred  │      │
│                                       └─────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Extract amplitude BEFORE any normalization**
   - Global/local pooling operations
   - Per-channel statistics (mean, std, max)
   - PCASL/VSASL ratio (encodes ATT-independence)

2. **Preserve amplitude through dedicated path**
   - No GroupNorm/BatchNorm on amplitude features
   - MLP to transform scalars → conditioning vector

3. **Inject via FiLM modulation**
   - Feature-wise Linear Modulation: `y = γ(amp) * x + β(amp)`
   - Modulates spatial features based on amplitude
   - Apply at bottleneck and/or decoder layers

4. **Amplitude-dependent output scaling**
   - CBF_pred = f(spatial_features) * g(amplitude_features)
   - The amplitude path directly influences CBF magnitude

### Alternative: Physics-Constrained Training

Even with the current architecture, physics constraints can help:

```python
# Physics loss: reconstruct signal from predicted parameters
pred_signal = kinetic_model(pred_cbf, pred_att)
physics_loss = |pred_signal - input_signal|

# The kinetic model IS sensitive to CBF amplitude
# So the network must learn correct CBF to minimize physics loss
```

**However**: This requires the network to solve an inverse problem through gradient descent, which is less direct than preserving amplitude information architecturally.

---

## Implementation Plan

### Phase 1: Architecture Redesign (Critical)

1. **Create `AmplitudeAwareSpatialASLNet`**
   - Add amplitude extraction module
   - Add FiLM conditioning layers (reuse from `enhanced_asl_network.py`)
   - Add amplitude-dependent output modulation

2. **Update `SpatialDataset`**
   - Return raw (unnormalized) signals alongside processed signals
   - Or compute amplitude features during loading

3. **Update training pipeline**
   - Pass amplitude features through network
   - Ensure gradient flow through amplitude path

### Phase 2: Loss Function Enhancement

1. **Enable physics loss** (`dc_weight > 0`)
   - Currently disabled in production config
   - Provides self-supervision for CBF amplitude

2. **Add amplitude prediction loss**
   - Predict mean signal amplitude as auxiliary task
   - Multi-task learning regularization

3. **Implement bias-reduced loss** (Mao et al., 2024)
   - Variance matching penalty
   - CRB-weighted loss terms

### Phase 3: Training Data Enhancement

1. **Domain randomization**
   - Vary M0, T1, labeling efficiency
   - Train on wide amplitude range

2. **In vivo-matched augmentation**
   - Match in vivo signal statistics
   - Add realistic artifacts

### Phase 4: Validation Protocol

1. **Amplitude sensitivity test**
   - Verify predictions change with input scale
   - Must show ~linear CBF scaling with amplitude

2. **Synthetic validation**
   - Known ground truth CBF/ATT
   - Test across full parameter range

3. **In vivo validation**
   - Compare GM/WM CBF values to literature
   - Test GM/WM contrast ratio
   - Compare with least-squares fitting

---

## Appendix: Mathematical Proof of Information Loss

### Setup
Signal model: `S(t) = A · k(t)` where `A = CBF · M0` and `k(t)` encodes ATT-dependent shape.

### After GroupNorm
GroupNorm computes:
- μ = mean over (channels, H, W)
- σ = std over (channels, H, W)

For a single pixel with temporal signal:
```
μ = (1/T) Σₜ S(t) = A · (1/T) Σₜ k(t) = A · k̄
σ² = (1/T) Σₜ (S(t) - μ)² = A² · var(k)
```

Normalized signal:
```
S_norm(t) = (S(t) - μ) / σ
          = (A·k(t) - A·k̄) / (A·√var(k))
          = (k(t) - k̄) / √var(k)
```

**The amplitude A = CBF · M0 cancels completely!**

### Information-Theoretic View
- Original mutual information: `I(S; CBF, ATT) > 0`
- After normalization: `I(S_norm; CBF) = 0` (CBF independent of normalized signal)
- But: `I(S_norm; ATT) > 0` (ATT still encoded in shape)

---

## References

1. Alsop et al. (2015). Recommended implementation of ASL for clinical applications. MRM.
2. Buxton et al. (1998). A general kinetic model for quantitative perfusion imaging with ASL. MRM.
3. Mao et al. (2024). Bias-reduced neural networks for parameter estimation in quantitative MRI. MRM.
4. Woods et al. (2019). A general framework for optimizing ASL MRI. MRM.

---

## Conclusion

The current `SpatialASLNet` architecture **fundamentally cannot estimate CBF accurately** because GroupNorm destroys the amplitude information that encodes CBF. The solution requires architectural changes to explicitly preserve and utilize amplitude information.

ATT estimation works reasonably well because ATT is encoded in the temporal shape, which survives normalization.

The recommended fix is the **Amplitude-Aware U-Net** architecture with explicit amplitude feature extraction and FiLM-based injection. This maintains the benefits of GroupNorm for training stability while preserving the critical amplitude information needed for CBF estimation.

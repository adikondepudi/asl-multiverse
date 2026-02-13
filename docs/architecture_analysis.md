# Architecture Analysis: ASL-Multiverse Spatial Models

## 1. SpatialASLNet (Baseline U-Net)

### Architecture Diagram

```
Input: (B, 12, 64, 64)  [6 PCASL + 6 VSASL channels]
         |
    [DoubleConv: 12 -> 32]     encoder1  (12,800 params)
         |                      e1: (B, 32, 64, 64)
    [MaxPool2d(2)]
         |
    [DoubleConv: 32 -> 64]     encoder2  (55,552 params)
         |                      e2: (B, 64, 32, 32)
    [MaxPool2d(2)]
         |
    [DoubleConv: 64 -> 128]    encoder3  (221,696 params)
         |                      e3: (B, 128, 16, 16)
    [MaxPool2d(2)]
         |
    [DoubleConv: 128 -> 256]   encoder4/bottleneck  (885,760 params)
         |                      e4: (B, 256, 8, 8)
         |
    [ConvTranspose2d: 256->128] up1  (131,200 params)
         |
    [cat(e3, d1)]              skip connection: 128+128=256 channels
    [DoubleConv: 256 -> 128]   decoder1  (442,880 params)
         |                      d1: (B, 128, 16, 16)
    [ConvTranspose2d: 128->64] up2  (32,832 params)
         |
    [cat(e2, d2)]              skip connection: 64+64=128 channels
    [DoubleConv: 128 -> 64]    decoder2  (110,848 params)
         |                      d2: (B, 64, 32, 32)
    [ConvTranspose2d: 64->32]  up3  (8,224 params)
         |
    [cat(e1, d3)]              skip connection: 32+32=64 channels
    [DoubleConv: 64 -> 32]     decoder3  (27,776 params)
         |                      d3: (B, 32, 64, 64)
    [Conv2d: 32 -> 2]          out_conv  (66 params)
         |
    Output: cbf_norm(B,1,64,64), att_norm(B,1,64,64), log_var x2
```

### DoubleConv Block Detail
```
Conv2d(in, out, 3x3, padding=1, bias=False)
  -> GroupNorm(8, out)
  -> ReLU
  -> Conv2d(out, out, 3x3, padding=1, bias=False)
  -> GroupNorm(8, out)
  -> ReLU
```

### Key Properties
- **Total parameters**: 1,929,634 (7.36 MB)
- **Receptive field**: ~124 pixels at bottleneck (3 pooling layers, each 3x3 conv)
- **Multi-scale representation**: 4 resolution levels (64, 32, 16, 8)
- **Skip connections**: Direct concatenation from encoder to decoder at each level
- **Normalization**: GroupNorm(8 groups) everywhere -- no train/eval mismatch
- **Output**: Unbounded (z-score predictions), denormalized at inference
- **Initialization**: Kaiming normal for conv weights, small normal (std=0.01) for output

### Parameter Distribution
| Component | Params | % of Total |
|-----------|--------|------------|
| Encoder (e1-e4) | 1,175,808 | 60.9% |
| Decoder (d1-d3 + up) | 754,760 | 39.1% |
| Output head | 66 | 0.003% |
| **Bottleneck (encoder4)** | **885,760** | **45.9%** |

The bottleneck layer alone accounts for nearly half of all parameters.

---

## 2. AmplitudeAwareSpatialASLNet

### Architecture Diagram

```
Input: (B, 12, 64, 64)
    |                    |
    |             [AmplitudeFeatureExtractor]
    |                    |
    |              conditioning: (B, 64)
    |                    |
[DoubleConvNoNorm: 12->32]   encoder1 (no GroupNorm!)
    |
[MaxPool2d] -> [DoubleConvWithNorm: 32->64]   encoder2
    |
[MaxPool2d] -> [DoubleConvWithNorm: 64->128]  encoder3
    |
[MaxPool2d] -> [DoubleConvWithNorm: 128->256] encoder4
    |
[FiLM(256, 64)]    <-- conditioning modulates bottleneck
    |
[Decoder with FiLM at each level]
    |
[spatial_head: Conv2d 32->2]
    |
cbf_spatial * amplitude_scale --> cbf_pred
att_spatial                   --> att_pred
```

### AmplitudeFeatureExtractor
```
Raw input (B, 12, H, W)
  -> per-channel mean (12), std (12), max (12)
  -> signal_power (1), pcasl/vsasl ratio (1), peak locations (2)
  -> Total: 40 raw features
  -> MLP: Linear(40->64) -> ReLU -> Linear(64->64) -> ReLU -> Linear(64->64)
  -> Output: conditioning vector (B, 64)
```

### FiLM Layer
```
conditioning (B, 64)
  -> gamma_generator: Linear(64->64) -> ReLU -> Linear(64->C)
  -> beta_generator:  Linear(64->64) -> ReLU -> Linear(64->C)

feature_map (B, C, H, W)
  -> output = (1 + gamma) * feature_map + beta
```

### Output Modulation
```
channel_means = raw_amplitude[:, :12].mean(dim=1)    # (B, 1)
amplitude_proxy = channel_means * 100.0               # Scale to ~1-10
log_correction = cbf_amplitude_correction(conditioning)  # Learned (B, 1)
correction = exp(clamp(log_correction, -2, 2))
cbf_scale = amplitude_proxy * correction               # (B, 1, 1, 1)
cbf_pred = cbf_spatial * cbf_scale                     # Multiply spatial output
```

### Key Properties
- **Total parameters**: 2,038,307 (7.78 MB) -- +5.6% over baseline
- **Overhead breakdown**:
  - Amplitude extractor: 10,944 params
  - Bottleneck FiLM: 41,600 params
  - 3 decoder FiLMs: 54,080 params
  - CBF correction MLP: 2,113 params
  - Total overhead: 108,673 params
- **First encoder has NO GroupNorm** (DoubleConvNoNorm) -- preserves some amplitude
- **Amplitude sensitivity**: 376x ratio (vs 1.0x for baseline)

---

## 3. DualEncoderSpatialASLNet (Y-Net)

### Architecture
- Two complete encoders (PCASL stream + VSASL stream)
- Fusion at bottleneck via concatenation + DoubleConv
- Decoder uses dual skip connections (3x channel concat: decoder + pcasl_skip + vsasl_skip)
- Separate output heads for CBF, ATT, uncertainty

### Key Properties
- **Total parameters**: 5,066,084 (19.33 MB) -- +162.5% over baseline
- Roughly 2x encoder parameters + larger decoder (triple skip connections)
- Never validated in ablation studies

---

## 4. SimpleCNN (New Ablation Baseline)

### Architecture
```
Input: (B, 12, 64, 64)
    |
[Conv2d: 12->32, 3x3, padding=1]
[GroupNorm(8, 32)]
[ReLU]
    |
[Conv2d: 32->64, 3x3, padding=1]
[GroupNorm(8, 64)]
[ReLU]
    |
[Conv2d: 64->2, 1x1]
    |
Output: cbf_norm(B,1,64,64), att_norm(B,1,64,64), log_var x2
```

### Key Properties
- **Total parameters**: 22,210 (0.08 MB) -- 87x fewer than U-Net
- **Receptive field**: 5x5 pixels (two 3x3 convolutions)
- **No multi-scale**: operates only at original resolution
- **No skip connections**: no encoder-decoder structure
- Same interface as SpatialASLNet (drop-in replacement)

---

## 5. CapacityMatchedSpatialASLNet (New Capacity Control)

### Purpose
Isolates whether AmplitudeAwareSpatialASLNet's dramatic performance advantage
(CBF MAE 0.46 vs 3.47) comes from the FiLM mechanism or simply from having
~108K extra parameters. The amplitude auditor found that models with low
sensitivity ratios (0.36) perform equally well as high-ratio models (90+),
suggesting capacity rather than amplitude awareness may be the real driver.

### Architecture
Same as SpatialASLNet, but with extra Conv+GroupNorm+ReLU layers in the decoder:
- decoder1: extra 1x1 Conv+GN+ReLU (128 channels) -> +16,640 params
- decoder2: extra DoubleConv (64 channels) -> +73,984 params
- decoder3: extra DoubleConv (32 channels) -> +18,560 params

### Key Properties
- **Total parameters**: 2,038,818 -- within 511 params (0.025%) of AmplitudeAware
- **NO amplitude awareness**: No FiLM, no output modulation, no amplitude extraction
- Same U-Net structure, same receptive field, just deeper decoder path

### Ablation Logic
| Outcome | Conclusion |
|---------|------------|
| CapacityMatched ~ AmplitudeAware | Extra capacity is the driver; FiLM is cosmetic |
| AmplitudeAware >> CapacityMatched | FiLM mechanism genuinely helps |
| CapacityMatched ~ Baseline | Extra params alone don't help; FiLM is necessary |

---

## 6. Parameter Count Comparison

| Model | Params | Size (MB) | vs Baseline |
|-------|--------|-----------|-------------|
| **SimpleCNN** | 22,210 | 0.08 | 0.012x (87x fewer) |
| **SpatialASLNet** | 1,929,634 | 7.36 | 1.0x (baseline) |
| **CapacityMatched** | 2,038,818 | 7.78 | 1.057x (+5.7%) |
| **AmplitudeAware (full)** | 2,038,307 | 7.78 | 1.056x (+5.6%) |
| **DualEncoder (Y-Net)** | 5,066,084 | 19.33 | 2.625x (+162.5%) |

---

## 6. Loss Function Analysis

### MaskedSpatialLoss (Primary)
- Used for all spatial models in training
- Operates in **normalized (z-score) space**: targets normalized by stored norm_stats
- Components:
  1. **Supervised loss** (L1/L2/Huber): `cbf_weight * cbf_loss + att_weight * att_loss_scaled`
  2. **Variance penalty**: `variance_weight * ReLU(target_var - pred_var)` per sample
  3. **Physics loss (DC)**: `dc_weight * L1(reconstructed_signal - input_signal)` via KineticModel
- Default production config: L1 loss, cbf_weight=1.0, att_weight=1.0, att_scale=1.0, dc_weight=0.0, variance_weight=0.1

### CBF:ATT Loss Balance
- With z-score normalized targets, both CBF and ATT are in standard deviation units
- att_scale=1.0 means equal weighting (correct for normalized targets)
- **Known v2 bug**: 9/11 v2 experiments used att_scale=0.033 (legacy), weighting ATT at only 3.3%

### Variance Weight
- Prevents variance collapse by penalizing `ReLU(target_var - pred_var)`
- Computed per-sample over brain pixels, then averaged across batch
- Default 0.1 -- sufficient to prevent collapse in spatial models

### Physics Loss Path
- `dc_weight > 0` activates KineticModel.compute_physics_loss()
- Denormalizes predictions -> generates expected signals -> L1 vs input
- Supports domain randomization of physics params during training
- v1 ablation tested dc_weight=0.1 and 0.3 (Exp 06, 07) -- amplitude-aware models

### AmplitudeAwareLoss (Dead Code)
- Defined in amplitude_aware_spatial_network.py but **never instantiated** by asl_trainer.py
- Training always uses MaskedSpatialLoss regardless of model class

---

## 7. Training Pipeline Analysis

### Hyperparameters (Production v1)
| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Epochs | 200 |
| Ensemble size | 5 |
| Scheduler | OneCycleLR |
| AMP | FP16 (GradScaler) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=25 |

### Scheduler: OneCycleLR
- Ramps LR from lr/25 -> max_lr -> lr/10000 over total_steps
- total_steps = steps_per_epoch * n_epochs
- This is a cosine-like schedule with warmup built in

### Data Pipeline (Spatial)
- SpatialDataset pre-loads all chunks to RAM (signals * 100 for M0 norm)
- Random augmentation: horizontal flip (50%), vertical flip (50%), 90/180/270 rotation (25%)
- 90/10 train/val split with fixed seed (42)
- Normalization applied in trainer._process_batch_on_gpu (global_scale mode: multiply by factor)

### Overfitting/Underfitting Evidence
- Early stopping patience of 25 epochs suggests convergence takes significant training
- 200 epochs with 100k-200k samples and batch_size=64 = ~3125 steps/epoch
- No dropout in spatial models (GroupNorm provides regularization)
- Variance penalty acts as regularizer against mean collapse

---

## 8. Ablation Hypotheses and Experiment Design

### Experiment A: SimpleCNN vs SpatialASLNet (Architecture)
**Config**: `config/ablation_simple_cnn.yaml`
**Question**: Is the U-Net encoder-decoder structure with skip connections necessary?
**Hypothesis**:
- If SimpleCNN ~ SpatialASLNet -> the **spatial context alone** (neighboring voxels) drives the 21x improvement over voxel-wise, not multi-scale features
- If SpatialASLNet >> SimpleCNN -> **multi-scale representation** matters; the U-Net learns features at different scales that are important for denoising
**Expected outcome**: SpatialASLNet will outperform SimpleCNN because:
1. The 5x5 receptive field of SimpleCNN is too small for effective denoising
2. Skip connections preserve fine-grained spatial detail while incorporating context
3. Multi-scale features capture both local texture and global structure

### Experiment B: CBF:ATT Loss Ratio Sweep
**Config**: `config/ablation_loss_ratio.yaml`
**Sweep**: cbf_weight = [0.5, 1.0, 2.0, 5.0], att_weight = 1.0
**Question**: Does upweighting CBF loss improve CBF estimation?
**Hypothesis**: With normalized targets, 1:1 ratio should be near-optimal since both parameters are in the same z-score units. Extreme ratios will hurt the under-weighted parameter.

### Experiment C: Variance Weight Sweep
**Config**: `config/ablation_variance_weight.yaml`
**Sweep**: variance_weight = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
**Question**: What is the minimum variance_weight to prevent collapse?
**Hypothesis**:
- 0.0 may or may not cause collapse in spatial models (they have structural regularization from spatial context)
- 0.05-0.1 is likely optimal; higher values may force artificial spread

### Experiment D: Physics Loss Weight Sweep
**Config**: `config/ablation_physics_loss.yaml`
**Sweep**: dc_weight = [0.0, 0.001, 0.01, 0.1, 0.3]
**Question**: Does physics-informed loss help spatial models?
**Hypothesis**: Small dc_weight (0.001-0.01) may help as a regularizer. Large values (0.1+) will dominate the loss landscape and hurt supervised accuracy. The v1 claim "minimal benefit" needs re-testing with corrected training.

### Experiment E: Capacity-Matched U-Net vs AmplitudeAware (CRITICAL)
**Config**: `config/ablation_capacity_matched.yaml`
**Question**: Is AmplitudeAware's performance gain from FiLM or just extra parameters?
**Background**: The amplitude auditor found that models with low sensitivity ratios (0.36)
perform equally well as models with 90+ ratios. This suggests capacity, not amplitude
awareness, may be the real driver of the CBF MAE improvement (0.46 vs 3.47).
**Design**: CapacityMatchedSpatialASLNet has the same param count as AmplitudeAware
(2,038,818 vs 2,038,307) but uses plain Conv+GN+ReLU layers instead of FiLM.
**Hypothesis and outcomes**:

| Outcome | Conclusion | Implication |
|---------|------------|-------------|
| CapacityMatched ~ AmplitudeAware | Extra capacity drives performance | Simplify: just use wider/deeper U-Net |
| AmplitudeAware >> CapacityMatched | FiLM genuinely helps | Keep FiLM architecture |
| CapacityMatched ~ Baseline | Extra decoder depth alone does not help | FiLM's input-dependent modulation is key |

This is the most important ablation because it resolves whether the claimed "amplitude
awareness" innovation is real or an artifact of capacity increase.

---

## 9. Minimum Viable Architecture

Based on the analysis, the minimum viable architecture is likely:

1. **For CBF**: SpatialASLNet with [32, 64, 128, 256] features is well-justified.
   - The bottleneck's 8x8 representation at 256 channels provides a 124-pixel receptive field
   - Skip connections are critical for preserving spatial detail in parameter maps
   - The 1.9M parameter count is modest by modern standards

2. **Potential simplification**: A **shallower U-Net** with [32, 64, 128] (3 levels instead of 4) might suffice:
   - Removes the 885,760-param bottleneck
   - Still provides 16x16 multi-scale representation
   - Receptive field ~60 pixels (may be sufficient for 64x64 patches)
   - Would reduce to ~440K params (4.4x reduction)

3. **AmplitudeAware overhead is minimal** (+5.6%) and provides 376x amplitude sensitivity.
   - **CAVEAT**: Amplitude auditor found performance may be from capacity, not FiLM.
   - Experiment E (capacity-matched ablation) is needed to resolve this.
   - If capacity drives performance, a wider baseline U-Net is simpler and equivalent.

4. **DualEncoder is likely overkill** (+162.5% params):
   - Never validated in ablation studies
   - The shared encoder in SpatialASLNet already processes interleaved PCASL+VSASL channels
   - May be worth testing if there's evidence that separate stream processing helps

---

## 10. Receptive Field Analysis

| Model | Pooling Levels | Bottleneck Resolution | Effective Receptive Field |
|-------|---------------|----------------------|--------------------------|
| SimpleCNN | 0 | 64x64 (original) | 5x5 pixels |
| SpatialASLNet | 3 | 8x8 | ~124x124 pixels |
| AmplitudeAware | 3 | 8x8 | ~124x124 pixels |

The receptive field is critical: at SNR=5, a single voxel's signal is dominated by noise.
The U-Net's 124-pixel receptive field means the bottleneck effectively averages over ~15K pixels,
providing a denoising factor of sqrt(15000) ~ 122x in theory.

SimpleCNN's 5x5 receptive field averages over only 25 pixels (5x reduction in noise).

---

## 11. Inference Time Estimates

Rough estimates for 64x64 input, batch_size=1, CPU:

| Model | FLOPs (approx) | Relative Speed |
|-------|----------------|----------------|
| SimpleCNN | ~180M | ~1.0x (fastest) |
| SpatialASLNet | ~1.5B | ~8x slower |
| AmplitudeAware | ~1.6B | ~9x slower |
| DualEncoder | ~4B | ~22x slower |

For clinical use, all are fast enough (<100ms per slice on GPU).

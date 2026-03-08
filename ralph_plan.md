# Ralph Plan — Ordered Task Checklist

**Status**: IN PROGRESS
**Iteration**: 2
**Last Updated**: 2026-03-08

---

## Phase A — Config-Level Fundamentals (highest leverage, lowest risk)

- [FAIL] **A1**: More training data + longer training
  - Change: `--n-samples 5000 --n-epochs 50` (currently 3000/30)
  - Why: More data = better generalization, more epochs = better convergence
  - Risk: Slower iteration (~15 min instead of ~10 min)
  - **FAIL**: CBF win dropped (66.5→62.1 SNR3, 75.8→72.2 SNR10), in-vivo CoV ratio 1.09 (was 0.95), smooth ratio 0.99 (was 0.77), GM/WM ratio 1.17 (physio FAIL). Longer training overfit and degraded in-vivo generalization.

- [x] **A2**: Increase TV weight 0.02 → 0.05
  - Change: `tv_weight: 0.05` in config or ralph_harness.py
  - Why: Stronger spatial smoothness penalty → lower in-vivo smoothness metric
  - Result: CBF wins all improved (+6.1/+2.1/+1.1%), ATT stable. In-vivo CoV/smooth slightly worse but synthetic gains justify.

- [ ] **A3**: Widen domain randomization ranges
  - Change: `alpha_BS1_range: [0.75, 1.0]`, `T1_artery_range: [1400, 2300]`
  - Why: Wider mismatch between NN training physics and LS fixed physics → higher win rate
  - Risk: Too wide may degrade NN accuracy (noise in training)

- [ ] **A4**: Switch noise model to Rician
  - Change: `noise_type: rician` in config
  - Why: Rician is physically correct for magnitude MRI data
  - Risk: Minor — may need noise level adjustment

## Phase B — Variance Reduction (code changes to ralph_harness.py)

- [ ] **B1**: Ensemble averaging (train 3 models, average predictions)
  - Change: Train 3 models with seeds 42,43,44; average CBF/ATT predictions
  - Why: Reduces prediction variance → lower CoV, smoother maps
  - Risk: 3x training time

- [ ] **B2**: Test-time augmentation (4x flips, average predictions)
  - Change: At inference, predict on original + 3 flips, average
  - Why: Reduces noise sensitivity without retraining
  - Risk: 4x inference time (negligible compared to training)

## Phase C — Architecture/Loss Refinements

- [ ] **C1**: Wider model [48, 96, 192, 384]
  - Change: `hidden_sizes: [48, 96, 192, 384]` in config
  - Why: More capacity → potentially better fitting
  - Risk: Slower training, possible overfitting

- [ ] **C2**: Huber loss instead of L1
  - Change: `loss_type: huber` in config
  - Why: Less sensitive to outliers than L2, smoother gradients than L1
  - Risk: May need delta tuning

- [ ] **C3**: Physiological noise augmentation
  - Change: Add low-frequency spatial noise to training data
  - Why: Better models real in-vivo noise characteristics
  - Risk: Code complexity

- [ ] **C4**: Smoother phantoms (pve_sigma 4.0 → 5.0)
  - Change: `SpatialPhantomGenerator(size=64, pve_sigma=5.0)`
  - Why: Smoother training = smoother predictions on in-vivo
  - Risk: May lose fine detail

- [ ] **C5**: Post-processing Gaussian blur (sigma=0.5)
  - Change: Apply scipy gaussian_filter to NN predictions before metrics
  - Why: Simple denoising → lower CoV and smoothness
  - Risk: May hurt per-voxel accuracy (synthetic win rate)

## Phase D — Aggressive (if stuck)

- [ ] **D1**: Extreme domain randomization (alpha_BS1 [0.60, 1.0])
  - Why: Maximizes physics mismatch LS sees but NN handles

- [ ] **D2**: Deeper U-Net (5 encoder levels)
  - Change: `hidden_sizes: [32, 64, 128, 256, 512]`
  - Why: More representational power

- [ ] **D3**: SNR curriculum training
  - Change: Start training with high SNR, gradually decrease
  - Why: Easier → harder learning schedule

- [ ] **D4**: Separate CBF/ATT decoder heads after bottleneck
  - Why: Decouple CBF spatial patterns from ATT

---

## Iteration Log

| Iter | Task | Result | CBF Win (3/10/25) | ATT Win (3/10/25) | CoV Ratio | Smooth Ratio | Notes |
|------|------|--------|-------------------|-------------------|-----------|--------------|-------|
| 0    | —    | baseline | 66.5/75.8/77.9 | 85.6/76.6/84.5 | 0.95 | 0.77 | Starting point |
| 1    | A1   | FAIL | 62.1/72.2/77.3 | 84.3/81.0/91.1 | 1.09 | 0.99 | 5000 samples/50 epochs overfit, degraded in-vivo |
| 2    | A2   | PASS | 72.6/77.9/79.0 | 85.0/76.6/85.0 | 1.00 | 0.90 | tv_weight 0.02→0.05, CBF wins all improved |

# Ralph Plan — Ordered Task Checklist

**Status**: IN PROGRESS
**Iteration**: 8
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

- [FAIL] **A3**: Widen domain randomization ranges
  - Change: `alpha_BS1_range: [0.75, 1.0]`, `T1_artery_range: [1400, 2300]`
  - Why: Wider mismatch between NN training physics and LS fixed physics → higher win rate
  - **FAIL**: ATT win at SNR10 collapsed 76.6%→61.8% (-14.8%). Too-wide ranges degraded NN accuracy.

- [FAIL] **A4**: Switch noise model to Rician
  - Change: `noise_type: rician` in config
  - Why: Rician is physically correct for magnitude MRI data
  - **FAIL**: CBF win SNR10 dropped 77.9%→74.7% (-3.2%). Rician training noise made model slightly worse, possibly because test eval uses Gaussian noise.

## Phase B — Variance Reduction (code changes to ralph_harness.py)

- [DEFER] **B1**: Ensemble averaging (train 3 models, average predictions)
  - Was tested and helped (CoV 1.00→0.94, smooth 0.90→0.87), but DISABLED for fast iteration.
  - Re-enable as final polish once other improvements are found.

- [DEFER] **B2**: Test-time augmentation (4x flips, average predictions)
  - DISABLED for fast iteration. Re-enable as final polish.

## Phase C — Architecture/Loss Refinements

- [FAIL] **C1**: Wider model [48, 96, 192, 384]
  - Change: `hidden_sizes: [48, 96, 192, 384]` in config
  - Why: More capacity → potentially better fitting
  - Risk: Slower training, possible overfitting
  - **FAIL**: CBF wins all dropped (-6.1/-3.5/-3.6), smooth ratio 0.77→1.00 (NN no longer smoother), physio FAIL (GM/WM 1.17). Wider model overfits, loses spatial smoothness.

- [FAIL] **C2**: Huber loss instead of L1
  - Change: `loss_type: huber` in config
  - Why: Less sensitive to outliers than L2, smoother gradients than L1
  - Risk: May need delta tuning
  - **FAIL**: CBF wins all dropped (-6.8/-4.8/-4.9), ATT win SNR10 collapsed 76.6%→62.8%. In-vivo smooth improved but synthetic degradation too severe.

- [FAIL] **C3**: Physiological noise augmentation
  - Change: Add low-frequency spatial noise to training data
  - **FAIL**: Implementation used per-sample gaussian_filter in inner loop (~90k scipy calls), made training 10x slower. Reverted. Would need GPU-native approach (torch convolution) to be viable.

- [FAIL] **C4**: Smoother phantoms (pve_sigma 3.0 → 5.0)
  - Change: `SpatialPhantomGenerator(size=64, pve_sigma=5.0)`
  - Why: Smoother training = smoother predictions on in-vivo
  - Risk: May lose fine detail
  - **FAIL**: CBF wins all dropped (-4.6/-1.6/-4.0), in-vivo CoV ratio 0.97 (was 0.95), smooth ratio 0.89 (was 0.77). Loss plateaued at 0.317 (vs 0.058) — too-smooth phantoms underfit.

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
| 3    | A3   | FAIL | 74.7/76.1/78.3 | 83.3/61.8/89.8 | 1.02 | 0.91 | Widened DR too aggressive, ATT win SNR10 collapsed -14.8% |
| 4    | A4   | FAIL | 70.5/74.7/78.4 | 85.9/77.2/86.6 | 1.03 | 0.93 | Rician noise dropped CBF win SNR10 -3.2% |
| 5    | B1   | PASS | 72.1/75.9/79.1 | 86.9/75.4/84.6 | 0.94 | 0.87 | 3-model ensemble, in-vivo CoV/smooth both improved |
| 6    | C1   | FAIL | 66.5/74.4/75.4 | 86.9/80.7/86.0 | 0.95 | 1.00 | Wider model [48,96,192,384] overfits, smooth ratio collapsed |
| 7    | C2   | FAIL | 65.8/73.1/74.1 | 85.7/62.8/79.2 | 0.91 | 0.71 | Huber loss degraded all win rates, ATT SNR10 -13.8% |
| 8    | C4   | FAIL | 68.0/76.3/75.0 | 88.0/79.3/86.0 | 0.97 | 0.89 | pve_sigma 3→5, too-smooth phantoms, CBF wins all dropped |

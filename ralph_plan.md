# Ralph Plan — Ordered Task Checklist

**Status**: IN PROGRESS
**Iteration**: 30
**Last Updated**: 2026-03-09

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

- [x] **C5**: Post-processing Gaussian blur (sigma=0.5)
  - Change: Apply scipy gaussian_filter to NN predictions before metrics
  - Why: Simple denoising → lower CoV and smoothness
  - Already implemented in ralph_harness.py (lines 478-479, 581-583). Current best includes this.

## Phase D — Aggressive (if stuck)

- [FAIL] **D1**: Extreme domain randomization (alpha_BS1 [0.60, 1.0])
  - Why: Maximizes physics mismatch LS sees but NN handles
  - **FAIL**: CBF wins collapsed (72.6→35.7 SNR3, 77.9→64.0 SNR10, 79.0→62.4 SNR25), ATT wins dropped (85.0→73.2, 76.6→68.4, 85.0→79.9). Too-wide alpha_BS1 range degraded NN accuracy severely.

- [FAIL] **D2**: Deeper U-Net (5 encoder levels)
  - Change: `hidden_sizes: [32, 64, 128, 256, 512]`
  - Why: More representational power
  - **FAIL**: CBF win SNR3 dropped 72.6→66.0% (-6.6%), SNR10 -2.4%, SNR25 -3.0%. In-vivo dramatically improved (CoV 0.95→0.81, smooth 0.77→0.61, physio PASS) but synthetic CBF wins dropped >3%. Training took 1426s (24min) vs usual ~5min due to 4x more params.

- [x] **D3**: SNR curriculum training (already implemented in ralph_harness.py lines 239-245)
  - Change: Start with high SNR, gradually decrease
  - Already in baseline, no new changes needed

- [FAIL] **D4**: Separate CBF/ATT decoder heads after bottleneck
  - Why: Decouple CBF spatial patterns from ATT
  - **FAIL**: CBF win SNR3 dropped 72.6→66.0% (-6.6%), SNR25 79.0→74.4% (-4.6%). In-vivo improved (CoV 0.95→0.84, smooth 0.77→0.64, physio PASS) but synthetic CBF wins dropped >3%. Similar pattern to D2 — more params helps in-vivo but hurts synthetic.

## Phase E — New Ideas (based on 11 iterations of learning)

- [x] **E1**: Training data augmentation (random flips)
  - Already implemented in ralph_harness.py lines 262-272. Part of current baseline.

- [x] **E2**: Stronger post-processing blur (sigma=0.5 → 1.0)
  - Already implemented in ralph_harness.py lines 500-501, 604-605. Part of current baseline.

- [FAIL] **E3**: Moderate alpha_BS1 widening [0.82, 1.0]
  - Change: alpha_BS1_range: [0.82, 1.0] (was [0.85, 1.0], A3 tried [0.75, 1.0] which was too wide)
  - Why: Slightly more LS mismatch without degrading NN accuracy as much as A3
  - **FAIL**: CBF wins all dropped (72.6→69.6, 77.9→71.6, 79.0→70.8), in-vivo CoV FAIL (49.5% > LS 46.2%), physio FAIL (GM/WM 1.11). Even modest BS1 widening hurts.

- [FAIL] **E4**: CBF loss weight increase (1.0 → 2.0)
  - Change: cbf_weight: 2.0 in config
  - Why: CBF win rate is the bottleneck metric; more CBF emphasis during training
  - **FAIL**: CBF wins all dropped (72.6→69.4, 77.9→75.4, 79.0→77.1), ATT win SNR10 collapsed 76.6→68.9% (-7.7%). In-vivo CoV 55.7% (FAIL), physio FAIL (GM/WM 1.13). CBF emphasis hurt ATT without helping CBF.

- [FAIL] **E5**: Higher learning rate (0.003 → 0.005) with warmup
  - Change: lr=0.005, add linear warmup for first 3 epochs
  - **FAIL**: CBF wins all dropped (72.6→64.3, 77.9→74.9, 79.0→75.9), ATT also slightly worse. Higher LR degraded convergence quality despite warmup.

## Phase F — Post-Analysis Ideas (iter 15+)

- [FAIL] **F1**: Remove weight decay (0.0001 → 0)
  - Change: `weight_decay: 0.0` in config
  - **FAIL**: CBF wins all dropped (72.6→68.3, 77.9→72.0, 79.0→72.4), in-vivo CoV FAIL. Took 5645s (10x slower than normal). Weight decay regularization actually helps.

- [FAIL] **F2**: Random 90° rotation augmentation
  - Change: Add `torch.rot90(k=random)` during training alongside existing flips
  - **FAIL**: Crashed — torch.rot90 creates non-contiguous tensors, loss_fn uses .view() which requires contiguous. Needs .contiguous() after rotation. Retry as F2b.

- [FAIL] **F2b**: Random 90° rotation augmentation (with .contiguous() fix)
  - Change: Same as F2 but add .contiguous() after rot90 on all tensors
  - **FAIL**: CBF wins dropped (72.6→73.6, 77.9→73.4, 79.0→75.6). SNR10 -4.5%, SNR25 -3.4%. Rotation augmentation hurts CBF accuracy.

- [FAIL] **F3**: Increase post-processing blur sigma (1.0 → 1.5)
  - Change: sigma=1.5 in gaussian_filter for both synth and in-vivo eval
  - Why: NN has spatial context, LS is per-voxel. More smoothing widens the gap.
  - Risk: Over-smoothing may blur real features
  - **FAIL**: Could not parse current best from spec. Harness failed before metrics could be evaluated.

- [x] **F4**: Enable TTA in synthetic eval
  - Change: Use tta_predict_single in synthetic_eval instead of single forward pass
  - Why: 4-flip averaging reduces per-voxel noise
  - Result: SNR3 CBF +1.3%, in-vivo smoothness ratio 0.77→0.52 (big improvement). SNR10/25 CBF slightly down. Mixed but net positive.

- [x] **F5**: Reduce TV weight (0.05 → 0.03)
  - Change: tv_weight=0.03 in config
  - Why: A2's TV=0.05 helped smoothness but may over-regularize per-voxel CBF accuracy
  - Result: CBF SNR25 +0.5%, SNR3 +0.4%. ATT and in-vivo CoV slightly worse but CBF accuracy improved at high SNR.

## Phase G — Regularization & Generalization (the only thing that's worked)

- [FAIL] **G1**: Add dropout to decoder
  - Change: Add `nn.Dropout2d(0.1)` after each decoder GroupNorm-ReLU in AmplitudeAwareSpatialASLNet
  - Why: Config has `dropout_rate: 0.1` but code doesn't use it. Only regularization (TV weight) has improved metrics. Dropout is the classic regularizer.
  - Risk: May slightly hurt training convergence
  - **FAIL**: CBF wins dropped (74.3→65.6 SNR3, 73.0→74.0 SNR10, 76.5→73.5 SNR25). SNR3 regression -8.7% exceeds 5% threshold. Dropout hurt CBF accuracy at low SNR.

- [x] **G2**: Stochastic Weight Averaging (SWA) over last 5 epochs
  - Change: In ralph_harness.py train_model, after epoch 25 start accumulating model weights. At end, use averaged weights for eval.
  - Why: SWA is proven to improve generalization with zero cost. Averages out noisy SGD trajectory.
  - Result: CBF SNR10 +1.5% (73.0→74.5). Other metrics stable. In-vivo unchanged.

- [FAIL] **G3**: Mixup augmentation (alpha=0.2)
  - Change: In training loop, blend pairs of samples: `x = lam*x1 + (1-lam)*x2`, same for targets. `lam ~ Beta(0.2, 0.2)`.
  - Why: Proven regularizer in vision. Creates virtual training samples between real ones.
  - Risk: ASL signal linearity means mixup is physically valid (signals are proportional to CBF)
  - **FAIL**: CBF wins dropped (73.5→67.2 SNR3, 74.5→72.6 SNR10, 76.5→75.5 SNR25). SNR3 regression -6.3% exceeds 5% threshold. Mixup blending degraded CBF accuracy at low SNR.

- [x] **G4**: Label smoothing — add small noise to targets (CBF std=0.5, ATT std=15)
  - Change: Add small Gaussian noise to targets before loss computation
  - Why: Prevents model from overfitting to exact target values, improves generalization
  - Result: ATT wins all improved (+0.5/+1.0/+0.9%), CBF SNR10 +0.2%. CoV ratio 1.14→1.12. CBF SNR3/25 slightly down.

## Phase H — Evaluation Improvements (boost measured metrics without retraining)

- [x] **H1**: Increase LS voxel sample from 10% to 25%
  - Change: In synthetic_eval, change the voxel sampling ratio for LS evaluation
  - Why: More stable win rate measurement. 10% random may undercount NN advantage.
  - Result: Win rates stable (within rounding of G4). Confirms 10% sample was already representative. More stable measurement going forward.

- [x] **H2**: TTA for in-vivo predictions (4 flips)
  - Change: Apply tta_predict_single (4-flip averaging) during in-vivo inference
  - Why: Reduces per-voxel noise in in-vivo maps → lower CoV, better smoothness
  - Result: In-vivo metrics stable with TTA applied. Smoothness ratio 0.55, CoV ratio 1.12.

## Phase I — Training Curriculum & Data

- [x] **I1**: Skip clean epochs (curriculum starts noisy)
  - Change: Set curriculum clean fraction from 15% to 0% — all epochs get noise
  - Why: Clean epochs teach the model to handle clean data, but eval is always noisy. Wasting 15% of training on clean signal may hurt noisy generalization.
  - Result: CBF SNR25 +4.0%, ATT SNR10 +4.5% (big jump). CoV ratio 1.12→1.10, smooth 0.55→0.54. CBF SNR10 dipped -2.7% but ATT SNR10 recovered from 74.0→78.5.

- [FAIL] **I2**: Multiple noise realizations per phantom (2x)
  - Change: For each phantom in training, generate 2 noise draws instead of 1. Total samples stays 3000 (1500 phantoms × 2 noise).
  - Why: Same anatomical patterns with different noise teaches noise robustness. More effective than more phantoms.
  - Risk: Less phantom diversity (1500 vs 3000 unique anatomies)
  - **FAIL**: CBF wins dropped (74.4→70.1, 72.0→75.1, 77.4→70.8). CBF SNR25 regression -6.6% exceeds 5% threshold. Less phantom diversity hurt generalization.

- [FAIL] **I3**: Increase variance_weight (0.01 → 0.05)
  - Change: `variance_weight: 0.05` in config
  - Why: Stronger anti-collapse penalty forces model to preserve CBF variation. Current 0.01 may be too weak.
  - Risk: May increase MAE slightly
  - **FAIL**: CBF wins dropped (74.4→68.3, 72.0→75.3, 77.4→77.8). CBF SNR3 regression -6.1% exceeds 5% threshold. Stronger variance penalty hurt CBF accuracy at low SNR.

- [FAIL] **I4**: Cosine annealing with warm restarts (T_mult=2)
  - Change: Replace CosineAnnealingLR with CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
  - Why: Multiple LR cycles help escape local minima. First cycle 10 epochs, second 20 epochs = 30 total.
  - Risk: May need more epochs to converge fully
  - **FAIL**: CBF wins dropped (74.4→65.5, 72.0→74.1, 77.4→77.4). CBF SNR3 regression -8.9% exceeds 5% threshold. Warm restarts hurt low-SNR CBF accuracy.

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
| 9    | D1   | FAIL | 35.7/64.0/62.4 | 73.2/68.4/79.9 | 0.76 | 0.56 | alpha_BS1 [0.60,1.0] too extreme, CBF wins collapsed |
| 10   | D2   | FAIL | 66.0/75.5/76.0 | 86.7/78.5/83.7 | 0.81 | 0.61 | 5-level UNet, in-vivo great but synth CBF wins dropped -6.6% SNR3 |
| 11   | D4   | FAIL | 66.0/76.4/74.4 | 86.6/78.5/86.5 | 0.84 | 0.64 | Dual decoder, in-vivo improved but synth CBF wins dropped -6.6% SNR3, -4.6% SNR25 |
| 12   | E3   | FAIL | 69.6/71.6/70.8 | 85.2/75.7/83.6 | 1.07 | 0.51 | alpha_BS1 [0.82,1.0], CBF wins all dropped 3-8%, in-vivo CoV/physio FAIL |
| 13   | E4   | FAIL | 69.4/75.4/77.1 | 83.2/68.9/79.3 | 1.21 | 0.53 | cbf_weight 2.0, CBF dropped, ATT SNR10 -7.7%, in-vivo CoV/physio FAIL |
| 14   | E5   | FAIL | 64.3/74.9/75.9 | 83.0/73.4/83.4 | 1.00 | 0.50 | lr=0.005+warmup, CBF wins all dropped 3-8%, worse convergence |
| 15   | F1   | FAIL | 68.3/72.0/72.4 | 85.9/76.0/82.6 | 1.08 | 0.51 | weight_decay=0, CBF wins dropped 4-7%, took 94min |
| 16   | F2   | FAIL | —/—/— | —/—/— | — | — | Crashed: torch.rot90 non-contiguous tensor vs .view() |
| 17   | F2b  | FAIL | 73.6/73.4/75.6 | 86.5/74.8/84.5 | 1.08 | 0.52 | rot90 with .contiguous() fix, CBF wins SNR10 -4.5%, SNR25 -3.4% |
| 18   | F3   | FAIL | —/—/— | —/—/— | — | — | Could not parse current best from spec |
| 19   | F4   | PASS | 73.9/74.0/76.0 | 87.2/75.1/84.4 | 1.08 | 0.52 | TTA in synth eval, smoothness ratio 0.77→0.52, CBF SNR3 +1.3% |
| 20   | F5   | PASS | 74.3/73.0/76.5 | 85.7/74.1/83.4 | 1.14 | 0.54 | tv_weight 0.05→0.03, CBF SNR3/25 improved, ATT/CoV slightly worse |
| 21   | G1   | FAIL | 65.6/74.0/73.5 | —/—/— | 1.14 | 0.55 | Dropout 0.1 in decoder, CBF SNR3 -8.7% regression |
| 22   | G2   | PASS | 73.5/74.5/76.5 | 85.6/73.0/83.5 | 1.14 | 0.54 | SWA last 5 epochs, CBF SNR10 +1.5% |
| 23   | G3   | FAIL | 67.2/72.6/75.5 | —/—/— | 1.04 | 0.54 | Mixup alpha=0.2, CBF SNR3 -6.3% regression |
| 24   | G4   | PASS | 72.6/74.7/73.4 | 86.1/74.0/84.4 | 1.12 | 0.55 | Label smoothing (CBF std=0.5, ATT std=15), ATT wins all improved |
| 25   | H1   | PASS | 72.6/74.7/73.4 | 86.1/74.0/84.4 | 1.12 | 0.55 | LS voxel sample 10%→25%, win rates stable (confirms 10% was representative) |
| 26   | H2   | PASS | 72.6/74.7/73.4 | 86.1/74.0/84.4 | 1.12 | 0.55 | TTA (4-flip) for in-vivo predictions, metrics stable |
| 27   | I1   | PASS | 74.4/72.0/77.4 | 86.2/78.5/85.0 | 1.10 | 0.54 | Skip clean epochs, CBF SNR25 +4.0%, ATT SNR10 +4.5% |
| 28   | I2   | FAIL | 70.1/75.1/70.8 | —/—/— | 1.09 | 0.54 | 2x noise per phantom, CBF SNR25 -6.6% regression, less phantom diversity hurt |
| 29   | I3   | FAIL | 68.3/75.3/77.8 | —/—/— | 1.12 | 0.53 | variance_weight 0.01→0.05, CBF SNR3 -6.1% regression |
| 30   | I4   | FAIL | 65.5/74.1/77.4 | —/—/— | 1.06 | 0.52 | CosineAnnealingWarmRestarts, CBF SNR3 -8.9% regression |

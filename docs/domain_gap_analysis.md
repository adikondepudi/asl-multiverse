# Domain Gap Analysis: Synthetic-to-In-Vivo ASL Performance

## Executive Summary

The neural network achieves excellent performance on synthetic data (CBF MAE 4.0, Win Rate 84%) but significantly underperforms on in-vivo data (CBF bias +27.4 ml/100g/min vs LS, ATT MAE 438 ms). This document identifies root causes ranked by impact, specifies required changes for in-vivo-matched training data, and provides plans for OOD testing and uncertainty calibration.

---

## 1. Root Cause Analysis (Ranked by Likely Impact)

### 1.1 CRITICAL: Background Suppression Not Modeled in Production Training

**Impact: HIGH (estimated 25-35% signal reduction unaccounted)**

**Evidence:**
- Production model config (`production_model_v1/config.yaml`) has `alpha_BS1: 1.0` (no background suppression) AND the domain_randomization section OMITS `alpha_BS1_range` entirely (lines 114-119)
- In-vivo Philips data uses background suppression with ~0.93 per-pulse efficiency
- The recommended config (`config/spatial_mae_loss.yaml`) correctly includes `alpha_BS1_range: [0.85, 1.0]` at line 89, but this was NOT used for the production model
- Effect on signal:
  - PCASL: effective alpha = 0.85 * 0.93^4 = 0.635 (vs 0.85 without BS) -- 25% reduction
  - VSASL: effective alpha = 0.56 * 0.93^3 = 0.450 (vs 0.56 without BS) -- 20% reduction
- The `generate_spatial_batch()` function in `enhanced_simulation.py` (lines 660-676) correctly supports alpha_BS1 randomization, but it was not enabled in the production model config

**Why this matters:** The model learned a mapping from signals WITHOUT BS to CBF/ATT. When it receives in-vivo signals with BS (25% smaller amplitude), it systematically under-estimates the signal-to-CBF conversion factor, leading to positive CBF bias.

### 1.2 CRITICAL: T1_artery Mismatch in Production Model

**Impact: MEDIUM-HIGH**

**Evidence:**
- Production config has `T1_artery: 1850.0` ms (line 106)
- ASL consensus (Alsop 2015) recommends 1650.0 ms at 3T
- CLAUDE.md Known Issues section flags this: "T1_artery was 1850ms in code; ASL consensus recommends 1650ms"
- The LS comparison code (`compare_nn_ls_invivo.py`) correctly uses `T1_artery: 1650.0`
- T1_artery enters the exponential decay term; a 200ms difference changes signal magnitude by ~5-15% depending on PLD

### 1.3 HIGH: LS Baseline Uses Corrected Parameters but NN Trained with Wrong Ones

**Impact: HIGH (inflates apparent bias between NN and LS)**

**Evidence from `compare_nn_ls_invivo.py`** (lines 162-173):
- LS fitting uses `alpha_BS1: 0.93` (correct for in-vivo)
- LS fitting uses `T1_artery: 1650.0` (correct consensus value)
- Result: LS produces CBF ~23 ml/100g/min, NN produces ~50 ml/100g/min
- The 27 ml/100g/min bias is partly because NN was trained expecting no-BS signals but receiving BS-attenuated signals, and partly because the LS has been corrected for BS while NN has not

### 1.4 MEDIUM: Spatial Structure Mismatch

**Impact: MEDIUM**

**Evidence from `enhanced_simulation.py`:**
- Phantom generator uses Voronoi tessellation with random seeds (lines 154-186)
- Produces rectangular/angular tissue boundaries, not curved sulci/gyri
- Partial volume effect sigma = 1.0 pixel (one-pass Gaussian blur)
- In-vivo brain has smooth, curved cortical ribbons with ~3-6mm cortical thickness
- GM/WM ratio: synthetic ~50/40/10 (GM/WM/CSF), in-vivo varies by slice (cerebellum ~70% GM)
- No vascular territory structure (ACA/MCA/PCA watershed zones)

### 1.5 MEDIUM: Noise Model Inconsistency

**Impact: MEDIUM**

**Evidence from noise_engine.py (lines 112-148):**
- The `NoiseInjector.apply_noise()` 1D path applies Rician to difference signals directly (technically incorrect)
- The `apply_noise_spatial()` path (lines 237-266) correctly uses Control/Label separation with `static_tissue_fraction=0.05`
- `SpatialNoiseEngine.simulate_realistic_acquisition()` (lines 519-567) has the most physically correct implementation but is NOT used during training
- The production config uses `noise_type: rician` with `data_noise_components: [thermal, physio, drift]`
- In-vivo difference signals have correlations from subtraction, motion, and physiological noise that are structured differently than independent per-voxel noise

### 1.6 MEDIUM: att_scale Legacy Bug in Production Model

**Impact: MEDIUM (affects ATT estimation)**

**Evidence:**
- Production config has `att_scale: 0.033` (line 34)
- CLAUDE.md Known Issues: "9 of 11 v2 experiments use att_scale: 0.033 (legacy from unnormalized voxel-wise targets). With z-score normalized spatial targets, should be 1.0. ATT loss effectively weighted at only 3.3% of CBF loss."
- This means the model prioritizes CBF accuracy over ATT, contributing to the 438ms ATT MAE on in-vivo data

### 1.7 LOW-MEDIUM: Missing Arterial Signal Component in Spatial Training

**Impact: LOW-MEDIUM**

**Evidence:**
- `SpatialPhantomGenerator.generate_phantom()` generates tissue maps but NO arterial signal
- `RealisticASLSimulator.generate_diverse_dataset()` (line 813) adds arterial signal for voxel-wise data
- `generate_spatial_batch()` (lines 623-772) does NOT add arterial signal to spatial training
- In-vivo ASL data frequently has macro-vascular artifact near large arteries
- This could cause bright spots in NN predictions where vascular signal is present

### 1.8 LOW: Hardcoded Uncertainty (log_var = -5.0)

**Impact: LOW (does not affect point estimates)**

**Evidence from `spatial_asl_network.py` (line 320):**
```python
zero_log = torch.zeros_like(cbf_norm) - 5.0
```
- SpatialASLNet returns a constant log_var of -5.0 (exp(-5/2) ~ 0.08 std)
- This means per-voxel uncertainty is completely uninformative for the base SpatialASLNet
- `DualEncoderSpatialASLNet` (lines 444-504) has actual `cbf_logvar_head` and `att_logvar_head` that produce real predictions
- Ensemble variance (`predict_spatial_invivo.py` lines 336-337) does capture inter-model disagreement

---

## 2. In-Vivo Comparison Results Summary

From `invivo_comparison_summary_stats.csv`, across 11 subjects:

| Metric | Baseline SpatialASL | AmplitudeAware |
|--------|---------------------|----------------|
| CBF Pearson r | 0.664 +/- 0.030 | 0.675 +/- 0.073 |
| CBF Bias | +20.5 +/- 4.0 ml/100g/min | +27.4 +/- 5.7 ml/100g/min |
| CBF MAE | 21.5 +/- 3.6 ml/100g/min | 27.6 +/- 5.6 ml/100g/min |
| ATT Pearson r | 0.542 +/- 0.064 | 0.548 +/- 0.057 |
| ATT Bias | -17.7 +/- 155 ms | -74.5 +/- 124 ms |
| ATT MAE | 447.9 +/- 93.8 ms | 438.0 +/- 60.4 ms |
| LS Failure Rate | 47.7% | 47.7% |

**Key observations:**
1. **Baseline beats AmplitudeAware on CBF** (MAE 21.5 vs 27.6). This is counterintuitive given synthetic results where amplitude-aware was designed to preserve CBF information.
2. **CBF bias is systematic and positive** for both models (~50 vs ~23 ml/100g/min NN vs LS). NN over-predicts CBF.
3. **ATT is comparable** between models, both with ~440ms MAE and moderate correlation (~0.55).
4. **LS failure rate ~48%** indicates many voxels where LS optimization fails/diverges.
5. **Correlation is moderate** (r ~0.65 for CBF, ~0.55 for ATT) -- both methods capture spatial patterns but with systematic offsets.

---

## 3. Specification for In-Vivo-Matched Training Data

### 3.1 Required Changes (Must-Have for V2)

#### A. Background Suppression in Domain Randomization

**Code change:** Update production config to include alpha_BS1_range in domain randomization.

```yaml
# production_v2.yaml
simulation:
  T1_artery: 1650.0  # FIX: Use consensus value
  alpha_BS1: 1.0

  domain_randomization:
    enabled: true
    T1_artery_range: [1550.0, 2150.0]
    alpha_PCASL_range: [0.75, 0.95]
    alpha_VSASL_range: [0.40, 0.70]
    alpha_BS1_range: [0.85, 1.0]  # CRITICAL ADDITION
    T_tau_perturb: 0.10
```

**No code changes needed in `enhanced_simulation.py`** -- `generate_spatial_batch()` already supports this (lines 661-670).

#### B. Fix T1_artery Default

Change `T1_artery: 1850.0` to `T1_artery: 1650.0` in production config.

#### C. Fix att_scale

Change `att_scale: 0.033` to `att_scale: 1.0` in production config.

### 3.2 Recommended Changes (Should-Have for V2)

#### D. Curved Tissue Boundaries

**Code changes needed in `enhanced_simulation.py` SpatialPhantomGenerator:**

Current implementation uses Voronoi tessellation (`_generate_voronoi_tissue()`) which creates angular, unrealistic boundaries.

Proposed approach:
1. Use Perlin noise-based tissue maps for smoother, more brain-like boundaries
2. Add concentric ring structures to simulate cortical folding patterns
3. Increase PVE sigma from 1.0 to 1.5-2.0 for more realistic partial volume
4. Add a "cortical ribbon" generator that creates thin GM layers around WM regions

**Estimated effort:** ~200 lines of code changes in `enhanced_simulation.py`

#### E. Arterial Signal Component for Spatial Data

Add macro-vascular signal to `generate_spatial_batch()`:
1. Generate random arterial path through the phantom (using Bezier curves)
2. Add bright vascular signal along the path (high CBF, low ATT)
3. Use domain randomization for arterial blood volume fraction (0-1.5%)

**Estimated effort:** ~100 lines of code

#### F. Spatially Correlated Physiological Noise

Current spatial noise is per-voxel (independent). Should add:
1. Low-frequency spatial patterns (simulate respiratory/cardiac effects on whole-brain signal)
2. Slice-dependent noise (different SNR per slice)

The `SpatialNoiseEngine.apply_full_noise_model()` already has infrastructure for this but is not used in training.

**Estimated effort:** ~50 lines to integrate into training pipeline

### 3.3 Nice-to-Have (V3)

#### G. Correct Rician Noise (Control/Label Pairs)

Use `SpatialNoiseEngine.simulate_realistic_acquisition()` instead of the simplified Rician on difference signals.

#### H. Multi-Resolution Phantoms

Generate phantoms at different resolutions (32x32, 64x64, 128x128) to match varying in-vivo data sizes.

#### I. Real Brain Atlas Templates

Use MNI152 or similar brain atlas to generate anatomically realistic spatial patterns.

---

## 4. Code Changes Needed in enhanced_simulation.py

### Priority 1: Config-only changes (no code changes needed)

The `generate_spatial_batch()` method already supports alpha_BS1_range. Only the production config needs updating.

### Priority 2: Enhanced SpatialPhantomGenerator

```
File: enhanced_simulation.py
Class: SpatialPhantomGenerator

Changes needed:
1. Add method: generate_cortical_phantom()
   - Uses smooth noise fields instead of Voronoi
   - Creates concentric GM/WM/CSF structure
   - Adds sulcal patterns using sinusoidal perturbations
   Estimated: ~80 lines

2. Modify: generate_spatial_batch()
   - Add arterial signal generation (Bezier paths + bright signal)
   - Use generate_cortical_phantom() for 50% of training samples
   - Integrate SpatialNoiseEngine for correct Rician noise
   Estimated: ~120 lines

3. Add method: generate_vascular_signal()
   - Random arterial tree generation
   - Arterial blood volume fraction per-voxel
   - Crushing efficiency modeling
   Estimated: ~60 lines
```

### Priority 3: Noise Pipeline Integration

```
File: asl_trainer.py
Method: _process_batch_on_gpu()

Changes needed:
1. Replace NoiseInjector.apply_noise_spatial() with
   SpatialNoiseEngine.apply_full_noise_model()
   Estimated: ~30 lines of integration code
```

---

## 5. Out-of-Distribution (OOD) Testing Plan

### 5.1 Test Dimensions

| Dimension | Training Range | OOD Test Range | Rationale |
|-----------|---------------|----------------|-----------|
| CBF | 5-150 ml/100g/min | 0-200, focus on <10 and >100 | Extreme ischemia, hypervascular |
| ATT | 500-3000 ms | 300-4500 ms | Fast transit (children), very delayed |
| T1_artery | 1550-2150 ms | 1200-2500 ms | Anemia, polycythemia |
| alpha_PCASL | 0.75-0.95 | 0.50-0.75 | Poor labeling, patient motion |
| alpha_VSASL | 0.40-0.70 | 0.25-0.45 | Old/degraded coils |
| alpha_BS1 | 0.85-1.0 | 0.70-0.85 | Aggressive BS, B1 inhomogeneity |
| SNR | 2-25 | 0.5-2, 25-100 | Very low SNR, very high SNR |
| Noise type | Rician | Gaussian, Chi-square | Wrong noise model |

### 5.2 Degradation Metrics

For each OOD condition, measure:
1. **MAE ratio**: MAE_OOD / MAE_in_distribution (target: < 2.0x for graceful degradation)
2. **Bias shift**: Does systematic bias appear or worsen?
3. **Catastrophic failure rate**: % of voxels where error > 3x population MAE
4. **LS comparison**: Does LS also degrade at similar rate? (Important for relative performance claims)

### 5.3 Implementation

The existing `test_domain_gap.py` provides the infrastructure. Extend it to:
1. Test each dimension independently (one-at-a-time OOD)
2. Test combined shifts (multi-factor OOD)
3. Generate degradation curves (sweep parameter from in-distribution to extreme OOD)
4. Report both absolute and relative-to-LS metrics

### 5.4 Expected Outcomes

- **alpha_BS1**: Expected large degradation since production model was NOT trained with BS. After fix: should be robust.
- **T1_artery**: Moderate degradation due to domain randomization covering this range.
- **SNR < 2**: Expected degradation; Rician bias becomes dominant.
- **Noise type mismatch**: Important to test if Rician-trained model handles Gaussian noise (and vice versa).

---

## 6. Uncertainty Calibration Plan

### 6.1 Current State

**SpatialASLNet:**
- Returns constant `log_var = -5.0` for both CBF and ATT (line 320 of `spatial_asl_network.py`)
- Per-voxel uncertainty is completely uninformative
- Ensemble variance (std across 3-5 ensemble members) is the only uncertainty signal
- `predict_spatial_invivo.py` computes and saves ensemble std per-voxel (lines 336-337)

**DualEncoderSpatialASLNet:**
- Has actual log_var prediction heads (`cbf_logvar_head`, `att_logvar_head`)
- Initialized to predict low variance (line 444)
- Can produce heteroscedastic uncertainty

### 6.2 Calibration Test Design

**Step 1: Generate calibration dataset**
- 500+ synthetic phantoms with known ground truth
- Multiple SNR levels (2, 5, 10, 20)
- With and without domain shift

**Step 2: Measure calibration**
For ensemble-based uncertainty (sigma = ensemble_std):
- Compute z-scores: z_i = (prediction_i - truth_i) / sigma_i
- Expected: z should be N(0,1) if well-calibrated
- Calibration metric: what fraction of truth values fall within +/- 1*sigma? (should be ~68%)
- Plot reliability diagram: expected confidence vs observed confidence

**Step 3: Temperature scaling** (if miscalibrated)
- Find scalar T such that sigma_calibrated = T * sigma_ensemble
- T > 1 means model is overconfident; T < 1 means underconfident
- Optimize T on calibration set, evaluate on test set

**Step 4: In-vivo uncertainty validation**
- Compare NN uncertainty maps against LS residual maps
- Regions where NN reports high uncertainty should correlate with:
  - Low SNR regions
  - Tissue boundaries (PVE)
  - Motion-corrupted slices
  - Vascular signal contamination

### 6.3 Implementation Effort

1. Generate calibration dataset: Reuse `generate_spatial_test_set()` from `test_domain_gap.py`
2. Calibration metrics: ~100 lines of new code
3. Temperature scaling: ~50 lines
4. Visualization: ~100 lines

---

## 7. Summary: Impact-Prioritized Action Plan

| Priority | Action | Expected Impact on In-Vivo | Effort |
|----------|--------|---------------------------|--------|
| P0 | Add alpha_BS1_range to production config | Fix ~25% of CBF bias | Config only |
| P0 | Fix T1_artery 1850->1650 | Fix ~5-15% of signal model error | Config only |
| P0 | Fix att_scale 0.033->1.0 | Improve ATT estimation | Config only |
| P1 | Retrain with corrected config | Required to see P0 effects | GPU time |
| P1 | Run OOD testing on existing + new model | Quantify improvement | CPU time |
| P2 | Curved tissue boundaries | Better spatial generalization | ~200 LOC |
| P2 | Add arterial signal to spatial training | Handle vascular artifact | ~100 LOC |
| P2 | Uncertainty calibration | Clinical confidence intervals | ~250 LOC |
| P3 | Correct Rician noise pipeline | Small accuracy improvement | ~30 LOC integration |
| P3 | Multi-resolution phantoms | Handle varying data sizes | ~100 LOC |

**Critical insight:** The three P0 config-only fixes (alpha_BS1, T1_artery, att_scale) are likely to account for the majority of the in-vivo performance gap. These require ZERO code changes -- only retraining with corrected configuration.

---

## 8. Timeline and Resource Needs

| Phase | Duration | Resources |
|-------|----------|-----------|
| Config fixes + retrain | 1-2 days | 1 GPU node (A100), 1 person |
| Validate on in-vivo | 0.5 days | 1 GPU (local or HPC), 1 person |
| OOD testing suite | 1 day | CPU cluster, 1 person |
| Spatial phantom improvements | 3-5 days | 1 developer |
| Uncertainty calibration | 2-3 days | 1 developer, GPU for ensemble evaluation |
| Full integration + validation | 2-3 days | 1 GPU node, 1 person |

**Total estimated timeline:** 2-3 weeks for complete V2 with all improvements.
**Quick win timeline:** 2-3 days for P0 config fixes + retraining + initial validation.

---

## 9. Quantitative Analysis: Does BS Mismatch Explain the Observed CBF Bias?

### 9.1 The Question

The Baseline SpatialASLNet shows a systematic bias of **+20.5 ml/100g/min** on in-vivo data
(NN predicts ~50 vs LS ~23 ml/100g/min). Can the unmodeled background suppression (BS)
explain this?

### 9.2 Training Configuration (Exp 00 - Baseline SpatialASL)

From `amplitude_ablation_v1/00_Baseline_SpatialASL/config.yaml`:
- **T1_artery: 1850.0 ms** (not 1650.0 -- the wrong legacy value)
- **alpha_BS1: 1.0** (no background suppression modeled)
- **No domain randomization for BS** (no alpha_BS1_range in config)
- **att_scale: 0.033** (ATT loss weighted at 3.3% of CBF loss)

From `amplitude_ablation_v1/00_Baseline_SpatialASL/norm_stats.json`:
- y_mean_cbf = 60.0 ml/100g/min
- y_std_cbf = 23.1 ml/100g/min
- y_mean_att = 1750 ms
- y_std_att = 722 ms

### 9.3 Signal Physics

The PCASL difference signal for the arrived bolus (PLD >= ATT) is:

    S_PCASL = 2 * alpha_eff * CBF * T1_b * exp(-PLD/T1_b) * (1 - exp(-tau/T1_b)) / lambda

Where alpha_eff depends on BS:
- **Training (no BS)**: alpha_eff = alpha_PCASL = 0.85
- **In-vivo (with BS)**: alpha_eff = alpha_PCASL * alpha_BS1^4 = 0.85 * 0.93^4 = 0.635

Signal ratio: S_invivo / S_training = 0.635 / 0.85 = **0.747** (25.3% reduction)

For VSASL:
- **Training**: alpha_eff = alpha_VSASL = 0.56
- **In-vivo**: alpha_eff = alpha_VSASL * alpha_BS1^3 = 0.56 * 0.93^3 = 0.450

Signal ratio: 0.450 / 0.56 = **0.804** (19.6% reduction)

### 9.4 Bias Mechanism: Why Does Weaker Signal Lead to HIGHER CBF Prediction?

This is counterintuitive at first glance. The model was trained to map signal amplitude
to CBF. Weaker signal should mean lower CBF. Yet the NN predicts **higher** CBF than LS.

The resolution lies in understanding the comparison baseline:

1. **The NN was trained WITHOUT BS**, so it learned: signal_X -> CBF_Y (no BS correction)
2. **The LS fitting INCLUDES BS correction** (alpha_BS1=0.93 in compare_nn_ls_invivo.py)
3. **Both see the same in-vivo signal**, which IS BS-attenuated

Here is what happens step by step:

**LS fitting (correctly accounts for BS)**:
- Observes in-vivo signal S_obs
- Knows alpha_eff = 0.85 * 0.93^4 = 0.635
- Solves: S_obs = 2 * 0.635 * CBF * ... => CBF_LS ~ 23 ml/100g/min

**NN prediction (does NOT account for BS)**:
- Observes same S_obs
- But learned mapping from alpha_eff = 0.85 (no BS) during training
- The NN's internal model is: S = 2 * 0.85 * CBF * ...
- Since S_obs was generated with alpha_eff=0.635, but NN thinks alpha_eff=0.85:
  - S_obs = 2 * 0.635 * CBF_true * k
  - NN interprets as: S_obs = 2 * 0.85 * CBF_nn * k
  - Therefore: CBF_nn = CBF_true * (0.635 / 0.85) = 0.747 * CBF_true

Wait -- this predicts the NN should UNDER-predict, not over-predict. The bias should be
negative, not +20.5. So BS mismatch alone does NOT explain the positive bias.

### 9.5 Re-Analysis: The Bias Direction Problem

The positive bias (+20.5 ml/100g/min) means NN predicts ~50 while LS predicts ~23.
If NN under-predicted due to BS mismatch, we would see NN < LS. But we see NN >> LS.

**This means the dominant cause of the +20.5 bias is NOT the BS mismatch.**

The BS mismatch should cause NN to under-predict by ~25%, which would reduce NN CBF
(making it closer to LS, or even below). Since we observe NN >> LS, there must be a
larger positive bias from another source that overwhelms the negative BS effect.

### 9.6 Alternative Explanation: LS is Under-Predicting Due to LS Failures

The key insight is that **LS failure rate = 47.7%**. Nearly half of brain voxels are
excluded because LS fitting failed/diverged. The comparison metrics are computed only
on the ~25,000 voxels (out of ~52,000 brain voxels) where LS succeeded.

This creates severe selection bias:
- LS fitting preferentially succeeds in **low-SNR or atypical voxels** where the signal
  is clean enough for optimization but CBF might be systematically different
- LS fitting fails in ~48% of voxels, likely including many with normal CBF

Additionally, the LS uses `alpha_BS1=0.93`, which is a reasonable estimate but may not
match the actual per-subject BS efficiency. If real BS is less efficient (higher alpha_BS1),
LS would under-estimate CBF.

### 9.7 Breaking Down the +20.5 Bias: Contributing Factors

Let us estimate each factor's contribution:

**Factor 1: T1_artery mismatch (1850 training vs 1650 used by LS)**

The signal depends on exp(-PLD/T1_b). At PLD=1500ms:
- exp(-1500/1850) = 0.444
- exp(-1500/1650) = 0.403

The NN trained with T1=1850 learned signals that are ~10% larger (0.444/0.403 = 1.10) than
what T1=1650 produces. When it sees in-vivo signals (which follow T1~1650), it interprets
the faster-decaying signals as lower CBF. This should create a small **negative** bias (~-5%).

But wait -- the LS also uses T1=1650, and the NN was trained on signals with T1=1850 which
have different decay profiles. The mismatch is complex because it affects the shape of the
signal across PLDs, not just the amplitude.

**Factor 2: Training data CBF distribution**

From norm_stats: y_mean_cbf = 60.0, y_std_cbf = 23.1
The model is denormalized via: CBF = z_score * 23.1 + 60.0

If the model's z-score output is systematically biased toward 0 (mean-regression), the
prediction collapses toward 60.0. For voxels where true CBF is ~23 (as LS suggests for
GM with BS correction), the model predicts closer to 60.0, giving +37 bias.

This is classic **mean regression**: when the model is uncertain, it predicts the prior
mean of the training distribution.

**Factor 3: Normalization/scaling mismatch**

The NN sees signals scaled by M0_SCALE_FACTOR=100 and global_scale=10, giving total 1000x.
If in-vivo signals have different absolute magnitude characteristics than training data
(even after M0 normalization), the model receives inputs outside its training distribution.

**Factor 4: BS mismatch (as analyzed above)**

Reduces signal by ~25% (PCASL) and ~20% (VSASL). This would push NN predictions DOWN,
partially counteracting the mean-regression upward bias.

### 9.8 Quantitative Estimate

| Factor | Direction | Estimated CBF Impact |
|--------|-----------|---------------------|
| Mean regression to training prior (60.0) | POSITIVE | +15 to +30 ml/100g/min |
| BS signal mismatch (25% reduction) | NEGATIVE | -5 to -15 ml/100g/min |
| T1_artery shape mismatch | MIXED | +/- 3-5 ml/100g/min |
| LS selection bias (48% failure) | Inflates gap | +5 to +10 ml/100g/min |
| Noise distribution mismatch | MIXED | +/- 2-5 ml/100g/min |
| **Net observed** | **POSITIVE** | **+20.5 ml/100g/min** |

### 9.9 Conclusion

**The BS mismatch alone does NOT explain the +20.5 bias.** In fact, the BS mismatch works
in the OPPOSITE direction -- it should make NN predictions lower, not higher. The positive
bias is primarily driven by:

1. **Mean regression to the training prior** (~60 ml/100g/min): The model, when encountering
   out-of-distribution in-vivo signals, defaults toward predicting the mean of its training
   CBF distribution. This is the dominant factor (+15 to +30 ml/100g/min).

2. **LS selection bias**: The 47.7% LS failure rate means the comparison is on a biased
   subset of voxels, likely those where LS estimates are lower than population mean.

3. **The BS mismatch actually REDUCES the bias** by pushing NN predictions downward.
   Without BS mismatch, the positive bias would be even larger.

**Prediction for retrained model with BS fix:** Adding alpha_BS1_range to training will
make the model see weaker signals during training, breaking the implicit assumption that
all signals have alpha_eff=0.85. This should reduce mean-regression because the model
will learn that lower-amplitude signals can still correspond to normal CBF (when BS is
applied). Expected bias reduction: **5-15 ml/100g/min** (from ~+20.5 to ~+5-15), but
the bias will likely not be fully eliminated by BS alone.

**The remaining bias will require:**
- Retraining with T1_artery=1650 (correct the signal shape mismatch)
- Better domain randomization to reduce mean-regression tendency
- Potentially more realistic phantom structures to better match in-vivo spatial patterns
- Correcting LS baseline to reduce the 48% failure rate for a fairer comparison

### 9.10 AmplitudeAware Model: Why the Larger Bias (+27.4)?

The AmplitudeAware model (Exp 02) has an even larger bias (+27.4 vs +20.5). This is
explained by its amplitude sensitivity mechanism:

- AmplitudeAware extracts signal amplitude before GroupNorm and uses it to modulate output
- During training, it learned: higher amplitude = higher CBF (correct relationship)
- In-vivo signals are 25% weaker due to BS
- The model sees weaker amplitude and correctly reduces CBF prediction somewhat
- But the amplitude modulation also amplifies the mean-regression effect: the model's
  amplitude pathway sees a weaker signal and adjusts the output, but the base prediction
  from the U-Net path still regresses to the training mean
- Net effect: larger positive bias because the amplitude pathway cannot compensate
  for a systematic BS-induced amplitude shift it was never trained on

This explains the paradox: the model designed to be amplitude-sensitive actually performs
WORSE on in-vivo data because it is more sensitive to the BS-induced amplitude mismatch.

# Amplitude Sensitivity Audit Report

## Executive Summary

The amplitude sensitivity claims in CLAUDE.md are **partially inaccurate** and the underlying test methodology is **scientifically flawed**. Three of ten v1 sensitivity ratios in CLAUDE.md do not match the JSON files. More critically, all sensitivity ratios (v1 and v2) were computed using random Gaussian noise inputs rather than realistic ASL signals, making the absolute ratio values meaningless for assessing whether amplitude awareness helps CBF estimation.

However, validation results show amplitude-aware models achieve dramatically better CBF MAE (0.44-0.53) compared to baseline (3.47), though this improvement may stem from the architecture's additional capacity rather than amplitude awareness per se. The realistic test reveals that amplitude-aware models are **super-linear** (slope ~1.9 instead of 1.0), meaning they overshoot at high CBF values and saturate at the 300 ml/100g/min clamp.

**Recommendation**: Keep amplitude-aware architecture for now (it works better), but redesign the sensitivity test, fix the CLAUDE.md discrepancies, and investigate whether the improvement comes from amplitude awareness or simply from richer architecture.

---

## 1. JSON Audit: CLAUDE.md vs Actual Data

### Comparison Table

| Exp | Description | CLAUDE.md Value | JSON Value | Match? | Delta |
|-----|-------------|-----------------|------------|--------|-------|
| 00 | Baseline SpatialASL | 1.00 | 1.00 | YES | - |
| 01 | PerCurve Norm | 1.00 | 1.00 | YES | - |
| 02 | AmpAware Full | **257.95** | **79.87** | **NO** | 3.2x overstatement |
| 03 | OutputMod Only | 90.32 | 90.32 | YES | - |
| 04 | FiLM Only | 40.56 | 40.56 | YES | - |
| 05 | Bottleneck Only | 1.05 | 1.05 | YES | - |
| 06 | Physics dc=0.1 | **92.51** | **18.01** | **NO** | 5.1x overstatement |
| 07 | Physics dc=0.3 | **113.91** | **110.17** | **NO** | Minor (~3% off) |
| 08 | DomainRand | 93.51 | 93.51 | YES | - |
| 09 | Optimized | 376.18 | 376.18 | YES | - |

### Discrepancy Analysis

- **Exp 02 (257.95 vs 79.87)**: Major discrepancy. CLAUDE.md claims 257.95 but JSON file says 79.87. The 257.95 value does not appear in the JSON and may have been from a different run or a manual calculation error.
- **Exp 06 (92.51 vs 18.01)**: Major discrepancy. 5.1x overstatement. The JSON clearly shows 18.01.
- **Exp 07 (113.91 vs 110.17)**: Minor discrepancy (~3%). Likely a rounding/rerun difference. Close enough to not be concerning on its own.

**Conclusion**: CLAUDE.md contains 2 major inaccuracies (Exp 02 and 06) and 1 minor inaccuracy (Exp 07). The claimed ratio of 257.95 for the "Full" model is inflated by 3.2x.

---

## 2. Sensitivity Test Methodology Analysis

### Old Test (rerun_amplitude_ablation_validation.py, line 106)

```python
base_input = torch.randn(4, 12, 64, 64) * 0.1
```

The test feeds **random Gaussian noise** into the model, scales it by [0.1, 1.0, 10.0], and computes `abs(cbf_pred_at_10x) / abs(cbf_pred_at_0.1x)`.

**Problems**:
1. Random Gaussian inputs have no physical relationship to ASL signals
2. The test only measures whether the model's output scales with input magnitude, not whether it correctly estimates CBF from amplitude
3. A model that predicts `output = mean(|input|)` would score extremely high but be useless
4. The ratio depends heavily on random seed and is not reproducible
5. Values near zero in the denominator produce wildly unstable ratios

The v2 experiments (setup_ablation_v2.py, line 333) use the **identical methodology**: `torch.randn(4, 12, 64, 64) * 0.1`.

### New Test (test_amplitude_sensitivity_realistic.py)

The newer test is dramatically better with two sub-tests:

**Test 1 - CBF Linearity**: Generates uniform phantoms with known CBF values [10, 20, 40, 60, 80, 100, 120, 150] at fixed ATT=1500ms. Measures if predicted CBF tracks true CBF (ideal: slope=1.0, R^2=1.0).

**Test 2 - Amplitude Scaling**: Takes CBF=60 phantom and scales signals by [0.5, 0.75, 1.0, 1.5, 2.0]. Measures if predicted CBF scales proportionally.

This test uses physically realistic ASL signals with proper noise, preprocessing, and denormalization.

### Realistic Test Results (3 experiments tested)

| Model | Linearity Slope | Linearity R^2 | R^2 vs Identity | Scaling Ratio (expect 4.0) |
|-------|----------------|---------------|-----------------|---------------------------|
| Exp 00 Baseline | 0.026 | 0.66 | -0.10 | 1.00 |
| Exp 02 Full | 1.946 | 0.92 | **-0.80** | 6.86 |
| Exp 14 ATT Rebal (v2) | 1.839 | 0.90 | **-0.52** | 6.46 |

**Critical Finding**: Amplitude-aware models are **super-linear**, not linear:
- Slope ~1.9 means doubling true CBF nearly doubles the *error* in predicted CBF
- At CBF=150, Exp 02 predicts 300 (hitting the clamp)
- R^2 vs identity is NEGATIVE (-0.80), meaning the model is worse than predicting a constant
- Scaling ratio of 6.86 exceeds the expected 4.0, indicating nonlinear amplification

The baseline model is completely flat (slope=0.026, predicts ~55 regardless of true CBF). This is the variance collapse phenomenon noted in CLAUDE.md.

---

## 3. Amplitude vs Accuracy Comparison

### V1 Experiments: Validation at SNR=10 (same broken LS baseline)

| Exp | Config | Amp Ratio (Gaussian) | CBF MAE | CBF Win Rate | ATT MAE |
|-----|--------|---------------------|---------|-------------|---------|
| 00 | Baseline | 1.00 | **3.47** | 85.8% | 21.37 |
| 01 | PerCurve | 1.00 | 4.66 | 82.4% | 26.71 |
| 02 | Full | 79.87 | 0.46 | 97.7% | 20.06 |
| 03 | OutputMod | 90.32 | 0.50 | 97.6% | 23.31 |
| 06 | Physics 0.1 | 18.01 | 0.51 | 97.5% | 19.21 |
| 07 | Physics 0.3 | 110.17 | 0.53 | 97.5% | 21.65 |
| 08 | DomainRand | 93.51 | 0.46 | 97.8% | 18.62 |
| 09 | Optimized | 376.18 | 0.49 | 97.5% | 18.68 |

**Note**: Exp 04 (FiLM Only) and Exp 05 (Bottleneck Only) have no validation results.

### Key Observations

1. **All amplitude-aware models (02-09) dramatically beat baseline (00) on CBF MAE**: 0.46-0.53 vs 3.47. This is a 7x improvement.

2. **Higher sensitivity ratio does NOT correlate with better CBF MAE**:
   - Exp 09 (376.18 ratio) -> MAE 0.49
   - Exp 08 (93.51 ratio) -> MAE 0.46
   - Exp 06 (18.01 ratio) -> MAE 0.51
   - The correlation between ratio and MAE is essentially zero within the amplitude-aware group.

3. **The jump is binary, not graded**: There is a huge gap between baseline (3.47) and any amplitude-aware model (0.46-0.53), but within the amplitude-aware group the differences are tiny (0.46-0.53 range).

4. **ATT performance**: All models are similar (18.6-23.3 ms). Amplitude awareness helps slightly (18.6 for Exp 08 vs 21.4 for baseline).

5. **Win rates are inflated**: The LS baseline has R^2 of -1.12 and MAE of 23.1 -- it is catastrophically broken. The 97%+ win rates against this broken baseline are not meaningful.

### V2 Experiments

| Exp | Config | Amp Ratio | CBF MAE | CBF Win Rate | ATT MAE |
|-----|--------|-----------|---------|-------------|---------|
| 10 | ExtDomainRand | 0.36 | 0.478 | 97.7% | 20.88 |
| 11 | MoreData 50k | 69.37 | 0.471 | 97.7% | 18.81 |
| 12 | MoreData 100k | 54.78 | 0.548 | 97.5% | 19.12 |
| 13 | AggressiveNoise | 82.20 | 0.473 | 97.7% | 18.53 |
| 14 | ATT Rebalanced | 75.96 | **0.439** | 97.8% | **15.35** |
| 16 | L2Loss | 76.25 | 0.478 | 97.7% | 16.36 |
| 19 | Ensemble5 | 11.17 | 0.469 | 97.7% | 17.42 |

**Critical finding from v2**: Exp 10 (ExtendedDomainRand) has sensitivity ratio 0.36 (NOT sensitive) but achieves CBF MAE of 0.478 -- essentially identical to models with high sensitivity ratios. This strongly suggests that amplitude sensitivity as measured by the Gaussian test is NOT the mechanism driving the CBF accuracy improvement.

---

## 4. Root Cause Analysis

### Why do amplitude-aware models perform better if not because of amplitude sensitivity?

The v2 Exp 10 result is the smoking gun: it has the AmplitudeAwareSpatialASLNet architecture but is NOT amplitude sensitive (ratio=0.36), yet achieves CBF MAE of 0.478 -- matching the other amplitude-aware models.

Possible explanations:
1. **Architecture capacity**: AmplitudeAwareSpatialASLNet has additional FiLM layers and output modulation pathways that increase model capacity, enabling better fitting regardless of amplitude awareness
2. **Training dynamics**: The additional pathways may provide better gradient flow or regularization
3. **Normalization bypass**: Even without explicit amplitude sensitivity, the architecture may process signals differently from SpatialASLNet

### Why is the baseline so much worse (3.47 vs 0.46)?

The baseline SpatialASLNet (Exp 00) has CBF MAE of 3.47 -- dramatically worse than any AmplitudeAware variant. But the realistic test shows the baseline predicts ~55 for ALL CBF inputs (slope=0.026). The validation MAE of 3.47 seems too good for a model predicting constants.

Possible explanations:
- Validation data may have a narrow CBF distribution (centered around 50-60), making constant predictions appear reasonable
- The spatial context in validation patches (with tissue boundaries) may provide some signal that the uniform phantom test cannot capture

---

## 5. Summary of Issues Found

### Accuracy Issues (CLAUDE.md)
1. Exp 02 sensitivity ratio: CLAUDE.md says 257.95, JSON says 79.87
2. Exp 06 sensitivity ratio: CLAUDE.md says 92.51, JSON says 18.01
3. Exp 07 sensitivity ratio: CLAUDE.md says 113.91, JSON says 110.17

### Methodological Issues
1. All sensitivity ratios computed with random Gaussian inputs, not ASL signals
2. Both v1 (rerun_amplitude_ablation_validation.py:106) and v2 (setup_ablation_v2.py:333) use identical flawed methodology
3. Sensitivity ratio has no correlation with CBF accuracy within amplitude-aware models
4. Exp 10 (v2) disproves the causal link: ratio=0.36 but MAE=0.478

### Scientific Validity Issues
1. Realistic test shows amplitude-aware models are super-linear (slope ~1.9 instead of 1.0)
2. R^2 vs identity is negative (-0.52 to -0.80), meaning predictions are worse than a constant
3. Models hit the 300 ml/100g/min clamp at CBF=150
4. Win rates (97%+) are against a broken LS baseline (LS MAE=23.1, R^2=-1.12)

---

## 6. Recommendations

### Immediate Actions
1. **Fix CLAUDE.md**: Correct the three wrong sensitivity ratios (Exp 02: 79.87, Exp 06: 18.01, Exp 07: 110.17)
2. **Add methodology caveat**: Note that sensitivity ratios were computed with random Gaussian inputs and are not scientifically meaningful
3. **Replace old test**: Use test_amplitude_sensitivity_realistic.py for all future sensitivity assessment

### Short-term Investigation
4. **Run realistic test on all v1 experiments** (currently only 3 have been tested)
5. **Run ablation on architecture**: Test SpatialASLNet with matched capacity (same number of parameters) to determine if improvement is from amplitude awareness or architecture capacity
6. **Fix super-linearity**: Investigate why amplitude-aware models overshoot (slope ~1.9). This suggests the output modulation is amplifying rather than linearly tracking

### Strategic Decision
7. **Keep amplitude-aware architecture** -- it produces better CBF MAE (0.46 vs 3.47), even if the mechanism may not be amplitude sensitivity per se
8. **Do NOT rely on sensitivity ratios** for model selection -- they correlate with nothing useful
9. **Fix the LS baseline** before drawing conclusions about win rates
10. **Rerun validation with corrected LS** to get meaningful win rates

---

## Appendix A: Data Sources

### V1 JSON Files
- `amplitude_ablation_v1/*/amplitude_sensitivity.json` (10 files)
- `amplitude_ablation_v1/*/validation_results/llm_analysis_report.json` (8 files; Exp 04, 05 missing)

### V2 Data
- `amplitude_ablation_v2/ablation_v2_summary.csv`
- `amplitude_ablation_v2/*/amplitude_sensitivity.json` (7 files)

### Realistic Test Results
- `results/amp_test/00_baseline/amplitude_sensitivity_realistic.json`
- `results/amp_test/02_full/amplitude_sensitivity_realistic.json`
- `results/amp_test/14_att/amplitude_sensitivity_realistic.json`

### Test Scripts
- `rerun_amplitude_ablation_validation.py` (v1 Gaussian test, line 106)
- `setup_ablation_v2.py` (v2 Gaussian test, line 333)
- `test_amplitude_sensitivity_realistic.py` (realistic test)

---

## Appendix B: Detailed Realistic Test Analysis

### B.1 Exp 00 Baseline SpatialASLNet -- Complete Variance Collapse

The baseline model exhibits total CBF variance collapse on the realistic test:

**Test 1 -- CBF Linearity (True CBF -> Predicted CBF):**

| True CBF | Pred CBF (mean) | Pred CBF (std) | Error | Pred ATT |
|----------|----------------|----------------|-------|----------|
| 10 | 50.91 | 0.40 | +40.91 | 1510 |
| 20 | 53.10 | 0.25 | +33.10 | 1525 |
| 40 | 54.40 | 0.15 | +14.40 | 1529 |
| 60 | 54.89 | 0.13 | -5.11 | 1532 |
| 80 | 55.21 | 0.08 | -24.79 | 1534 |
| 100 | 55.37 | 0.08 | -44.63 | 1535 |
| 120 | 55.45 | 0.05 | -64.55 | 1535 |
| 150 | 55.49 | 0.04 | -94.51 | 1536 |

The model predicts CBF in the range [50.9, 55.5] regardless of true CBF (10 to 150). This is a textbook case of variance collapse -- the model learned to predict the training distribution mean (~55, close to the norm_stats `y_mean_cbf = 59.98`) and ignores the input amplitude entirely.

Key metrics:
- Slope: 0.026 (essentially zero; ideal is 1.0)
- R^2 vs identity: -0.10 (worse than predicting a constant)
- Prediction range: 4.6 ml/100g/min across a 140 ml/100g/min true range

**Test 2 -- Amplitude Scaling (CBF=60, signals scaled):**

| Scale | Pred CBF | Expected CBF |
|-------|----------|-------------|
| 0.5x | 54.95 | 30 |
| 0.75x | 54.95 | 45 |
| 1.0x | 54.95 | 60 |
| 1.5x | 54.95 | 90 |
| 2.0x | 54.95 | 120 |

Perfectly flat: the model outputs 54.95 regardless of input scaling. GroupNorm completely strips amplitude information as expected.

**ATT performance**: Interestingly, ATT predictions are reasonable (1510-1536 ms vs true 1500 ms), confirming that the temporal *shape* of the ASL curve is preserved through GroupNorm even though amplitude is lost.

### B.2 Exp 02 AmpAware Full -- Super-Linear Response

**Test 1 -- CBF Linearity:**

| True CBF | Pred CBF (mean) | Pred CBF (std) | Error | Ratio (Pred/True) |
|----------|----------------|----------------|-------|--------------------|
| 10 | 17.75 | 0.10 | +7.75 | 1.77 |
| 20 | 20.00 | 0.05 | +0.00 | 1.00 |
| 40 | 40.25 | 0.05 | +0.25 | 1.01 |
| 60 | 59.98 | 0.01 | -0.02 | 1.00 |
| 80 | 86.93 | 0.02 | +6.93 | 1.09 |
| 100 | 132.93 | 0.03 | +32.93 | 1.33 |
| 120 | 203.02 | 0.03 | +83.02 | 1.69 |
| 150 | 300.00 | 0.00 | +150.00 | 2.00 (CLAMPED) |

**Analysis of the super-linearity pattern:**
- CBF 20-60: Nearly perfect (ratio 1.00-1.01). The model is accurate in the training distribution center.
- CBF 10: Over-predicts by 77%. Low-CBF signals are weak and the model may interpret low amplitude as noise rather than low CBF.
- CBF 80+: Increasingly over-predicts. At CBF=100, predicts 133 (33% over). At CBF=120, predicts 203 (69% over). At CBF=150, hits the 300 clamp.
- The model is accurate in the narrow range [20, 60] but diverges super-linearly outside this range.

**Root cause -- Training data distribution mismatch:**
The training data CBF distribution from `enhanced_simulation.py` is:
- Gray matter: 50-70 ml/100g/min (majority of voxels)
- White matter: 18-28 ml/100g/min
- Norm stats: mean=59.98, std=23.08

This means the training distribution is concentrated in [20, 70] with most mass around [50, 70]. The model has never seen CBF=100+ during training (except possibly rare tumor_hyper samples at 90-150), so its response is uncalibrated outside the training range.

The z-score normalization compounds this: CBF=150 maps to z=(150-59.98)/23.08 = 3.9 standard deviations above the mean. The model has almost no training data at z > 1.5, so its output is extrapolation.

**Test 2 -- Amplitude Scaling (CBF=60):**

| Scale | Pred CBF | Expected (linear) | Ratio to 1.0x |
|-------|----------|--------------------|----------------|
| 0.5x | 29.75 | 30 | 0.50 |
| 0.75x | 45.19 | 45 | 0.75 |
| 1.0x | 59.97 | 60 | 1.00 |
| 1.5x | 107.60 | 90 | 1.79 |
| 2.0x | 203.98 | 120 | 3.40 |

Scaling up by 2x (equivalent to CBF=120) gives 204 instead of 120 -- matching the CBF linearity test result. The super-linearity is consistent: the model treats increased amplitude as mapping to z-scores further from the mean, and extrapolates nonlinearly.

Scaling down by 0.5x (equivalent to CBF=30) gives 29.75, very close to expected 30. This confirms the model is well-calibrated within the training range.

### B.3 Exp 14 ATT Rebalanced (v2) -- Same Pattern

| True CBF | Pred CBF | Error | Ratio |
|----------|----------|-------|-------|
| 10 | 20.43 | +10.43 | 2.04 |
| 20 | 20.85 | +0.85 | 1.04 |
| 40 | 39.61 | -0.39 | 0.99 |
| 60 | 59.88 | -0.12 | 1.00 |
| 80 | 83.94 | +3.94 | 1.05 |
| 100 | 123.84 | +23.84 | 1.24 |
| 120 | 181.87 | +61.87 | 1.52 |
| 150 | 295.61 | +145.61 | 1.97 (near clamp) |

Slightly less super-linear than Exp 02 (CBF=100: +24% vs +33%), but same fundamental pattern. ATT rebalancing did not fix the extrapolation problem.

### B.4 Why the Baseline Has 3.47 MAE Despite Predicting Constants

The apparent contradiction: the baseline predicts ~55 for all inputs, yet achieves CBF MAE of 3.47 in validation. This is explained by the training data distribution:

- Training/validation CBF: mean=59.98, std=23.08
- Most voxels are gray matter (CBF 50-70) or white matter (CBF 18-28)
- The overall voxel-weighted mean is ~55-60

If the model predicts ~55, the error distribution is:
- Gray matter (50-70): error = |55-60| = ~5, but many voxels near 55 have error ~0
- White matter (18-28): error = |55-23| = ~32, but fewer white matter voxels

The volume-weighted MAE of 3.47 implies a heavy gray matter weighting in the validation patches. This is plausible: gray matter typically comprises 60-80% of brain voxels in the phantom, and the masking further biases toward higher-signal regions.

**This means the 3.47 MAE understates the problem.** For white matter (CBF ~23), the model predicts ~55, an error of 32 ml/100g/min. For gray matter (CBF ~60), it predicts ~55, an error of only 5. The aggregate MAE hides tissue-specific failure.

---

## Appendix C: Super-Linearity Root Cause Analysis

### C.1 The Mechanism

The super-linearity arises from the interaction of three factors:

1. **Narrow training distribution**: CBF is concentrated in [20, 70] during training. The z-scored target space spans roughly [-1.7, 0.4] standard deviations for this range. The model has dense training data here.

2. **Output modulation amplifies extrapolation**: The AmplitudeAwareSpatialASLNet extracts amplitude statistics before GroupNorm and uses them to modulate the output via FiLM layers and output scaling. When input amplitude corresponds to CBF=100+ (outside training range), the modulation factor has no learned calibration -- it extrapolates the learned mapping, which happens to be super-linear.

3. **Unbounded output**: The model predicts unbounded z-scores (`spatial_asl_network.py:364`), relying only on the 300 ml/100g/min clamp for bounding. In the training range, this works fine. Outside it, the z-score extrapolation can produce arbitrarily large values.

### C.2 Why This Matters

For the current validation setup (CBF 20-70 training range, same distribution for validation), the super-linearity is invisible -- the model is accurate within range. But for clinical use:

- **Hyper-perfusion** (infection, tumor): CBF can reach 100-150+ ml/100g/min. The model would dramatically over-predict.
- **Post-contrast CBF**: Could appear higher than baseline. The model would exaggerate the difference.
- **Heterogeneous tissues**: Extreme values would distort spatial maps.

### C.3 Potential Fixes

1. **Expand training CBF range**: Include samples up to 150-200 ml/100g/min. The `tumor_hyper` category (90-150) exists but may be too rare in training.
2. **Output bounding**: Apply a bounded activation (sigmoid scaled to [0, 200]) instead of unbounded z-scores. This prevents extrapolation but may reduce accuracy near bounds.
3. **Calibrate the output modulation**: The modulation pathway learns a mapping from amplitude to CBF scaling. If this mapping is trained on a narrow range, it extrapolates poorly. Training with wider CBF ranges would calibrate it.
4. **Post-hoc correction**: Apply a nonlinear correction to the output based on the observed super-linearity curve.

### C.4 Clamp Inconsistency

The realistic test (`test_amplitude_sensitivity_realistic.py:293`) uses `clamp(0, 300)` for CBF, while the production denormalization function (`spatial_asl_network.py:847`) uses `clamp(0, 200)`. This means:
- Realistic test allows predictions up to 300 (Exp 02 hits this at CBF=150)
- Production code would clamp at 200 (Exp 02 would be clamped at CBF=120+)
- This inconsistency should be resolved; 200 is more physiologically appropriate for most applications

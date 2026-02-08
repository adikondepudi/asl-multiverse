# Validation Plan: P3 Re-Validation with Corrected Baselines

## 1. Current Validation Pipeline State

### What Works

**validate.py** (the main validation script):
- Auto-detects spatial vs voxel-wise model architecture from checkpoint keys
- For spatial models: generates synthetic 2D phantoms via `SpatialPhantomGenerator`
- Runs NN inference (ensemble averaging) and voxel-by-voxel LS fitting
- Computes: MAE, RMSE, Bias, R2, Failure Rate, Win Rate
- Has `_bootstrap_ci()` and `_bootstrap_ci_winrate()` for confidence intervals
- Has multi-SNR capability (SNR = [3, 5, 10, 15, 25]) with `run_spatial_validation(multi_snr=True)`
- Also computes Smoothed-LS baseline (Gaussian sigma=2.0) at SNR=10
- Saves LLM report (JSON + Markdown) and interactive plot data

**validate_spatial.py** (simpler spatial-only script):
- Loads pre-generated spatial chunk data from `.npz` files
- Runs NN inference only (no LS comparison)
- Computes basic MAE, RMSE, Bias, Correlation
- Does NOT compute win rates, bootstrap CIs, or LS comparison

**validation_metrics.py**:
- Comprehensive metrics library: Bland-Altman, ICC, CCC, SSIM, CoV, Win Rate
- `test_win_rate_significance()` (Wilcoxon/paired-t) is DEFINED but NEVER CALLED
- `compute_all_metrics()` produces a full comparison report but is UNUSED by validate.py

**rerun_amplitude_ablation_validation.py**:
- Fixed the v1 model filename bug (ensemble_model_*.pt vs spatial_model_*.pt)
- Runs amplitude sensitivity tests correctly
- Calls validate.py for full validation

**compare_nn_ls_invivo.py**:
- In-vivo comparison pipeline
- Uses CORRECTED LS parameters: T1_artery=1650, alpha_BS1=0.93
- Supports parallel LS fitting
- Supports optional BASIL/Bayesian baseline

### What's Missing or Broken

1. **No multi-SNR results exist**: All validation results use only SNR=10. No experiment has a `multi_snr_results.json`.

2. **LS parameters in validate.py use CONFIG values, not corrected values**:
   - validate.py line 96-101: `T1_artery=self.config.get('T1_artery', 1650.0)`, `alpha_BS1` comes from `self.params.alpha_BS1`
   - Exp 00 research_config.json: `T1_artery: 1850.0`, `alpha_BS1: 1.0`
   - This means ALL existing validation results use T1_artery=1850 and alpha_BS1=1.0 for LS
   - compare_nn_ls_invivo.py already uses correct values (T1_artery=1650, alpha_BS1=0.93)

3. **LS results are identical across experiments**: Exp 00 and Exp 14 show EXACTLY the same LS numbers (CBF MAE=23.11, ATT MAE=383.76). This is because:
   - Same phantom seeds (np.random.seed(phantom_idx))
   - Same LS parameters (from config, which inherited v1 defaults)
   - Same SNR=10

4. **interactive_plot_data.json is empty** for spatial validation (only populated for voxel-wise scenarios A/B/C/D)

5. **test_win_rate_significance() never called**: No statistical significance testing on win rates

6. **No tissue-stratified metrics**: Phantom metadata includes tissue labels but no per-tissue metrics computed

7. **LS subsamples at 1:10 ratio**: `sample_indices = brain_indices[::10]` -- only fits LS on 10% of brain voxels to save time. NN is evaluated on all voxels. Win rate compares NN at LS voxel locations vs LS.

8. **No multi-start LS**: Single-start LS with grid search initialization. May miss global optimum for some voxels.

9. **v2 att_scale bug**: 9 of 11 v2 experiments use `att_scale: 0.033` (legacy). Only Exp 14 (att_scale=1.0) and Exp 20 (att_scale=1.0) are unaffected.

10. **Win rates are inflated**: LS with T1_artery=1850 and alpha_BS1=1.0 produces catastrophically bad results (CBF MAE=23, ATT MAE=384ms). This makes NN win rates ~85-98% meaninglessly high.

---

## 2. Exact Blockers from P1

### Must Complete Before Re-Validation

| Blocker | Description | Impact on Validation |
|---------|-------------|---------------------|
| **P1a: LS Physics Params** | Fix T1_artery (1850->1650), add alpha_BS1 correction | LS baseline completely wrong without this |
| **P1e: Unit Consistency** | Verify CBF units (ml/100g/min vs ml/g/s) and ATT units (ms) are consistent | MAE numbers meaningless if units inconsistent |

### Desirable Before Re-Validation (but not blocking)

| Enhancement | Description | Impact |
|-------------|-------------|--------|
| P1b: Multi-Start LS | Run LS from multiple initial guesses | Better LS baseline, fairer comparison |
| P1c: Smoothed-LS | Already implemented (sigma=2.0) but only at SNR=10 | Need at all SNR levels |
| P1f: Statistical Testing | Call `test_win_rate_significance()` | Publication-quality claims |

### Corrected LS Parameters for Re-Validation

Based on `compare_nn_ls_invivo.py` (lines 162-173) which already has the correct values:

```python
ls_params = {
    'T1_artery': 1650.0,     # 3T consensus (Alsop 2015), was 1850.0
    'T_tau': 1800.0,          # Unchanged
    'alpha_PCASL': 0.85,      # Unchanged
    'alpha_VSASL': 0.56,      # Unchanged
    'T2_factor': 1.0,         # Unchanged
    'alpha_BS1': 1.0           # For synthetic data with NO background suppression
}
```

IMPORTANT: For synthetic phantom validation, alpha_BS1 should remain 1.0 because:
- Phantoms are generated WITHOUT background suppression
- The LS fitter should match the data generation process
- alpha_BS1=0.93 is only for IN-VIVO data where BS is applied

The main fix for synthetic validation is T1_artery: 1850 -> 1650.

---

## 3. Planned Validation Runs

### Phase 3a: Corrected Synthetic Validation

#### Run 1: Exp 00 Baseline SpatialASL (HIGHEST PRIORITY)

**Purpose**: Establish TRUE performance of baseline model against corrected LS.

**Parameters**:
- Model: amplitude_ablation_v1/00_Baseline_SpatialASL
- LS params: T1_artery=1650, alpha_BS1=1.0 (synthetic, no BS)
- SNR levels: [2, 3, 5, 8, 10, 15, 20, 25]
- Phantoms per SNR: 50 (SNR=10), 20 (others)
- Metrics: MAE, Bias, Win Rate, all with bootstrap 95% CI
- Additional: Wilcoxon test on win rate, tissue-stratified metrics

**Expected outcome**: LS MAE will DROP dramatically (from 23 to perhaps 5-10 for CBF). Win rates will decrease from 85% to perhaps 55-75%.

#### Run 2: Exp 14 ATT_Rebalanced (HIGH PRIORITY)

**Purpose**: Best v2 model, only one with correct att_scale=1.0.

**Parameters**: Same as Run 1 but with:
- Model: amplitude_ablation_v2/14_ATT_Rebalanced
- Note: This model has CBF MAE=0.44 (may seem too good -- need to verify units and denormalization)

#### Run 3: Exp 18 LongerTraining (MEDIUM PRIORITY)

**Purpose**: Never validated, 89-epoch model may be best available.

**Parameters**: Same as Run 1 but with:
- Model: amplitude_ablation_v2/18_LongerTraining
- Note: Partial training (89/100 epochs) but loss was still improving

#### Run 4: Exp 20 BestCombo (MEDIUM PRIORITY)

**Purpose**: Kitchen-sink experiment, never validated.

**Parameters**: Same as Run 1 but with:
- Model: amplitude_ablation_v2/20_BestCombo
- Note: Only 51/100 epochs, likely undertrained
- Has 5 ensembles vs 3

#### Run 5: Smoothed-LS Sweep (LOW PRIORITY)

**Purpose**: Find optimal smoothing sigma for LS baseline.

**Parameters**:
- Sigma values: [0.5, 1.0, 1.5, 2.0, 3.0]
- SNR levels: [3, 5, 10]
- Compare smoothed-LS vs NN at each combination

### Phase 3b: In-Vivo Re-Validation

**Purpose**: Re-run compare_nn_ls_invivo.py with alpha_BS1 sweep.

**alpha_BS1 sweep**: [0.85, 0.90, 0.93, 0.95, 1.0]

For each alpha_BS1 value:
- Run LS fitting on all subjects
- Compare NN vs LS
- Identify which alpha_BS1 gives most reasonable CBF values (expected ~40-60 ml/100g/min for GM)

**Blocked by**: In-vivo data must be accessible on local machine or HPC.

### Phase 3c: Multi-SNR Crossover Analysis

**Purpose**: Find SNR threshold where NN advantage disappears.

**SNR levels**: [2, 3, 5, 8, 10, 15, 20, 25, 50, 100]

**Hypothesis**: At high SNR (>25), LS should eventually match or beat NN since noise is minimal and the model-based approach has correct physics.

**Metrics at each SNR**:
- CBF: MAE, Bias, Win Rate (each with 95% CI)
- ATT: MAE, Bias, Win Rate (each with 95% CI)
- Statistical significance: Wilcoxon test on paired errors

### Phase 3d: Tissue-Stratified Metrics

**Purpose**: Understand per-tissue performance differences.

**Stratification**:
The `SpatialPhantomGenerator.generate_phantom()` returns a metadata dict with tissue information. We need to:

1. Check if metadata includes per-voxel tissue labels (from the segmentation map)
2. If yes: separate voxels by GM, WM, pathology (tumor/stroke), boundary regions
3. If no: infer tissue type from ground truth CBF/ATT:
   - GM: CBF 50-70, ATT 1000-1600
   - WM: CBF 18-28, ATT 1200-1800
   - Pathology: CBF <15 or >90, or ATT >2000

**Metrics per tissue type**: MAE, Bias, Win Rate

---

## 4. Template for corrected_baseline_comparison.json

```json
{
  "experiment": "00_Baseline_SpatialASL",
  "model_path": "amplitude_ablation_v1/00_Baseline_SpatialASL",
  "validation_date": "2026-02-XX",
  "phantom_config": {
    "phantom_size": 64,
    "n_phantoms_per_snr": {"default": 20, "snr_10": 50},
    "phantom_generator": "SpatialPhantomGenerator",
    "include_pathology": true
  },
  "ls_params": {
    "T1_artery": 1650.0,
    "T_tau": 1800.0,
    "alpha_PCASL": 0.85,
    "alpha_VSASL": 0.56,
    "T2_factor": 1.0,
    "alpha_BS1": 1.0,
    "note": "alpha_BS1=1.0 for synthetic data (no background suppression)"
  },
  "snr_results": {
    "3": {
      "nn_cbf_mae": null,
      "nn_cbf_mae_ci": [null, null],
      "nn_cbf_bias": null,
      "ls_cbf_mae": null,
      "ls_cbf_mae_ci": [null, null],
      "cbf_win_rate": null,
      "cbf_win_rate_ci": [null, null],
      "cbf_wilcoxon_p": null,
      "cbf_effect_size": null,
      "nn_att_mae": null,
      "nn_att_mae_ci": [null, null],
      "nn_att_bias": null,
      "ls_att_mae": null,
      "ls_att_mae_ci": [null, null],
      "att_win_rate": null,
      "att_win_rate_ci": [null, null],
      "att_wilcoxon_p": null,
      "att_effect_size": null,
      "n_nn_voxels": null,
      "n_ls_voxels": null
    },
    "5": { "...": "same structure" },
    "10": { "...": "same structure" },
    "15": { "...": "same structure" },
    "25": { "...": "same structure" }
  },
  "smoothed_ls_results": {
    "sigma_0.5": { "snr_10": { "...": "same metrics structure" } },
    "sigma_1.0": { "...": "..." },
    "sigma_2.0": { "...": "..." },
    "sigma_3.0": { "...": "..." }
  },
  "tissue_stratified": {
    "gray_matter": {
      "snr_10": {
        "nn_cbf_mae": null,
        "ls_cbf_mae": null,
        "cbf_win_rate": null,
        "nn_att_mae": null,
        "ls_att_mae": null,
        "att_win_rate": null,
        "n_voxels": null
      }
    },
    "white_matter": { "...": "same structure" },
    "pathology": { "...": "same structure" },
    "boundary": { "...": "same structure" }
  },
  "comparison_vs_original": {
    "original_ls_params": {
      "T1_artery": 1850.0,
      "alpha_BS1": 1.0
    },
    "original_snr10_results": {
      "cbf_win_rate": 0.858,
      "ls_cbf_mae": 23.11,
      "nn_cbf_mae": 3.47,
      "att_win_rate": 0.961,
      "ls_att_mae": 383.76
    },
    "corrected_snr10_results": {
      "cbf_win_rate": null,
      "ls_cbf_mae": null,
      "nn_cbf_mae": null,
      "att_win_rate": null,
      "ls_att_mae": null
    },
    "win_rate_change": {
      "cbf": null,
      "att": null,
      "note": "Negative = NN advantage decreased (as expected with corrected LS)"
    }
  }
}
```

---

## 5. Models to Validate (Priority Order)

### Tier 1: Must Validate

| Priority | Model | Path | Reason |
|----------|-------|------|--------|
| 1 | Exp 00 Baseline SpatialASL | amplitude_ablation_v1/00_Baseline_SpatialASL | Cleanest comparison, known good model |
| 2 | Exp 14 ATT_Rebalanced | amplitude_ablation_v2/14_ATT_Rebalanced | Best v2 model, correct att_scale=1.0 |

### Tier 2: Should Validate

| Priority | Model | Path | Reason |
|----------|-------|------|--------|
| 3 | Exp 18 LongerTraining | amplitude_ablation_v2/18_LongerTraining | Never validated, 89-epoch model |
| 4 | Exp 20 BestCombo | amplitude_ablation_v2/20_BestCombo | Never validated, kitchen-sink approach |
| 5 | Exp 09 Optimized | amplitude_ablation_v1/09_AmpAware_Optimized | Highest amplitude sensitivity (376x) |

### Tier 3: Nice to Have

| Priority | Model | Path | Reason |
|----------|-------|------|--------|
| 6 | Exp 13 AggressiveNoise | amplitude_ablation_v2/13_AggressiveNoise | Highest v2 amplitude sensitivity (82x) |
| 7 | Exp 02 AmpAware_Full | amplitude_ablation_v1/02_AmpAware_Full | First amplitude-aware model |

---

## 6. Key Questions This Validation Will Answer

1. **What is the TRUE NN advantage over correctly-parameterized LS?**
   - Current: CBF win rate 85%, ATT win rate 96% (against broken LS)
   - Expected: CBF win rate 55-75%? ATT win rate 70-90%?

2. **At what SNR does the NN advantage disappear?**
   - Hypothesis: NN dominates at SNR<10, LS catches up at SNR>20

3. **Does the NN advantage hold for all tissue types?**
   - GM (high CBF, moderate ATT): likely strong NN advantage
   - WM (low CBF, high ATT): unclear
   - Pathology: critical clinical question

4. **Is smoothed-LS competitive with NN?**
   - Smoothed-LS gets spatial averaging similar to NN's U-Net
   - If smoothed-LS matches NN, the NN adds no unique value

5. **Does amplitude-aware architecture matter after LS correction?**
   - If corrected LS is much better, amplitude sensitivity may be less important

---

## 7. Technical Notes for Implementation

### Modifying validate.py for Corrected LS

The key change needed in `validate.py` is to override LS parameters independently of the model's training config:

```python
# In _run_spatial_at_snr, replace:
ls_params = {
    'T1_artery': self.params.T1_artery,  # Uses config value (1850 for v1)
    ...
}

# With:
ls_params = {
    'T1_artery': corrected_T1_artery,  # Override: 1650.0
    ...
}
```

This is a P1 task (foundation-lead). The validation-lead should NOT modify validate.py directly but should use the corrected version once P1 delivers it.

### Adding Statistical Significance Testing

In `_compute_snr_metrics_with_ci`, add:
```python
from validation_metrics import test_win_rate_significance
sig_result = test_win_rate_significance(nn_cbf_errors, ls_cbf_errors, method='wilcoxon')
```

### Adding Tissue Stratification

The `SpatialPhantomGenerator.generate_phantom()` method needs to return tissue labels. Check the `metadata` dict returned alongside `true_cbf_map` and `true_att_map`.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Corrected LS dramatically reduces NN advantage | HIGH | HIGH | This is the point -- we need honest numbers |
| Smoothed-LS matches NN | MEDIUM | HIGH | Still valuable to know; NN has speed advantage |
| v2 models all have att_scale bug | CONFIRMED (9/11) | MEDIUM | Focus on Exp 14 (correct att_scale) |
| Exp 18/20 partially trained models underperform | MEDIUM | LOW | Note training status in results |
| Runtime too long for multi-SNR + multi-model | MEDIUM | LOW | Prioritize Tier 1 models, parallelize on HPC |

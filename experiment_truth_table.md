# Experiment Truth Table

**Generated**: 2026-02-08
**Purpose**: Ground truth status of all 21 experiments across v1 (10 experiments) and v2 (11 experiments).

---

## V1 Experiments (amplitude_ablation_v1/)

All v1 experiments use: `att_scale=0.033`, `T1_artery=1850.0`, `alpha_BS1=1.0`

| Exp | Name | Model Class | Training | Validation | Amp JSON | Amp Ratio | Amp Sensitive | Ensemble Size |
|-----|------|-------------|----------|------------|----------|-----------|---------------|---------------|
| 00 | Baseline_SpatialASL | SpatialASLNet | YES (3 models) | YES | YES | 1.00 | NO | 3 |
| 01 | PerCurve_Norm | SpatialASLNet | YES (3 models) | YES | YES | 1.00 | NO | 3 |
| 02 | AmpAware_Full | AmplitudeAwareSpatialASLNet | YES (3 models) | YES | YES | **79.87** | YES | 3 |
| 03 | AmpAware_OutputMod_Only | AmplitudeAwareSpatialASLNet | YES (3 models) | YES | YES | 90.32 | YES | 3 |
| 04 | AmpAware_FiLM_Only | AmplitudeAwareSpatialASLNet | YES (3 models) | YES | YES | 40.56 | YES | 3 |
| 05 | AmpAware_Bottleneck_Only | AmplitudeAwareSpatialASLNet | YES (3 models) | YES | YES | 1.05 | NO | 3 |
| 06 | AmpAware_Physics_0p1 | AmplitudeAwareSpatialASLNet | YES (3 models) | YES | YES | **18.01** | YES | 3 |
| 07 | AmpAware_Physics_0p3 | AmplitudeAwareSpatialASLNet | YES (3 models) | YES | YES | **110.17** | YES | 3 |
| 08 | AmpAware_DomainRand | AmplitudeAwareSpatialASLNet | YES (3 models) | YES | YES | 93.51 | YES | 3 |
| 09 | AmpAware_Optimized | AmplitudeAwareSpatialASLNet | YES (3 models) | YES | YES | 376.18 | YES | 3 |

### V1 Amplitude Sensitivity Discrepancies vs CLAUDE.md

| Exp | CLAUDE.md Value | JSON Value | Match? | Notes |
|-----|----------------|------------|--------|-------|
| 00 | 1.00 | 1.00 | YES | |
| 01 | 1.00 | 1.00 | YES | |
| 02 | **257.95** | **79.87** | **NO** | CLAUDE.md overstates by 3.2x |
| 03 | 90.32 | 90.32 | YES | |
| 04 | 40.56 | 40.56 | YES | |
| 05 | 1.05 | 1.05 | YES | |
| 06 | **92.51** | **18.01** | **NO** | CLAUDE.md overstates by 5.1x |
| 07 | **113.91** | **110.17** | **NO** | Close but not exact (3.4% off) |
| 08 | 93.51 | 93.51 | YES | |
| 09 | 376.18 | 376.18 | YES | |

**Note**: The JSON files lack the `used_trained_model` flag that `rerun_amplitude_ablation_validation.py` would have added, suggesting the rerun script either was never executed successfully or the JSON files were overwritten afterward. The discrepancies for Exp 02, 06, and 07 are significant and suggest CLAUDE.md was updated with values from a different run (possibly the rerun) that no longer matches the current JSON files on disk.

### V1 Validation Results (from llm_analysis_report.json, SNR=10)

| Exp | NN CBF MAE | NN CBF Win% | NN ATT MAE | NN ATT Win% | Source |
|-----|------------|-------------|------------|-------------|--------|
| 00 | 3.47 | 85.8% | 21.37 | 96.1% | llm_analysis_report.json |

**CLAUDE.md reports**: CBF MAE=4.01, Win=84.2%, ATT MAE=21.81, Win=95.8% -- these do NOT exactly match the JSON files on disk. The differences are small (5-15%) but non-zero.

---

## V2 Experiments (amplitude_ablation_v2/)

All v2 experiments use: `T1_artery=1850.0`, `alpha_BS1=1.0`, model class `AmplitudeAwareSpatialASLNet`

| Exp | Name | att_scale | loss_type | Training | Validation | Amp JSON | Amp Ratio | Status |
|-----|------|-----------|-----------|----------|------------|----------|-----------|--------|
| 10 | ExtendedDomainRand | 0.033 | l1 | YES (3) | YES | YES | 0.36 | COMPLETE |
| 11 | MoreData_50k | 0.033 | l1 | YES (3) | YES | YES | 69.37 | COMPLETE |
| 12 | MoreData_100k | 0.033 | l1 | YES (3) | YES | YES | 54.78 | COMPLETE |
| 13 | AggressiveNoise | 0.033 | l1 | YES (3) | YES | YES | 82.20 | COMPLETE |
| 14 | ATT_Rebalanced | **1.0** | l1 | YES (3) | YES | YES | 75.96 | COMPLETE |
| 15 | HuberLoss | 0.033 | huber | **NO** | NO | NO | N/A | **FAILED** (data loading hang) |
| 16 | L2Loss | 0.033 | l2 | YES (3) | YES | YES | 76.25 | COMPLETE |
| 17 | LargerModel | 0.033 | l1 | **NO** | NO | NO | N/A | **FAILED** (data loading hang) |
| 18 | LongerTraining | 0.033 | l1 | YES (3) | **NO** | **NO** | N/A | **PARTIAL** (89/100 epochs, time limit) |
| 19 | Ensemble5 | 0.033 | l1 | YES (5) | YES | YES | 11.17 | COMPLETE |
| 20 | BestCombo | **1.0** | l1 | YES (5) | **NO** | **NO** | N/A | **PARTIAL** (51/100 epochs, time limit) |

### V2 att_scale Bug Analysis

CLAUDE.md states "9 of 11 v2 experiments use att_scale: 0.033". Actual count from config files:

- **att_scale=0.033**: Exp 10, 11, 12, 13, 15, 16, 17, 18, 19 = **9 experiments** (correct)
- **att_scale=1.0**: Exp 14, 20 = **2 experiments**

With z-score normalized spatial targets, att_scale should be 1.0. The 9 experiments with 0.033 have ATT loss effectively weighted at only 3.3% of CBF loss.

### V2 Failed/Partial Summary

| Exp | Failure Cause | Models Exist? | Recoverable? |
|-----|--------------|---------------|--------------|
| 15 | Data loading hang on gpu108 | NO | Re-run needed |
| 17 | Data loading hang on gpu104 | NO | Re-run needed |
| 18 | 6h SLURM time limit (89/100 epochs) | YES (3 models, well-converged) | Run validation + amp test only |
| 20 | 6h SLURM time limit (51/100 epochs) | YES (5 models, undertrained) | Re-run with 14h limit or validate as-is |

### V2 Validation Results (from llm_analysis_report.json, SNR=10)

**IMPORTANT**: CBF MAE values below are in physical units (ml/100g/min) after denormalization via validate.py (lines 746-747). ATT MAE values are in ms. LS values are identical across all experiments because LS fitting is model-independent and uses the same validation phantoms.

| Exp | NN CBF MAE | NN CBF Win% | NN ATT MAE | NN ATT Win% | LS CBF MAE | LS ATT MAE |
|-----|------------|-------------|------------|-------------|------------|------------|
| 10 | 0.48 | 97.7% | 20.88 | 96.4% | 23.11 | 383.76 |
| 11 | 0.47 | 97.7% | 18.81 | 96.8% | 23.11 | 383.76 |
| 12 | 0.55 | 97.5% | 19.12 | 96.8% | 23.11 | 383.76 |
| 13 | 0.47 | 97.7% | 18.53 | 96.8% | 23.11 | 383.76 |
| 14 | 0.44 | 97.8% | 15.35 | 97.2% | 23.11 | 383.76 |
| 16 | 0.48 | 97.7% | 16.36 | 97.1% | 23.11 | 383.76 |
| 19 | 0.47 | 97.7% | 17.42 | 97.2% | 23.11 | 383.76 |

**LS Baseline Warning**: LS CBF MAE=23.11 and ATT MAE=383.76 are catastrophically high because the LS fitter uses `alpha_BS1=1.0` (no background suppression) and `T1_artery=1850` (non-consensus). This inflates NN win rates to 97%+. These win rates will drop significantly once the LS baseline is corrected.

---

## Global Issues Affecting All Experiments

1. **att_scale=0.033**: All v1 experiments and 9/11 v2 experiments use the legacy att_scale value. Only v2 Exp 14 and 20 use the correct value of 1.0.

2. **T1_artery=1850**: ALL experiments (v1 and v2) use T1_artery=1850ms. The ASL consensus (Alsop 2015) recommends 1650ms at 3T.

3. **alpha_BS1=1.0**: ALL experiments use alpha_BS1=1.0 (no background suppression) in both data generation AND LS fitting. This means the NN-vs-LS comparison is fair (both see no-BS data), but neither matches real in-vivo conditions.

4. **Broken LS Baseline**: The LS fitter produces catastrophically high errors (CBF MAE>23, ATT MAE>383), making NN win rates unreliable indicators of true model quality.

5. **Domain Randomization Only Affects DC Loss Path**: Domain randomization parameters in config are only used by KineticModel during DC loss computation. When dc_weight=0 (the default and setting in all experiments), domain randomization has NO effect on training. The data generation in `generate_clean_library.py` uses fixed ASLParameters without domain randomization.

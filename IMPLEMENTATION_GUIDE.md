# Production Phase - Implementation Guide

**Date**: February 5, 2026
**Status**: Phase 1 (Bug Fixes) COMPLETE, Phase 2+ Ready to Start

---

## What Was Fixed

### Bug Fix #1: validate_spatial.py Model Class Detection ✅
**File**: `validate_spatial.py`, lines 98-147
**Problem**: Script hardcoded loading as SpatialASLNet, breaking AmplitudeAwareSpatialASLNet
**Fix Applied**:
- ✅ Added model class detection from config
- ✅ Instantiate correct model (SpatialASLNet or AmplitudeAwareSpatialASLNet)
- ✅ Pass amplitude-aware flags (use_film_at_bottleneck, etc.)
- ✅ Better error handling with traceback logging

**Impact**:
- ✅ Exp 04 & 05 validation should now work
- ✅ All future AmplitudeAware experiments will validate properly
- ✅ Eliminates state_dict mismatch errors

**How to Verify**:
```bash
cd /Users/adikondepudi/Desktop/asl-multiverse

# Re-validate Exp 04 (should now succeed)
python validate_spatial.py /path/to/amplitude_ablation_v1/04_AmpAware_FiLM_Only

# Re-validate Exp 05 (should now succeed)
python validate_spatial.py /path/to/amplitude_ablation_v1/05_AmpAware_Bottleneck_Only

# Re-validate Exp 09 (ensure it still works)
python validate_spatial.py /path/to/amplitude_ablation_v1/09_AmpAware_Optimized
```

### Config Files Created ✅
- ✅ `config/production_v2.yaml` - Locked production configuration
- ✅ `PRODUCTION_PHASE_PLAN.md` - Complete implementation roadmap
- ✅ `IMPLEMENTATION_GUIDE.md` - This file

---

## Production V2 Configuration

### What is Locked?
```yaml
✅ LOCKED (DO NOT CHANGE):
- model_class_name: "AmplitudeAwareSpatialASLNet"
- use_amplitude_output_modulation: true
- normalization_mode: "global_scale"
- loss_type: "l1", dc_weight: 0.0
- n_ensembles: 3

⚠️  CRITICAL (Only change with good reason):
- learning_rate: 0.0001
- batch_size: 32
- dropout_rate: 0.1
- variance_weight: 0.1
- domain_randomization: true
```

### Expected Performance
```
Simulation (SNR=10):
  CBF: MAE 0.49 ml/100g/min, Win Rate 97.5%
  ATT: MAE 18.7 ms, Win Rate 96.8%
  Amplitude Sensitivity: 376.2×

In-Vivo (11 subjects):
  CBF: ICC 0.9999, r=0.675, Bias +27.4 ml/100g/min
  ATT: ICC 0.921, r=0.548, Bias -74.5 ms
  Voxel Coverage: 100% (vs LS 52%)
```

---

## Next: Advanced Ablations (Exp 10-20)

### Why These Experiments?
The current best model (Exp 09) is excellent on **simulation**, but shows:
1. **ATT gap**: 18.7 ms → 439 ms in-vivo (23× worse)
2. **CBF bias**: Perfect in sim, +27 ml/100g/min in-vivo
3. **Limited spatial context**: Only 64×64 patches
4. **No confidence bounds**: Hard to know when predictions are unreliable

These experiments aim to **close these gaps**.

### Experiment Design Strategy

#### Exp 10-12: ATT Improvements
**Hypothesis**: ATT estimation can be dramatically improved with architecture changes

**Exp 10 - Separate Decoder Branches**
```yaml
# Config changes from production_v2
# Split network: [Shared Encoder] → [CBF Decoder] + [ATT Decoder]
decoder_architecture: "separate"  # Instead of "shared"
expected_improvement: "+15-25% ATT accuracy"
```

**Exp 11 - ATT-Specific Conditioning**
```yaml
# Different amplitude features for ATT (ATT depends on ratio, not mean)
amplitude_path_att_only: true
att_conditioning_features: ['pcasl_vsasl_ratio', 'temporal_slope', 'peak_delay']
expected_improvement: "+20% ATT accuracy"
```

**Exp 12 - Adaptive Loss Weighting**
```yaml
# Dynamically adjust CBF vs ATT loss weight based on validation performance
adaptive_loss_weighting: true
att_weight_schedule: "inverse_validation_loss"  # Lower weight = easier task focus
expected_improvement: "+10% both metrics"
```

#### Exp 13-14: Domain Gap Reduction
**Hypothesis**: Stronger domain randomization and adaptation reduce in-vivo bias

**Exp 13 - Extended Domain Randomization**
```yaml
# Randomize MORE parameters (acquisition, physiology, scanner effects)
alpha_BS1_range: [0.7, 1.0]  # Wider BS variation
motion_corruption_enabled: true
slice_delay_variation: true
expected_improvement: "-10 ml/100g/min bias"
```

**Exp 14 - Domain Adaptation Fine-tuning**
```yaml
# Fine-tune on small in-vivo dataset
fine_tuning_enabled: true
fine_tuning_dataset: "real_patient_sample"  # 100 in-vivo voxels
fine_tuning_epochs: 5
expected_improvement: "-15 ml/100g/min bias"
```

#### Exp 15-16: Spatial Context Expansion
**Hypothesis**: Larger patches capture more spatial structure for ATT

**Exp 15 - 128×128 Patches**
```yaml
# Double patch size (64×64 → 128×128)
spatial_patch_size: 128  # Instead of 64
hidden_sizes: [32, 64, 128, 256, 512]  # Add 5th level
expected_improvement: "+15% ATT"
```

**Exp 16 - Multi-Scale Fusion**
```yaml
# Process 64×64 and 128×128 in parallel, fuse predictions
multi_scale: true
scales: [64, 128]
fusion_method: "attention"
expected_improvement: "+20% ATT, stable CBF"
```

#### Exp 17-18: Uncertainty Quantification
**Hypothesis**: Better uncertainty enables OOD detection and confidence bounds

**Exp 17 - MC Dropout**
```yaml
# High dropout at test time for sampling
mc_dropout_enabled: true
dropout_rate_train: 0.1
dropout_rate_test: 0.5  # Sample N forward passes
n_samples: 10  # Number of Monte Carlo samples
expected_improvement: "Confidence bounds ±2-3 standard errors"
```

**Exp 18 - Ensemble Disagreement**
```yaml
# Use 3-model disagreement as uncertainty indicator
ensemble_uncertainty: true
n_ensembles: 3  # Keep existing
uncertainty_metric: "variance"  # Disagreement between models
expected_improvement: "95% confidence intervals"
```

#### Exp 19-20: Robustness & Self-Supervised
**Hypothesis**: Extended SNR range and self-supervised pretraining improve robustness

**Exp 19 - Extended SNR Training**
```yaml
# Train on extreme SNRs (real-world range)
training_noise_levels: [0.5, 1, 2, 5, 10, 15, 20, 25, 50]  # Wider range
expected_improvement: "Robust at SNR < 2, > 25"
```

**Exp 20 - Self-Supervised Denoising Pre-training**
```yaml
# Pre-train denoising head, then train main task
pretraining_task: "denoising"  # MSE loss on clean signal recovery
pretraining_epochs: 20
main_training_epochs: 50
expected_improvement: "+10% robustness, maybe +5% accuracy"
```

---

## Running Exp 10-20

### Command Template
```bash
cd /Users/adikondepudi/Desktop/asl-multiverse

# Create experiment config (copy production_v2.yaml, modify one parameter)
cp config/production_v2.yaml config/exp_10_separate_decoders.yaml
# Edit: decoder_architecture: "separate"

# Train
python main.py config/exp_10_separate_decoders.yaml --stage 2 --output-dir ./results/exp_10

# Validate
python validate_spatial.py ./results/exp_10

# Analyze
# Compare validation metrics to baseline (Exp 09)
```

### Parallel Execution Strategy
```bash
# Week 2: Train Exp 10, 11, 12 in parallel (if you have 3 GPUs)
# Each takes ~6 hours
python main.py config/exp_10.yaml --stage 2 --output-dir ./results/exp_10 &
python main.py config/exp_11.yaml --stage 2 --output-dir ./results/exp_11 &
python main.py config/exp_12.yaml --stage 2 --output-dir ./results/exp_12 &

# Week 3: Train Exp 13-16
# Week 4: Train Exp 17-20
```

---

## Success Criteria for Each Experiment

### Baseline (Exp 09): Must Match
```
CBF MAE: 0.49 ± 0.02
ATT MAE: 18.7 ± 0.5
Amplitude Sensitivity: 376 ± 10
```

### Exp 10-12 Success: One Must Improve ATT
```
✅ Success if ANY of:
  - ATT MAE < 17.0 ms (→10% improvement)
  - ATT Win Rate > 97% (→+0.2%)
  - R² > 0.994 (from 0.993)
```

### Exp 13-14 Success: Must Reduce In-Vivo Bias
```
✅ Success if ANY of:
  - Bias < 20 ml/100g/min (from +27.4)
  - Validation IAE < 25 ml/100g/min
```

### Exp 15-16 Success: Better ATT with Stable CBF
```
✅ Success if:
  - ATT improves: MAE < 17 ms
  - CBF stable: MAE still < 0.52 ml/100g/min
```

### Exp 17-20 Success: Quantifiable Improvements
```
✅ Success if:
  - Exp 17: 95% CI width < 2 MAE
  - Exp 18: OOD AUROC > 0.85
  - Exp 19: MAE stable at SNR 0.5-50
  - Exp 20: +5% accuracy from pretraining
```

---

## Expected New Baseline (After Exp 10-20)

If top improvements (Exp 10, 13, 15) all work:
```
Optimistic: CBF 0.46 MAE, ATT 14 ms MAE, +10% better
Conservative: CBF 0.49 MAE, ATT 16 ms MAE, +5% better
Realistic: CBF 0.48 MAE, ATT 17.5 ms MAE, some improvement
```

---

## Troubleshooting Guide

### Issue: "Model fails to load - missing key cbf_amplitude_correction"
**Cause**: Loading old Exp 04/05 saved model with fixed validate script
**Solution**:
```bash
# Exp 04 & 05 don't have this layer, validate script now handles it
python validate_spatial.py ./amplitude_ablation_v1/04_AmpAware_FiLM_Only
```

### Issue: New experiment validation fails
**Cause**: Inconsistent config flags between training and validation
**Solution**:
1. Ensure config file is in experiment directory
2. Check research_config.json has model_class_name set
3. Run: `python validate_spatial.py ./results/exp_10 --debug` for details

### Issue: ATT doesn't improve despite architectural changes
**Cause**: ATT is fundamentally harder (weaker physical signal)
**Solution**: Try weighted loss (higher att_weight) or separate architecture

---

## Checkpoints

### Phase 1: Bug Fixes ✅ COMPLETE
- ✅ Fixed validate_spatial.py
- ✅ Created production_v2.yaml
- ✅ All Exp 00-09 ready to re-validate

### Phase 2: Early Ablations (Week 2)
- [ ] Implement Exp 10 (Separate Decoders)
- [ ] Implement Exp 11 (ATT Conditioning)
- [ ] Implement Exp 12 (Adaptive Loss)
- [ ] Validate all three

### Phase 3: Domain & Spatial (Week 3)
- [ ] Implement Exp 13 (Extended Domain Rand)
- [ ] Implement Exp 14 (Domain Adaptation)
- [ ] Implement Exp 15 (128×128 Patches)
- [ ] Implement Exp 16 (Multi-Scale)

### Phase 4: Advanced Features (Week 4)
- [ ] Implement Exp 17 (MC Dropout)
- [ ] Implement Exp 18 (Ensemble Disagreement)
- [ ] Implement Exp 19 (Extended SNR)
- [ ] Implement Exp 20 (Self-Supervised)

### Phase 5: Analysis & New Baseline
- [ ] Analyze all results
- [ ] Rank by improvement
- [ ] Select best (or top 3)
- [ ] Set as new baseline
- [ ] Update all future work to new baseline

---

## Files Ready to Use

```
✅ config/production_v2.yaml      - Locked production config
✅ validate_spatial.py             - Fixed validation script
✅ PRODUCTION_PHASE_PLAN.md        - Full roadmap
✅ IMPLEMENTATION_GUIDE.md         - This file
```

**Next Action**: Start Phase 2 - Implement Exp 10-12

**Estimated Timeline**: 4 weeks for full Phase 4 completion

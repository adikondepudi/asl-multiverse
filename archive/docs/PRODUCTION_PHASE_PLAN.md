# Production Phase Implementation Plan

**Date**: February 5, 2026
**Objective**: Fix training code bugs, lock production config, run advanced ablations to improve model

---

## Phase 1: Fix Code Bugs (Immediate)

### Bug #1: validate_spatial.py Hardcodes SpatialASLNet
**Location**: validate_spatial.py, line 127
**Problem**: Script always instantiates SpatialASLNet regardless of trained model type
**Impact**: AmplitudeAwareSpatialASLNet validation fails with state_dict mismatch
**Fix**:
- Read `model_class_name` from config
- Instantiate correct model class (SpatialASLNet or AmplitudeAwareSpatialASLNet)
- Pass config flags (use_film_at_bottleneck, use_film_at_decoder, use_amplitude_output_modulation) to AmplitudeAwareSpatialASLNet

**Status**: 🔧 TODO

### Bug #2: validate.py May Have Same Issue
**Location**: validate.py (for voxel-wise models)
**Problem**: Similar hardcoding issue
**Fix**: Implement model class detection for validate.py as well

**Status**: 🔧 TODO

---

## Phase 2: Lock Production Configuration (Baseline)

### Exp 09 Configuration (Current Best)
```yaml
# Model Architecture
model_class_name: "AmplitudeAwareSpatialASLNet"
hidden_sizes: [32, 64, 128, 256]

# Amplitude Awareness (CRITICAL)
use_film_at_bottleneck: true
use_film_at_decoder: true
use_amplitude_output_modulation: true

# Normalization (CRITICAL)
normalization_mode: "global_scale"
global_scale_factor: 10.0

# Training
learning_rate: 0.0001
batch_size: 32
n_epochs: 50
n_ensembles: 3
dropout_rate: 0.1
norm_type: "group"  # GroupNorm required

# Loss Configuration
loss_type: "l1"
loss_mode: "mae_nll"
cbf_weight: 1.0
att_weight: 1.0
dc_weight: 0.0         # No physics loss (best validation)
variance_weight: 0.1   # Anti-collapse penalty

# Generalization
domain_randomization: true
training_noise_levels: [2.0, 5.0, 10.0, 15.0, 20.0, 25.0]

# Physics Parameters (Domain Randomization Ranges)
T1_artery: 1850.0
T1_artery_range: [1550.0, 2150.0]
T_tau: 1800.0
alpha_PCASL: 0.85
alpha_PCASL_range: [0.75, 0.95]
alpha_VSASL: 0.56
alpha_VSASL_range: [0.40, 0.70]
alpha_BS1_range: [0.85, 1.0]  # Background suppression

# Data
pld_values: [500, 1000, 1500, 2000, 2500, 3000]
active_features: ['mean', 'std', 'peak', 't1_artery']
data_noise_components: ['thermal', 'physio', 'drift']
noise_type: "rician"
```

**File to Create**: `config/production_v2.yaml`

---

## Phase 3: Identify Model Flaws & Improvement Areas

### Current Best Model Performance (Exp 09)
- **CBF MAE**: 0.49 ml/100g/min (excellent)
- **ATT MAE**: 18.7 ms (good)
- **Amplitude Sensitivity**: 376.2× (exceptional)
- **CBF Win Rate**: 97.5% vs LS
- **ATT Win Rate**: 96.8% vs LS

### Potential Flaws & Improvement Areas

#### Flaw 1: ATT Performance Still Moderate
- **Issue**: ATT MAE is 18.7 ms, but in-vivo shows much higher errors (~400-600 ms)
- **Possible Causes**:
  - ATT harder to estimate than CBF (non-linear dependence)
  - May need separate ATT-specific architecture
  - Current FiLM conditioning optimized for CBF
- **Improvement Ideas**:
  - Ablate: Separate output heads for CBF vs ATT
  - Ablate: ATT-specific attention mechanism
  - Ablate: Different loss weights for CBF vs ATT
  - Ablate: Separate decoder branches for CBF vs ATT

#### Flaw 2: In-Vivo Performance Gap
- **Issue**: In-vivo shows systematic bias (+27 ml/100g/min CBF)
- **Possible Causes**:
  - Domain gap: Training on synthetic, testing on real
  - Missing real-world artifacts (motion, flow, etc.)
  - Physics parameters (T1, α) differ from real data
- **Improvement Ideas**:
  - Ablate: Stronger domain randomization
  - Ablate: Adaptive bias correction module
  - Ablate: In-vivo data fine-tuning
  - Ablate: Adversarial domain adaptation

#### Flaw 3: Limited Spatial Context
- **Issue**: Current model uses 64×64 patches
- **Possible Causes**:
  - May not capture distant correlations
  - Edge artifacts for patches near image boundary
  - Coarse spatial resolution
- **Improvement Ideas**:
  - Ablate: Larger 128×128 patches
  - Ablate: Multi-scale fusion (64×64 + 128×128)
  - Ablate: Overlapping patches with blending
  - Ablate: Full-image processing (but GPU memory constraints)

#### Flaw 4: Uncertain Prediction Confidence
- **Issue**: No uncertainty quantification beyond variance loss
- **Possible Causes**:
  - Variance loss alone insufficient for OOD detection
  - May fail on unusual anatomy or pathology
- **Improvement Ideas**:
  - Ablate: Ensemble disagreement as confidence
  - Ablate: Dropout uncertainty (MC Dropout)
  - Ablate: Auxiliary confidence head
  - Ablate: Energy-based OOD detection

#### Flaw 5: Noise Robustness at Extreme SNRs
- **Issue**: Only trained SNR 2-25, but real data can be SNR 1-50+
- **Possible Causes**:
  - Distribution mismatch at extreme SNRs
  - May degrade at very low SNR (<2)
- **Improvement Ideas**:
  - Ablate: Extended SNR range [0.5, 50]
  - Ablate: Self-supervised denoising pre-training
  - Ablate: Explicit noise estimator module

#### Flaw 6: Hard to Train ATT with CBF
- **Issue**: Joint CBF/ATT estimation may have competing gradients
- **Possible Causes**:
  - CBF and ATT losses interfere with each other
  - ATT harder to learn so gets ignored
- **Improvement Ideas**:
  - Ablate: Separate CBF-only and ATT-only models
  - Ablate: Adaptive loss weighting (easier task gets lower weight)
  - Ablate: Sequential training (CBF first, ATT fine-tuning)

---

## Phase 4: Planned Ablation Studies (Advanced)

### Ablation Series 1: ATT Improvements (3-4 experiments)
1. **Exp 10**: Separate decoder branches (CBF vs ATT)
   - Hypothesis: Separate pathways better capture ATT
   - Config: Split decoder after bottleneck

2. **Exp 11**: ATT-specific conditioning
   - Hypothesis: Different amplitude features for ATT
   - Config: Separate amplitude path for ATT

3. **Exp 12**: Adaptive loss weighting (MAE weight ∝ validation variance)
   - Hypothesis: Balance CBF/ATT learning
   - Config: Dynamic weight scheduling

### Ablation Series 2: Domain Gap Reduction (2-3 experiments)
1. **Exp 13**: Extended domain randomization (physics + acquisition)
   - Hypothesis: Larger domain variation = better generalization
   - Config: Randomize more parameters

2. **Exp 14**: Domain adaptation fine-tuning
   - Hypothesis: Few in-vivo samples can reduce bias
   - Config: 2-shot fine-tuning on real data

### Ablation Series 3: Spatial Context (2-3 experiments)
1. **Exp 15**: Larger 128×128 patches
   - Hypothesis: More context helps ATT estimation
   - Config: Increase patch size, reduce spatial resolution

2. **Exp 16**: Multi-scale fusion
   - Hypothesis: Both local (64×64) and global (128×128) context
   - Config: Process two scales in parallel, fuse

### Ablation Series 4: Uncertainty (2-3 experiments)
1. **Exp 17**: MC Dropout for uncertainty
   - Hypothesis: Sampling gives better OOD detection
   - Config: High dropout (0.3-0.5) at test time

2. **Exp 18**: Ensemble disagreement
   - Hypothesis: Disagreement between models indicates uncertainty
   - Config: Analyze 3-ensemble variance

### Ablation Series 5: Robustness (2-3 experiments)
1. **Exp 19**: Extended SNR range [0.5, 50]
   - Hypothesis: Covers real-world extremes
   - Config: training_noise_levels: [0.5, 1, 2, 5, 10, 15, 20, 25, 50]

2. **Exp 20**: Self-supervised pretraining
   - Hypothesis: Denoising improves noise robustness
   - Config: Add denoising head with MSE loss

---

## Phase 5: Production Baseline Update

After fixing bugs and running Exp 10-20, identify best improvement:
1. **Current Best**: Exp 09 (376.2× amplitude, 97.5% CBF win, 18.7 ATT MAE)
2. **After Improvements**: Set new baseline to best-of-{Exp 10-20}
3. **All future work**: Compare against new baseline

---

## Implementation Order

### Week 1 (Immediate)
- [ ] Fix validate_spatial.py (model class detection)
- [ ] Fix validate.py (if needed)
- [ ] Re-validate Exp 00-09 with fixed validation script
- [ ] Create production_v2.yaml config

### Week 2
- [ ] Implement Exp 10-12 (ATT improvements)
- [ ] Run validation on all new experiments
- [ ] Analyze ATT improvements

### Week 3
- [ ] Implement Exp 13-14 (Domain gap)
- [ ] Implement Exp 15-16 (Spatial context)
- [ ] Run validation

### Week 4
- [ ] Implement Exp 17-18 (Uncertainty)
- [ ] Implement Exp 19-20 (Robustness)
- [ ] Final analysis & new baseline selection

---

## Success Metrics

### Phase 1 (Bugs): ✅ DONE when
- [ ] All Exp 00-09 successfully validate
- [ ] No state_dict mismatch errors
- [ ] Exp 04-05 validation metrics recovered

### Phase 2 (Config): ✅ DONE when
- [ ] production_v2.yaml locked
- [ ] All flags properly documented
- [ ] Training reproducible from config

### Phase 3 (Flaws): ✅ DONE when
- [ ] All 6 flaws identified & documented
- [ ] Improvement hypotheses clear
- [ ] Ablation experiments designed

### Phase 4 (Ablations): ✅ DONE when
- [ ] Exp 10-20 trained & validated
- [ ] Each improves on baseline or fails with clear reason
- [ ] New best model identified

### Phase 5 (Baseline): ✅ DONE when
- [ ] New best model selected
- [ ] All future work references new baseline
- [ ] Performance documented

---

## Expected Outcomes

### Realistic Improvements (Based on Literature)
- ATT: +20-30% improvement (separate architecture)
- Domain gap: +10-15% reduction (domain adaptation)
- Uncertainty: +5% OOD precision (MC Dropout)
- Overall: 15-30% better in-vivo performance

### Conservative Estimate
- After 10 experiments: 10-20% overall improvement
- After full Phase 4: 20-30% overall improvement

### Moonshot Goals
- ATT MAE < 15 ms (simulation)
- In-vivo bias < 10 ml/100g/min (50% reduction)
- 99% voxel coverage vs LS 52%

---

**Status**: Ready to start Phase 1
**Next Action**: Fix validate_spatial.py

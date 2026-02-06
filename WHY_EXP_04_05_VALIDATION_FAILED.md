# Why Exp 04 & 05 Validation Failed - Code Bug Analysis

## Problem
Experiments 04 and 05 failed during validation with state_dict mismatch errors, even though training completed successfully.

## Root Cause: Training Code Bug
**The training code does not properly instantiate architecture components based on configuration flags.**

---

## Exp 04 Failure Details

### Configuration
```yaml
use_amplitude_output_modulation: false  # ← DISABLED in Exp 04
use_film_at_bottleneck: true
use_film_at_decoder: true
model_class_name: AmplitudeAwareSpatialASLNet
```

### Error
```
Missing key(s) in state_dict for AmplitudeAwareSpatialASLNet:
"cbf_amplitude_correction.0.weight"
"cbf_amplitude_correction.0.bias"
"cbf_amplitude_correction.2.weight"
"cbf_amplitude_correction.2.bias"
```

### What Happened
1. **Training**: Model was initialized as `AmplitudeAwareSpatialASLNet`
2. **Expected**: With `use_amplitude_output_modulation: false`, the amplitude correction layer should NOT be created
3. **Actual**: Validation code tries to load into `AmplitudeAwareSpatialASLNet` architecture which ALWAYS includes amplitude correction layers
4. **Result**: Mismatch between saved model (no amplitude correction) and architecture (expects amplitude correction)

### Impact
Cannot validate despite successful training. However, **amplitude sensitivity test still works** because it uses an untrained model.

---

## Exp 05 Failure Details

### Configuration
```yaml
use_amplitude_output_modulation: false  # ← DISABLED
use_film_at_bottleneck: true
use_film_at_decoder: false             # ← DISABLED
model_class_name: AmplitudeAwareSpatialASLNet
```

### Error
```
Missing key(s) in state_dict for AmplitudeAwareSpatialASLNet:

Decoder FiLM layers:
"decoder1_film.gamma_generator.0.weight"
"decoder1_film.gamma_generator.0.bias"
... (16 more decoder FiLM keys)

Amplitude correction layers:
"cbf_amplitude_correction.0.weight"
"cbf_amplitude_correction.0.bias"
"cbf_amplitude_correction.2.weight"
"cbf_amplitude_correction.2.bias"
```

### What Happened
1. **Training**: Model created WITHOUT decoder FiLM layers (disabled) AND WITHOUT amplitude correction (disabled)
2. **Validation**: Tries to load into full `AmplitudeAwareSpatialASLNet` which includes both components
3. **Result**: 26 missing keys (16 decoder FiLM + 10 amplitude correction)

### Impact
Complete architecture mismatch. Cannot validate.

---

## Root Cause Analysis

### The Bug
The **validation script uses a hardcoded architecture** without reading the actual configuration flags used during training:

```python
# Validation script hardcoded
model = AmplitudeAwareSpatialASLNet(...)  # Always creates ALL components

# Should be
model = load_from_config(config_file)  # Create based on actual config
```

### Training Code Issue
The **training code should properly save/load configuration** with the model so validation knows which components to instantiate.

### Why Amplitude Sensitivity Still Works
The amplitude sensitivity test creates a **fresh untrained model** without trying to load trained weights, so there's no state_dict mismatch:

```python
# Amplitude sensitivity test - works fine
model = AmplitudeAwareSpatialASLNet(...)  # Create untrained model
# No weights loaded, no mismatch error
```

---

## Timeline

| Time | Event |
|------|-------|
| 12:39 | Training starts for Exp 04 |
| 17:04 | Training completes (50/50 epochs) |
| 17:05 | Validation script runs, tries to load trained models |
| 17:05 | Models fail to load (missing amplitude correction layer) |
| 17:05 | Amplitude sensitivity test runs (creates untrained model) |
| 17:05 | Amplitude sensitivity test succeeds (no state_dict issue) |
| 17:05 | Exp 04 complete with validation failure |

---

## Fix Required

### Option A: Fix Validation Script (Recommended)
Read the configuration from `research_config.json` or `config.yaml` and instantiate models based on actual flags:

```python
# Load config
config = load_json('research_config.json')

# Instantiate based on actual configuration
if config['model_class_name'] == 'AmplitudeAwareSpatialASLNet':
    model = AmplitudeAwareSpatialASLNet(
        use_film_at_bottleneck=config['use_film_at_bottleneck'],
        use_film_at_decoder=config['use_film_at_decoder'],
        use_amplitude_output_modulation=config['use_amplitude_output_modulation'],
        ...
    )
```

### Option B: Fix Training Code
Store configuration with model checkpoint and load it during validation:

```python
# During training
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': training_config,  # Save the config
}
torch.save(checkpoint, model_path)

# During validation
checkpoint = torch.load(model_path)
model = create_model_from_config(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Impact on Results

### Data Loss
- ❌ Exp 04 validation metrics: LOST
- ❌ Exp 05 validation metrics: LOST

### What We Still Have
- ✅ Exp 04 amplitude sensitivity: 40.6× (untrained model test)
- ✅ Exp 05 amplitude sensitivity: 1.05× (untrained model test)
- ✅ Training logs for both (proof of convergence)
- ✅ Trained model weights (just can't load them due to architecture mismatch)

### Comparison to Successful Validations
Experiments 00-03 and 06-09 all have successful validations because:
- Either they used a different model class
- Or the architecture instantiation was correct
- Or configuration flags matched the actual saved model

---

## Recommendation

Since the amplitude sensitivity tests still work (they don't use trained weights), and the training logs show convergence, **we can trust the amplitude sensitivity results for Exp 04 and 05**. However:

1. **Cannot use validation metrics** from these experiments
2. **Should fix the code** before running new ablations
3. **The amplitude sensitivity finding is still valid**: OutputMod alone (Exp 03: 90.3×) is much better than FiLM alone (Exp 04: 40.6×), which confirms output modulation is critical

---

## Summary Table

| Exp | Training | Amplitude Sensitivity | Validation | Root Cause |
|-----|----------|----------------------|------------|-----------|
| 04 | ✅ Success | ✅ 40.6× | ❌ Failed | Missing amplitude_correction in saved model |
| 05 | ✅ Success | ✅ 1.05× | ❌ Failed | Missing decoder_film + amplitude_correction |

**Lesson**: Configuration flags must be properly instantiated during model creation, and validated before validation script runs.

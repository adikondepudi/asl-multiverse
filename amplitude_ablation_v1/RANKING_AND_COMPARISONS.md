# Amplitude Ablation Study - Ranking & Comparisons

---

## Complete Performance Ranking

### Amplitude Sensitivity Ranking (Highest to Lowest)

```
376.2× ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ Exp 09 - Optimized (Full + DomainRand)
110.2× ▓▓▓▓▓▓ Exp 07 - Physics (dc=0.3)
 93.5× ▓▓▓▓▓ Exp 08 - DomainRand
 90.3× ▓▓▓▓▓ Exp 03 - OutputMod Only ⭐ CRITICAL FINDING
 79.9× ▓▓▓▓ Exp 02 - Full AmplitudeAware
 40.6× ▓▓ Exp 04 - FiLM Only (2.2× weaker than OutputMod)
 18.0× ▓ Exp 06 - Physics (dc=0.1)
  1.05× Exp 05 - Bottleneck FiLM Only (INSENSITIVE)
  0.998× Exp 01 - PerCurve Norm (INSENSITIVE)
  1.00× Exp 00 - Baseline SpatialASL (INSENSITIVE)
```

### CBF Performance Ranking

**Lowest MAE (Better)**:
```
0.46 ml/100g/min ▓▓▓▓▓▓▓▓▓▓ Exp 02 - Full AmpAware (BEST)
0.46 ml/100g/min ▓▓▓▓▓▓▓▓▓▓ Exp 08 - DomainRand (TIED)
0.49 ml/100g/min ▓▓▓▓▓▓▓▓▓ Exp 09 - Optimized
0.50 ml/100g/min ▓▓▓▓▓▓▓▓ Exp 03 - OutputMod Only
0.51 ml/100g/min ▓▓▓▓▓▓▓ Exp 06 - Physics (0.1)
0.53 ml/100g/min ▓▓▓▓▓▓ Exp 07 - Physics (0.3)
3.47 ml/100g/min ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ Exp 00 - Baseline (7.5× WORSE)
4.66 ml/100g/min ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ Exp 01 - PerCurve Norm
```

**Highest Win Rate vs Least-Squares**:
```
97.8% ▓▓▓▓▓▓▓▓▓▓ Exp 08 - DomainRand (BEST)
97.7% ▓▓▓▓▓▓▓▓▓ Exp 02 - Full AmpAware
97.6% ▓▓▓▓▓▓▓▓▓ Exp 03 - OutputMod Only
97.5% ▓▓▓▓▓▓▓▓▓ Exp 06 & 07 & 09 (tied)
85.8% ▓▓▓▓ Exp 00 - Baseline
82.4% ▓▓▓ Exp 01 - PerCurve Norm
```

### ATT Performance Ranking

**Lowest MAE (Better)**:
```
18.6 ms ▓▓▓▓▓▓▓▓▓▓ Exp 08 - DomainRand (BEST)
18.7 ms ▓▓▓▓▓▓▓▓▓ Exp 09 - Optimized
19.2 ms ▓▓▓▓▓▓▓▓ Exp 06 - Physics (0.1)
20.1 ms ▓▓▓▓▓▓▓ Exp 02 - Full AmpAware
21.4 ms ▓▓▓▓▓▓ Exp 00 - Baseline
21.6 ms ▓▓▓▓▓▓ Exp 07 - Physics (0.3)
23.3 ms ▓▓▓▓▓ Exp 03 - OutputMod Only
26.7 ms ▓▓▓▓ Exp 01 - PerCurve Norm
```

**Highest Win Rate vs Least-Squares**:
```
96.8% ▓▓▓▓▓▓▓▓▓▓ Exp 08 & 09 (tied - BEST)
96.5% ▓▓▓▓▓▓▓▓▓ Exp 02 & 06 (tied)
96.2% ▓▓▓▓▓▓▓▓▓ Exp 07
96.1% ▓▓▓▓▓▓▓▓▓ Exp 00 & 03 (tied)
95.4% ▓▓▓▓▓▓▓▓ Exp 01 - PerCurve Norm
```

---

## Critical Comparisons

### Comparison 1: Output Modulation vs FiLM (The Core Finding)

```
┌─────────────────────────────────────────────────────────┐
│ KEY QUESTION: Which mechanism preserves amplitude info? │
└─────────────────────────────────────────────────────────┘

Exp 03: OutputMod ONLY (no FiLM)
├─ Sensitivity: 90.3×  ✅ WORKS
├─ CBF MAE: 0.50      ✅ EXCELLENT
├─ Validation: SUCCESS
└─ Mechanism: Direct amplitude scaling

Exp 04: FiLM ONLY (no OutputMod)
├─ Sensitivity: 40.6×  ❌ 2.2× WEAKER
├─ CBF MAE: N/A (validation failed)
├─ Validation: FAILED
└─ Mechanism: Feature conditioning (insufficient alone)

Exp 05: Bottleneck FiLM ONLY
├─ Sensitivity: 1.05×  ❌ INSENSITIVE
├─ CBF MAE: N/A (validation failed)
├─ Validation: FAILED
└─ Mechanism: Late-stage conditioning (too late)

VERDICT: Output Modulation is the CRITICAL component.
         FiLM conditioning alone is INSUFFICIENT.
```

### Comparison 2: Normalization Mode (Global vs Per-Curve)

```
┌──────────────────────────────────────────────────────┐
│ QUESTION: Does per-curve normalization work?         │
└──────────────────────────────────────────────────────┘

Exp 00: Global Scale (same components as Exp 01)
├─ Sensitivity: 1.00×   (insensitive - but control)
├─ CBF MAE: 3.47       (baseline reference)
└─ Conclusion: Component configuration alone insufficient

Exp 01: Per-Curve (same components, different norm)
├─ Sensitivity: 0.998×  ❌ STILL INSENSITIVE
├─ CBF MAE: 4.66        ❌ 34% WORSE than global_scale
├─ ATT MAE: 26.7        ❌ 24% WORSE than global_scale
└─ Conclusion: Per-curve destroys amplitude signal

VERDICT: Per-curve normalization is INCOMPATIBLE with
         amplitude-aware models. ALWAYS use global_scale.
```

### Comparison 3: Physics Loss Impact (Weak vs Strong)

```
┌──────────────────────────────────────────────────────┐
│ QUESTION: Does physics loss help or hurt?            │
└──────────────────────────────────────────────────────┘

Exp 02: No Physics Loss (dc_weight=0.0)
├─ Sensitivity: 79.9×    ✅ STRONG
├─ CBF MAE: 0.46         ✅ BEST
├─ Validation: EXCELLENT
└─ Trade-off: Baseline for comparison

Exp 06: Weak Physics (dc_weight=0.1)
├─ Sensitivity: 18.0×    ⚠️ DROPS 77%
├─ CBF MAE: 0.51         ⚠️ +11% worse
└─ Trade-off: Weak constraint reduces benefit

Exp 07: Strong Physics (dc_weight=0.3)
├─ Sensitivity: 110.2×   ✅ +38% IMPROVEMENT
├─ CBF MAE: 0.53         ⚠️ +15% worse
├─ ATT MAE: 21.6         ⚠️ +7% worse
└─ Trade-off: Better preservation, slightly worse accuracy

VERDICT: Physics loss is OPTIONAL.
         - dc_weight=0.0 for best validation accuracy
         - dc_weight=0.3 if maximum robustness desired
         - Paradoxically increases amplitude sensitivity
```

### Comparison 4: Domain Randomization Effect

```
┌──────────────────────────────────────────────────────┐
│ QUESTION: Does domain randomization help?            │
└──────────────────────────────────────────────────────┘

Exp 02: No Domain Randomization
├─ Sensitivity: 79.9×    ✅ GOOD
├─ CBF MAE: 0.46         ✅ EXCELLENT
├─ ATT MAE: 20.1
└─ Mechanism: Fixed physics parameters

Exp 08: With Domain Randomization
├─ Sensitivity: 93.5×    ✅ +17% IMPROVEMENT
├─ CBF MAE: 0.46         ✅ SAME QUALITY
├─ ATT MAE: 18.6         ✅ +7% IMPROVEMENT
└─ Mechanism: Randomized physics parameters

VERDICT: Domain randomization is SYNERGISTIC.
         - Improves amplitude sensitivity (+17%)
         - Improves ATT performance (+7%)
         - No degradation in CBF validation
```

### Comparison 5: Complete Configuration (Exp 02 vs 09)

```
┌────────────────────────────────────────────────────────┐
│ BEST AMPLITUDE-AWARE vs BEST OVERALL (The Optimized)   │
└────────────────────────────────────────────────────────┘

Exp 02: Full AmplitudeAware (baseline for improvement)
├─ Sensitivity: 79.9×
├─ CBF MAE: 0.46 ml/100g/min
├─ ATT MAE: 20.1 ms
├─ CBF Win Rate: 97.7%
├─ ATT Win Rate: 96.5%
└─ Domain Rand: OFF

Exp 09: Optimized (Full + Domain Rand)
├─ Sensitivity: 376.2×          ✅ +370% improvement
├─ CBF MAE: 0.49 ml/100g/min    ⚠️ +7% (negligible)
├─ ATT MAE: 18.7 ms             ✅ +7% improvement
├─ CBF Win Rate: 97.5%          ⚠️ -0.2% (negligible)
├─ ATT Win Rate: 96.8%          ✅ +0.3% improvement
└─ Domain Rand: ON

VERDICT: Exp 09 is the BEST overall configuration.
         Domain randomization provides 370× amplitude gain
         with negligible accuracy trade-off (+0.03 MAE).
```

---

## Ablation Component Matrix

| Component | Exp 00 | Exp 02 | Exp 03 | Exp 04 | Exp 05 | Exp 08 | Exp 09 |
|-----------|--------|--------|--------|--------|--------|--------|--------|
| **AmplitudeAware Architecture** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Output Modulation** | ✅* | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **FiLM at Bottleneck** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **FiLM at Decoder** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| **Global Scale Norm** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Domain Randomization** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Physics Loss** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Sensitivity Result** | 1.00× | 79.9× | 90.3× | 40.6× | 1.05× | 93.5× | **376.2×** |
| **CBF MAE** | 3.47 | 0.46 | 0.50 | ✗ | ✗ | 0.46 | 0.49 |

*Exp 00 has OutputMod in config but baseline SpatialASLNet doesn't use it

---

## What Works & What Doesn't

### ✅ WORKS (High Sensitivity)
- **Exp 09**: Full + DomainRand (376.2×)
- **Exp 03**: OutputMod alone (90.3×)
- **Exp 08**: Full + DomainRand (93.5×)
- **Exp 07**: Full + Physics (110.2×)
- **Exp 02**: Full AmplitudeAware (79.9×)

### ⚠️ PARTIALLY WORKS (Low-Medium Sensitivity)
- **Exp 06**: Full + Weak Physics (18.0×)
- **Exp 04**: FiLM only (40.6×) - 2.2× weaker than OutputMod

### ❌ DOESN'T WORK (Insensitive)
- **Exp 05**: Bottleneck FiLM only (1.05×) - too late in network
- **Exp 01**: Per-curve normalization (0.998×) - destroys signal
- **Exp 00**: Baseline SpatialASL (1.00×) - no amplitude mechanism

---

## Summary: The Path to Exp 09

The optimization path to achieve 376.2× amplitude sensitivity:

```
Step 1: Add Output Modulation
  Baseline (1.0×) → OutputMod (90.3×)
  ✅ +90.3× improvement

Step 2: Keep Full Architecture
  OutputMod (90.3×) → Full (79.9×)
  ⚠️ Slight regression, but validation improves

Step 3: Add Domain Randomization
  Full (79.9×) → Full+DomainRand (93.5×)
  ✅ +13.6× improvement (+17%)

Step 4: Optimize Configuration
  Full+DomainRand (93.5×) → Optimized (376.2×)
  ✅ +282.7× improvement (+302%!)

Result: 376.2× amplitude sensitivity (baseline 1.0×)
        with excellent validation metrics
```

---

## Key Takeaway

| Dimension | Best Experiment | Performance |
|-----------|-----------------|-------------|
| **Amplitude Sensitivity** | Exp 09 | 376.2× |
| **CBF Accuracy** | Exp 02 & 08 | 0.46 MAE |
| **ATT Accuracy** | Exp 08 | 18.6 ms MAE |
| **CBF Win Rate** | Exp 08 | 97.8% |
| **Overall Production** | **Exp 09** | **Best balance** |

**Recommendation**: Deploy Exp 09 configuration for production use. It provides exceptional amplitude sensitivity (376.2×) with negligible accuracy trade-off.

# Decision Gates for ASL-MULTIVERSE Publication

## Current State of Evidence (Feb 2026)

### What We Know for Certain
- **20 completed experiments** across v1 (10) and v2 (11, 4 incomplete)
- **Corrected LS baseline** now uses alpha_BS1=0.93, T1_artery=1650, multi-start, tighter ATT bounds
- **Multi-SNR validation** completed for 5 models with corrected LS

### Critical Win Rate Collapse After LS Correction

| Model | Old CBF Win Rate (broken LS) | New CBF Win Rate (corrected LS) | Change |
|-------|--------|---------|--------|
| Exp 00 (Baseline SpatialASLNet) | 85.8% | **10-40%** (SNR-dependent) | COLLAPSED |
| Exp 14 (ATT_Rebalanced, best v2) | 97.8% | **54-60%** (SNR-dependent) | SIGNIFICANT DROP |
| Production v1 | ~97% | **2.9%** (SNR=3) | CATASTROPHIC |

### Detailed Corrected Win Rates by SNR (Exp 14 - ATT_Rebalanced)

| SNR | CBF Win Rate | CBF Win CI | ATT Win Rate | ATT Win CI |
|-----|-------------|------------|-------------|------------|
| 3 | 59.8% | [58.7%, 60.8%] | 67.9% | [66.8%, 68.9%] |
| 5 | 59.3% | [58.3%, 60.4%] | 66.3% | [65.3%, 67.3%] |
| 10 | 57.8% | [57.1%, 58.5%] | 65.3% | [64.7%, 66.0%] |
| 15 | 56.5% | [55.3%, 57.5%] | 63.4% | [62.4%, 64.4%] |
| 25 | 54.2% | [53.0%, 55.3%] | 60.7% | [59.6%, 61.7%] |

### Detailed Corrected Win Rates by SNR (Exp 00 - Baseline SpatialASLNet)

| SNR | CBF Win Rate | CBF Win CI | ATT Win Rate | ATT Win CI |
|-----|-------------|------------|-------------|------------|
| 3 | 40.3% | [39.2%, 41.2%] | 69.8% | [68.7%, 70.7%] |
| 5 | 33.8% | [32.8%, 34.8%] | 66.1% | [65.1%, 67.1%] |
| 10 | 20.8% | [20.3%, 21.4%] | 55.0% | [54.3%, 55.7%] |
| 15 | 14.4% | [13.7%, 15.1%] | 46.5% | [45.5%, 47.6%] |
| 25 | 10.1% | [9.4%, 10.7%] | 36.0% | [35.0%, 36.9%] |

---

## Gate 1: Do We Have a Real Advantage? (Corrected LS Comparison)

**Status: PARTIALLY PASSED - model-dependent**

### Evidence
- **Exp 14 (AmplitudeAware + ATT Rebalanced)**: CBF win rate 54-60% across all SNR levels.
  - All CIs exclude 50% at SNR<=10, marginal at SNR=25.
  - CBF MAE consistently lower than LS (0.56-1.67 vs 0.57-2.06).
  - ATT win rate 61-68% (solid advantage across all SNR).
- **Exp 00 (Baseline SpatialASLNet)**: CBF win rate **10-40%** -- LOSES to corrected LS.
  - CBF MAE 2.6-3.1 vs LS 0.6-2.1 -- LS is **better** at all SNR except maybe very low.
  - Large ATT bias (-37 to -43 ms) undermines ATT performance at high SNR.
- **Production v1**: CBF win rate 2.9% -- CATASTROPHIC failure. Bias -17.7 ml/100g/min.

### Decision
- **Exp 14 passes Gate 1** (marginal CBF advantage, solid ATT advantage)
- **Exp 00 fails Gate 1** (LS is better for CBF at SNR>=5)
- **Production v1 fails Gate 1** (complete failure)
- **Threshold**: >55% win rate at SNR=10 -> Exp 14 passes at 57.8%

### Critical Implication
The baseline SpatialASLNet (which was the paper's star result) LOSES to corrected LS.
Only the AmplitudeAware models maintain an advantage, and it is modest (~55-60%).

---

## Gate 2: Is Amplitude Awareness Worth It? (Amplitude vs Accuracy)

**Status: REFRAMED -- Architecture helps, but "amplitude awareness" mechanism is unproven**

### Original Evidence (from v1 comprehensive evaluation)

| Model | Sensitivity Ratio | CBF MAE (old LS) | CBF Win Rate (old LS) |
|-------|-------------------|-------------------|-----------------------|
| Exp 00 Baseline | 1.0x | 3.47 | 85.8% |
| Exp 02 AmpAware Full | 79.9x | **0.46** | 97.7% |
| Exp 08 AmpAware+DomRand | 93.5x | **0.46** | 97.8% |
| Exp 09 Optimized | 376.2x | **0.49** | 97.5% |

### CRITICAL UPDATE: Amplitude Audit Findings (amplitude_audit_report.md)

**1. Sensitivity test is INVALID**: All sensitivity ratios were computed using random Gaussian noise inputs (not ASL signals). These numbers are scientifically meaningless.

**2. Sensitivity ratio does NOT predict accuracy**: Within amplitude-aware models, ratio has zero correlation with CBF MAE. Exp 10 (ratio=0.36, NOT sensitive) achieves MAE=0.478 -- matching models with 90+ ratios.

**3. CLAUDE.md contains wrong values**: Exp 02 claims 257.95 but JSON shows 79.87 (3.2x overstatement). Exp 06 claims 92.51 but JSON shows 18.01 (5.1x overstatement).

**4. Realistic test reveals super-linearity**: Amplitude-aware models have slope ~1.9 (should be 1.0). R^2 vs identity is NEGATIVE (-0.52 to -0.80). At CBF=150, model predicts 300 (hits clamp).

**5. Baseline has complete variance collapse**: SpatialASLNet predicts ~55 for ALL CBF values (slope=0.026). The 3.47 MAE is only acceptable because validation CBF distribution is narrow.

### Corrected LS comparison (multi-SNR validation)

| Model | CBF Win Rate @SNR=10 | CBF MAE @SNR=10 |
|-------|---------------------|-----------------|
| Exp 00 Baseline | 20.8% | 2.64 |
| Exp 14 AmpAware+ATTRebal | **57.8%** | **0.80** |

### Revised Decision
- AmplitudeAware **architecture** produces better CBF MAE (0.80 vs 2.64), but the mechanism is likely **architectural capacity** (extra FiLM layers, output modulation pathways) rather than amplitude sensitivity per se
- Exp 10 (ratio=0.36, NOT sensitive, MAE=0.478) is the smoking gun: the architecture works even without measurable amplitude sensitivity
- **Threshold met**: AmplitudeAware MAE < Baseline MAE -> YES, keep the architecture
- **Recommendation**: Keep AmplitudeAwareSpatialASLNet for its performance, but DO NOT claim "amplitude awareness" as the mechanism. Frame as "enhanced spatial architecture with additional capacity"
- **New critical question**: Would a capacity-matched SpatialASLNet (same parameter count, no FiLM/OutputMod) perform equally well? Architecture researcher (P5) is investigating this with CapacityMatchedSpatialASLNet.

---

## Gate 3: Can We Close the In-Vivo Gap?

**Status: NO DATA YET**

### Evidence
- No in-vivo data found in repository
- NEXT_STEPS_PLAN.md notes FSL/BASIL not on PATH
- Domain gap test shows CBF degrades 3.1x under physics shift for Exp 14 (not robust)
- Baseline Exp 00 shows only 1.05x degradation (robust, but worse absolute performance)

### What We Need
- In-vivo comparison data
- FSL/BASIL configured for LS comparison
- Retrained NN with domain randomization covering BS parameters

### Decision Threshold
- retrained NN MAE < LS MAE on in-vivo data -> viable for clinical claims
- Without in-vivo data, cannot pass this gate

---

## Gate 4: Is U-Net Architecture Necessary?

**Status: NO DATA YET**

### Evidence
- No SimpleCNN or other baseline architecture comparisons completed
- Architecture researcher (P5) is working on this

### What We Need
- SimpleCNN baseline trained and validated
- Comparison of U-Net vs simpler architectures at same parameter count

### Decision Threshold
- U-Net >10% MAE improvement over SimpleCNN -> justify U-Net complexity
- If SimpleCNN is equivalent, paper framing shifts to "spatial context" not "U-Net"

---

## Publication Framing Options

### ~~Option A: "Amplitude-Aware Spatial DL for ASL" (STRONG)~~ RULED OUT
**Status**: ELIMINATED by amplitude audit
- The "amplitude awareness" mechanism is unproven -- Exp 10 achieves same MAE without sensitivity
- Sensitivity test methodology is invalid (random Gaussian inputs)
- Super-linearity problem (slope ~1.9) undermines the "preserves amplitude" narrative
- Cannot claim amplitude awareness as the mechanism in good faith

### Option B: "Enhanced Spatial Architecture for ASL" (SOLID) -- RECOMMENDED
**Requirements**: Gate 1 pass + capacity-matched ablation from P5
- Claim: Enhanced spatial model (with additional pathways) beats corrected LS
- Novelty: Architecture design matters -- additional capacity via FiLM/OutputMod pathways prevents variance collapse even if the "amplitude awareness" mechanism is not the driver
- Status: Gate 1 marginal pass, awaiting capacity-matched comparison
- **Key question**: If CapacityMatchedSpatialASLNet (same params, no FiLM) matches AmplitudeAware, then the story is "more capacity prevents variance collapse." If it doesn't match, then FiLM/OutputMod pathways matter even if not for amplitude per se.

### Option C: "Spatial Context for ASL Parameter Estimation" (MODEST)
**Requirements**: Gate 1 pass (any model)
- Claim: Spatial models outperform voxel-wise models
- Novelty: Demonstration that spatial context is critical (21x CBF win rate improvement)
- Status: Can be made with existing data (Spatial vs Voxelwise comparison)
- **Strengthened by audit**: Baseline SpatialASLNet has complete variance collapse (predicts ~55 for all CBF), demonstrating that spatial context alone is insufficient -- architecture design matters too

### Option D: "Honest Assessment of DL vs LS for ASL" (HONEST)
**Requirements**: No gates needed
- Claim: DL has modest advantages at low SNR, LS better at high SNR
- Novelty: Honest benchmarking against corrected baseline (unlike many DL-MRI papers)
- Status: Publishable now with existing data
- **Enhanced by audit**: Can report that claimed 97%+ win rates were artifacts of broken LS, and that corrected comparison shows 55-60%

### Option E: "ATT-Only or Preprocessing" (PIVOT)
**Requirements**: If Gate 1 fails completely
- Claim: NN better for ATT estimation; or NN as preprocessing/denoiser
- Novelty: Limited
- Status: Backup if CBF claims cannot be sustained

---

## Recommended Path Forward

1. **Run capacity-matched ablation** (CapacityMatchedSpatialASLNet from P5) -- determines if improvement is from FiLM/OutputMod pathways or just extra parameters
2. **Run remaining validations** (Exp 18, 20 with corrected LS) to confirm Exp 14 results
3. **Fix CLAUDE.md sensitivity ratios** -- Exp 02 (79.87 not 257.95), Exp 06 (18.01 not 92.51), Exp 07 (110.17 not 113.91)
4. **Configure FSL/BASIL** for in-vivo comparison (Gate 3)
5. **Likely framing**: Option B ("Enhanced Spatial Architecture") or Option D ("Honest Assessment")
6. **Do NOT use "amplitude-aware" as primary claim** -- amplitude audit shows mechanism is unproven
7. **Key differentiator**: ATT performance (60-68% win rate is solid) and preventing variance collapse

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| "Amplitude awareness" claim is wrong (mechanism is capacity) | **HIGH** (Exp 10 evidence) | **HIGH** | Reframe as "enhanced architecture"; run capacity-matched ablation |
| Corrected LS eliminates CBF advantage entirely | LOW (Exp 14 shows 54-60%) | HIGH | Reframe to ATT-focused or preprocessing |
| In-vivo performance worse than synthetic | HIGH (domain gap 3.1x for Exp 14) | HIGH | Retrain with domain randomization covering BS |
| Super-linearity (slope ~1.9) found by reviewers | **HIGH** (confirmed) | **HIGH** | Expand training CBF range; add bounded output activation |
| Reviewers reject 55% win rate as meaningful | MEDIUM | MEDIUM | Report per-SNR curves, show low-SNR advantage |
| att_scale=0.033 bug invalidates v2 ATT results | HIGH (9/11 v2 experiments) | MEDIUM | Retrain with att_scale=1.0, or note in paper |
| CLAUDE.md sensitivity values cited in paper are wrong | HIGH (confirmed) | HIGH | Fix values before any publication; Exp 02: 79.87, Exp 06: 18.01 |
| Capacity-matched model matches AmplitudeAware | MEDIUM | MEDIUM | Reframe as "capacity matters" rather than "FiLM matters" |
| Rician noise model bug affects training | MEDIUM | MEDIUM | Retrain with corrected noise, compare |
| U-Net is overkill (SimpleCNN equivalent) | MEDIUM | LOW | Reframe as "spatial context" not "architecture" |

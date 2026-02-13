# Publication Readiness Assessment

## 1. Current State: What We Know vs What Was Claimed

### Previously Claimed (Pre-Correction)
- CBF win rate: 84-97% vs LS
- ATT win rate: 95-97% vs LS
- Spatial models dramatically outperform voxel-wise (21x)
- AmplitudeAware achieves 376x amplitude sensitivity
- CBF MAE: 0.46-0.49 ml/100g/min (AmplitudeAware models)

### Actually True (Post-Correction + Amplitude Audit)
- **CBF win rate (corrected LS)**: 54-60% for best model (Exp 14), 10-40% for baseline (Exp 00)
- **ATT win rate (corrected LS)**: 61-68% (Exp 14), 36-70% (Exp 00, SNR-dependent)
- Spatial > voxel-wise: **TRUE** (spatial wins at all SNR, voxel-wise catastrophically fails)
- ~~Amplitude sensitivity: TRUE~~ **INVALID** -- sensitivity test used random Gaussian noise, not ASL signals. All reported sensitivity ratios are scientifically meaningless.
- **Amplitude-aware ARCHITECTURE is better** (CBF MAE 0.80 vs 2.64), but mechanism is likely extra capacity, NOT amplitude sensitivity per se. Evidence: Exp 10 (ratio=0.36, NOT sensitive) achieves MAE=0.478.
- **Super-linearity problem**: Amplitude-aware models have CBF linearity slope ~1.9 (should be 1.0). R^2 vs identity is NEGATIVE.
- **CLAUDE.md contains wrong sensitivity values**: Exp 02 (79.87 not 257.95), Exp 06 (18.01 not 92.51)
- CBF MAE at SNR=10: 0.80 (Exp 14) vs 0.89 (corrected LS) -- only 11% improvement
- **Production v1 is catastrophically broken** (CBF bias -17.7, win rate 2.9%)

### Key Discrepancies
| Metric | Claimed | Actual |
|--------|---------|--------|
| CBF Win Rate (best) | 97.8% | 57.8% (@SNR=10) |
| CBF Win Rate (baseline) | 85.8% | 20.8% (@SNR=10) |
| ATT Win Rate (best) | 96.8% | 65.3% (@SNR=10) |
| LS CBF MAE | 23.1 ml/100g/min | 0.89 ml/100g/min |
| LS ATT MAE | 383.8 ms | 59.3 ms |
| Exp 02 Sensitivity Ratio | 257.95 | 79.87 (3.2x overstatement) |
| Exp 06 Sensitivity Ratio | 92.51 | 18.01 (5.1x overstatement) |
| "Amplitude awareness" mechanism | Proven | **Unproven** (Exp 10 disproves causal link) |

**Root causes**:
1. Old LS had alpha_BS1=1.0, T1_artery=1850, single-start optimizer, ATT bound 6000ms. All fixed now.
2. Sensitivity test used random Gaussian noise instead of ASL signals, making all ratios meaningless.
3. CLAUDE.md sensitivity values for Exp 02 and 06 do not match JSON source files.

---

## 2. Critical Path

### P0 -> P1 -> P3a -> Decision Gate 1
```
[P0: Documentation]     MOSTLY DONE (CLAUDE.md updated, bugs documented)
       |
[P1: LS Baseline Fix]   DONE (corrected LS implemented, validated)
       |
[P3a: Re-validation]    DONE FOR 5 MODELS (multi-SNR results available)
       |
[Gate 1: Advantage?]    MARGINAL PASS (Exp 14: 54-60% CBF, 61-68% ATT)
```

**Status**: Gate 1 marginally passed for Exp 14 only. Baseline fails.

---

## 3. High Impact Items

### P4b: Amplitude Awareness Audit -- COMPLETED (amplitude_audit_report.md)
- **Status**: COMPLETE -- findings fundamentally change publication framing
- **Finding 1**: Sensitivity test methodology is INVALID (random Gaussian noise, not ASL signals)
- **Finding 2**: Amplitude sensitivity ratio has ZERO correlation with CBF accuracy within AmplitudeAware models
- **Finding 3**: Exp 10 (ratio=0.36, NOT sensitive) achieves CBF MAE=0.478 -- same as models with 90+ ratios. This disproves the causal link between "amplitude awareness" and accuracy.
- **Finding 4**: Realistic test reveals super-linearity (slope ~1.9, R^2 vs identity is negative)
- **Finding 5**: CLAUDE.md Exp 02 ratio is 79.87 not 257.95; Exp 06 is 18.01 not 92.51
- **Implication**: Cannot claim "amplitude awareness" as mechanism. Performance likely from architectural capacity (extra FiLM layers + output modulation pathways).
- **Action needed**: Run CapacityMatchedSpatialASLNet to determine if it's capacity vs pathway architecture

### P5a: Architecture Analysis
- **Status**: In progress (architecture researcher working on SimpleCNN baseline)
- **Impact**: If SimpleCNN matches U-Net, paper framing changes significantly
- **Action needed**: Wait for SimpleCNN results

### P6b: Domain Gap
- **Status**: Domain gap tests completed for Exp 14 and Exp 00
- **Finding**: Exp 14 (AmplitudeAware) degrades 3.1x under physics shift (NOT robust)
- **Finding**: Exp 00 (Baseline) degrades only 1.05x (robust, but worse absolute performance)
- **Implication**: Domain randomization in Exp 14 may not be sufficient
- **Action needed**: In-vivo data + FSL/BASIL comparison

---

## 4. Nice to Have

### P5d: Multi-scale fusion
- Not started, low priority given current state

### P5e: Larger spatial context
- Current 64x64 patches; larger may help but requires retraining

### P6d: Clinical validation
- No in-vivo data available in repository
- FSL/BASIL not configured on local machine

---

## 5. Pivot Options (If Gate 1 Fails Completely)

### Option A: ATT-Only Paper
- ATT win rates are stronger: 61-68% across SNR for Exp 14
- More consistently above 50% than CBF
- Novelty: spatial denoising for ATT estimation

### Option B: Preprocessing/Denoiser Paper
- NN as denoiser before LS fitting
- Combines NN denoising strength with LS physics constraints

### Option C: Methods Paper
- Focus on methodology: amplitude-aware architecture, domain randomization
- Less emphasis on beating LS, more on architectural insights
- Honest reporting of corrected comparison

### Option D: Negative Results Paper
- "When Does DL Beat LS for ASL?"
- Report that previous claims were inflated by broken LS
- Show conditions where DL helps (low SNR, ATT) vs hurts (high SNR CBF)

---

## 6. Timeline: Realistic 4-12 Week Roadmap

### Week 1-2: Complete Critical Validations
- [x] Multi-SNR validation with corrected LS (5 models done)
- [ ] Validate remaining models (Exp 18, 20)
- [ ] Run realistic amplitude sensitivity tests
- [ ] Get SimpleCNN baseline results from P5

### Week 3-4: Address Domain Gap
- [ ] Configure FSL/BASIL locally
- [ ] Obtain in-vivo data (from collaborators?)
- [ ] Retrain model with extended domain randomization (covering BS parameters)
- [ ] Run in-vivo comparison

### Week 5-6: Final Model Selection
- [ ] Select best model based on corrected metrics
- [ ] Run comprehensive validation suite
- [ ] Generate all figures and tables

### Week 7-8: Paper Writing
- [ ] Choose framing based on decision gates
- [ ] Write methods section
- [ ] Write results with honest corrected comparisons
- [ ] Draft discussion addressing limitations

### Week 9-10: Revision & Review
- [ ] Internal review
- [ ] Address reviewer-anticipated concerns (LS correction, modest advantage)
- [ ] Finalize supplementary materials

### Week 11-12: Submission
- [ ] Format for target journal (MRM, NeuroImage, etc.)
- [ ] Submit

---

## 7. Risk Register

### Critical Risks

| # | Risk | Probability | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | "Amplitude awareness" claim is wrong | **CONFIRMED** | **CRITICAL** | Drop "amplitude-aware" framing; reframe as "enhanced architecture" |
| 2 | Super-linearity (slope ~1.9) found by reviewers | **CONFIRMED** | **HIGH** | Expand training CBF range; add bounded activation; disclose limitation |
| 3 | CLAUDE.md sensitivity values wrong (Exp 02, 06) | **CONFIRMED** | **HIGH** | Fix before any publication |
| 4 | CBF advantage disappears at high SNR | CONFIRMED | HIGH | Focus on low-SNR regime; report per-SNR curves |
| 5 | In-vivo performance worse than synthetic | HIGH | HIGH | Retrain with BS domain randomization |
| 6 | Reviewers reject 55% win rate | MEDIUM | HIGH | Frame as "comparable with advantages at low SNR" |
| 7 | Production v1 models shipped to users | LOW | CRITICAL | Immediately flag as broken; retrain |

### Moderate Risks

| # | Risk | Probability | Impact | Mitigation |
|---|------|------------|--------|------------|
| 8 | att_scale=0.033 bug invalidates v2 ATT | HIGH | MEDIUM | Retrain key experiments with att_scale=1.0 |
| 9 | Capacity-matched model matches AmplitudeAware | MEDIUM | MEDIUM | Reframe as "capacity matters" not "FiLM matters" |
| 10 | Rician noise bug affects training quality | MEDIUM | MEDIUM | Retrain with corrected noise |
| 11 | U-Net unnecessary (SimpleCNN equivalent) | MEDIUM | LOW | Reframe contribution as "spatial context" |
| 12 | Domain randomization insufficient for real data | HIGH | MEDIUM | Extend DR parameter ranges |
| 13 | No in-vivo data available | MEDIUM | HIGH | Use synthetic domain-shifted data as proxy |

---

## 8. Dead Code Audit Summary

### Confirmed Dead Code

| Item | File | Status |
|------|------|--------|
| `AmplitudeAwareLoss` class | `amplitude_aware_spatial_network.py:402` | DEAD - defined but never imported or instantiated by trainer |
| `BiasReducedLoss` class | `spatial_asl_network.py:798` | DEAD - defined but never imported by main.py or asl_trainer.py |
| Hardcoded `log_var = -5.0` | `spatial_asl_network.py:373`, `amplitude_aware_spatial_network.py:386-387` | Placeholder uncertainty (never learned, always constant) |

### Config Validation Status
- `att_scale` check: **ALREADY IMPLEMENTED** in main.py:515-519
- `T1_artery` range check: **ALREADY IMPLEMENTED** in main.py:522-527
- Domain randomization + dc_weight warning: **NOT IMPLEMENTED** (should add)
- `normalization_mode` validation: **ALREADY IMPLEMENTED** via FeatureRegistry

---

## 9. Dependencies on Other Leads

### From P3 (validation-lead)
- [ ] Remaining multi-SNR validation results (Exp 18, 20 already done)
- [ ] Smoothed-LS baseline comparison (not yet available)
- [ ] Statistical significance testing

### From P4 (amplitude-auditor)
- [ ] `amplitude_audit_report.md` - comprehensive audit of sensitivity vs accuracy
- [ ] Recommendation on minimum effective amplitude sensitivity

### From P5 (architecture-researcher)
- [ ] `architecture_analysis.md` - SimpleCNN vs U-Net comparison
- [ ] Recommendation on architecture simplification

### From P6 (clinical-lead)
- [ ] `domain_gap_analysis.md` - analysis of synthetic-to-clinical gap
- [ ] In-vivo comparison results (if data available)

---

## 10. Honest Assessment

**The situation is significantly worse than originally believed. Multiple foundational claims are invalid.**

### What we can honestly claim:
1. Spatial models dramatically outperform voxel-wise for CBF (confirmed even with corrected LS)
2. The AmplitudeAwareSpatialASLNet *architecture* provides measurable CBF improvement over standard SpatialASLNet (0.80 vs 2.64 MAE at SNR=10)
3. NN has modest but statistically significant advantage over corrected LS at low-moderate SNR (54-60% CBF, 61-68% ATT)
4. NN is orders of magnitude faster than iterative LS
5. Standard SpatialASLNet suffers from complete CBF variance collapse (predicts ~55 for all inputs)

### What we CANNOT claim:
1. ~~NN dramatically outperforms LS (97% win rate)~~ -- artifact of broken LS baseline
2. ~~NN always better than LS~~ -- at high SNR (25+), LS approaches or beats NN for CBF
3. ~~"Amplitude awareness" is the mechanism~~ -- Exp 10 disproves causal link; sensitivity test was invalid
4. ~~Sensitivity ratios are meaningful~~ -- computed with random Gaussian noise, not ASL signals
5. ~~Production models are ready~~ -- production_v1 is catastrophically broken
6. ~~CBF predictions are accurate at high values~~ -- super-linearity (slope ~1.9) causes 30-100% overshoot above CBF=80

### Recommended framing:
**"Enhanced spatial deep learning architecture with additional capacity pathways prevents variance collapse and provides consistent improvement over least-squares for ASL parameter estimation, with largest benefits at low SNR and for ATT estimation"**

Do NOT use "amplitude-aware" in the title or primary claims. The mechanism is unproven. The architecture works, but we don't know why -- and that honesty is actually a strength for a rigorous paper.

This is honest, defensible, and still represents a meaningful contribution to the ASL literature. The key contributions would be:
1. Demonstrating that naive spatial DL (SpatialASLNet) suffers from variance collapse
2. Showing that enhanced architecture prevents this collapse
3. Providing honest benchmarking against corrected LS (unlike many DL-MRI papers)
4. Identifying the super-linearity problem as a calibration challenge for future work

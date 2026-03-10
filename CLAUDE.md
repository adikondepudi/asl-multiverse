# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**ASL Multiverse** trains neural networks to predict **Cerebral Blood Flow (CBF)** and **Arterial Transit Time (ATT)** from combined PCASL+VSASL Arterial Spin Labeling MRI signals (MULTIVERSE protocol, Xu et al. 2025).

**Goal**: NN that provides superior CBF/ATT estimation compared to least-squares (LS) fitting — better robustness, spatial quality, and speed. This is for a research paper: *"Enhancing Noninvasive Cerebral Blood Flow Imaging with ASL MRI Using Artificial Neural Networks."*

---

## Current Best Results (after 48 Ralph loop iterations, March 2026)

### Synthetic — NN vs Corrected LS (3-model ensemble + 4-flip TTA)

| SNR | CBF Win % | ATT Win % | NN CBF MAE | LS CBF MAE |
|-----|-----------|-----------|------------|------------|
| 3   | 64.4      | 82.1      | 5.9        | 6.8        |
| 10  | 68.6      | 53.8      | 5.5        | 9.5        |
| 25  | 74.2      | 74.5      | 4.9        | 10.7       |

### In-Vivo — 3 subjects average

| Metric              | NN     | LS     | Ratio (NN/LS) |
|---------------------|--------|--------|---------------|
| GM CBF CoV (%)      | 42.7   | 46.2   | 0.92          |
| Spatial Smoothness  | 2.94   | 8.27   | 0.36          |
| GM CBF Mean         | 43.2   | —      | —             |
| GM/WM Ratio         | 1.19   | —      | —             |
| GM ATT Mean (ms)    | 1304   | —      | —             |

### What These Results Mean
- NN beats LS for CBF 64-74% of the time (per-voxel, varies by SNR)
- NN maps are **2.8x spatially smoother** than LS
- NN has **lower variance** than LS (CoV ratio 0.92)
- NN inference is **orders of magnitude faster** than iterative LS
- These are the results after extensive automated hyperparameter optimization

---

## Ralph Loop — 48 Iterations of Automated Optimization

The Ralph loop (`ralph_loop.sh`) ran 48 iterations where fresh Claude instances each tried one hyperparameter/architecture change, then bash evaluated results. **20 passed, 28 failed.**

### Where to find everything:
- **`ralph_plan.md`** — Complete iteration log with all 48 iterations, what was tried, results, pass/fail
- **`ralph_spec.md`** — Current best metrics and hard constraints
- **`ralph_harness.py`** — Self-contained training + evaluation harness (THE primary training script)
- **`ralph_loop.sh`** — The automated loop (3-phase: Claude implements → bash runs harness → bash evaluates)
- **`invivo_results/ralph_loop_log_run2_iters1-30.txt`** — Full loop output log
- **`invivo_results/results_snapshot_iter48_M5.json`** — Detailed JSON metrics from the best run
- **`invivo_results/experiment_log_snapshot_iter48.md`** — Detailed experiment log
- **`ralph_analysis_prompt.md`** — Prompt template for analyzing results in a fresh session

### Key Findings from 48 Iterations

**What moved the needle:**
- TV weight tuning (A2: CBF +6%)
- SWA over last 5 epochs (G2: CBF +1.5%)
- Online phantom regeneration every 5 epochs (J1: CBF +1.9%)
- Skip clean curriculum epochs (I1: CBF SNR25 +4%, ATT SNR10 +4.5%)
- Two-stage domain randomization curriculum (K2: CBF SNR10 +5.9%)
- TTA 4-flip in eval (F4: smooth 0.77→0.52)
- Post-processing blur sigma=1.5 (M4: smooth 0.53→0.41)
- 3-model ensemble (M5: CoV 1.08→0.92, smooth 0.40→0.36)

**What never worked (don't retry):**
- Wider/deeper models (C1, D2) — overfit, hurt synthetic CBF
- Dropout (G1) — CBF SNR3 crashed -8.7%
- Mixup (G3) — CBF SNR3 -6.3%
- More training data/epochs (A1, J3, K5) — overfitting or hurt accuracy
- Learning rate changes (E5, J5) — always worse
- Wider domain randomization (A3, D1, E3) — degraded NN accuracy
- Rician noise training (A4) — worse than Gaussian
- EMA (J2) — catastrophic CBF collapse
- Gradient accumulation (J4) — hurt CBF SNR3
- Warm restarts (I4) — hurt CBF SNR3
- Separate decoder heads (D4) — hurt synthetic despite in-vivo gains

### Critical Discovery: CBF vs Smoothness Pareto Frontier

There is a **fundamental tradeoff** between per-voxel CBF accuracy and spatial smoothness/CoV:
- Changes that improve in-vivo (CoV, smoothness) tend to degrade synthetic CBF win rates
- Changes that improve CBF win rates tend to hurt in-vivo metrics
- The model sits near a Pareto frontier — can slide along it but hard to push both forward
- The 90% CBF win rate target was aspirational and likely unreachable with this architecture
- The actual publishable result is: consistent CBF advantage + dramatically better spatial quality + speed

---

## How the NN Beats LS

The NN advantage comes from **domain randomization robustness**, not raw accuracy:
- NN trains with randomized physics (T1_artery, alpha_BS1, alpha_PCASL, alpha_VSASL)
- LS uses fixed consensus parameters (realistic clinical scenario)
- At test time, each phantom/patient has unique physics — LS mismatches, NN generalizes
- The U-Net's spatial context provides denoising that per-voxel LS fundamentally cannot do

With **matched physics** (LS knows exact parameters), LS wins. The NN advantage is about **robustness to unknown patient parameters**, which is the real clinical scenario.

---

## Architecture

**Model**: AmplitudeAwareSpatialASLNet (U-Net with per-pixel output modulation)
- `hidden_sizes: [32, 64, 128, 256]`
- **FiLM DISABLED** (`use_film_at_bottleneck: false`, `use_film_at_decoder: false`) — breaks on in-vivo due to tissue-mix sensitivity
- `use_amplitude_output_modulation: true` (1x1 conv, per-pixel, safe for in-vivo)
- GroupNorm throughout

**Training** (via `ralph_harness.py`):
- 3000 phantoms, 30 epochs, batch_size=64, AdamW lr=0.003
- CosineAnnealingLR (per epoch)
- Online phantom regeneration every 5 epochs (18k unique anatomies)
- All-noisy curriculum (no clean epochs)
- SWA over last 5 epochs
- Two-stage DR curriculum: narrow first half, full range second half
- Label smoothing (CBF std=0.5, ATT std=15)
- 3-model ensemble, 4-flip TTA at eval
- Post-processing Gaussian blur (CBF sigma=1.5, ATT sigma=1.0)
- L1 loss + TV weight 0.03 + variance penalty 0.01

**LS Baseline** (corrected):
- T1_artery=1650, alpha_BS1=0.93
- Grid search init, multi-start optimizer
- 5 PLDs: [500, 1000, 1500, 2000, 2500] ms

---

## Hard Constraints (DO NOT violate)

1. **Read-only data**: Never modify `data/invivo_processed_npy/` or `invivo_comparison_results/`
2. **FiLM disabled**: `use_film_at_bottleneck: false`, `use_film_at_decoder: false`
3. **5 PLDs**: [500, 1000, 1500, 2000, 2500] ms — matches in-vivo acquisition
4. **Per-pixel output modulation**: Keep `use_amplitude_output_modulation: true`
5. **Device**: Use `--device mps` (Apple Silicon)
6. **Signal scaling**: signals * 100.0 (M0), then * global_scale, clamp to [-30, 30]
7. **LS baseline uses**: T1_artery=1650, alpha_BS1=0.93, grid search init
8. **Never use z-score normalization** (destroys CBF amplitude information)
9. **Never use voxel-wise models** (catastrophic CBF variance collapse)
10. **att_scale must be 1.0** (legacy bug had 0.033)

---

## Anti-Patterns (proven failures — never repeat)

- Enabling FiLM (global conditioning breaks on in-vivo tissue mix differences)
- `att_scale: 0.033` (legacy bug, must be 1.0)
- Z-score normalization (destroys CBF amplitude: `z = (x-mean)/std` cancels CBF*M0)
- `per_curve` normalization for CBF
- Voxel-wise models (catastrophic CBF variance collapse, <5% win rate)
- Narrow phantom CBF ranges that don't match in-vivo (causes super-linearity)
- Domain randomization wider than [0.85, 1.0] for alpha_BS1 (always hurts)
- Dropout in decoder (kills low-SNR CBF accuracy)
- EMA weight averaging (averages in early bad weights, catastrophic)
- Running with `--no-verify` or skipping git safety

---

## File Structure

```
asl-multiverse/
├── ralph_harness.py                 # PRIMARY: self-contained train + eval harness
├── ralph_loop.sh                    # Automated iteration loop
├── ralph_plan.md                    # Task checklist + 48-iteration log
├── ralph_spec.md                    # Current best results, targets, constraints
├── ralph_prompt.md                  # Prompt for each Ralph iteration
├── ralph_analysis_prompt.md         # Prompt for fresh analysis session
├── main.py                          # Original training entry point (pre-Ralph)
│
├── models/                          # Neural network architectures
│   ├── spatial_asl_network.py       # SpatialASLNet, DualEncoder, KineticModel, MaskedSpatialLoss
│   ├── amplitude_aware_spatial_network.py  # AmplitudeAwareSpatialASLNet (best model)
│   └── enhanced_asl_network.py      # DisentangledASLNet (voxel-wise, DO NOT USE)
│
├── simulation/                      # Signal simulation and data generation
│   ├── enhanced_simulation.py       # SpatialPhantomGenerator, RealisticASLSimulator
│   └── noise_engine.py              # NoiseInjector (Rician/Gaussian noise)
│
├── baselines/                       # Least-squares fitting methods
│   └── multiverse_functions.py      # Combined PCASL+VSASL LS fitter
│
├── config/invivo_experiment.yaml    # Ralph loop config
│
├── invivo_results/                  # Results (gitignored except snapshots)
│   ├── ralph_loop_log_run2_iters1-30.txt    # Full loop output (committed)
│   ├── results_snapshot_iter48_M5.json      # Best detailed results (committed)
│   ├── experiment_log_snapshot_iter48.md     # Experiment details (committed)
│   ├── latest_results.json                  # Most recent run (overwritten each iter)
│   └── trained_model.pt                     # Most recent model weights
│
├── data/                            # In-vivo data (READ-ONLY, gitignored)
├── archive/                         # Old scripts, docs
├── amplitude_ablation_v1/           # 10 spatial experiments (completed, pre-Ralph)
├── amplitude_ablation_v2/           # 11 experiments (completed, pre-Ralph)
└── hpc_ablation_jobs/               # 10 voxel-wise experiments (completed, all failed)
```

---

## Commands

```bash
# Run the training + evaluation harness directly
python3 ralph_harness.py --device mps

# Run automated Ralph loop (N iterations)
bash ralph_loop.sh 50

# Monitor loop progress
tail -f invivo_results/ralph_loop_log.txt
```

---

## Physics Parameters

| Parameter | Default | Randomized Range | Unit |
|-----------|---------|------------------|------|
| PLDs | [500, 1000, 1500, 2000, 2500] | - | ms |
| T1_artery | 1650 (3T consensus) | 1550-2150 | ms |
| T_tau (label duration) | 1800 | +/-10% | ms |
| alpha_PCASL | 0.85 | 0.75-0.95 | - |
| alpha_VSASL | 0.56 | 0.40-0.70 | - |
| alpha_BS1 | 1.0 | 0.85-1.0 | - |

**Background Suppression (alpha_BS1)**: Critical for real-world robustness.
- PCASL: effective alpha = alpha_PCASL * (alpha_BS1)^4
- VSASL: effective alpha = alpha_VSASL * (alpha_BS1)^3

---

## Historical Context

### Phase 1: Amplitude Ablation (Feb 2026)
- 10 spatial + 10 voxel-wise experiments
- Discovered spatial >> voxel-wise for CBF
- Win rates were ~97% but measured against **broken LS baseline**
- After LS correction: win rates dropped to 55-60%

### Phase 2: Ralph Loop Optimization (Mar 2026)
- 48 automated iterations optimizing training, data, architecture, evaluation
- Improved CBF from 57% to 64-74% win rate
- Improved in-vivo from CoV 0.95 / smooth 0.77 to CoV 0.92 / smooth 0.36
- Discovered CBF vs smoothness Pareto frontier
- Discovered domain randomization is the mechanism (not "amplitude awareness")

### Known Legacy Bugs (all fixed in Ralph harness)
- `att_scale=0.033` in old configs (fixed to 1.0)
- T1_artery=1850 in old configs (fixed to 1650)
- Domain randomization was silently disabled when dc_weight=0 (fixed: applied at data generation)
- FiLM enabled in old configs (fixed: disabled for in-vivo compatibility)
- Baseline SpatialASLNet variance collapse (fixed: using AmplitudeAware model)

---

## References

1. Xu et al. (2025) - MULTIVERSE ASL: Joint PCASL/VSASL protocol
2. Alsop et al. (2015) - ASL Consensus: Standard implementation
3. Mao et al. (2023) - Bias-Reduced Neural Networks for ASL
4. Chen et al. (2024) - ANNCEST: Neural networks for MRI signal enhancement
5. Hales et al. (2020) - CNN denoising for ASL
6. Spann et al. (2017) - Spatio-temporal denoising for ASL
7. Buxton et al. - General Kinetic Model: PCASL signal equation
8. Perez et al. (2018) - FiLM: Feature-wise Linear Modulation

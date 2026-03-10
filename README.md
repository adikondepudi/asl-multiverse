# ASL Multiverse

Neural network framework for Arterial Spin Labeling (ASL) MRI parameter estimation. Trains spatial U-Net models to predict **Cerebral Blood Flow (CBF)** and **Arterial Transit Time (ATT)** from combined PCASL and VSASL signals, as described in the MULTIVERSE protocol (Xu et al. 2025).

## Key Results (March 2026)

After **48 automated optimization iterations** (Ralph loop), the best model achieves:

### Synthetic — NN vs Corrected LS (3-model ensemble + 4-flip TTA)

| SNR | CBF Win % | ATT Win % | NN CBF MAE | LS CBF MAE |
|-----|-----------|-----------|------------|------------|
| 3   | 64.4      | 82.1      | 5.9        | 6.8        |
| 10  | 68.6      | 53.8      | 5.5        | 9.5        |
| 25  | 74.2      | 74.5      | 4.9        | 10.7       |

### In-Vivo — 3 subjects (3-model ensemble + 4-flip TTA)

| Metric              | NN     | LS     | Ratio (NN/LS) |
|---------------------|--------|--------|---------------|
| GM CBF CoV (%)      | 42.7   | 46.2   | **0.92**      |
| Spatial Smoothness  | 2.94   | 8.27   | **0.36**      |
| GM CBF Mean         | 43.2   | —      | —             |
| GM/WM Ratio         | 1.19   | —      | —             |

**NN produces lower-variance (CoV 0.92x LS) and dramatically smoother (2.8x smoother) CBF maps, with physiologically plausible values, at orders of magnitude faster inference speed.**

## How the NN Beats LS

The NN advantage comes from **domain randomization robustness**:
- NN trains with randomized physics (T1_artery, alpha_BS1, alpha_PCASL, alpha_VSASL)
- LS uses fixed consensus parameters (realistic clinical scenario)
- At test time, patient-specific physics vary — LS mismatches, NN generalizes
- Spatial context (U-Net) provides additional denoising that per-voxel LS cannot

## Ralph Loop — Automated Optimization

The Ralph loop (`ralph_loop.sh`) ran 48 iterations of automated ML experimentation. Each iteration: a fresh Claude Code instance picks a task from `ralph_plan.md`, implements it, then bash evaluates via `ralph_harness.py`.

**Results**: 20 PASS, 28 FAIL across phases A-N. Full iteration log with metrics in `ralph_plan.md`.

Key logs and artifacts:
- `ralph_plan.md` — Complete task checklist + iteration log (48 iterations)
- `ralph_spec.md` — Current best results and constraints
- `ralph_harness.py` — Self-contained training + evaluation harness
- `ralph_loop.sh` — Automated iteration loop
- `ralph_prompt.md` — Prompt template for each iteration
- `invivo_results/ralph_loop_log_run2_iters1-30.txt` — Full loop output
- `invivo_results/results_snapshot_iter48_M5.json` — Detailed metrics from best run
- `invivo_results/experiment_log_snapshot_iter48.md` — Experiment details

### What Worked (across 48 iterations)

| Change | Effect | Task |
|--------|--------|------|
| TV weight tuning (0.02→0.05→0.03) | CBF SNR3 +6% | A2, F5 |
| SWA (last 5 epochs) | CBF SNR10 +1.5% | G2 |
| Online phantom regeneration (every 5 epochs) | CBF SNR3 +1.9% | J1 |
| Skip clean curriculum epochs | CBF SNR25 +4%, ATT SNR10 +4.5% | I1 |
| Two-stage DR curriculum | CBF SNR10 +5.9%, CoV 1.15→1.07 | K2 |
| TTA (4-flip) in eval | Smooth 0.77→0.52 | F4 |
| Post-processing blur (sigma=1.5) | Smooth 0.53→0.41 | M4 |
| 3-model ensemble | CoV 1.08→0.92, smooth 0.40→0.36 | M5 |

### What Never Worked

Wider/deeper models, dropout, mixup, more training data, learning rate changes, wider domain randomization, Rician noise, gradient accumulation, EMA, warm restarts.

### Key Discovery: CBF-Smoothness Pareto Frontier

There is a fundamental tradeoff between per-voxel CBF accuracy and spatial smoothness. Changes that improve in-vivo metrics (CoV, smoothness) tend to degrade synthetic CBF win rates, and vice versa. The model is near the Pareto frontier — you can slide along it but not easily push both forward.

## Project Structure

```
asl-multiverse/
├── ralph_harness.py                 # Self-contained train + eval harness
├── ralph_loop.sh                    # Automated iteration loop
├── ralph_plan.md                    # Task checklist + 48-iteration log
├── ralph_spec.md                    # Current best results, targets, constraints
├── ralph_prompt.md                  # Prompt for each Ralph iteration
├── ralph_analysis_prompt.md         # Prompt to analyze results in fresh session
├── main.py                          # Original training entry point
├── CLAUDE.md                        # AI assistant context
│
├── models/                          # Neural network architectures
│   ├── spatial_asl_network.py       # SpatialASLNet, DualEncoder, KineticModel
│   ├── amplitude_aware_spatial_network.py  # AmplitudeAwareSpatialASLNet
│   └── enhanced_asl_network.py      # DisentangledASLNet (voxel-wise, not recommended)
│
├── simulation/                      # Signal simulation and data generation
│   ├── enhanced_simulation.py       # SpatialPhantomGenerator
│   └── noise_engine.py              # NoiseInjector
│
├── baselines/                       # Least-squares fitting methods
│   └── multiverse_functions.py      # Combined PCASL+VSASL LS fitter
│
├── config/                          # YAML experiment configs
│   └── invivo_experiment.yaml       # Ralph loop config
│
├── invivo_results/                  # Results and logs (gitignored, snapshots committed)
│   ├── ralph_loop_log_run2_iters1-30.txt    # Full loop output
│   ├── results_snapshot_iter48_M5.json      # Best detailed results
│   ├── experiment_log_snapshot_iter48.md     # Experiment details
│   ├── latest_results.json                  # Most recent run (overwritten each iter)
│   └── trained_model.pt                     # Most recent model weights
│
├── data/                            # In-vivo data (READ-ONLY)
├── archive/                         # Old scripts, docs, shell scripts
├── amplitude_ablation_v1/           # 10 spatial experiments (completed)
├── amplitude_ablation_v2/           # 11 experiments (completed)
└── hpc_ablation_jobs/               # 10 voxel-wise experiments (completed)
```

## Quick Start — Ralph Harness

The Ralph harness is the primary way to train and evaluate models:

```bash
# Single training + evaluation run
python3 ralph_harness.py --device mps

# Run the automated optimization loop (spawns fresh Claude instances)
bash ralph_loop.sh 50

# Monitor loop progress
tail -f invivo_results/ralph_loop_log.txt
```

## References

1. Xu et al. (2025) - MULTIVERSE ASL: Joint PCASL/VSASL protocol
2. Alsop et al. (2015) - ASL Consensus: Standard implementation
3. Mao et al. (2023) - Bias-Reduced Neural Networks for ASL
4. Buxton et al. - General Kinetic Model: PCASL signal equation
5. Chen et al. (2024) - ANNCEST: Neural networks for MRI signal enhancement
6. Hales et al. (2020) - CNN denoising for ASL
7. Spann et al. (2017) - Spatio-temporal denoising for ASL

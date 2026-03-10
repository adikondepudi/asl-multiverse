# Ralph Spec — Problem Definition

## Current Best Results (updated by each successful iteration)

### Synthetic (per-voxel win rate, NN vs domain-randomized LS)
NOTE: Single model with 4-flip TTA. Ensemble adds ~5% to metrics.
| SNR | CBF Win % | ATT Win % | CBF Slope | CBF Bias |
|-----|-----------|-----------|-----------|----------|
| 3   | 62.6      | 80.1      | 0.69      | -3.2     |
| 10  | 71.2      | 58.3      | 0.81      | -0.7     |
| 25  | 75.5      | 74.8      | 0.90      | +2.3     |

### In-Vivo (3 subjects average)
NOTE: Single model with 4-flip TTA. Ensemble improves CoV ratio by ~0.06, smooth by ~0.03.
| Metric              | NN     | LS     | Ratio (NN/LS) |
|---------------------|--------|--------|---------------|
| GM CBF CoV (%)      | 50.6   | 46.2   | 1.10          |
| Spatial Smoothness  | 3.39   | 8.27   | 0.41          |
| GM CBF Mean         | 45.4   | —      | —             |
| GM/WM Ratio         | 1.10   | —      | —             |
| GM ATT Mean (ms)    | 1324   | —      | —             |

## Targets (hard — no fallback)
- **Synthetic CBF win rate > 90%** at ALL SNR levels (3, 10, 25)
- **Synthetic ATT win rate > 90%** at ALL SNR levels
- **In-vivo CoV ratio < 0.50** (NN CoV < 50% of LS CoV, i.e. NN CoV < ~23%)
- **In-vivo smoothness ratio < 0.50** (NN smooth < 50% of LS, i.e. < ~4.1)
- **Concordance**: synthetic and in-vivo both say NN wins
- **Physiological plausibility**: GM CBF 15-120, GM/WM 1.2-6.0, GM ATT 400-3000

## Hard Constraints (DO NOT violate)
1. **Read-only data**: Never modify `data/invivo_processed_npy/` or `invivo_comparison_results/`
2. **FiLM disabled**: `use_film_at_bottleneck: false`, `use_film_at_decoder: false` (breaks on in-vivo)
3. **5 PLDs**: [500, 1000, 1500, 2000, 2500] ms — matches in-vivo acquisition
4. **Per-pixel output modulation**: Keep `use_amplitude_output_modulation: true`
5. **Device**: Use `--device mps` (Apple Silicon)
6. **Signal scaling**: signals * 100.0 (M0), then * global_scale, clamp to [-30, 30]
7. **LS baseline uses**: T1_artery=1650, alpha_BS1=0.93, grid search init

## Modifiable Files
- `config/invivo_experiment.yaml` — hyperparams, physics, domain randomization
- `ralph_harness.py` — training loop, data generation, evaluation logic
- `models/amplitude_aware_spatial_network.py` — model architecture
- `models/spatial_asl_network.py` — loss function, base U-Net
- `simulation/enhanced_simulation.py` — phantom generation, tissue ranges
- `simulation/noise_engine.py` — noise models

## Key Insight
NN beats LS via **domain randomization robustness**. LS uses fixed consensus parameters.
NN trains with randomized physics (T1_artery, alpha_BS1, alpha_PCASL, alpha_VSASL).
At test time, each phantom gets random physics — LS mismatches, NN generalizes.
Wider domain randomization = bigger NN advantage.

## Anti-Patterns (never repeat)
- Enabling FiLM (global conditioning breaks on in-vivo tissue mix differences)
- Using `att_scale: 0.033` (legacy bug, must be 1.0)
- Using z-score normalization (destroys CBF amplitude information)
- Using `per_curve` normalization for CBF
- Training voxel-wise models (catastrophic CBF variance collapse)
- Narrow phantom CBF ranges that don't match in-vivo (causes super-linearity)
- Running with `--no-verify` or skipping git safety

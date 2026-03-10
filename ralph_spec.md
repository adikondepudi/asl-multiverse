# Ralph Spec — Problem Definition

## Current Best Results (updated by each successful iteration)

### Synthetic (per-voxel win rate, NN vs domain-randomized LS)
NOTE: 3-model ensemble with 4-flip TTA.
| SNR | CBF Win % | ATT Win % | CBF Slope | CBF Bias |
|-----|-----------|-----------|-----------|----------|
| 3   | 64.4      | 82.1      | 0.71      | -3.6     |
| 10  | 68.6      | 53.8      | 0.82      | -0.7     |
| 25  | 74.2      | 74.5      | 0.91      | +2.1     |

### In-Vivo (3 subjects average)
NOTE: 3-model ensemble with 4-flip TTA.
| Metric              | NN     | LS     | Ratio (NN/LS) |
|---------------------|--------|--------|---------------|
| GM CBF CoV (%)      | 42.7   | 46.2   | 0.92          |
| Spatial Smoothness  | 2.94   | 8.27   | 0.36          |
| GM CBF Mean         | 43.2   | —      | —             |
| GM/WM Ratio         | 1.19   | —      | —             |
| GM ATT Mean (ms)    | 1304   | —      | —             |

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

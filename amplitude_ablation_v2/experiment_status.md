# Amplitude Ablation V2: Experiment Status Report

Generated: 2026-02-07

## Overview

The v2 ablation study ran 11 experiments (Exp 10-20) on the HPC cluster using SLURM.
All jobs were submitted on 2026-02-06 with a **6-hour wall-clock time limit** (`#SBATCH --time=6:00:00`).
All jobs ran on Tesla T4 GPUs with 64 GB RAM and 8 CPUs.

Each SLURM job was designed to perform three stages sequentially:
1. **Training** (spatial model training with ensembles)
2. **Validation** (validate against least-squares baseline)
3. **Amplitude sensitivity test** (scale input by [0.1x, 1x, 10x] and measure CBF response)

---

## Experiment Status Summary

| Exp | Name | Status | Models | Validation | Amp Test | Failure Cause |
|-----|------|--------|--------|------------|----------|---------------|
| 10 | ExtendedDomainRand | COMPLETE | Yes (3) | Yes | Yes | - |
| 11 | MoreData_50k | COMPLETE | Yes (3) | Yes | Yes | - |
| 12 | MoreData_100k | COMPLETE | Yes (3) | Yes | Yes | - |
| 13 | AggressiveNoise | COMPLETE | Yes (3) | Yes | Yes | - |
| 14 | ATT_Rebalanced | COMPLETE | Yes (3) | Yes | Yes | - |
| 15 | HuberLoss | **FAILED** | **No** | No | No | Data loading hang + time limit |
| 16 | L2Loss | COMPLETE | Yes (3) | Yes | Yes | - |
| 17 | LargerModel | **FAILED** | **No** | No | No | Data loading hang + time limit |
| 18 | LongerTraining | **PARTIAL** | Yes (3) | **No** | **No** | Time limit (89/100 epochs) |
| 19 | Ensemble5 | COMPLETE | Yes (3) | Yes | Yes | - |
| 20 | BestCombo | **PARTIAL** | Yes (5) | **No** | **No** | Time limit (51/100 epochs) |

---

## Detailed Analysis of Failed/Partial Experiments

### Exp 15: HuberLoss -- FAILED (No Models)

**Hypothesis**: Huber loss is more robust to outliers than L1, may help ATT in difficult voxels

**Config differences from baseline (Exp 10)**:
- `loss_type: huber` (vs `l1`)
- Standard domain randomization ranges (narrower than Exp 10)
- Otherwise identical: 50 epochs, 20k samples, 3 ensembles, hidden_sizes [32,64,128,256]

**What happened**:
- Job 2692517 started on gpu108 at 16:08:55
- Config validation and normalization stats completed normally by 16:10:42
- Reached `[SPATIAL MODE] initializing lazy-loading SpatialDataset from asl_spatial_dataset_ablation_v2...`
- **No further output was produced** -- the process hung during dataset loading
- Job was killed at 22:09:14 due to the 6-hour time limit
- Error log shows: `slurmstepd: error: *** JOB 2692517 ON gpu108 CANCELLED AT 2026-02-06T22:09:14 DUE TO TIME LIMIT ***`

**Likely cause**: The dataset loading process hung silently. This is the same shared
dataset (`asl_spatial_dataset_ablation_v2`) used by all experiments. The hang occurred
after W&B initialization but before the "[SpatialDataset] Pre-loading" message that
normally appears next. This suggests an I/O or filesystem issue on the specific compute
node (gpu108), not a config-related error. The Huber loss config change itself would not
affect data loading at all.

**Directory contents**: config.yaml, norm_stats.json, research_config.json, run.slurm,
slurm output/error files. **No trained_models/ directory, no validation_results/, no
amplitude_sensitivity.json.**

---

### Exp 17: LargerModel -- FAILED (No Models)

**Hypothesis**: Doubling model capacity [64,128,256,512] may capture more complex spatial patterns

**Config differences from baseline**:
- `hidden_sizes: [64, 128, 256, 512]` (vs `[32, 64, 128, 256]` -- 2x capacity)
- Standard domain randomization ranges
- Otherwise identical: 50 epochs, 20k samples, 3 ensembles, loss_type l1

**What happened**:
- Job 2692519 started on gpu104 at 17:00:41
- Config validation and normalization stats completed normally by 17:02:50
- Reached `[SPATIAL MODE] initializing lazy-loading SpatialDataset from asl_spatial_dataset_ablation_v2...`
- **No further output was produced** -- the process hung during dataset loading
- Job was killed at 23:00:45 due to the 6-hour time limit
- Error log shows: `slurmstepd: error: *** JOB 2692519 ON gpu104 CANCELLED AT 2026-02-06T23:00:45 DUE TO TIME LIMIT ***`

**Likely cause**: Same failure mode as Exp 15 -- silent hang during dataset loading.
Different node (gpu104 vs gpu108) but same symptom. The larger model capacity would
not affect data loading. This points to a shared filesystem or resource contention
issue that affected multiple nodes. Both Exp 15 and 17 were likely competing with
other experiments for NFS/Lustre access to the same large dataset directory.

**Directory contents**: config.yaml, norm_stats.json, research_config.json, run.slurm,
slurm output/error files. **No trained_models/ directory, no validation_results/, no
amplitude_sensitivity.json.**

---

### Exp 18: LongerTraining -- PARTIAL (Models Exist, No Validation)

**Hypothesis**: 100 epochs (vs 50) with higher patience may find better minima, especially for ATT

**Config differences from baseline**:
- `n_epochs: 100` (vs 50 -- 2x training duration)
- `early_stopping_patience: 25` (vs 15)
- Otherwise identical: 20k samples, 3 ensembles, hidden_sizes [32,64,128,256], loss_type l1

**What happened**:
- Job 2692520 started on gpu120 at 17:02:11
- Dataset loaded successfully (100k samples loaded, subsetted to 20k)
- Training began at 17:24:04 with ~3.8 min/epoch
- Training progressed well, reaching epoch 89/100 at 22:59:00
- **Job killed at 23:02:15 due to 6-hour time limit** -- 11 epochs short of completion
- Best validation losses at time of kill: ensemble_0=0.003156, ensemble_1=0.003109, ensemble_2=0.003302
- The training was still finding improvements (new best models saved as late as epoch 89)

**Time budget analysis**:
- Data loading: ~22 min (17:02 to 17:24)
- Training 89 epochs: ~5h 35min (17:24 to 22:59)
- Estimated total for 100 epochs: ~6h 17min
- But the SLURM job also needed time for validation + amplitude test (~30min)
- **The 6-hour limit was insufficient for 100 epochs + validation + amp test**

**Models**: 3 ensemble models exist in `trained_models/` (8.2 MB each). These are
best-of-89-epochs models and are likely high quality -- loss had converged well.

**Needs**: Validation run and amplitude sensitivity test using the existing models.

---

### Exp 20: BestCombo -- PARTIAL (Models Exist, No Validation)

**Hypothesis**: Combine extended domain rand + 100k data + aggressive noise + ATT rebalancing + longer training + 5 ensembles

**Config (combination of best settings from other experiments)**:
- `n_epochs: 100` (from Exp 18)
- `n_ensembles: 5` (from Exp 19)
- `num_samples_to_load: 100000` (from Exp 12)
- `att_scale: 1.0, att_weight: 2.0` (from Exp 14)
- `early_stopping_patience: 25`
- `hidden_sizes: [32, 64, 128, 256]`
- Aggressive noise config: SNR range [1.0, 30.0], physio_amp up to 0.2, spike_prob 0.05,
  spike_magnitude up to 8.0, spatial_noise_sigma 1.2, motion simulation (from Exp 13)
- Extended domain randomization: T1_artery [1400, 2300], alpha_PCASL [0.65, 0.98],
  alpha_VSASL [0.3, 0.75], alpha_BS1 [0.7, 1.0], T_tau_perturb 0.15 (from Exp 10)
- Additional noise components: thermal, physio, drift, **spikes** (4 components vs 3)

**What happened**:
- Job 2692522 started on gpu119 at 17:28:13
- Dataset loaded successfully (100k samples, 19.7 GB RAM, took ~10 min)
- Training began at 17:40:42 with ~6.8 min/epoch (slower due to 5 ensembles + 100k data)
- Training progressed well, reaching epoch 51/100 at 23:25:23
- **Job killed at 23:28:14 due to 6-hour time limit** -- 49 epochs short of completion
- Best validation losses at epoch 51: ensemble_0=0.027158, ensemble_1=0.022058,
  ensemble_2=0.033682, ensemble_3=0.024066, ensemble_4=0.026483
- Loss was still decreasing steadily (no plateau reached)

**Time budget analysis**:
- Data loading: ~12 min
- Training 51 epochs: ~5h 45min
- Estimated total for 100 epochs: ~11h 20min
- **The 6-hour limit was grossly insufficient -- needed ~12 hours minimum**
- This is the most resource-intensive experiment: 5 ensembles x 100 epochs x 100k samples

**Models**: 5 ensemble models exist in `trained_models/` (8.2 MB each). These represent
best-of-51-epochs. Loss was still actively decreasing, so these models are likely
**undertrained** compared to what the full 100-epoch run would produce. However, they
may still be usable for preliminary validation.

**Needs**: Either (a) re-run with higher time limit (12+ hours), or (b) validate the
existing partially-trained models as-is, noting they are sub-optimal.

---

## Completed Experiments: Results Summary

Reference baseline (Exp 09 from v1): CBF MAE=0.49, ATT MAE=18.7, CBF Win=97.5%, ATT Win=96.8%

| Exp | Name | CBF MAE | CBF Win% | ATT MAE | ATT Win% | Amp Ratio | Amp Sensitive |
|-----|------|---------|----------|---------|----------|-----------|---------------|
| 10 | ExtendedDomainRand | 0.48 | 97.7% | 20.88 | 96.4% | 0.36 | No |
| 11 | MoreData_50k | 0.47 | 97.7% | 18.81 | 96.8% | 69.37 | Yes |
| 12 | MoreData_100k | 0.55 | 97.5% | 19.12 | 96.8% | 54.78 | Yes |
| 13 | AggressiveNoise | 0.47 | 97.7% | 18.53 | 96.8% | 82.20 | Yes |
| 14 | ATT_Rebalanced | 0.44 | 97.8% | 15.35 | 97.2% | 75.96 | Yes |
| 16 | L2Loss | 0.48 | 97.7% | 16.36 | 97.1% | 76.25 | Yes |
| 19 | Ensemble5 | 0.47 | 97.7% | 17.42 | 97.2% | 11.17 | Yes |

**Best performers**:
- **CBF**: Exp 14 (ATT_Rebalanced) -- lowest CBF MAE (0.44) and highest CBF win rate (97.8%)
- **ATT**: Exp 14 (ATT_Rebalanced) -- lowest ATT MAE (15.35) and highest ATT win rate (97.2%)
- **Amplitude sensitivity**: Exp 13 (AggressiveNoise) -- highest ratio (82.20)

**Notable finding**: Exp 10 (ExtendedDomainRand) lost amplitude sensitivity entirely
(ratio=0.36), despite using the same AmplitudeAwareSpatialASLNet architecture. This
suggests overly wide domain randomization ranges can interfere with the output
modulation mechanism.

---

## Action Items

### High Priority
1. **Exp 18 (LongerTraining)**: Run validation and amplitude sensitivity test on existing
   models. Training reached epoch 89/100 with well-converged loss -- models are likely
   production quality.

2. **Exp 20 (BestCombo)**: Run validation and amplitude sensitivity test on existing
   models. Note that training only reached epoch 51/100, so models may be undertrained.
   Consider re-submitting with `--time=14:00:00` to allow full training.

### Medium Priority
3. **Exp 15 (HuberLoss)**: Re-submit with same config. The failure was due to a data
   loading hang (likely I/O contention), not a config error. The 6-hour time limit is
   sufficient for 50 epochs with 20k samples.

4. **Exp 17 (LargerModel)**: Re-submit with same config. Same data loading hang as
   Exp 15. May need slightly more time due to larger model, but 6 hours should suffice
   for 50 epochs. Consider `--time=8:00:00` to be safe given the 2x model capacity.

### Low Priority
5. **Exp 20 (BestCombo)**: If re-running, increase SLURM time limit to at least 14 hours
   (`--time=14:00:00`). The 5-ensemble x 100-epoch x 100k-sample combination requires
   approximately 11-12 hours for training alone, plus ~30 min for validation and
   amplitude testing.

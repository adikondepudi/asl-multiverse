# Next Steps: Local Validation Execution Plan

All code changes from P0-P6 are committed. Now run validations locally to get corrected numbers.

## Available Locally
- All v1 models (10 experiments, 3 ensembles each)
- All v2 models except Exp 15/17 (9 experiments with models)
- Production v1 models (5 ensembles)
- FSL is reportedly installed but `oxford_asl` is NOT on PATH — need to find/source it
- No in-vivo data found in the repo directory

## Priority 1: Re-validate with Fixed LS Baseline

The LS baseline now has corrected alpha_BS1=0.93, T1_artery=1650, multi-start optimizer, tightened ATT bounds [100,4000], widened CBF bounds [0,200]. This is the most important thing to run — it will show whether the ~97% win rates hold up or drop.

### Run on Exp 14 (best v2 model — ATT_Rebalanced):
```bash
python validate_spatial.py amplitude_ablation_v2/14_ATT_Rebalanced
```
This now automatically runs multi-SNR validation at [3, 5, 10, 15, 25] with bootstrap CIs.

### Run on production_v1:
```bash
python validate_spatial.py production_model_v1
```

### Run on v1 baseline (Exp 00):
```bash
python validate_spatial.py amplitude_ablation_v1/00_Baseline_SpatialASL
```

## Priority 2: Validate Never-Validated Experiments

### Exp 18 (LongerTraining) — 89/100 epochs, models likely good:
```bash
python validate_spatial.py amplitude_ablation_v2/18_LongerTraining
```

### Exp 20 (BestCombo) — 51/100 epochs, possibly undertrained:
```bash
python validate_spatial.py amplitude_ablation_v2/20_BestCombo
```

## Priority 3: Run New Diagnostic Scripts

### Realistic amplitude sensitivity (replaces random-Gaussian test):
```bash
# On baseline (should show NOT sensitive)
python test_amplitude_sensitivity_realistic.py --run_dir amplitude_ablation_v1/00_Baseline_SpatialASL --output_dir results/amp_test/00_baseline

# On best amplitude-aware model
python test_amplitude_sensitivity_realistic.py --run_dir amplitude_ablation_v1/02_AmpAware_Full --output_dir results/amp_test/02_full

# On Exp 14
python test_amplitude_sensitivity_realistic.py --run_dir amplitude_ablation_v2/14_ATT_Rebalanced --output_dir results/amp_test/14_att
```

### Domain gap test:
```bash
python test_domain_gap.py --run_dir amplitude_ablation_v2/14_ATT_Rebalanced --output_dir results/domain_gap/14_att
python test_domain_gap.py --run_dir amplitude_ablation_v1/00_Baseline_SpatialASL --output_dir results/domain_gap/00_baseline
```

### Weight audit (quick, runs in seconds):
```bash
python audit_model_weights.py --scan_dir amplitude_ablation_v2/ --output results/v2_weight_audit.json
```

## Priority 4: Find and Configure FSL/BASIL

The user said FSL is installed locally but `oxford_asl` and `basil` aren't on PATH. Need to:
1. Find FSL: `mdfind -name "fsldir"` or check `~/.bash_profile`, `~/.zshrc` for FSLDIR
2. Source FSL: `source $FSLDIR/etc/fslconf/fsl.sh`
3. Then re-run in-vivo comparison with `--basil` flag

## Priority 5: Re-submit Failed HPC Jobs

Exp 15 (HuberLoss) and Exp 17 (LargerModel) failed due to I/O contention. Just resubmit:
```bash
cd amplitude_ablation_v2/15_HuberLoss && sbatch run.slurm
cd amplitude_ablation_v2/17_LargerModel && sbatch run.slurm
```

For Exp 20 (BestCombo), if re-running for full 100 epochs, edit `run.slurm` to change `--time=6:00:00` to `--time=14:00:00` first.

## What to Look For in Results

### Re-validation (Priority 1)
- **Win rates**: Were ~97% with broken LS. With fixed LS they should drop. If they stay >70%, the NN advantage is real. If they drop below 50%, the previous claims were artifacts.
- **CBF MAE**: Compare NN vs LS at each SNR level. The multi-SNR curve shows where NN advantage emerges.
- **Bootstrap CIs**: Check if win rate CIs exclude 50% — this is the statistical significance test.
- **Smoothed-LS**: This new baseline isolates spatial smoothing benefit from DL benefit.

### Amplitude sensitivity (Priority 3)
- **CBF linearity slope**: 1.0 = perfect, <0.1 = amplitude invariant
- **Scaling ratio**: 4.0 expected for 2x/0.5x, 1.0 = invariant

### Domain gap (Priority 3)
- **Degradation ratio**: <1.5x = robust, 1.5-2.0x = moderate, >2.0x = sensitive
- Compare models with vs without domain randomization

## Parallelization Strategy

These are independent and can run simultaneously:
- validate_spatial.py on Exp 14 (longest — multi-SNR with 50 phantoms per SNR)
- validate_spatial.py on Exp 18
- validate_spatial.py on Exp 20
- test_amplitude_sensitivity_realistic.py on Exp 00
- test_domain_gap.py on Exp 14
- audit_model_weights.py on v2/ (fastest, seconds)

Use background processes or separate terminals. CPU-only inference on spatial models takes ~5-30 min per validation depending on phantom count.

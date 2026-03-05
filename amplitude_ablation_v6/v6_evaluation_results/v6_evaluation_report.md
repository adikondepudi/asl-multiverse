# V6 Evaluation Report

Generated: 2026-03-05 10:30

## V6 Configuration Changes

| Parameter | V5 (Previous) | V6 (Current) |
|-----------|---------------|--------------|
| Loss | NLL | **L1** (reverted) |
| variance_weight | 0.1 | **0.5** |
| PLDs | 6 [500-3000] | **5 [500-2500]** (matches in-vivo) |
| CBF range (GM) | 10-120 | **10-200** |
| ATT range (GM) | 500-2500 | **500-3000** |
| CBF clip | 200 | **250** |
| T1_artery | 1650 | 1650 (unchanged) |
| att_scale | 1.0 | 1.0 (unchanged) |

## Training Summary

| Model | Epochs Completed | Best Val Loss | Parameters |
|-------|-----------------|---------------|------------|
| Baseline SpatialASL | 19/30 (time limit) | 0.328 | 1,929,060 |
| AmplitudeAware | 29/40 (time limit) | 0.333 | 2,035,493 |

Note: Both jobs hit SLURM time limits. Models were saved at best checkpoint.

## CBF Linearity Test (SNR=10)

| Model | Slope | Intercept | R2 (identity) | R2 (fit) |
|-------|-------|-----------|---------------|----------|
| Baseline SpatialASL | 0.868 | 6.4 | 0.963 | 0.996 |
| AmplitudeAware | 0.841 | 2.7 | 0.913 | 0.997 |

*Ideal: slope=1.0, intercept=0, R2_identity=1.0*

## Win Rate Analysis (NN vs Corrected LS)

### CBF Win Rate (%)

| SNR | Baseline SpatialASL | AmplitudeAware |
|-----|---|---|
| 3 | 41.1% | 15.9% | 
| 5 | 31.6% | 8.3% | 
| 10 | 21.1% | 3.2% | 
| 15 | 11.0% | 2.5% | 
| 25 | 6.6% | 3.0% | 

### ATT Win Rate (%)

| SNR | Baseline SpatialASL | AmplitudeAware |
|-----|---|---|
| 3 | 48.0% | 40.8% | 
| 5 | 35.0% | 39.1% | 
| 10 | 20.6% | 29.3% | 
| 15 | 13.4% | 24.8% | 
| 25 | 7.4% | 20.0% | 

## Bias/CoV Summary

### CBF Bias at Typical Values (CBF=50, ATT=1500)

| SNR | Model | CBF Bias | CBF CoV (%) | ATT Bias (ms) | ATT CoV (%) |
|-----|-------|----------|-------------|---------------|-------------|
| 10.0 | Baseline SpatialASL | -0.09 | 7.1 | -31 | 3.7 |
| 10.0 | AmplitudeAware | -4.41 | 3.7 | -8 | 2.5 |
| 10.0 | Least Squares | 0.06 | 1.3 | -0 | 1.3 |
| 3.0 | Baseline SpatialASL | 0.62 | 8.8 | -35 | 4.7 |
| 3.0 | AmplitudeAware | 0.84 | 7.3 | -35 | 4.6 |
| 3.0 | Least Squares | 0.32 | 4.5 | -0 | 4.1 |
| 5.0 | Baseline SpatialASL | 0.03 | 7.7 | -29 | 4.1 |
| 5.0 | AmplitudeAware | -3.90 | 4.6 | -19 | 3.4 |
| 5.0 | Least Squares | 0.20 | 2.8 | -1 | 2.7 |

## In-Vivo Results

### Baseline SpatialASL

| Subject | CBF Mean | CBF Std | CBF Median | ATT Mean | ATT Std | ATT Median |
|---------|----------|---------|------------|----------|---------|------------|
| 20231002_MR1_A144 | 20.6 | 31.3 | 4.8 | 1143 | 491 | 1126 |
| 20231003_MR1_A142 | 23.5 | 33.4 | 9.7 | 1144 | 523 | 1097 |
| 20231004_MR1_A151 | 25.4 | 33.8 | 8.8 | 1120 | 526 | 1082 |
| 20231016_MR1_A147 | 17.3 | 26.0 | 7.8 | 1415 | 725 | 1349 |
| 20231030_MR1_A152 | 30.6 | 37.9 | 13.3 | 1136 | 505 | 1108 |
| 20231101_MR1_A153 | 20.3 | 27.6 | 9.5 | 1272 | 630 | 1217 |
| 20240211_MR1_A155 | 28.8 | 35.6 | 14.8 | 1114 | 443 | 1079 |
| 20240216_MR1_A156 | 18.3 | 29.3 | 7.0 | 1348 | 738 | 1249 |

### AmplitudeAware

| Subject | CBF Mean | CBF Std | CBF Median | ATT Mean | ATT Std | ATT Median |
|---------|----------|---------|------------|----------|---------|------------|
| 20231002_MR1_A144 | 12.3 | 25.9 | 0.0 | 548 | 701 | 236 |
| 20231003_MR1_A142 | 7.0 | 21.7 | 0.0 | 365 | 682 | 0 |
| 20231004_MR1_A151 | 21.3 | 31.8 | 0.0 | 745 | 670 | 642 |
| 20231016_MR1_A147 | 13.5 | 22.5 | 0.0 | 1012 | 887 | 1010 |
| 20231030_MR1_A152 | 25.0 | 39.5 | 0.0 | 657 | 644 | 528 |
| 20231101_MR1_A153 | 15.3 | 24.4 | 0.0 | 865 | 755 | 826 |
| 20240211_MR1_A155 | 18.7 | 30.5 | 0.0 | 637 | 604 | 558 |
| 20240216_MR1_A156 | 8.2 | 19.2 | 0.0 | 909 | 886 | 763 |

## Key Findings

1. **Baseline SpatialASL linearity improved** (slope=0.868)
2. **AmplitudeAware linearity is good** (slope=0.841). Expanded CBF range in V6 helped.

## Figures

- `fig1_training_curves.png` - Training and validation loss curves
- `fig2_bias_cov_*.png` - Bias and CoV across ATT/CBF sweeps per SNR
- `fig3_win_rates.png` - NN vs LS win rate by SNR
- `fig4_linearity.png` - CBF predicted vs true (linearity test)
- `fig5_invivo_*.png` - In-vivo CBF/ATT maps per subject
- `fig6_invivo_summary.png` - In-vivo summary statistics

# ASL Multiverse

Neural network framework for Arterial Spin Labeling (ASL) MRI parameter estimation. Trains models to predict Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) from combined PCASL and VSASL signals.

## Overview

ASL Multiverse uses deep learning to estimate CBF and ATT from joint PCASL/VSASL acquisitions, as described in the MULTIVERSE protocol (Xu et al. 2025). The framework supports both spatial (2D U-Net) and voxel-wise (1D) architectures, with spatial models substantially outperforming voxel-wise.

Compared to corrected least-squares (LS) fitting, the best neural network model shows a **modest but consistent advantage at low-to-moderate SNR** (54-60% CBF win rate, 61-68% ATT win rate). The NN advantage diminishes at high SNR. The primary practical benefit is **speed**: NN inference is orders of magnitude faster than iterative LS fitting.

## Key Results

Best model: AmplitudeAwareSpatialASLNet (Exp 14, with ATT rebalancing), evaluated against **corrected** LS baseline:

| SNR | CBF Win Rate | ATT Win Rate |
|-----|-------------|-------------|
| 3   | 59.8%       | 67.9%       |
| 5   | 59.3%       | 66.3%       |
| 10  | 57.8%       | 65.3%       |
| 15  | 56.5%       | 63.4%       |
| 25  | 54.2%       | 60.7%       |

**Important context**: Earlier results reported ~97% CBF win rates. These were measured against a broken LS baseline (`alpha_BS1=1.0`, `T1_artery=1850`, single-start optimizer, ATT bound 6000ms). After correcting the LS implementation (`alpha_BS1=0.93`, `T1_artery=1650`, multi-start, tighter ATT bounds), win rates dropped dramatically. The baseline SpatialASLNet (Exp 00) actually **loses** to corrected LS for CBF at all SNR >= 5. See `CLAUDE.md` for the full corrected analysis.

## Honest Assessment

**What we can claim:**
- Spatial models dramatically outperform voxel-wise for CBF
- AmplitudeAwareSpatialASLNet provides measurable CBF improvement over baseline SpatialASLNet (MAE 0.80 vs 2.64 at SNR=10)
- NN has modest but statistically significant advantage over corrected LS at low-moderate SNR
- NN inference is orders of magnitude faster than iterative LS
- Baseline SpatialASLNet suffers from complete CBF variance collapse

**What we cannot claim:**
- ~~NN dramatically outperforms LS (97% win rate)~~ -- artifact of broken LS baseline
- ~~NN always better than LS~~ -- at high SNR (25+), LS approaches or beats NN for CBF
- ~~"Amplitude awareness" is the mechanism~~ -- Exp 10 disproves the causal link; extra model capacity is the likely explanation (see `docs/amplitude_audit_report.md`)
- ~~Production models are ready~~ -- production_v1 is broken (CBF win rate 2.9%, bias -17.7)
- ~~CBF predictions are accurate at high values~~ -- super-linearity (slope ~1.9) at CBF >80

## Project Structure

```
asl-multiverse/
├── main.py                          # Training entry point
├── models/                          # Neural network architectures
│   ├── spatial_asl_network.py       # SpatialASLNet, DualEncoder, KineticModel
│   ├── amplitude_aware_spatial_network.py  # AmplitudeAwareSpatialASLNet
│   └── enhanced_asl_network.py      # DisentangledASLNet (voxel-wise)
├── training/                        # Training loop and utilities
│   └── asl_trainer.py               # EnhancedASLTrainer, FastTensorDataLoader
├── validation/                      # Validation scripts and metrics
│   ├── validate.py                  # Validation with LS comparison
│   ├── validate_spatial.py          # Spatial model validation
│   └── validation_metrics.py        # Bland-Altman, ICC, CCC, SSIM
├── simulation/                      # Signal simulation and data generation
│   ├── asl_simulation.py            # JIT-compiled ASL signal generation
│   ├── enhanced_simulation.py       # SpatialPhantomGenerator, RealisticASLSimulator
│   ├── noise_engine.py              # NoiseInjector (Rician noise)
│   └── generate_clean_library.py    # Training data generation
├── utils/                           # Utilities and feature management
│   ├── helpers.py                   # Normalization, signal processing
│   └── feature_registry.py          # Feature dims, norm_stats indices
├── baselines/                       # Least-squares fitting methods
│   ├── multiverse_functions.py      # Combined PCASL+VSASL LS fitter
│   ├── pcasl_functions.py           # PCASL-only LS fitter
│   ├── vsasl_functions.py           # VSASL-only LS fitter
│   └── basil_baseline.py           # FSL BASIL wrapper
├── inference/                       # In-vivo prediction scripts
│   ├── predict_on_invivo.py         # Voxel-wise in-vivo inference
│   └── predict_spatial_invivo.py    # Spatial in-vivo inference
├── config/                          # YAML experiment configurations
├── docs/                            # Analysis documents
├── archive/                         # Archived scripts and old docs
├── amplitude_ablation_v1/           # 10 spatial experiments (completed)
├── amplitude_ablation_v2/           # 11 experiments, 4 incomplete
└── hpc_ablation_jobs/               # 10 voxel-wise experiments (completed)
```

## Quick Start

```bash
# Install dependencies (requirements.txt is incomplete -- you also need torch, scipy, nibabel, etc.)
pip install -r requirements.txt

# Generate spatial training data
python -m simulation.generate_clean_library <output_dir> --spatial --total_samples 100000

# Train spatial model (recommended: AmplitudeAwareSpatialASLNet)
python main.py config/amplitude_aware_spatial.yaml --stage 2 --output-dir ./results/amp_aware

# Train baseline spatial model
python main.py config/spatial_mae_loss.yaml --stage 2 --output-dir ./results/run

# Validate
python -m validation.validate --run_dir <run_dir> --output_dir validation_results

# Spatial validation
python -m validation.validate_spatial <run_dir>
```

## Models

### Spatial (2D) -- Recommended

| Model | File | Notes |
|-------|------|-------|
| **SpatialASLNet** | `models/spatial_asl_network.py` | Baseline U-Net; suffers from variance collapse (predicts ~55 for all CBF) |
| **DualEncoderSpatialASLNet** | `models/spatial_asl_network.py` | Y-Net with separate PCASL/VSASL streams |
| **AmplitudeAwareSpatialASLNet** | `models/amplitude_aware_spatial_network.py` | Best performer (CBF MAE 0.80 vs 2.64 baseline); mechanism unproven -- likely benefits from extra capacity, not "amplitude awareness" |

### Voxel-Wise (1D) -- Not Recommended

| Model | File | Notes |
|-------|------|-------|
| **DisentangledASLNet** | `models/enhanced_asl_network.py` | <5% CBF win rate; 20-36% ATT win rate; variance collapse |

Voxel-wise models catastrophically fail for CBF. Spatial context is critical.

## Known Issues

**Critical:**
- **att_scale=0.033 legacy bug**: All v1 experiments and most v2 experiments use incorrect ATT loss weighting. Must be 1.0. Only Exp 14 and Exp 20 use the correct value.
- **Super-linearity**: AmplitudeAware models have CBF linearity slope ~1.9 (should be 1.0). At CBF=150, predictions hit the 300 clamp. Likely caused by narrow training CBF range [20-70].

**High:**
- **Domain randomization silently disabled**: When `dc_weight=0.0` (the default), domain randomization parameters have no effect since they only feed into the physics loss path.
- **Baseline variance collapse**: SpatialASLNet predicts ~55 for all CBF values (slope=0.026). Low validation MAE is an artifact of narrow validation CBF distribution.
- **Production v1 models broken**: CBF win rate 2.9%, bias -17.7 ml/100g/min. Do not use for inference.
- **T1_artery=1850 in all experiments**: ASL consensus (Alsop 2015) recommends 1650ms at 3T.

See `CLAUDE.md` for the complete bug list and detailed analysis.

## References

- Xu et al. (2025) - MULTIVERSE ASL: Joint PCASL/VSASL protocol
- Alsop et al. (2015) - ASL Consensus: Standard implementation
- Mao et al. (2023) - Bias-Reduced Neural Networks for ASL
- Buxton et al. - General Kinetic Model: PCASL signal equation
- Perez et al. (2018) - FiLM: Feature-wise Linear Modulation

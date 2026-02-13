# ASL Multiverse

Neural network framework for Arterial Spin Labeling (ASL) MRI parameter estimation. Trains models to predict Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) from combined PCASL and VSASL signals.

## Project Structure

```
asl-multiverse/
├── main.py                    # Training entry point
├── models/                    # Neural network architectures
├── training/                  # Training loop (EnhancedASLTrainer)
├── validation/                # Validation scripts and metrics
├── simulation/                # ASL signal simulation and data generation
├── utils/                     # Helpers and feature registry
├── baselines/                 # Least-squares fitting methods
├── inference/                 # In-vivo prediction scripts
├── config/                    # YAML experiment configurations
├── docs/                      # Analysis documents
└── archive/                   # Archived scripts and old docs
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate spatial training data
python -m simulation.generate_clean_library <output_dir> --spatial --total_samples 100000

# Train model
python main.py config/spatial_mae_loss.yaml --stage 2 --output-dir ./results/run

# Validate
python -m validation.validate --run_dir <run_dir> --output_dir validation_results
```

## Key Results

Best model (AmplitudeAwareSpatialASLNet, Exp 14) achieves:
- CBF win rate vs corrected LS: 54-60% (SNR-dependent)
- ATT win rate vs corrected LS: 61-68%
- CBF MAE: 0.80 ml/100g/min at SNR=10

See `CLAUDE.md` for detailed findings and `docs/` for analysis reports.

## References

- Xu et al. (2025) - MULTIVERSE ASL: Joint PCASL/VSASL protocol
- Alsop et al. (2015) - ASL Consensus: Standard implementation

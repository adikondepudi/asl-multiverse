# ASL Analysis and Processing Framework

A comprehensive framework for processing and analyzing Arterial Spin Labeling (ASL) MRI data, supporting multiple ASL techniques including VSASL, PCASL, and MULTIVERSE approaches.

## Features

- Multiple ASL processing methods support:
  - Velocity-Selective ASL (VSASL)
  - Pseudo-Continuous ASL (PCASL)
  - MULTIVERSE (combined PCASL+VSASL)
- Advanced neural network model for parameter estimation
- Comprehensive evaluation and validation tools
- Hyperparameter optimization utilities
- Automated quality control metrics
- Cross-validation support

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── asl_pipeline.py         # Main ASL processing pipeline
├── asl_trainer.py          # Training utilities and model management
├── cli.py                  # Command-line interface
├── config/
│   └── default.yaml       # Default configuration settings
├── enhanced_asl_network.py # Enhanced neural network implementation
├── evaluation/            # Evaluation and validation utilities
├── utils/                # Utility functions
└── README.md
```

## Quick Start

1. Basic usage with CLI:
```bash
# Train a model with default configuration
python cli.py train --config config/default.yaml

# Evaluate a trained model
python cli.py evaluate --model-path path/to/model --data-path path/to/data
```

2. Using the Python API:
```python
from asl_simulation import ASLSimulator, ASLParameters
from asl_trainer import EnhancedASLTrainer
from evaluation import EnhancedEvaluator

# Initialize simulator and trainer
simulator = ASLSimulator(ASLParameters())
trainer = EnhancedASLTrainer(input_size=12)  # 6 PLDs * 2 (PCASL + VSASL)

# Train the model
train_loader, val_loader = trainer.prepare_curriculum_data(simulator)
trainer.train_ensemble(train_loader, val_loader)
```

## Configuration

The framework can be configured using YAML files. Key configuration parameters include:

```yaml
training:
  batch_size: 256
  learning_rate: 0.001
  hidden_sizes: [256, 128, 64]
  n_epochs: 200
  n_ensembles: 5
  dropout_rate: 0.1
  norm_type: "batch"

data:
  n_samples: 20000
  val_split: 0.2
  pld_range: [500, 3000, 500]
```

## Main Components

### ASL Pipeline

The `ASLDataProcessor` class in `asl_pipeline.py` provides a complete pipeline for ASL data processing:

```python
from asl_pipeline import ASLDataProcessor

processor = ASLDataProcessor()
processor.load_nifti("path/to/data.nii")
processor.motion_correction()
maps = processor.compute_perfusion_map(method='multiverse')
```

### Training

The framework supports curriculum learning and ensemble training:

```python
from asl_trainer import EnhancedASLTrainer

trainer = EnhancedASLTrainer(
    input_size=12,
    hidden_sizes=[256, 128, 64],
    n_ensembles=5
)

# Train with curriculum learning
train_loaders, val_loader = trainer.prepare_curriculum_data(simulator)
trainer.train_ensemble(train_loaders, val_loader)
```

### Evaluation

Comprehensive evaluation tools are available:

```python
from evaluation import EnhancedEvaluator

evaluator = EnhancedEvaluator()
results = evaluator.evaluate_model(trainer, test_signals, test_params)
```

## API Reference

### ASLSimulator

```python
simulator = ASLSimulator(params=ASLParameters())
signals = simulator.generate_synthetic_data(
    plds=np.arange(500, 3001, 500),
    att_values=np.array([800, 1600, 2400]),
    n_noise=50
)
```

### EnhancedASLNet

```python
from enhanced_asl_network import EnhancedASLNet

model = EnhancedASLNet(
    input_size=12,
    hidden_sizes=[256, 128, 64],
    n_plds=6,
    dropout_rate=0.1,
    norm_type='batch'
)
```

## Testing

Run the test suite:
```bash
python -m unittest test_all.py
```

For specific test categories:
```bash
python test_all.py test
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[License Information]

## Citation

If you use this framework in your research, please cite:

[Citation Information]

## Support

For questions and support, please [create an issue](link-to-issues) in the repository.
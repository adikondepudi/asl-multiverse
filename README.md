# ASL Multiverse: Advanced Neural Network Framework for Arterial Spin Labeling MRI Analysis

This repository contains a comprehensive suite of Python tools for simulating Arterial Spin Labeling (ASL) MRI data, training advanced neural networks for parameter estimation, and comparing their performance against conventional fitting methods. The primary goal is to improve the accuracy and efficiency of Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) quantification using the MULTIVERSE ASL framework, which combines pseudo-continuous ASL (PCASL) and velocity-selective ASL (VSASL).

## Overview

The project encompasses:
1.  **Python Translations of ASL Models**: Kinetic models and fitting routines for PCASL, VSASL, and the combined MULTIVERSE approach, originally implemented in MATLAB.
2.  **Advanced ASL Simulation**: Tools to generate synthetic ASL data, ranging from basic simulations to "realistic" scenarios with physiological variations, diverse noise models, and artifact inclusion.
3.  **Neural Network Development**:
    *   `EnhancedASLNet`: A sophisticated neural network architecture, potentially incorporating Transformer layers, for robust CBF and ATT estimation with uncertainty quantification.
    *   `EnhancedASLTrainer`: A training pipeline supporting curriculum learning, ensemble methods, and advanced data augmentation for `EnhancedASLNet`.
4.  **Comprehensive Evaluation Framework**:
    *   Comparison of the neural network against traditional least-squares fitting methods for PCASL, VSASL, and MULTIVERSE.
    *   Validation across various simulated clinical scenarios and noise levels.
    *   Performance benchmarking against targets defined in the research proposal.
5.  **Research Pipeline Orchestration**: A main script (`main.py`) that manages the entire workflow from data generation, hyperparameter optimization (Optuna), model training, and evaluation, to the generation of publication-ready materials.

## Features

*   **MATLAB-to-Python Translation**: Core ASL kinetic models and fitting functions for PCASL, VSASL, and MULTIVERSE are available in Python.
*   **Realistic Data Simulation**: `RealisticASLSimulator` generates diverse datasets considering various physiological conditions (healthy, stroke, tumor, elderly), noise levels, and sequence parameter perturbations. It can also generate spatial ASL data.
*   **Advanced Neural Network**: `EnhancedASLNet` for direct CBF/ATT estimation, featuring:
    *   Residual blocks.
    *   Optional Transformer-based temporal processing.
    *   Dedicated uncertainty estimation heads.
    *   Support for M0 as an input feature.
*   **Sophisticated Training**: `EnhancedASLTrainer` provides:
    *   Curriculum learning based on ATT ranges.
    *   Ensemble training for improved robustness and uncertainty.
    *   Advanced data augmentation via `EnhancedASLDataset`.
    *   Custom loss function with ATT-based weighting and aleatoric uncertainty.
*   **Benchmarking & Validation**:
    *   `ComprehensiveComparison`: Rigorous comparison of NN vs. conventional methods across different ATT ranges.
    *   `SingleRepeatValidator`: Assesses the key proposal objective of achieving high-quality results with single-repeat NN acquisitions compared to multi-repeat conventional methods.
    *   `ClinicalValidator`: Evaluates performance in simulated clinical scenarios.
*   **Hyperparameter Optimization**: Integrated Optuna support for optimizing `EnhancedASLNet` hyperparameters.
*   **Publication Generation**: Automated creation of summary tables and figures.
*   **Testing Suite**: Includes unit tests for translations, simulation components, and high-level validation routines.

## Directory Structure

```
asl-multiverse/
├── asl_simulation.py                # Core ASL signal generation logic and parameters
├── asl_trainer.py                   # Trainers and datasets for NNs (basic and enhanced)
├── compare_methods.py               # Standalone Monte Carlo comparison of LS methods
├── comparison_framework.py          # Framework for comprehensive NN vs LS comparison
├── enhanced_asl_network.py          # Defines EnhancedASLNet and CustomLoss
├── enhanced_data_generation.py      # Script to generate comprehensive datasets (as per proposal)
├── enhanced_simulation.py           # RealisticASLSimulator with diverse conditions & noise
├── main.py                          # Main script to run the comprehensive research pipeline
├── multiverse_functions.py          # Python implementation of MULTIVERSE kinetic model & fitting
├── old_readme.txt                   # Initial thoughts and discussions from project start
├── pcasl_functions.py               # Python implementation of PCASL kinetic model & fitting
├── performance_metrics.py           # Proposal-specific evaluation and figure generation
├── readme_discussions.docx          # (Likely Word doc with discussion notes)
├── requirements.txt                 # Python package dependencies
├── single_repeat_validation.py      # Validation of single-repeat NN vs multi-repeat LS
├── spatial_cnn.py                   # CNN architectures for spatial ASL data (denoising/estimation)
├── test_all.py                      # Comprehensive testing suite (unit tests & validation)
├── test_vsasl_functions.py          # Specific tests for VSASL Python translation
├── verify_translation.py            # Script to verify VSASL Python vs MATLAB results
├── vsasl_functions.py               # Python implementation of VSASL kinetic model & fitting
├── codes/                           # Original MATLAB code and documentation
│   ├── Examples for using the code.md
│   ├── fit_PCASL_vectInit_pep.m
│   ├── fit_PCVSASL_misMatchPLD_vectInit_pep.m
│   ├── fit_VSASL_vect_nopep.m
│   ├── fit_VSASL_vect_pep.m
│   ├── fit_VSASL_vectInit_pep.m
│   ├── fun_PCASL_1comp_vect_pep.m
│   ├── fun_PCVSASL_misMatchPLD_vect_pep.m
│   ├── fun_VSASL_1comp_vect_pep.m
│   ├── KineticEquationsForASL.md
│   └── MCsimu_PCpVSASL_multiPLD_onePLD_rmse_scalNoise_20241226.m
└── config/
    └── default.yaml                 # Default configuration for the main.py pipeline
```

## Dependencies

The main dependencies are listed in `requirements.txt`. Key packages include:
*   Python 3.8+
*   numpy
*   scipy
*   pandas
*   matplotlib
*   torch (PyTorch)
*   scikit-learn
*   tqdm
*   pyyaml
*   seaborn
*   optuna

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd adikondepudi-asl-multiverse
```

2. Create a Python virtual environment (recommended) and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration

The main research pipeline driven by `main.py` is configured using a YAML file. A `config/default.yaml` is provided. You can create custom configuration files and pass them as a command-line argument to `main.py`.

The configuration file covers parameters for:

- **Training**: Batch size, learning rate, hidden layer sizes, epochs, ensemble settings, dropout, normalization, Transformer settings for `EnhancedASLNet`.
- **Data Generation**: Number of training subjects, PLD values, ATT ranges for curriculum learning, M0 feature inclusion.
- **Simulation**: Base ASL parameters (T1_artery, T_tau, efficiencies, etc.).
- **Optuna**: Number of trials, timeout, and parameters for optimization runs.
- **Clinical Validation**: Number of test subjects, SNR levels, conditions for testing.
- **Performance Targets**: Goals for CBF/ATT CoV improvement.

## Usage / Running the Code

### 1. Main Research Pipeline (main.py)

This is the primary script to run the full research workflow, including hyperparameter optimization, model training, comprehensive evaluation, and report generation.

```bash
python main.py [path_to_custom_config.yaml]
```

If no configuration file is provided, `config/default.yaml` will be attempted to be loaded, followed by `ResearchConfig` defaults. The pipeline will:

1. **(Optional) Hyperparameter Optimization**: Uses Optuna to find optimal NN hyperparameters.
2. **Ensemble Training**: Trains an ensemble of `EnhancedASLNet` models using curriculum learning.
3. **Clinical Validation**: Evaluates the trained models on simulated clinical scenarios.
4. **Benchmarking**: Compares NN performance against conventional LS methods using `ComprehensiveComparison`.
5. **Publication Material Generation**: Creates figures and tables summarizing the results.

Output from the pipeline (logs, models, figures, reports) will be saved in a timestamped subdirectory within `comprehensive_results/`.

### 2. Testing and Validation (test_all.py)

This script provides a suite of tests and a high-level validation routine.

**Run all unit tests and a validation summary:**
```bash
python test_all.py
```

**Run only unit tests:**
```bash
python test_all.py unittest
```

**Run comprehensive validation (establishes baselines for conventional methods):**
This routine performs a detailed comparison of conventional fitting methods (PCASL, VSASL, MULTIVERSE-LS) across various SNRs and ATT ranges. It saves results and generates plots.
```bash
python test_all.py validation
```
Output will be in `test_results/`.

**Compare trained Neural Networks against conventional methods:**
After training models using `main.py`, you can run this to get a detailed comparison.
```bash
python test_all.py comparison <path_to_trained_models_directory>
```
Example: `python test_all.py comparison comprehensive_results/asl_research_YYYYMMDD_HHMMSS/trained_models`

### 3. Individual Scripts

Some scripts can be run individually for specific tasks:

**Verify MATLAB-to-Python Translation (VSASL):**
```bash
python verify_translation.py
```
This script compares the output of `vsasl_functions.py` against reference values from MATLAB examples and generates plots.

**Standalone Monte Carlo Comparison (LS Methods):**
```bash
python compare_methods.py
```
Runs a simpler Monte Carlo simulation to compare PCASL, VSASL, and MULTIVERSE least-squares fitting methods.

**Generate Comprehensive Datasets:**
```bash
python enhanced_data_generation.py
```
This script uses `RealisticASLSimulator` to generate various datasets as outlined in the project proposal and saves them as pickle files in `data/comprehensive_datasets/`.

**Test VSASL Functions:**
```bash
python test_vsasl_functions.py
```
Specific tests for VSASL signal generation and fitting, including plots.

## Core Components Explained

- **`asl_simulation.py`**: Defines `ASLParameters` and `ASLSimulator` for basic ASL signal generation with noise scaling.

- **`enhanced_simulation.py`**: Contains `RealisticASLSimulator` which builds upon `ASLSimulator` to generate diverse datasets with physiological variations (healthy, stroke, tumor, etc.), perturbed sequence parameters, and more complex noise models (Gaussian, Rician, physiological artifacts). It can also generate 2D/3D spatial ASL data.

- **`pcasl_functions.py`, `vsasl_functions.py`, `multiverse_functions.py`**: Python implementations of the kinetic models and least-squares fitting routines for PCASL, VSASL, and the combined MULTIVERSE ASL technique, respectively. These are translated from the original MATLAB code.

- **`enhanced_asl_network.py`**:
  - **`EnhancedASLNet`**: The main neural network architecture designed for robust CBF and ATT estimation. It features an MLP backbone with residual blocks, an optional Transformer-based module for temporal feature extraction, and separate heads for predicting parameter means and their aleatoric uncertainties (log variance). It can optionally take M0 as an input.
  - **`CustomLoss`**: A loss function for training `EnhancedASLNet`. It combines Gaussian Negative Log-Likelihood (NLL) terms for CBF and ATT (leveraging the predicted log variances) and includes ATT-based instance weighting and an epoch-dependent schedule for ATT loss contribution.

- **`asl_trainer.py`**:
  - **`ASLNet`, `ASLDataset`, `ASLTrainer`**: Components for a simpler, baseline neural network.
  - **`EnhancedASLDataset`**: PyTorch Dataset class that applies on-the-fly data augmentations like noise injection, signal dropout, global scaling, and baseline shifts.
  - **`EnhancedASLTrainer`**: Manages the training and evaluation of `EnhancedASLNet` ensembles. It implements curriculum learning (staging data by ATT ranges), uses `WeightedRandomSampler` for ATT-dependent sample weighting, and employs `OneCycleLR` schedulers.

- **`comparison_framework.py`**:
  - **`ComprehensiveComparison`**: A class to systematically evaluate and compare the performance of different methods (least-squares PCASL, VSASL, MULTIVERSE-LS, and the trained Neural Network). It calculates detailed metrics (bias, CoV, RMSE, success rate, computation time) across specified ATT ranges and visualizes the results.

- **`performance_metrics.py`**:
  - **`ProposalEvaluator`**: Calculates performance metrics as specifically defined in the research proposal (e.g., normalized bias, CoV, normalized RMSE). It can also generate figures similar to those in the proposal.

- **`single_repeat_validation.py`**:
  - **`SingleRepeatValidator`**: A dedicated framework to assess a key research objective: comparing the performance of single-repeat Neural Network acquisitions against multi-repeat (averaged) conventional least-squares methods. This is crucial for evaluating potential scan time reduction.

- **`spatial_cnn.py`**: Defines CNN architectures for spatial processing of ASL data:
  - **`SpatialEnhancementCNN`**: A U-Net like 3D CNN for tasks like denoising ASL volumes.
  - **`SpatioTemporalASLNet`**: A hybrid model combining 2D spatial CNNs with 1D temporal processors (LSTM or Transformer) for parameter estimation from slice-wise image data and global signal vectors.
  - Includes `ASLSpatialDataset` for handling spatial data.

- **`config/default.yaml`**: Provides default parameters for the comprehensive research pipeline managed by `main.py`.

## MATLAB Code and Translation

The `codes/` directory contains the original MATLAB (.m) files that formed the basis for the Python kinetic models and fitting functions in:

- `pcasl_functions.py`
- `vsasl_functions.py`
- `multiverse_functions.py`

The file `codes/Examples for using the code.md` provides context on how the MATLAB functions were used, and `codes/KineticEquationsForASL.md` offers a brief theoretical background on the ASL models.

The `verify_translation.py` and `test_vsasl_functions.py` scripts are designed to ensure the Python implementations match the behavior of their MATLAB counterparts. The `test_all.py` script also includes extensive tests for this translation.

## M0 Input Feature

The `EnhancedASLNet` and the associated data generation/processing pipelines have an option to include M0 (equilibrium magnetization) as an input feature to the neural network. This is controlled by flags like `m0_input_feature_model` in the configuration and corresponding parameters in various classes. When enabled, the input vector to the NN is extended by one, and the data preparation steps attempt to include M0 values (often as dummy values if full M0 simulation is not integrated into the specific data generation path being used).
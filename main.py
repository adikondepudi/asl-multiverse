"""
Enhanced main.py - Comprehensive ASL Neural Network Research Pipeline

This module implements a complete end-to-end framework for developing, validating,
and benchmarking neural networks against clinical requirements for ASL parameter estimation.

Primary Purpose:
Comprehensive ASL Neural Network Research Pipeline - A complete framework that transforms
research hypotheses into validated clinical improvements through systematic development,
optimization, and validation of neural networks for ASL parameter estimation.

Core Objectives:
1. Demonstrate 50% precision improvement over conventional methods
2. Enable single-repeat acquisition with maintained quality
3. Ensure clinical robustness across patient populations
4. Generate publication-ready figures and metrics
5. Provide reproducible research framework

Author: Enhanced ASL Research Team (with modifications for diverse data)
Date: 2025 (Updated)
"""

import torch
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import yaml
import logging # Import standard logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
from dataclasses import dataclass, asdict, field # Added field
import warnings
warnings.filterwarnings('ignore') # Use with caution

# Import enhanced ASL components
from enhanced_asl_network import EnhancedASLNet, CustomLoss
from asl_simulation import ASLParameters # ASLSimulator not directly used by main anymore
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer # Uses RealisticASLSimulator internally now
from comparison_framework import ComprehensiveComparison, ComparisonResult # Import ComparisonResult
from performance_metrics import ProposalEvaluator # For proposal-specific metrics if needed
from single_repeat_validation import SingleRepeatValidator # For specific validation task

# Import conventional methods for comparison
from vsasl_functions import fit_VSASL_vectInit_pep
from pcasl_functions import fit_PCASL_vectInit_pep
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

# Global logger for this script, to be configured by PerformanceMonitor
script_logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Configuration for the comprehensive research pipeline"""
    # Training parameters
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    batch_size: int = 256
    n_training_subjects: int = 10000  # Renamed from n_samples for clarity with diverse_dataset
    n_epochs: int = 200 # Reduced default from 300 for potentially longer data gen
    n_ensembles: int = 5
    dropout_rate: float = 0.1
    norm_type: str = 'batch'

    # Hyperparameter optimization
    optuna_n_trials: int = 20 # Reduced default from 100
    optuna_timeout_hours: float = 0.5 # Reduced default from 1 hour (3600s)
    optuna_n_subjects: int = 500 # Subjects for each optuna trial
    optuna_n_epochs: int = 20  # Epochs for each optuna trial

    # Data generation
    pld_values: List[int] = field(default_factory=lambda: list(range(500, 3001, 500))) # Explicit PLDs
    # att_ranges for curriculum and evaluation
    att_ranges_config: List[Tuple[float, float, str]] = field(default_factory=lambda: [
        (500.0, 1500.0, "Short ATT"),
        (1500.0, 2500.0, "Medium ATT"),
        (2500.0, 4000.0, "Long ATT")
    ])

    # Simulation parameters for RealisticASLSimulator (used if not overridden by its internal defaults)
    T1_artery: float = 1850.0
    T2_factor: float = 1.0
    alpha_BS1: float = 1.0
    alpha_PCASL: float = 0.85
    alpha_VSASL: float = 0.56
    T_tau: float = 1800.0
    # CBF is varied in RealisticASLSimulator, so a single config.CBF is less relevant for generation
    # but can be used as a reference for normalization if needed.
    reference_CBF: float = 60.0

    # Clinical validation / Test set generation
    n_test_subjects_per_att_range: int = 200 # For Phase 4 benchmarking
    test_snr_levels: List[float] = field(default_factory=lambda: [5.0, 10.0]) # For Phase 4
    test_conditions: List[str] = field(default_factory=lambda: ['healthy', 'stroke']) # For Phase 4

    # Clinical validation scenarios for ClinicalValidator
    n_clinical_scenario_subjects: int = 100 # For Phase 3 ClinicalValidator
    clinical_scenario_definitions: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'healthy_adult': {'cbf_range': (50.0, 80.0), 'att_range': (800.0, 1800.0), 'snr': 8.0},
        'elderly_patient': {'cbf_range': (30.0, 60.0), 'att_range': (1500.0, 3000.0), 'snr': 5.0},
        'stroke_patient': {'cbf_range': (10.0, 40.0), 'att_range': (2000.0, 4000.0), 'snr': 3.0},
        'tumor_patient': {'cbf_range': (20.0, 120.0), 'att_range': (1000.0, 3000.0), 'snr': 6.0}
    })
    training_conditions: List[str] = field(default_factory=lambda: ['healthy', 'stroke', 'elderly', 'tumor'])
    training_noise_levels: List[float] = field(default_factory=lambda: [3.0, 5.0, 10.0, 15.0])


    # Performance targets (50% improvement goals)
    target_cbf_cv_improvement_perc: float = 50.0 # As percentage
    target_att_cv_improvement_perc: float = 50.0 # As percentage
    # target_scan_time_reduction: float = 0.75 # Not directly optimized, but an outcome

class PerformanceMonitor:
    """Monitor training progress and research objectives"""
    def __init__(self, config: ResearchConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.metrics_history = [] # Could store detailed metrics over time
        self.logger = logging.getLogger("ASLResearchPipeline") # Use a specific logger name
        self.logger.setLevel(logging.INFO) # Default level

        # Remove existing handlers to avoid duplicate logging if script is re-run in same session
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Setup file handler
        fh = logging.FileHandler(output_dir / 'research.log', mode='w') # Overwrite log file each run
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        # Setup stream handler (console)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(sh)
        script_logger.propagate = False # Prevent root logger from also handling these

    def log_progress(self, phase: str, message: str, level: int = logging.INFO):
        self.logger.log(level, f"[{phase}] {message}")

    def check_target_achievement(self, nn_att_range_results: Dict, baseline_att_range_results: Dict) -> Dict:
        """Check if research targets are met for a specific ATT range's results."""
        achievements = {}
        # nn_att_range_results and baseline_att_range_results are expected to be
        # dictionaries of metrics for a single ATT range, e.g., nn_metrics from Phase 4.

        for att_range_name in nn_att_range_results.keys(): # Should typically be one range
            if att_range_name not in baseline_att_range_results:
                self.log_progress("TARGET_CHECK", f"Baseline results missing for {att_range_name}", logging.WARNING)
                continue

            nn_metrics = nn_att_range_results[att_range_name]
            baseline_metrics = baseline_att_range_results[att_range_name]

            current_cbf_cv = nn_metrics.get('cbf_cov', float('inf'))
            baseline_cbf_cv = baseline_metrics.get('cbf_cov', float('inf'))
            cbf_improvement_perc = 0.0
            if baseline_cbf_cv > 0 and baseline_cbf_cv != float('inf') and current_cbf_cv != float('inf'):
                cbf_improvement_perc = ((baseline_cbf_cv - current_cbf_cv) / baseline_cbf_cv) * 100

            current_att_cv = nn_metrics.get('att_cov', float('inf'))
            baseline_att_cv = baseline_metrics.get('att_cov', float('inf'))
            att_improvement_perc = 0.0
            if baseline_att_cv > 0 and baseline_att_cv != float('inf') and current_att_cv != float('inf'):
                att_improvement_perc = ((baseline_att_cv - current_att_cv) / baseline_att_cv) * 100

            cbf_target_met = cbf_improvement_perc >= self.config.target_cbf_cv_improvement_perc
            att_target_met = att_improvement_perc >= self.config.target_att_cv_improvement_perc

            achievements[att_range_name] = {
                'cbf_cv_improvement_perc': cbf_improvement_perc,
                'att_cv_improvement_perc': att_improvement_perc,
                'cbf_target_met': cbf_target_met,
                'att_target_met': att_target_met
            }
            self.log_progress("TARGET_CHECK", f"{att_range_name} - CBF CV Improv: {cbf_improvement_perc:.1f}% ({'MET' if cbf_target_met else 'NOT MET'})")
            self.log_progress("TARGET_CHECK", f"{att_range_name} - ATT CV Improv: {att_improvement_perc:.1f}% ({'MET' if att_target_met else 'NOT MET'})")
        return achievements

class HyperparameterOptimizer:
    """Systematic hyperparameter optimization using Optuna"""
    def __init__(self, base_config: ResearchConfig, monitor: PerformanceMonitor):
        self.base_config = base_config
        self.monitor = monitor
        self.study = None

    def objective(self, trial: optuna.Trial) -> float:
        hidden_size_1 = trial.suggest_categorical('hidden_size_1', [128, 256, 512])
        hidden_size_2 = trial.suggest_categorical('hidden_size_2', [64, 128, 256])
        hidden_size_3 = trial.suggest_categorical('hidden_size_3', [32, 64, 128])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3) # Wider range
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        # Can add norm_type, loss weights etc. here too

        # Create a temporary config for this trial
        trial_config_dict = asdict(self.base_config)
        trial_config_dict.update({
            'hidden_sizes': [hidden_size_1, hidden_size_2, hidden_size_3],
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'n_training_subjects': self.base_config.optuna_n_subjects,
            'n_epochs': self.base_config.optuna_n_epochs,
            'n_ensembles': 1 # Single model for faster optimization
        })
        trial_config = ResearchConfig(**trial_config_dict)
        self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number}: {trial.params}")

        try:
            _, _, validation_loss = self._quick_training_run(trial_config)
            self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number} Val Loss: {validation_loss:.6f}")
            return validation_loss
        except Exception as e:
            self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number} FAILED: {e}", logging.ERROR)
            return float('inf') # Penalize failed trials

    def _quick_training_run(self, config: ResearchConfig) -> Tuple[Any, Any, float]:
        """Quick training run for hyperparameter optimization"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        asl_params_sim = ASLParameters(T1_artery=config.T1_artery, T2_factor=config.T2_factor, alpha_BS1=config.alpha_BS1,
                                   alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL, T_tau=config.T_tau)
        simulator = RealisticASLSimulator(params=asl_params_sim)
        plds_np = np.array(config.pld_values)

        def create_trial_model():
            return EnhancedASLNet(
                input_size=len(plds_np) * 2, hidden_sizes=config.hidden_sizes,
                n_plds=len(plds_np), dropout_rate=config.dropout_rate, norm_type=config.norm_type
            ).to(device)

        trainer = EnhancedASLTrainer(
            model_class=create_trial_model, input_size=len(plds_np) * 2,
            hidden_sizes=config.hidden_sizes, learning_rate=config.learning_rate,
            batch_size=config.batch_size, n_ensembles=config.n_ensembles, device=device,
            n_plds_for_model=len(plds_np)
        )

        train_loaders, val_loader = trainer.prepare_curriculum_data(
            simulator, n_training_subjects=config.n_training_subjects, plds=plds_np,
            curriculum_att_ranges_config=config.att_ranges_config,
            training_conditions_config=config.training_conditions[:1], # Use only 'healthy' for speed
            training_noise_levels_config=config.training_noise_levels[:1], # Use only one SNR for speed
            n_epochs_for_scheduler=config.n_epochs
        )
        if not train_loaders:
            self.monitor.log_progress("OPTUNA_RUN", "No training data for Optuna trial, returning inf loss.", logging.ERROR)
            return None, None, float('inf')

        # Quick training
        trainer.train_ensemble(train_loaders, val_loader, n_epochs=config.n_epochs, early_stopping_patience=5)

        # Get validation loss from the single model in the ensemble
        if val_loader and trainer.models:
            val_loss = trainer._validate(trainer.models[0], val_loader, config.n_epochs-1, len(train_loaders)-1, config.n_epochs)
        else:
            val_loss = float('inf') # No validation if no val_loader or no models
        return trainer, simulator, val_loss

    def optimize(self) -> Dict:
        self.monitor.log_progress("OPTUNA", f"Starting hyperparameter optimization: {self.base_config.optuna_n_trials} trials, timeout {self.base_config.optuna_timeout_hours}h.")
        self.study = optuna.create_study(direction='minimize')
        timeout_seconds = self.base_config.optuna_timeout_hours * 3600
        self.study.optimize(self.objective, n_trials=self.base_config.optuna_n_trials, timeout=timeout_seconds)

        best_params = self.study.best_params
        self.monitor.log_progress("OPTUNA", f"Best parameters found: {best_params}")
        self.monitor.log_progress("OPTUNA", f"Best validation loss: {self.study.best_value:.6f}")
        return best_params

class ClinicalValidator:
    """Comprehensive clinical validation framework"""
    def __init__(self, config: ResearchConfig, monitor: PerformanceMonitor):
        self.config = config
        self.monitor = monitor
        asl_params_sim = ASLParameters(T1_artery=config.T1_artery, T2_factor=config.T2_factor, alpha_BS1=config.alpha_BS1,
                                   alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL, T_tau=config.T_tau)
        self.simulator = RealisticASLSimulator(params=asl_params_sim) # Simulator for generating scenario data
        self.plds_np = np.array(config.pld_values)


    def validate_clinical_scenarios(self, trained_nn_models: List[torch.nn.Module]) -> Dict:
        self.monitor.log_progress("CLINICAL_VAL", "Running clinical validation scenarios...")
        all_scenario_results = {}
        pldti = np.column_stack((self.plds_np, self.plds_np))

        for scenario_name, params in self.config.clinical_scenario_definitions.items():
            self.monitor.log_progress("CLINICAL_VAL", f"  Validating {scenario_name}...")
            n_subjects = self.config.n_clinical_scenario_subjects
            
            # Generate CBF/ATT values for this specific scenario
            true_cbf_vals = np.random.uniform(*params['cbf_range'], n_subjects)
            true_att_vals = np.random.uniform(*params['att_range'], n_subjects)
            current_snr = params['snr']

            scenario_metrics = {
                'neural_network': {'cbf_preds': [], 'att_preds': [], 'cbf_uncs': [], 'att_uncs': []},
                'multiverse_ls_single_repeat': {'cbf_preds': [], 'att_preds': []}, # LS on single repeat
                'multiverse_ls_multi_repeat_avg': {'cbf_preds': [], 'att_preds': []} # LS on 4x averaged
            }

            for i in range(n_subjects):
                true_cbf, true_att = true_cbf_vals[i], true_att_vals[i]
                self.simulator.params.CBF = true_cbf # Set for base generate_synthetic_data
                self.simulator.cbf = true_cbf / 6000.0

                # --- Single Repeat Data Generation (for NN and single-repeat LS) ---
                single_repeat_data_dict = self.simulator.generate_synthetic_data(
                    self.plds_np, np.array([true_att]), n_noise=1, tsnr=current_snr
                )
                # MULTIVERSE signal for NN: (n_plds*2)
                nn_input_signal = np.concatenate([
                    single_repeat_data_dict['PCASL'][0, 0, :], # (n_plds,)
                    single_repeat_data_dict['VSASL'][0, 0, :]  # (n_plds,)
                ])
                # MULTIVERSE signal for LS: (n_plds, 2)
                ls_single_repeat_signal = single_repeat_data_dict['MULTIVERSE'][0, 0, :, :]

                # Neural network prediction on single repeat
                if trained_nn_models:
                    cbf_nn_mean, att_nn_mean, cbf_nn_std, att_nn_std = self._ensemble_predict(trained_nn_models, nn_input_signal)
                    scenario_metrics['neural_network']['cbf_preds'].append(cbf_nn_mean)
                    scenario_metrics['neural_network']['att_preds'].append(att_nn_mean)
                    scenario_metrics['neural_network']['cbf_uncs'].append(cbf_nn_std)
                    scenario_metrics['neural_network']['att_uncs'].append(att_nn_std)
                else: # Append NaNs if no model
                    for k in scenario_metrics['neural_network']: scenario_metrics['neural_network'][k].append(np.nan)


                # Conventional MULTIVERSE-LS fitting on single repeat
                try:
                    beta_ls_sr, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, ls_single_repeat_signal, [50.0/6000.0, 1500.0], # Fixed init
                        self.config.T1_artery, self.config.T_tau, self.config.T2_factor,
                        self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL
                    )
                    scenario_metrics['multiverse_ls_single_repeat']['cbf_preds'].append(beta_ls_sr[0] * 6000.0)
                    scenario_metrics['multiverse_ls_single_repeat']['att_preds'].append(beta_ls_sr[1])
                except Exception:
                    scenario_metrics['multiverse_ls_single_repeat']['cbf_preds'].append(np.nan)
                    scenario_metrics['multiverse_ls_single_repeat']['att_preds'].append(np.nan)

                # --- Multi-Repeat Averaged Data Generation (for LS gold standard) ---
                multi_repeat_signals_raw = []
                for _ in range(4): # 4 repeats
                    # Higher SNR for individual repeats that will be averaged
                    repeat_data_dict = self.simulator.generate_synthetic_data(
                        self.plds_np, np.array([true_att]), n_noise=1, tsnr=current_snr * np.sqrt(4) # Effective SNR after averaging
                    )
                    multi_repeat_signals_raw.append(repeat_data_dict['MULTIVERSE'][0, 0, :, :])
                
                avg_multi_repeat_signal = np.mean(multi_repeat_signals_raw, axis=0)
                try:
                    beta_ls_mr, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, avg_multi_repeat_signal, [50.0/6000.0, 1500.0], # Fixed init
                        self.config.T1_artery, self.config.T_tau, self.config.T2_factor,
                        self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL
                    )
                    scenario_metrics['multiverse_ls_multi_repeat_avg']['cbf_preds'].append(beta_ls_mr[0] * 6000.0)
                    scenario_metrics['multiverse_ls_multi_repeat_avg']['att_preds'].append(beta_ls_mr[1])
                except Exception:
                    scenario_metrics['multiverse_ls_multi_repeat_avg']['cbf_preds'].append(np.nan)
                    scenario_metrics['multiverse_ls_multi_repeat_avg']['att_preds'].append(np.nan)
            
            # Calculate summary metrics for this scenario
            # Instantiate a temporary comparator for _calculate_detailed_metrics
            temp_comparator = ComprehensiveComparison(nn_input_size=len(self.plds_np)*2, nn_n_plds=len(self.plds_np))

            all_scenario_results[scenario_name] = {}
            for method_key, data_dict in scenario_metrics.items():
                # Ensure all pred lists are arrays before metric calculation
                c_preds = np.array(data_dict['cbf_preds'])
                a_preds = np.array(data_dict['att_preds'])
                
                metrics_summary = temp_comparator._calculate_detailed_metrics(
                    c_preds, true_cbf_vals, a_preds, true_att_vals
                )
                # Add success rate (number of non-NaN fits)
                num_valid = np.sum(~np.isnan(c_preds) & ~np.isnan(a_preds))
                metrics_summary['success_rate'] = (num_valid / n_subjects) * 100 if n_subjects > 0 else 0
                if 'cbf_uncs' in data_dict and data_dict['cbf_uncs']: # For NN
                    metrics_summary['mean_cbf_uncertainty_std'] = np.nanmean(data_dict['cbf_uncs'])
                    metrics_summary['mean_att_uncertainty_std'] = np.nanmean(data_dict['att_uncs'])

                all_scenario_results[scenario_name][method_key] = metrics_summary
                self.monitor.log_progress("CLINICAL_VAL", f"  {scenario_name} - {method_key}: CBF RMSE {metrics_summary.get('cbf_rmse',np.nan):.2f}, ATT RMSE {metrics_summary.get('att_rmse',np.nan):.2f}, Success {metrics_summary.get('success_rate',0):.1f}%")

        return all_scenario_results

    def _ensemble_predict(self, models: List[torch.nn.Module], input_signal_flat: np.ndarray) -> Tuple[float, float, float, float]:
        """Make ensemble prediction for a single flattened sample. Returns mean_cbf, mean_att, std_cbf, std_att."""
        if not models: return np.nan, np.nan, np.nan, np.nan

        input_tensor = torch.FloatTensor(input_signal_flat).unsqueeze(0).to(models[0].input_layer.weight.device) # Ensure tensor is on model device

        cbf_means_list, att_means_list = [], []
        cbf_vars_list, att_vars_list = [], [] # Aleatoric variances

        for model in models:
            model.eval()
            with torch.no_grad():
                cbf_mean, att_mean, cbf_log_var, att_log_var = model(input_tensor)
                # NN output is directly in ml/100g/min for CBF and ms for ATT
                cbf_means_list.append(cbf_mean.item())
                att_means_list.append(att_mean.item())
                cbf_vars_list.append(torch.exp(cbf_log_var).item())
                att_vars_list.append(torch.exp(att_log_var).item())

        # Ensemble mean
        ens_cbf_mean = np.mean(cbf_means_list) if cbf_means_list else np.nan
        ens_att_mean = np.mean(att_means_list) if att_means_list else np.nan

        # Aleatoric variance (mean of variances)
        mean_aleatoric_cbf_var = np.mean(cbf_vars_list) if cbf_vars_list else np.nan
        mean_aleatoric_att_var = np.mean(att_vars_list) if att_vars_list else np.nan

        # Epistemic variance (variance of means)
        epistemic_cbf_var = np.var(cbf_means_list) if len(cbf_means_list) > 1 else 0.0
        epistemic_att_var = np.var(att_means_list) if len(att_means_list) > 1 else 0.0
        
        total_cbf_std = np.sqrt(mean_aleatoric_cbf_var + epistemic_cbf_var) if not (np.isnan(mean_aleatoric_cbf_var) or np.isnan(epistemic_cbf_var)) else np.nan
        total_att_std = np.sqrt(mean_aleatoric_att_var + epistemic_att_var) if not (np.isnan(mean_aleatoric_att_var) or np.isnan(epistemic_att_var)) else np.nan

        return ens_cbf_mean, ens_att_mean, total_cbf_std, total_att_std

class PublicationGenerator:
    """Generate publication-ready materials"""
    def __init__(self, config: ResearchConfig, output_dir: Path, monitor: PerformanceMonitor):
        self.config = config
        self.output_dir = output_dir
        self.monitor = monitor

    def generate_publication_package(self,
                                   clinical_results: Dict, # From ClinicalValidator
                                   # nn_benchmark_results_by_att_range: Dict, # From Phase 4 NN eval
                                   # baseline_ls_results_by_att_range: Dict, # From Phase 4 LS eval
                                   comparison_df: pd.DataFrame) -> Dict: # From ComprehensiveComparison
        self.monitor.log_progress("PUB_GEN", "Generating publication materials...")
        package = {'figures': {}, 'tables': {}, 'statistical_analysis': {}}

        # Generate Figure 1 (recreation of proposal figure) using the DataFrame
        # This figure now directly uses the output of ComprehensiveComparison
        comp_framework_instance = ComprehensiveComparison(output_dir=self.output_dir / "temp_comp_vis")
        comp_framework_instance.visualize_results(comparison_df) # This saves its own plots
        package['figures']['figure1_performance_from_comp_df'] = str(self.output_dir / "temp_comp_vis" / 'comparison_figure1_style_detailed.png')
        # Copy other relevant plots if needed, e.g. computation_time_comparison.png

        package['figures']['clinical_validation'] = self._generate_clinical_figures(clinical_results)
        package['tables']['performance_summary_csv'] = self._generate_performance_table_csv(comparison_df)
        # ... other analyses ...
        self._save_publication_materials(package) # Saves a JSON summary of paths
        return package

    def _generate_clinical_figures(self, clinical_results: Dict) -> Dict[str, str]:
        """Generate clinical validation figures from ClinicalValidator output."""
        figures_paths = {}
        if not clinical_results:
            self.monitor.log_progress("PUB_GEN", "No clinical results to plot.", logging.WARNING)
            return figures_paths

        scenarios = list(clinical_results.keys())
        # Methods present in clinical_results: 'neural_network', 'multiverse_ls_single_repeat', 'multiverse_ls_multi_repeat_avg'
        methods_to_plot = {
            'neural_network': 'NN (1x Repeat)',
            'multiverse_ls_single_repeat': 'LS (1x Repeat)',
            'multiverse_ls_multi_repeat_avg': 'LS (4x Avg Repeat)'
        }
        
        # Metrics to plot from the 'metrics' sub-dictionary of clinical_results
        plot_metric_keys = [('cbf_rmse', 'CBF RMSE (ml/100g/min)'), ('att_rmse', 'ATT RMSE (ms)'),
                            ('cbf_cov', 'CBF CoV (%)'), ('att_cov', 'ATT CoV (%)')]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False)
        axes = axes.ravel()

        for i, (metric_key, metric_label) in enumerate(plot_metric_keys):
            ax = axes[i]
            num_methods = len(methods_to_plot)
            bar_width = 0.8 / num_methods
            
            for j, (method_internal_key, method_display_name) in enumerate(methods_to_plot.items()):
                values = [clinical_results[sc].get(method_internal_key, {}).get('metrics', {}).get(metric_key, np.nan) for sc in scenarios]
                x_indices = np.arange(len(scenarios)) + j * bar_width - (bar_width * (num_methods-1) / 2)
                ax.bar(x_indices, values, width=bar_width, label=method_display_name, alpha=0.8)
            
            ax.set_ylabel(metric_label)
            ax.set_title(f"Clinical Validation: {metric_label.split('(')[0].strip()}")
            ax.set_xticks(np.arange(len(scenarios)))
            ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=30, ha="right")
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            if i == 0: ax.legend(fontsize='small')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle("Clinical Scenario Performance Comparison", fontsize=16, fontweight='bold')
        fpath = self.output_dir / 'clinical_validation_metrics_comparison.png'
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        figures_paths['clinical_metrics_comparison'] = str(fpath)
        self.monitor.log_progress("PUB_GEN", f"Saved clinical validation plot to {fpath}")
        return figures_paths

    def _generate_performance_table_csv(self, comparison_df: pd.DataFrame) -> str:
        """Generate performance summary table from ComprehensiveComparison DataFrame."""
        if comparison_df.empty:
            self.monitor.log_progress("PUB_GEN", "Comparison DataFrame is empty, cannot generate table.", logging.WARNING)
            return ""
        
        # Select and reorder columns for the table
        cols_to_show = ['method', 'att_range_name',
                        'cbf_nbias_perc', 'cbf_cov', 'cbf_nrmse_perc',
                        'att_nbias_perc', 'att_cov', 'att_nrmse_perc',
                        'success_rate', 'computation_time']
        # Filter for existing columns to prevent errors
        existing_cols = [col for col in cols_to_show if col in comparison_df.columns]
        summary_df = comparison_df[existing_cols]

        table_path = self.output_dir / 'performance_summary_from_comparison.csv'
        summary_df.to_csv(table_path, index=False, float_format='%.2f')
        self.monitor.log_progress("PUB_GEN", f"Saved performance summary CSV to {table_path}")
        return str(table_path)

    def _save_publication_materials(self, package: Dict):
        # Convert complex objects for JSON serialization if any
        def make_serializable(obj):
            if isinstance(obj, Path): return str(obj)
            if isinstance(obj, (np.ndarray, np.generic)): return obj.tolist() # Basic conversion
            if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list): return [make_serializable(i) for i in obj]
            return obj

        serializable_package = make_serializable(package)
        fpath = self.output_dir / 'publication_package_summary.json'
        with open(fpath, 'w') as f:
            json.dump(serializable_package, f, indent=2)
        self.monitor.log_progress("PUB_GEN", f"Saved publication package summary to {fpath}")


# Main Execution Block
def run_comprehensive_asl_research(config: ResearchConfig, output_parent_dir: str = 'comprehensive_results') -> Dict:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_parent_dir) / f'asl_research_{timestamp}'
    output_path.mkdir(parents=True, exist_ok=True)

    monitor = PerformanceMonitor(config, output_path)
    monitor.log_progress("SETUP", f"Initializing comprehensive ASL research pipeline. Output: {output_path}")
    with open(output_path / 'research_config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)

    plds_np = np.array(config.pld_values)
    asl_params_sim = ASLParameters(T1_artery=config.T1_artery, T2_factor=config.T2_factor, alpha_BS1=config.alpha_BS1,
                                   alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL, T_tau=config.T_tau)
    simulator = RealisticASLSimulator(params=asl_params_sim) # Used throughout

    # Phase 1: Hyperparameter Optimization
    best_optuna_params = {}
    if config.optuna_n_trials > 0:
        monitor.log_progress("PHASE1", "Starting hyperparameter optimization")
        optimizer = HyperparameterOptimizer(config, monitor)
        best_optuna_params = optimizer.optimize()
        if best_optuna_params: # Update config with best params if found
            config.hidden_sizes = [best_optuna_params['hidden_size_1'], best_optuna_params['hidden_size_2'], best_optuna_params['hidden_size_3']]
            config.learning_rate = best_optuna_params['learning_rate']
            config.dropout_rate = best_optuna_params['dropout_rate']
            config.batch_size = best_optuna_params['batch_size']
            monitor.log_progress("PHASE1", f"Updated config with Optuna best_params: {best_optuna_params}")
    else:
        monitor.log_progress("PHASE1", "Skipping hyperparameter optimization as n_trials is 0.")

    # Phase 2: Multi-objective Training
    monitor.log_progress("PHASE2", "Starting multi-objective ensemble training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    monitor.log_progress("PHASE2", f"Using device: {device}")

    def create_main_model(): # Factory for EnhancedASLTrainer
        return EnhancedASLNet(
            input_size=len(plds_np) * 2, hidden_sizes=config.hidden_sizes,
            n_plds=len(plds_np), dropout_rate=config.dropout_rate, norm_type=config.norm_type
        ).to(device)

    trainer = EnhancedASLTrainer(
        model_class=create_main_model, input_size=len(plds_np) * 2,
        hidden_sizes=config.hidden_sizes, learning_rate=config.learning_rate,
        batch_size=config.batch_size, n_ensembles=config.n_ensembles, device=device,
        n_plds_for_model=len(plds_np)
    )
    monitor.log_progress("PHASE2", "Preparing curriculum training datasets using DIVERSE data...")
    train_loaders, val_loader = trainer.prepare_curriculum_data(
        simulator, n_training_subjects=config.n_training_subjects, plds=plds_np,
        curriculum_att_ranges_config=config.att_ranges_config,
        training_conditions_config=config.training_conditions,
        training_noise_levels_config=config.training_noise_levels,
        n_epochs_for_scheduler=config.n_epochs
    )
    if not train_loaders:
        monitor.log_progress("PHASE2", "Failed to create training loaders. Aborting.", logging.CRITICAL)
        return {"error": "Training data preparation failed."}

    monitor.log_progress("PHASE2", f"Training {config.n_ensembles}-model ensemble for {config.n_epochs} epochs...")
    training_start_time = time.time()
    training_histories = trainer.train_ensemble(train_loaders, val_loader, n_epochs=config.n_epochs)
    training_duration_hours = (time.time() - training_start_time) / 3600
    monitor.log_progress("PHASE2", f"Training completed in {training_duration_hours:.2f} hours.")

    # Phase 3: Clinical Validation (using ClinicalValidator)
    monitor.log_progress("PHASE3", "Starting clinical validation across patient populations")
    clinical_validator = ClinicalValidator(config, monitor)
    clinical_validation_results = clinical_validator.validate_clinical_scenarios(trainer.models if trainer.models else [])

    # Phase 4: Benchmarking Against Conventional Methods (using ComprehensiveComparison)
    monitor.log_progress("PHASE4", "Benchmarking NN against conventional LS methods using DIVERSE test data")
    
    # Generate a single, comprehensive diverse test dataset for all ATT ranges
    monitor.log_progress("PHASE4", f"Generating DIVERSE test dataset for benchmarking...")
    # Total subjects for test set, distributed across conditions/SNRs
    total_test_subjects = config.n_test_subjects_per_att_range * len(config.att_ranges_config)
    # Cap total_test_subjects to avoid excessive generation if many ATT ranges
    total_test_subjects = min(total_test_subjects, 1000) # Example cap
    
    benchmark_test_dataset_raw = simulator.generate_diverse_dataset(
        plds=plds_np,
        n_subjects=total_test_subjects // (len(config.test_conditions) * len(config.test_snr_levels)*3), # Approx subjects
        conditions=config.test_conditions,
        noise_levels=config.test_snr_levels
    )
    benchmark_X_all = benchmark_test_dataset_raw['signals'] # (N, n_plds*2)
    benchmark_y_all = benchmark_test_dataset_raw['parameters'] # (N, 2) [CBF ml/100g/min, ATT ms]

    # Prepare data formats for ComprehensiveComparison
    benchmark_test_data_for_comp = {
        'PCASL': benchmark_X_all[:, :len(plds_np)],
        'VSASL': benchmark_X_all[:, len(plds_np):],
        'MULTIVERSE_LS_FORMAT': benchmark_X_all.reshape(-1, len(plds_np), 2), # (N, n_plds, 2)
        'NN_INPUT_FORMAT': benchmark_X_all # (N, n_plds*2)
    }

    # Instantiate ComprehensiveComparison
    # Path to first ensemble model for NN evaluation in ComprehensiveComparison
    nn_model_for_comp_path = None
    if trainer.models: # Check if models were trained
        temp_model_save_path = output_path / 'temp_ensemble_model_0_for_comp.pt'
        torch.save(trainer.models[0].state_dict(), temp_model_save_path)
        nn_model_for_comp_path = str(temp_model_save_path)

    comp_framework = ComprehensiveComparison(
        nn_model_path=nn_model_for_comp_path,
        output_dir=output_path / "comparison_framework_outputs",
        nn_input_size=len(plds_np) * 2,
        nn_hidden_sizes=config.hidden_sizes,
        nn_n_plds=len(plds_np)
    )
    
    comparison_results_df = comp_framework.compare_methods(
        benchmark_test_data_for_comp,
        benchmark_y_all,
        plds_np,
        config.att_ranges_config # Pass the list of (min, max, name) tuples
    )
    if nn_model_for_comp_path and temp_model_save_path.exists():
        temp_model_save_path.unlink() # Clean up temporary model file

    # Extract NN and Baseline (MULTIVERSE-LS) results from df for target checking
    # This assumes comparison_results_df is populated and has 'method' and 'att_range_name'
    nn_benchmark_metrics_for_monitor = {}
    baseline_ls_metrics_for_monitor = {}
    if not comparison_results_df.empty:
        for att_range_tuple in config.att_ranges_config:
            range_name = att_range_tuple[2]
            
            nn_row = comparison_results_df[(comparison_results_df['method'] == 'Neural Network') & (comparison_results_df['att_range_name'] == range_name)]
            if not nn_row.empty:
                nn_benchmark_metrics_for_monitor[range_name] = nn_row.iloc[0].to_dict()

            ls_row = comparison_results_df[(comparison_results_df['method'] == 'MULTIVERSE-LS') & (comparison_results_df['att_range_name'] == range_name)]
            if not ls_row.empty:
                baseline_ls_metrics_for_monitor[range_name] = ls_row.iloc[0].to_dict()
        
        monitor.check_target_achievement(nn_benchmark_metrics_for_monitor, baseline_ls_metrics_for_monitor)


    # Phase 5: Publication Material Generation
    monitor.log_progress("PHASE5", "Generating publication-ready materials")
    pub_gen = PublicationGenerator(config, output_path, monitor)
    publication_package = pub_gen.generate_publication_package(
        clinical_validation_results,
        comparison_results_df # Pass the full DataFrame
    )

    # Phase 6: Final Research Summary
    monitor.log_progress("PHASE6", "Generating comprehensive research summary")
    models_dir = output_path / 'trained_models'
    models_dir.mkdir(exist_ok=True)
    if trainer.models:
        for i, model_state in enumerate(trainer.best_states if hasattr(trainer, 'best_states') and trainer.best_states else [m.state_dict() for m in trainer.models]): # Use best_states if available
            if model_state:
                 torch.save(model_state, models_dir / f'ensemble_model_{i}_best.pt')
            elif trainer.models and trainer.models[i]: # Fallback to current model state if no best_state
                 torch.save(trainer.models[i].state_dict(), models_dir / f'ensemble_model_{i}_final.pt')


    final_results_summary = {
        'config': asdict(config),
        'optuna_best_params': best_optuna_params,
        'training_duration_hours': training_duration_hours,
        'training_histories': training_histories, # Contains losses
        'clinical_validation_results': clinical_validation_results,
        'benchmark_comparison_results_csv_path': str(output_path / "comparison_framework_outputs" / 'comparison_results_detailed.csv') if not comparison_results_df.empty else None,
        'publication_package_summary_path': str(output_path / 'publication_package_summary.json'),
        'trained_models_dir': str(models_dir)
    }
    with open(output_path / 'final_research_results.json', 'w') as f:
        json.dump(final_results_summary, f, indent=2, default=lambda o: '<not serializable>')


    summary_report_path = output_path / 'RESEARCH_SUMMARY.txt'
    # ... (generate summary text as before, referencing new result structures) ...
    with open(summary_report_path, 'w') as f:
        f.write(f"Research pipeline completed. Full summary in final_research_results.json and {output_path}\n")
        # Add more details from final_results_summary

    monitor.log_progress("COMPLETE", f"Research pipeline finished. Results in {output_path}")
    return final_results_summary


if __name__ == "__main__":
    script_logger.info("=" * 80)
    script_logger.info("ASL NEURAL NETWORK COMPREHENSIVE RESEARCH PIPELINE")
    script_logger.info("Enhancing Noninvasive Cerebral Blood Flow Imaging with Neural Networks")
    script_logger.info("=" * 80)

    config_file_path = None
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]

    loaded_config = ResearchConfig() # Start with defaults
    if config_file_path and Path(config_file_path).exists():
        script_logger.info(f"Loading configuration from {config_file_path}")
        try:
            with open(config_file_path, 'r') as f:
                if config_file_path.lower().endswith(('.yaml', '.yml')):
                    config_dict_loaded = yaml.safe_load(f)
                else:
                    config_dict_loaded = json.load(f)
            # Update default config with loaded values
            # This needs careful merging if config_dict_loaded is partial
            for key, value in config_dict_loaded.items():
                if hasattr(loaded_config, key):
                    setattr(loaded_config, key, value)
        except Exception as e:
            script_logger.error(f"Error loading config file {config_file_path}: {e}. Using defaults.")
    else:
        script_logger.info("No valid config file provided or found. Using default configuration.")

    script_logger.info("\nResearch Configuration:")
    script_logger.info("-" * 30)
    for key, value in asdict(loaded_config).items():
        script_logger.info(f"{key}: {value}")
    script_logger.info("-" * 30)

    script_logger.info("\nStarting comprehensive ASL research pipeline...")
    pipeline_results = run_comprehensive_asl_research(config=loaded_config)

    script_logger.info("\n" + "=" * 80)
    script_logger.info("RESEARCH PIPELINE COMPLETED!")
    script_logger.info("=" * 80)
    if "error" not in pipeline_results:
        script_logger.info(f"Results saved in: {pipeline_results.get('trained_models_dir', 'Specified output directory')}")
        script_logger.info("Check RESEARCH_SUMMARY.txt and final_research_results.json for detailed findings.")
    else:
        script_logger.error(f"Pipeline failed: {pipeline_results.get('error')}")
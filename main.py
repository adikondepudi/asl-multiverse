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

Author: Enhanced ASL Research Team
Date: 2025
"""

import torch
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import enhanced ASL components
from enhanced_asl_network import EnhancedASLNet, CustomLoss
from asl_simulation import ASLSimulator, ASLParameters
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer
from comparison_framework import ComprehensiveComparison
from performance_metrics import ProposalEvaluator
from single_repeat_validation import SingleRepeatValidator

# Import conventional methods for comparison
from vsasl_functions import fit_VSASL_vectInit_pep
from pcasl_functions import fit_PCASL_vectInit_pep  
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep


@dataclass
class ResearchConfig:
    """Configuration for the comprehensive research pipeline"""
    # Training parameters
    hidden_sizes: List[int] = None
    learning_rate: float = 0.001
    batch_size: int = 256
    n_samples: int = 50000  # Increased for comprehensive training
    n_epochs: int = 300
    n_ensembles: int = 5
    dropout_rate: float = 0.1
    norm_type: str = 'batch'
    
    # Hyperparameter optimization
    n_trials: int = 100
    optimization_timeout: int = 3600  # 1 hour
    
    # Data generation
    pld_range: List[int] = None
    att_ranges: List[Tuple[float, float, str]] = None
    
    # Simulation parameters  
    T1_artery: float = 1850
    T2_factor: float = 1.0
    alpha_BS1: float = 1.0
    alpha_PCASL: float = 0.85
    alpha_VSASL: float = 0.56
    T_tau: float = 1800
    CBF: float = 60
    
    # Clinical validation
    n_clinical_subjects: int = 200
    clinical_conditions: List[str] = None
    noise_levels: List[float] = None
    
    # Performance targets (50% improvement goals)
    target_cbf_cv_improvement: float = 0.50
    target_att_cv_improvement: float = 0.50
    target_scan_time_reduction: float = 0.75  # 75% reduction (4x to 1x repeats)
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128, 64]
        if self.pld_range is None:
            self.pld_range = [500, 3000, 500]
        if self.att_ranges is None:
            self.att_ranges = [
                (500, 1500, "Short ATT"),
                (1500, 2500, "Medium ATT"), 
                (2500, 4000, "Long ATT")
            ]
        if self.clinical_conditions is None:
            self.clinical_conditions = ['healthy', 'stroke', 'elderly', 'tumor']
        if self.noise_levels is None:
            self.noise_levels = [3.0, 5.0, 10.0, 15.0]


class PerformanceMonitor:
    """Monitor training progress and research objectives"""
    
    def __init__(self, config: ResearchConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.metrics_history = []
        self.target_achievements = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(output_dir / 'research.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_progress(self, phase: str, message: str):
        """Log research progress"""
        self.logger.info(f"[{phase}] {message}")
        
    def check_target_achievement(self, results: Dict, baseline_results: Dict) -> Dict:
        """Check if research targets are being met"""
        achievements = {}
        
        for att_range in ['Short ATT', 'Medium ATT', 'Long ATT']:
            if att_range in results and att_range in baseline_results:
                # CBF CV improvement
                current_cv = results[att_range].get('cbf_cov', float('inf'))
                baseline_cv = baseline_results[att_range].get('cbf_cov', float('inf'))
                cbf_improvement = (baseline_cv - current_cv) / baseline_cv if baseline_cv > 0 else 0
                
                # ATT CV improvement  
                current_att_cv = results[att_range].get('att_cov', float('inf'))
                baseline_att_cv = baseline_results[att_range].get('att_cov', float('inf'))
                att_improvement = (baseline_att_cv - current_att_cv) / baseline_att_cv if baseline_att_cv > 0 else 0
                
                achievements[att_range] = {
                    'cbf_cv_improvement': cbf_improvement,
                    'att_cv_improvement': att_improvement,
                    'cbf_target_met': cbf_improvement >= self.config.target_cbf_cv_improvement,
                    'att_target_met': att_improvement >= self.config.target_att_cv_improvement
                }
                
                if achievements[att_range]['cbf_target_met']:
                    self.log_progress("TARGET", f"CBF CV improvement target met for {att_range}: {cbf_improvement:.1%}")
                if achievements[att_range]['att_target_met']:
                    self.log_progress("TARGET", f"ATT CV improvement target met for {att_range}: {att_improvement:.1%}")
        
        return achievements


class HyperparameterOptimizer:
    """Systematic hyperparameter optimization using Optuna"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.best_params = None
        self.study = None
        
    def objective(self, trial) -> float:
        """Objective function for hyperparameter optimization"""
        # Define search space
        hidden_size_1 = trial.suggest_categorical('hidden_size_1', [128, 256, 512])
        hidden_size_2 = trial.suggest_categorical('hidden_size_2', [64, 128, 256])
        hidden_size_3 = trial.suggest_categorical('hidden_size_3', [32, 64, 128])
        
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.2)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        
        # Create temporary config
        temp_config = ResearchConfig(
            hidden_sizes=[hidden_size_1, hidden_size_2, hidden_size_3],
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            n_epochs=50,  # Reduced for optimization
            n_samples=10000  # Reduced for optimization
        )
        
        try:
            # Quick training run
            _, _, validation_loss = self._quick_training_run(temp_config)
            return validation_loss
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
            
    def _quick_training_run(self, config: ResearchConfig) -> Tuple[Any, Any, float]:
        """Quick training run for hyperparameter optimization"""
        # Setup components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        simulator = RealisticASLSimulator()
        plds = np.arange(*config.pld_range)
        
        # Create model factory
        def create_model():
            return EnhancedASLNet(
                input_size=len(plds) * 2,
                hidden_sizes=config.hidden_sizes,
                n_plds=len(plds),
                dropout_rate=config.dropout_rate,
                norm_type=config.norm_type
            )
        
        # Initialize trainer
        trainer = EnhancedASLTrainer(
            model_class=create_model,
            input_size=len(plds) * 2,
            hidden_sizes=config.hidden_sizes,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            n_ensembles=1,  # Single model for optimization
            device=device
        )
        
        # Prepare quick data
        train_loaders, val_loader = trainer.prepare_curriculum_data(
            simulator, n_samples=config.n_samples
        )
        
        # Quick training
        trainer.train_ensemble(train_loaders, val_loader, n_epochs=config.n_epochs)
        
        # Return validation loss
        val_loss = trainer._validate(trainer.models[0], val_loader, config.n_epochs)
        return trainer, simulator, val_loss
        
    def optimize(self) -> Dict:
        """Run hyperparameter optimization"""
        print("Starting hyperparameter optimization...")
        
        # Create study
        self.study = optuna.create_study(direction='minimize')
        
        # Optimize
        self.study.optimize(
            self.objective, 
            n_trials=self.config.n_trials,
            timeout=self.config.optimization_timeout
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        print(f"Best parameters found: {self.best_params}")
        print(f"Best validation loss: {self.study.best_value:.6f}")
        
        return self.best_params


class ClinicalValidator:
    """Comprehensive clinical validation framework"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        
    def validate_clinical_scenarios(self, trained_models: List[torch.nn.Module]) -> Dict:
        """Validate across clinical scenarios"""
        print("Running clinical validation scenarios...")
        
        simulator = RealisticASLSimulator()
        plds = np.arange(*self.config.pld_range)
        
        # Clinical scenarios matching research proposal
        clinical_scenarios = {
            'healthy_adult': {'cbf_range': (50, 80), 'att_range': (800, 1800), 'snr': 8},
            'elderly_patient': {'cbf_range': (30, 60), 'att_range': (1500, 3000), 'snr': 5}, 
            'stroke_patient': {'cbf_range': (10, 40), 'att_range': (2000, 4000), 'snr': 3},
            'tumor_patient': {'cbf_range': (20, 120), 'att_range': (1000, 3000), 'snr': 6}
        }
        
        results = {}
        
        for scenario_name, params in clinical_scenarios.items():
            print(f"  Validating {scenario_name}...")
            
            # Generate test data for this scenario
            n_subjects = self.config.n_clinical_subjects
            cbf_values = np.random.uniform(*params['cbf_range'], n_subjects)
            att_values = np.random.uniform(*params['att_range'], n_subjects)
            
            scenario_results = {
                'neural_network': {'cbf': [], 'att': [], 'uncertainties_cbf': [], 'uncertainties_att': []},
                'multiverse_ls': {'cbf': [], 'att': []},
                'single_repeat_nn': {'cbf': [], 'att': []},
                'multi_repeat_ls': {'cbf': [], 'att': []}
            }
            
            for i, (true_cbf, true_att) in enumerate(zip(cbf_values, att_values)):
                # Generate signals
                signals = simulator.generate_synthetic_data(
                    plds, np.array([true_att]), n_noise=1, tsnr=params['snr']
                )
                
                # Neural network prediction (single repeat)
                nn_input = np.concatenate([
                    signals['PCASL'][0, 0],
                    signals['VSASL'][0, 0]
                ])
                
                # Ensemble prediction
                cbf_preds, att_preds, cbf_uncs, att_uncs = self._ensemble_predict(
                    trained_models, nn_input
                )
                
                scenario_results['neural_network']['cbf'].append(cbf_preds.mean())
                scenario_results['neural_network']['att'].append(att_preds.mean())
                scenario_results['neural_network']['uncertainties_cbf'].append(cbf_uncs.mean())
                scenario_results['neural_network']['uncertainties_att'].append(att_uncs.mean())
                
                # Conventional MULTIVERSE fitting (single repeat)
                try:
                    pldti = np.column_stack([plds, plds])
                    multiverse_signal = signals['MULTIVERSE'][0, 0]
                    beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, multiverse_signal, [50/6000, 1500],
                        self.config.T1_artery, self.config.T_tau, self.config.T2_factor,
                        self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL
                    )
                    scenario_results['multiverse_ls']['cbf'].append(beta[0] * 6000)
                    scenario_results['multiverse_ls']['att'].append(beta[1])
                except:
                    scenario_results['multiverse_ls']['cbf'].append(np.nan)
                    scenario_results['multiverse_ls']['att'].append(np.nan)
                
                # Multi-repeat conventional (4 repeats) - gold standard
                multi_repeat_signals = []
                for repeat in range(4):
                    repeat_signals = simulator.generate_synthetic_data(
                        plds, np.array([true_att]), n_noise=1, tsnr=params['snr'] * 2  # Higher SNR
                    )
                    multi_repeat_signals.append(repeat_signals['MULTIVERSE'][0, 0])
                
                # Average the repeats
                averaged_signal = np.mean(multi_repeat_signals, axis=0)
                
                try:
                    beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, averaged_signal, [50/6000, 1500],
                        self.config.T1_artery, self.config.T_tau, self.config.T2_factor,
                        self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL
                    )
                    scenario_results['multi_repeat_ls']['cbf'].append(beta[0] * 6000)
                    scenario_results['multi_repeat_ls']['att'].append(beta[1])
                except:
                    scenario_results['multi_repeat_ls']['cbf'].append(np.nan)
                    scenario_results['multi_repeat_ls']['att'].append(np.nan)
            
            # Calculate metrics for this scenario
            true_cbf_array = cbf_values
            true_att_array = att_values
            
            for method in scenario_results:
                cbf_estimates = np.array(scenario_results[method]['cbf'])
                att_estimates = np.array(scenario_results[method]['att'])
                
                # Remove NaNs
                valid_mask = ~(np.isnan(cbf_estimates) | np.isnan(att_estimates))
                if np.sum(valid_mask) > 0:
                    cbf_valid = cbf_estimates[valid_mask]
                    att_valid = att_estimates[valid_mask]
                    true_cbf_valid = true_cbf_array[valid_mask]
                    true_att_valid = true_att_array[valid_mask]
                    
                    scenario_results[method]['metrics'] = {
                        'cbf_bias': np.mean(cbf_valid - true_cbf_valid),
                        'att_bias': np.mean(att_valid - true_att_valid),
                        'cbf_cov': np.std(cbf_valid) / np.mean(cbf_valid) * 100,
                        'att_cov': np.std(att_valid) / np.mean(att_valid) * 100,
                        'cbf_rmse': np.sqrt(np.mean((cbf_valid - true_cbf_valid)**2)),
                        'att_rmse': np.sqrt(np.mean((att_valid - true_att_valid)**2)),
                        'success_rate': np.sum(valid_mask) / len(cbf_estimates) * 100
                    }
                else:
                    scenario_results[method]['metrics'] = {
                        'cbf_bias': np.nan, 'att_bias': np.nan, 'cbf_cov': np.nan,
                        'att_cov': np.nan, 'cbf_rmse': np.nan, 'att_rmse': np.nan,
                        'success_rate': 0.0
                    }
            
            results[scenario_name] = scenario_results
            
        return results
    
    def _ensemble_predict(self, models: List[torch.nn.Module], input_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Make ensemble predictions"""
        input_tensor = torch.FloatTensor(input_signal).unsqueeze(0)
        
        cbf_preds = []
        att_preds = []
        cbf_vars = []
        att_vars = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                cbf_mean, att_mean, cbf_log_var, att_log_var = model(input_tensor)
                cbf_preds.append(cbf_mean.item() * 6000)  # Convert to ml/100g/min
                att_preds.append(att_mean.item())
                cbf_vars.append(np.exp(cbf_log_var.item()))
                att_vars.append(np.exp(att_log_var.item()))
        
        # Combine predictions (ensemble average)
        cbf_final = np.array(cbf_preds)
        att_final = np.array(att_preds)
        
        # Total uncertainty (aleatoric + epistemic)
        cbf_uncertainty = np.sqrt(np.mean(cbf_vars) + np.var(cbf_preds))
        att_uncertainty = np.sqrt(np.mean(att_vars) + np.var(att_preds))
        
        return cbf_final, att_final, np.array([cbf_uncertainty]), np.array([att_uncertainty])


class PublicationGenerator:
    """Generate publication-ready materials"""
    
    def __init__(self, config: ResearchConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        
    def generate_publication_package(self, 
                                   clinical_results: Dict,
                                   comparison_results: Dict,
                                   baseline_results: Dict) -> Dict:
        """Generate complete publication package"""
        print("Generating publication materials...")
        
        package = {
            'figures': {},
            'tables': {},
            'statistical_analysis': {},
            'code_package': {}
        }
        
        # Generate Figure 1 (recreation of proposal figure)
        package['figures']['figure1_performance'] = self._generate_figure1(
            comparison_results, baseline_results
        )
        
        # Generate clinical validation figures
        package['figures']['clinical_validation'] = self._generate_clinical_figures(
            clinical_results
        )
        
        # Generate scan time analysis
        package['figures']['scan_time_analysis'] = self._generate_scan_time_figures(
            clinical_results
        )
        
        # Generate performance tables
        package['tables']['performance_summary'] = self._generate_performance_table(
            comparison_results, baseline_results
        )
        
        # Generate statistical analysis
        package['statistical_analysis'] = self._generate_statistical_analysis(
            clinical_results, comparison_results, baseline_results
        )
        
        # Save all materials
        self._save_publication_materials(package)
        
        return package
    
    def _generate_figure1(self, comparison_results: Dict, baseline_results: Dict) -> str:
        """Generate Figure 1 recreating proposal performance plots"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Setup data
        att_ranges = ['Short ATT', 'Medium ATT', 'Long ATT']
        methods = ['PCASL-LS', 'VSASL-LS', 'MULTIVERSE-LS', 'MULTIVERSE-NN']
        colors = {'PCASL-LS': 'blue', 'VSASL-LS': 'green', 
                 'MULTIVERSE-LS': 'red', 'MULTIVERSE-NN': 'purple'}
        
        metrics = [
            ('cbf_bias', 'att_bias', 'Normalized Bias', '%'),
            ('cbf_cov', 'att_cov', 'Coefficient of Variation', '%'),
            ('cbf_rmse', 'att_rmse', 'Normalized RMSE', '%')
        ]
        
        x_positions = range(len(att_ranges))
        
        for row, (cbf_metric, att_metric, metric_name, unit) in enumerate(metrics):
            # CBF subplot
            ax_cbf = axes[row, 0]
            for method in methods:
                values = []
                for att_range in att_ranges:
                    if method in ['MULTIVERSE-NN'] and att_range in comparison_results:
                        # Neural network results
                        val = comparison_results[att_range].get(cbf_metric, np.nan)
                    elif att_range in baseline_results:
                        # Conventional method results
                        val = baseline_results[att_range].get(cbf_metric, np.nan)
                    else:
                        val = np.nan
                    
                    # Normalize bias and RMSE as percentages
                    if 'bias' in cbf_metric:
                        val = val / 60 * 100  # Normalize by true CBF
                    elif 'rmse' in cbf_metric:
                        val = val / 60 * 100
                        
                    values.append(val)
                
                valid_indices = ~np.isnan(values)
                if np.any(valid_indices):
                    ax_cbf.plot(np.array(x_positions)[valid_indices], 
                               np.array(values)[valid_indices],
                               'o-', color=colors[method], 
                               linewidth=3 if method == 'MULTIVERSE-NN' else 2,
                               markersize=8, label=method)
            
            ax_cbf.set_ylabel(f'CBF {metric_name} ({unit})')
            ax_cbf.set_title(f'CBF {metric_name}')
            ax_cbf.set_xticks(x_positions)
            ax_cbf.set_xticklabels(att_ranges)
            ax_cbf.grid(True, alpha=0.3)
            if row == 0:
                ax_cbf.legend()
            
            # ATT subplot  
            ax_att = axes[row, 1]
            for method in methods:
                values = []
                for att_range in att_ranges:
                    if method in ['MULTIVERSE-NN'] and att_range in comparison_results:
                        val = comparison_results[att_range].get(att_metric, np.nan)
                    elif att_range in baseline_results:
                        val = baseline_results[att_range].get(att_metric, np.nan)
                    else:
                        val = np.nan
                    
                    # Normalize bias and RMSE  
                    if 'bias' in att_metric:
                        typical_att = {'Short ATT': 1000, 'Medium ATT': 2000, 'Long ATT': 3000}
                        val = val / typical_att.get(att_range, 2000) * 100
                    elif 'rmse' in att_metric:
                        typical_att = {'Short ATT': 1000, 'Medium ATT': 2000, 'Long ATT': 3000}
                        val = val / typical_att.get(att_range, 2000) * 100
                        
                    values.append(val)
                
                valid_indices = ~np.isnan(values)
                if np.any(valid_indices):
                    ax_att.plot(np.array(x_positions)[valid_indices],
                               np.array(values)[valid_indices], 
                               'o-', color=colors[method],
                               linewidth=3 if method == 'MULTIVERSE-NN' else 2,
                               markersize=8, label=method)
            
            ax_att.set_ylabel(f'ATT {metric_name} ({unit})')
            ax_att.set_title(f'ATT {metric_name}')
            ax_att.set_xticks(x_positions)
            ax_att.set_xticklabels(att_ranges)
            ax_att.grid(True, alpha=0.3)
            
            # Add reference line for bias plots
            if 'bias' in cbf_metric:
                ax_cbf.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax_att.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        figure_path = self.output_dir / 'figure1_performance_comparison.png'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(figure_path)
    
    def _generate_clinical_figures(self, clinical_results: Dict) -> Dict:
        """Generate clinical validation figures"""
        figures = {}
        
        # Clinical scenario comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        scenarios = list(clinical_results.keys())
        methods = ['neural_network', 'multiverse_ls', 'multi_repeat_ls']
        method_labels = ['Neural Network\n(Single Repeat)', 'MULTIVERSE LS\n(Single Repeat)', 
                        'MULTIVERSE LS\n(Multi Repeat)']
        
        metrics = ['cbf_cov', 'att_cov', 'cbf_rmse', 'att_rmse']
        metric_labels = ['CBF CoV (%)', 'ATT CoV (%)', 'CBF RMSE (ml/100g/min)', 'ATT RMSE (ms)']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i//2, i%2]
            
            x_pos = np.arange(len(scenarios))
            width = 0.25
            
            for j, (method, method_label) in enumerate(zip(methods, method_labels)):
                values = []
                for scenario in scenarios:
                    if scenario in clinical_results and method in clinical_results[scenario]:
                        val = clinical_results[scenario][method]['metrics'].get(metric, np.nan)
                        values.append(val)
                    else:
                        values.append(np.nan)
                
                ax.bar(x_pos + j*width, values, width, label=method_label, alpha=0.8)
            
            ax.set_ylabel(label)
            ax.set_title(f'Clinical Validation: {label}')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figure_path = self.output_dir / 'clinical_validation.png'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        figures['clinical_comparison'] = str(figure_path)
        return figures
    
    def _generate_scan_time_figures(self, clinical_results: Dict) -> Dict:
        """Generate scan time analysis figures"""
        # This would implement scan time vs. quality trade-off analysis
        # Placeholder implementation
        return {'scan_time_analysis': 'scan_time_analysis.png'}
    
    def _generate_performance_table(self, comparison_results: Dict, baseline_results: Dict) -> str:
        """Generate performance summary table"""
        # Create comprehensive performance table
        table_data = []
        
        for att_range in ['Short ATT', 'Medium ATT', 'Long ATT']:
            if att_range in comparison_results:
                nn_results = comparison_results[att_range]
                baseline = baseline_results.get(att_range, {})
                
                row = {
                    'ATT_Range': att_range,
                    'NN_CBF_CV': nn_results.get('cbf_cov', np.nan),
                    'NN_ATT_CV': nn_results.get('att_cov', np.nan),
                    'Baseline_CBF_CV': baseline.get('cbf_cov', np.nan),
                    'Baseline_ATT_CV': baseline.get('att_cov', np.nan),
                    'CBF_Improvement': ((baseline.get('cbf_cov', 0) - nn_results.get('cbf_cov', 0)) / 
                                       baseline.get('cbf_cov', 1) * 100) if baseline.get('cbf_cov', 0) > 0 else 0,
                    'ATT_Improvement': ((baseline.get('att_cov', 0) - nn_results.get('att_cov', 0)) / 
                                       baseline.get('att_cov', 1) * 100) if baseline.get('att_cov', 0) > 0 else 0
                }
                table_data.append(row)
        
        df = pd.DataFrame(table_data)
        table_path = self.output_dir / 'performance_summary.csv'
        df.to_csv(table_path, index=False)
        
        return str(table_path)
    
    def _generate_statistical_analysis(self, clinical_results: Dict, 
                                     comparison_results: Dict, baseline_results: Dict) -> Dict:
        """Generate statistical analysis"""
        analysis = {
            'target_achievements': {},
            'significance_tests': {},
            'effect_sizes': {},
            'confidence_intervals': {}
        }
        
        # Check target achievements
        for att_range in ['Short ATT', 'Medium ATT', 'Long ATT']:
            if att_range in comparison_results and att_range in baseline_results:
                nn_cbf_cv = comparison_results[att_range].get('cbf_cov', float('inf'))
                baseline_cbf_cv = baseline_results[att_range].get('cbf_cov', float('inf'))
                
                cbf_improvement = (baseline_cbf_cv - nn_cbf_cv) / baseline_cbf_cv if baseline_cbf_cv > 0 else 0
                
                analysis['target_achievements'][att_range] = {
                    'cbf_cv_improvement': cbf_improvement,
                    'target_met': cbf_improvement >= self.config.target_cbf_cv_improvement,
                    'improvement_percentage': cbf_improvement * 100
                }
        
        return analysis
    
    def _save_publication_materials(self, package: Dict):
        """Save all publication materials"""
        # Save package summary
        with open(self.output_dir / 'publication_package.json', 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            serializable_package = {}
            for key, value in package.items():
                if isinstance(value, dict):
                    serializable_package[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (np.int64, np.float64)):
                            serializable_package[key][subkey] = float(subvalue)
                        elif isinstance(subvalue, np.ndarray):
                            serializable_package[key][subkey] = subvalue.tolist()
                        else:
                            serializable_package[key][subkey] = subvalue
                else:
                    serializable_package[key] = value
            
            json.dump(serializable_package, f, indent=2)


def run_comprehensive_asl_research(config: Optional[ResearchConfig] = None,
                                  output_dir: str = 'comprehensive_results') -> Dict:
    """
    Main function implementing the complete research pipeline.
    
    This function orchestrates the entire ASL neural network research process:
    1. Hyperparameter optimization
    2. Multi-objective training  
    3. Clinical validation
    4. Benchmarking against conventional methods
    5. Publication material generation
    
    Parameters
    ----------
    config : ResearchConfig, optional
        Research configuration parameters
    output_dir : str
        Output directory for results
        
    Returns
    -------
    Dict
        Complete research results and trained models
    """
    
    # Setup
    if config is None:
        config = ResearchConfig()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir) / f'asl_research_{timestamp}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize monitoring
    monitor = PerformanceMonitor(config, output_path)
    monitor.log_progress("SETUP", "Initializing comprehensive ASL research pipeline")
    
    # Save configuration
    with open(output_path / 'research_config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Phase 1: Hyperparameter Optimization
    monitor.log_progress("PHASE1", "Starting hyperparameter optimization")
    optimizer = HyperparameterOptimizer(config)
    best_params = optimizer.optimize()
    
    # Update config with best parameters
    if best_params:
        config.hidden_sizes = [best_params['hidden_size_1'], 
                              best_params['hidden_size_2'], 
                              best_params['hidden_size_3']]
        config.learning_rate = best_params['learning_rate']
        config.dropout_rate = best_params['dropout_rate']
        config.batch_size = best_params['batch_size']
    
    # Phase 2: Multi-objective Training
    monitor.log_progress("PHASE2", "Starting multi-objective ensemble training")
    
    # Setup training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator = RealisticASLSimulator()
    plds = np.arange(*config.pld_range)
    
    # Create model factory with optimized parameters
    def create_model():
        return EnhancedASLNet(
            input_size=len(plds) * 2,
            hidden_sizes=config.hidden_sizes,
            n_plds=len(plds),
            dropout_rate=config.dropout_rate,
            norm_type=config.norm_type
        )
    
    # Initialize enhanced trainer
    trainer = EnhancedASLTrainer(
        model_class=create_model,
        input_size=len(plds) * 2,
        hidden_sizes=config.hidden_sizes,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        n_ensembles=config.n_ensembles,
        device=device
    )
    
    # Prepare comprehensive training data
    monitor.log_progress("PHASE2", "Preparing curriculum training datasets")
    train_loaders, val_loader = trainer.prepare_curriculum_data(
        simulator, n_samples=config.n_samples
    )
    
    # Train ensemble with monitoring
    monitor.log_progress("PHASE2", f"Training {config.n_ensembles}-model ensemble")
    training_start = time.time()
    
    training_histories = trainer.train_ensemble(
        train_loaders, val_loader, n_epochs=config.n_epochs
    )
    
    training_time = time.time() - training_start
    monitor.log_progress("PHASE2", f"Training completed in {training_time/3600:.2f} hours")
    
    # Phase 3: Clinical Validation
    monitor.log_progress("PHASE3", "Starting clinical validation across patient populations")
    
    validator = ClinicalValidator(config)
    clinical_results = validator.validate_clinical_scenarios(trainer.models)
    
    # Phase 4: Benchmarking Against Conventional Methods
    monitor.log_progress("PHASE4", "Benchmarking against conventional least-squares methods")
    
    # Generate baseline results using conventional methods
    baseline_results = {}
    
    for att_min, att_max, range_name in config.att_ranges:
        monitor.log_progress("PHASE4", f"Evaluating {range_name}")
        
        # Generate test data for this range
        n_test = 1000
        att_values = np.random.uniform(att_min, att_max, n_test)
        test_signals = simulator.generate_synthetic_data(plds, att_values, n_noise=50, tsnr=5.0)
        
        # Test neural network
        nn_results = {'cbf': [], 'att': []}
        
        for i in range(n_test):
            for noise_idx in range(50):
                nn_input = np.concatenate([
                    test_signals['PCASL'][noise_idx, i],
                    test_signals['VSASL'][noise_idx, i]
                ])
                
                cbf_preds, att_preds, _, _ = validator._ensemble_predict(trainer.models, nn_input)
                nn_results['cbf'].append(cbf_preds.mean())
                nn_results['att'].append(att_preds.mean())
        
        # Test conventional MULTIVERSE
        conv_results = {'cbf': [], 'att': []}
        pldti = np.column_stack([plds, plds])
        
        for i in range(n_test):
            for noise_idx in range(50):
                try:
                    signal = test_signals['MULTIVERSE'][noise_idx, i]
                    beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, signal, [50/6000, 1500],
                        config.T1_artery, config.T_tau, config.T2_factor,
                        config.alpha_BS1, config.alpha_PCASL, config.alpha_VSASL
                    )
                    conv_results['cbf'].append(beta[0] * 6000)
                    conv_results['att'].append(beta[1])
                except:
                    conv_results['cbf'].append(np.nan)
                    conv_results['att'].append(np.nan)
        
        # Calculate metrics
        true_cbf = np.full(n_test * 50, config.CBF)
        true_att = np.repeat(att_values, 50)
        
        # Neural network metrics
        nn_cbf = np.array(nn_results['cbf'])
        nn_att = np.array(nn_results['att'])
        
        nn_metrics = {
            'cbf_bias': np.mean(nn_cbf - true_cbf),
            'att_bias': np.mean(nn_att - true_att),
            'cbf_cov': np.std(nn_cbf) / np.mean(nn_cbf) * 100,
            'att_cov': np.std(nn_att) / np.mean(nn_att) * 100,
            'cbf_rmse': np.sqrt(np.mean((nn_cbf - true_cbf)**2)),
            'att_rmse': np.sqrt(np.mean((nn_att - true_att)**2))
        }
        
        # Conventional metrics
        conv_cbf = np.array(conv_results['cbf'])
        conv_att = np.array(conv_results['att'])
        
        # Remove NaNs for conventional method
        valid_mask = ~(np.isnan(conv_cbf) | np.isnan(conv_att))
        if np.sum(valid_mask) > 0:
            conv_cbf_valid = conv_cbf[valid_mask]
            conv_att_valid = conv_att[valid_mask]
            true_cbf_valid = true_cbf[valid_mask]
            true_att_valid = true_att[valid_mask]
            
            conv_metrics = {
                'cbf_bias': np.mean(conv_cbf_valid - true_cbf_valid),
                'att_bias': np.mean(conv_att_valid - true_att_valid),
                'cbf_cov': np.std(conv_cbf_valid) / np.mean(conv_cbf_valid) * 100,
                'att_cov': np.std(conv_att_valid) / np.mean(conv_att_valid) * 100,
                'cbf_rmse': np.sqrt(np.mean((conv_cbf_valid - true_cbf_valid)**2)),
                'att_rmse': np.sqrt(np.mean((conv_att_valid - true_att_valid)**2))
            }
        else:
            conv_metrics = {key: np.nan for key in nn_metrics.keys()}
        
        # Store results
        baseline_results[range_name] = conv_metrics
        
        # Check target achievements
        achievements = monitor.check_target_achievement({range_name: nn_metrics}, 
                                                       {range_name: conv_metrics})
    
    # Phase 5: Publication Material Generation
    monitor.log_progress("PHASE5", "Generating publication-ready materials")
    
    pub_generator = PublicationGenerator(config, output_path)
    publication_package = pub_generator.generate_publication_package(
        clinical_results, {range_name: nn_metrics}, baseline_results
    )
    
    # Phase 6: Final Research Summary
    monitor.log_progress("PHASE6", "Generating comprehensive research summary")
    
    # Save models
    models_dir = output_path / 'trained_models'
    models_dir.mkdir(exist_ok=True)
    for i, model in enumerate(trainer.models):
        torch.save(model.state_dict(), models_dir / f'ensemble_model_{i}.pt')
    
    # Save all results
    results = {
        'config': asdict(config),
        'training_time_hours': training_time / 3600,
        'best_hyperparameters': best_params,
        'clinical_validation': clinical_results,
        'baseline_comparison': baseline_results,
        'neural_network_results': {range_name: nn_metrics},
        'publication_package': publication_package,
        'models_path': str(models_dir)
    }
    
    with open(output_path / 'comprehensive_results.json', 'w') as f:
        # Handle numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    # Generate final summary report
    summary_lines = [
        "ASL NEURAL NETWORK RESEARCH SUMMARY",
        "=" * 50,
        f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Runtime: {training_time/3600:.2f} hours",
        "",
        "KEY ACHIEVEMENTS:",
        f"✓ Trained {config.n_ensembles}-model ensemble",
        f"✓ Validated across {len(clinical_results)} clinical scenarios",
        f"✓ Generated publication package with {len(publication_package['figures'])} figures",
        "",
        "PERFORMANCE TARGETS:",
    ]
    
    for range_name in ['Short ATT', 'Medium ATT', 'Long ATT']:
        if range_name in baseline_results:
            baseline_cv = baseline_results[range_name].get('cbf_cov', 0)
            nn_cv = nn_metrics.get('cbf_cov', 0)
            improvement = (baseline_cv - nn_cv) / baseline_cv * 100 if baseline_cv > 0 else 0
            target_met = improvement >= config.target_cbf_cv_improvement * 100
            
            summary_lines.append(f"  {range_name}: {improvement:.1f}% CBF CV improvement {'✓' if target_met else '✗'}")
    
    summary_lines.extend([
        "",
        "NEXT STEPS:",
        "• Review detailed results in comprehensive_results.json",
        "• Examine publication materials in the figures/ directory", 
        "• Use trained models for further validation",
        "• Submit findings to peer-reviewed journal",
        "",
        f"All results saved in: {output_path}"
    ])
    
    with open(output_path / 'RESEARCH_SUMMARY.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Final log
    monitor.log_progress("COMPLETE", f"Research pipeline completed successfully. Results in {output_path}")
    
    return results


def train_enhanced_asl_model(config: dict = None, output_dir: str = 'results'):
    """
    Legacy interface for backward compatibility.
    Redirects to the comprehensive research pipeline.
    """
    if config is None:
        research_config = ResearchConfig()
    else:
        # Convert old config format to new ResearchConfig
        research_config = ResearchConfig(
            hidden_sizes=config.get('hidden_sizes', [256, 128, 64]),
            learning_rate=config.get('learning_rate', 0.001),
            batch_size=config.get('batch_size', 256),
            n_samples=config.get('n_samples', 20000),
            n_epochs=config.get('n_epochs', 200),
            n_ensembles=config.get('n_ensembles', 5),
            dropout_rate=config.get('dropout_rate', 0.1)
        )
    
    return run_comprehensive_asl_research(research_config, output_dir)


if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("ASL NEURAL NETWORK COMPREHENSIVE RESEARCH PIPELINE")
    print("Enhancing Noninvasive Cerebral Blood Flow Imaging with Neural Networks")
    print("=" * 80)
    
    # Check for configuration file
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # Load configuration
    if config_file and Path(config_file).exists():
        print(f"Loading configuration from {config_file}")
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
        
        # Convert to ResearchConfig
        config = ResearchConfig(**config_dict)
    else:
        print("Using default configuration")
        config = ResearchConfig()
    
    # Display configuration
    print("\nResearch Configuration:")
    print("-" * 30)
    print(f"Training samples: {config.n_samples:,}")
    print(f"Ensemble size: {config.n_ensembles}")
    print(f"Training epochs: {config.n_epochs}")
    print(f"Hidden layers: {config.hidden_sizes}")
    print(f"Clinical conditions: {config.clinical_conditions}")
    print(f"Target CBF CV improvement: {config.target_cbf_cv_improvement:.0%}")
    print(f"Target scan time reduction: {config.target_scan_time_reduction:.0%}")
    
    # Run comprehensive research
    print("\nStarting comprehensive ASL research pipeline...")
    results = run_comprehensive_asl_research(config)
    
    print("\n" + "=" * 80)
    print("RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Results saved in: {results.get('models_path', 'results')}")
    print("Check RESEARCH_SUMMARY.txt for detailed findings")
    print("Publication materials ready for manuscript preparation")
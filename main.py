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
# import matplotlib.pyplot as plt # No longer needed for saving plots
# import seaborn as sns # No longer needed for saving plots
from tqdm import tqdm
import optuna
from dataclasses import dataclass, asdict, field
import sys
import warnings
import wandb 
import joblib 
import math # For log_var defaults if not in config for some reason

warnings.filterwarnings('ignore', category=UserWarning) # Filter UserWarning from PyTorch/Optuna etc.

from enhanced_asl_network import EnhancedASLNet, CustomLoss
from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer # Contains EnhancedASLDataset
from comparison_framework import ComprehensiveComparison, ComparisonResult # ComparisonResult not directly used here
from performance_metrics import ProposalEvaluator # Not directly used in main flow, but available
from single_repeat_validation import SingleRepeatValidator, run_single_repeat_validation_main

from vsasl_functions import fit_VSASL_vectInit_pep
from pcasl_functions import fit_PCASL_vectInit_pep
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

script_logger = logging.getLogger(__name__) # Use __name__ for module-level logger


@dataclass
class ResearchConfig:
    # Training parameters
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    batch_size: int = 256
    n_training_subjects: int = 10000 
    training_n_epochs: int = 200
    n_ensembles: int = 5
    dropout_rate: float = 0.1
    norm_type: str = 'batch'
    
    m0_input_feature_model: bool = False

    use_transformer_temporal_model: bool = True
    use_focused_transformer_model: bool = False 
    transformer_d_model: int = 64              
    transformer_d_model_focused: int = 32      
    transformer_nhead_model: int = 4
    transformer_nlayers_model: int = 2
    
    log_var_cbf_min: float = -6.0
    log_var_cbf_max: float = 7.0
    log_var_att_min: float = -2.0
    log_var_att_max: float = 14.0

    loss_weight_cbf: float = 1.0
    loss_weight_att: float = 1.0
    loss_log_var_reg_lambda: float = 0.0

    optuna_n_trials: int = 20
    optuna_timeout_hours: float = 0.5
    optuna_n_subjects: int = 500 
    optuna_n_epochs: int = 20    
    optuna_study_name: str = "asl_multiverse_hpo"

    pld_values: List[int] = field(default_factory=lambda: list(range(500, 3001, 500)))
    att_ranges_config: List[Tuple[float, float, str]] = field(default_factory=lambda: [
        (500.0, 1500.0, "Short ATT"),
        (1500.0, 2500.0, "Medium ATT"),
        (2500.0, 4000.0, "Long ATT")
    ])
    include_m0_in_training_data: bool = False 

    T1_artery: float = 1850.0; T2_factor: float = 1.0; alpha_BS1: float = 1.0
    alpha_PCASL: float = 0.85; alpha_VSASL: float = 0.56; T_tau: float = 1800.0
    reference_CBF: float = 60.0 

    n_test_subjects_per_att_range: int = 200 
    test_snr_levels: List[float] = field(default_factory=lambda: [5.0, 10.0])
    test_conditions: List[str] = field(default_factory=lambda: ['healthy', 'stroke'])
    
    n_clinical_scenario_subjects: int = 100 
    clinical_scenario_definitions: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'healthy_adult': {'cbf_range': (50.0, 80.0), 'att_range': (800.0, 1800.0), 'snr': 8.0},
        'elderly_patient': {'cbf_range': (30.0, 60.0), 'att_range': (1500.0, 3000.0), 'snr': 5.0},
        'stroke_patient': {'cbf_range': (10.0, 40.0), 'att_range': (2000.0, 4000.0), 'snr': 3.0},
        'tumor_patient': {'cbf_range': (20.0, 120.0), 'att_range': (1000.0, 3000.0), 'snr': 6.0}
    })
    training_conditions: List[str] = field(default_factory=lambda: ['healthy', 'stroke', 'elderly', 'tumor'])
    training_noise_levels: List[float] = field(default_factory=lambda: [3.0, 5.0, 10.0, 15.0])

    target_cbf_cv_improvement_perc: float = 50.0
    target_att_cv_improvement_perc: float = 50.0

    wandb_project: str = "asl-multiverse-project"
    wandb_entity: Optional[str] = None 

# Helper function for normalization (can be moved to a utils.py later)
def apply_normalization_to_input_flat(flat_signal: np.ndarray, 
                                      norm_stats: Dict, 
                                      num_plds_per_modality: int, 
                                      has_m0: bool) -> np.ndarray:
    if not norm_stats or not isinstance(norm_stats, dict): return flat_signal 

    pcasl_signal_part = flat_signal[:num_plds_per_modality]
    vsasl_signal_part = flat_signal[num_plds_per_modality : num_plds_per_modality*2]
    
    pcasl_norm = (pcasl_signal_part - norm_stats.get('pcasl_mean', 0)) / norm_stats.get('pcasl_std', 1)
    vsasl_norm = (vsasl_signal_part - norm_stats.get('vsasl_mean', 0)) / norm_stats.get('vsasl_std', 1)
    
    normalized_parts = [pcasl_norm, vsasl_norm]
    
    if has_m0:
        m0_signal_part = flat_signal[num_plds_per_modality*2:] # Assumes M0 is at the end
        if m0_signal_part.size > 0 : # Ensure M0 part exists
            m0_norm = (m0_signal_part - norm_stats.get('m0_mean', 0)) / norm_stats.get('m0_std', 1)
            normalized_parts.append(m0_norm)
        else: # Should not happen if has_m0 is true and data is consistent
            script_logger.warning("has_m0 is true, but M0 part is missing in flat_signal for normalization.")
            
    return np.concatenate(normalized_parts)


class PerformanceMonitor:
    def __init__(self, config: ResearchConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger("ASLResearchPipeline") # Consistent logger name
        # Clear existing handlers to avoid duplicate logging if this is re-instantiated
        for handler in self.logger.handlers[:]: self.logger.removeHandler(handler)
        
        self.logger.setLevel(logging.INFO) # Set level for this specific logger instance
        
        fh = logging.FileHandler(output_dir / 'research.log', mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        
        sh = logging.StreamHandler(sys.stdout) # Log to stdout
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(sh)
        self.logger.propagate = False # Prevent propagation to root logger if root is also configured

    def log_progress(self, phase: str, message: str, level: int = logging.INFO):
        self.logger.log(level, f"[{phase}] {message}")
        if wandb.run: 
             wandb.log({f"Progress/{phase.replace(' ', '_')}": message}, step=wandb.run.step if wandb.run.step is not None else 0)


    def check_target_achievement(self, nn_att_range_results: Dict, baseline_att_range_results: Dict) -> Dict:
        achievements = {}
        for att_range_name_key in nn_att_range_results.keys():
            if att_range_name_key not in baseline_att_range_results:
                self.log_progress("TARGET_CHECK", f"Baseline results missing for {att_range_name_key}", logging.WARNING)
                continue
            nn_metrics_dict, baseline_metrics_dict = nn_att_range_results[att_range_name_key], baseline_att_range_results[att_range_name_key]
            
            current_cbf_cv_val = nn_metrics_dict.get('cbf_cov', float('inf'))
            baseline_cbf_cv_val = baseline_metrics_dict.get('cbf_cov', float('inf'))
            cbf_improvement_val = ((baseline_cbf_cv_val - current_cbf_cv_val) / baseline_cbf_cv_val) * 100 if baseline_cbf_cv_val > 0 and not np.isinf(baseline_cbf_cv_val) and not np.isinf(current_cbf_cv_val) else 0.0
            
            current_att_cv_val = nn_metrics_dict.get('att_cov', float('inf'))
            baseline_att_cv_val = baseline_metrics_dict.get('att_cov', float('inf'))
            att_improvement_val = ((baseline_att_cv_val - current_att_cv_val) / baseline_att_cv_val) * 100 if baseline_att_cv_val > 0 and not np.isinf(baseline_att_cv_val) and not np.isinf(current_att_cv_val) else 0.0
            
            cbf_target_met_flag, att_target_met_flag = cbf_improvement_val >= self.config.target_cbf_cv_improvement_perc, att_improvement_val >= self.config.target_att_cv_improvement_perc
            achievements[att_range_name_key] = {'cbf_cv_improvement_perc': cbf_improvement_val, 'att_cv_improvement_perc': att_improvement_val, 'cbf_target_met': cbf_target_met_flag, 'att_target_met': att_target_met_flag}
            self.log_progress("TARGET_CHECK", f"{att_range_name_key} - CBF CV Improv: {cbf_improvement_val:.1f}% ({'MET' if cbf_target_met_flag else 'NOT MET'})")
            self.log_progress("TARGET_CHECK", f"{att_range_name_key} - ATT CV Improv: {att_improvement_val:.1f}% ({'MET' if att_target_met_flag else 'NOT MET'})")
            if wandb.run:
                wandb.summary[f'TargetAchieved/{att_range_name_key}/CBF_CV_Improvement'] = cbf_improvement_val
                wandb.summary[f'TargetAchieved/{att_range_name_key}/ATT_CV_Improvement'] = att_improvement_val
        return achievements

class HyperparameterOptimizer:
    def __init__(self, base_config: ResearchConfig, monitor: PerformanceMonitor, output_dir: Path):
        self.base_config = base_config
        self.monitor = monitor
        self.output_dir = output_dir
        self.study = None
        self.main_wandb_run_id = wandb.run.id if wandb.run else None
        self.main_wandb_project = wandb.run.project if wandb.run else self.base_config.wandb_project
        self.main_wandb_entity = wandb.run.entity if wandb.run else self.base_config.wandb_entity


    def objective(self, trial: optuna.Trial) -> float:
        hidden_size_1 = trial.suggest_categorical('hidden_size_1', [128, 256, 512])
        hidden_size_2 = trial.suggest_categorical('hidden_size_2', [64, 128, 256])
        hidden_size_3 = trial.suggest_categorical('hidden_size_3', [32, 64, 128])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        
        # Keep transformer architecture fixed based on base_config for HPO simplicity for now
        # Or, add them to Optuna suggestions if desired:
        # use_focused_transformer = trial.suggest_categorical('use_focused_transformer', [True, False])
        # transformer_d_model = trial.suggest_categorical('transformer_d_model', [32, 64, 128]) 
        # transformer_d_model_focused = trial.suggest_categorical('transformer_d_model_focused', [16, 32, 64])
        # transformer_nhead = trial.suggest_categorical('transformer_nhead', [2, 4, 8])
        # transformer_nlayers = trial.suggest_categorical('transformer_nlayers', [1, 2, 3])


        trial_config_dict = asdict(self.base_config) # Start with base
        trial_optuna_params = { # Optuna-specific overrides for this trial
            'hidden_sizes': [hidden_size_1, hidden_size_2, hidden_size_3],
            'learning_rate': learning_rate, 'dropout_rate': dropout_rate, 'batch_size': batch_size,
            # If optimizing transformer params:
            # 'use_focused_transformer_model': use_focused_transformer,
            # 'transformer_d_model': transformer_d_model,
            # 'transformer_d_model_focused': transformer_d_model_focused,
            # 'transformer_nhead_model': transformer_nhead,
            # 'transformer_nlayers_model': transformer_nlayers,
        }
        trial_config_dict.update(trial_optuna_params)
        # Override dataset size and epochs for faster HPO trials
        trial_config_dict.update({ 
            'n_training_subjects': self.base_config.optuna_n_subjects,
            'training_n_epochs': self.base_config.optuna_n_epochs, 'n_ensembles': 1,
        })
        trial_run_config = ResearchConfig(**trial_config_dict)
        self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number}: Params {trial.params}")

        # W&B handling for Optuna trials
        if self.main_wandb_run_id and wandb.run and wandb.run.id == self.main_wandb_run_id :
            wandb.finish(quiet=True) # Temporarily finish main run
        
        trial_wandb_run = wandb.init(
            project=self.main_wandb_project, entity=self.main_wandb_entity,
            group=self.base_config.optuna_study_name, name=f"trial_{trial.number}_{wandb.util.generate_id()[:4]}",
            config=trial_optuna_params, reinit=True, job_type="hpo_trial"
        )
        
        try:
            _, _, validation_metrics_dict = self._quick_training_run(trial_run_config)
            validation_loss_val = validation_metrics_dict.get('val_loss', float('inf'))
            self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number} Val Loss: {validation_loss_val:.6f}")
            if trial_wandb_run:
                trial_wandb_run.summary['final_validation_loss'] = validation_loss_val
                trial_wandb_run.finish(quiet=True)
            return validation_loss_val
        except Exception as e:
            self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number} FAILED: {e}", logging.ERROR)
            if trial_wandb_run:
                trial_wandb_run.summary['status'] = 'failed'
                trial_wandb_run.finish(quiet=True)
            return float('inf') # Prune failed trials
        finally: # Ensure main W&B run is restored if it was active
            if self.main_wandb_run_id and (wandb.run is None or wandb.run.id != self.main_wandb_run_id):
                wandb.init(project=self.main_wandb_project, entity=self.main_wandb_entity,
                           id=self.main_wandb_run_id, resume="must") # Removed quiet=True


    def _quick_training_run(self, config_obj: ResearchConfig) -> Tuple[Any, Any, Dict[str, float]]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        asl_params_for_sim = ASLParameters(T1_artery=config_obj.T1_artery, T_tau=config_obj.T_tau, 
                                   alpha_PCASL=config_obj.alpha_PCASL, alpha_VSASL=config_obj.alpha_VSASL)
        simulator_obj = RealisticASLSimulator(params=asl_params_for_sim)
        plds_numpy_arr = np.array(config_obj.pld_values)
        base_nn_input_size = len(plds_numpy_arr) * 2
        
        # Model config for this trial (passed to EnhancedASLNet and CustomLoss via EnhancedASLTrainer)
        trial_model_config = {
            'hidden_sizes': config_obj.hidden_sizes, 'dropout_rate': config_obj.dropout_rate, 
            'norm_type': config_obj.norm_type,
            'use_transformer_temporal': config_obj.use_transformer_temporal_model,
            'use_focused_transformer': config_obj.use_focused_transformer_model,
            'transformer_d_model': config_obj.transformer_d_model,
            'transformer_d_model_focused': config_obj.transformer_d_model_focused,
            'transformer_nhead': config_obj.transformer_nhead_model,
            'transformer_nlayers': config_obj.transformer_nlayers_model,
            'm0_input_feature': config_obj.m0_input_feature_model,
            'log_var_cbf_min': config_obj.log_var_cbf_min, 'log_var_cbf_max': config_obj.log_var_cbf_max,
            'log_var_att_min': config_obj.log_var_att_min, 'log_var_att_max': config_obj.log_var_att_max,
            'loss_weight_cbf': config_obj.loss_weight_cbf, 
            'loss_weight_att': config_obj.loss_weight_att,
            'loss_log_var_reg_lambda': config_obj.loss_log_var_reg_lambda,
            'n_plds': len(plds_numpy_arr) # Add n_plds to model_config
        }

        def create_hpo_model(**kwargs_from_trainer): # Closure to capture trial_model_config
            # Map config keys to EnhancedASLNet constructor parameter names
            # and select only valid parameters for the model.
            net_params_map = {
                'use_transformer_temporal_model': 'use_transformer_temporal',
                'use_focused_transformer_model': 'use_focused_transformer',
                'transformer_nhead_model': 'transformer_nhead',
                'transformer_nlayers_model': 'transformer_nlayers',
                'm0_input_feature_model': 'm0_input_feature',
            }
            # Parameters that have direct name match or are always needed from config
            ENHANCED_ASL_NET_DIRECT_PARAMS = [
                'hidden_sizes', 'n_plds', 'dropout_rate', 'norm_type',
                'transformer_d_model', 'transformer_d_model_focused',
                'log_var_cbf_min', 'log_var_cbf_max',
                'log_var_att_min', 'log_var_att_max'
            ]
            
            model_specific_kwargs = {}
            # Apply mapping for keys that differ
            for config_key, net_key in net_params_map.items():
                if config_key in kwargs_from_trainer:
                    model_specific_kwargs[net_key] = kwargs_from_trainer[config_key]
            
            # Add directly named parameters
            for param_key in ENHANCED_ASL_NET_DIRECT_PARAMS:
                if param_key in kwargs_from_trainer:
                    model_specific_kwargs[param_key] = kwargs_from_trainer[param_key]
            
            return EnhancedASLNet(input_size=base_nn_input_size, **model_specific_kwargs).to(device)

        trainer_obj = EnhancedASLTrainer(model_config=trial_model_config, 
                                     model_class=create_hpo_model, 
                                     input_size=base_nn_input_size + (1 if config_obj.m0_input_feature_model else 0),
                                     learning_rate=config_obj.learning_rate, batch_size=config_obj.batch_size, 
                                     n_ensembles=config_obj.n_ensembles, device=device,
                                     n_plds_for_model=len(plds_numpy_arr), 
                                     m0_input_feature_model=config_obj.m0_input_feature_model)
        
        train_loaders_list, val_loader_obj, _ = trainer_obj.prepare_curriculum_data( # norm_stats not used in HPO loop
            simulator_obj, n_training_subjects=config_obj.n_training_subjects, plds=plds_numpy_arr,
            curriculum_att_ranges_config=config_obj.att_ranges_config,
            training_conditions_config=config_obj.training_conditions[:1], # Limit conditions for speed
            training_noise_levels_config=config_obj.training_noise_levels[:1], # Limit noise for speed
            n_epochs_for_scheduler=config_obj.training_n_epochs,
            include_m0_in_data=config_obj.include_m0_in_training_data
        )
        if not train_loaders_list:
            self.monitor.log_progress("OPTUNA_RUN", "No training data for HPO trial.", logging.ERROR)
            return None, None, {'val_loss': float('inf')}
        
        history = trainer_obj.train_ensemble(train_loaders_list, val_loader_obj, n_epochs=config_obj.training_n_epochs, early_stopping_patience=5)
        
        final_val_metrics_dict = {'val_loss': float('inf')} # Default
        # Extract validation loss from the first (and only) model in the ensemble for HPO
        if val_loader_obj and history['all_histories'] and 0 in history['all_histories'] and history['all_histories'][0]['val_metrics']:
            # Get the val_loss from the last epoch of the first model's validation metrics
            last_val_epoch_metrics = history['all_histories'][0]['val_metrics'][-1]
            final_val_metrics_dict = last_val_epoch_metrics 
        elif history.get('final_mean_val_loss') is not None and not np.isnan(history['final_mean_val_loss']):
             final_val_metrics_dict['val_loss'] = history['final_mean_val_loss']


        return trainer_obj, simulator_obj, final_val_metrics_dict


    def optimize(self) -> Dict:
        self.monitor.log_progress("OPTUNA", f"Starting HPO: {self.base_config.optuna_n_trials} trials, timeout {self.base_config.optuna_timeout_hours}h.")
        self.study = optuna.create_study(direction='minimize', study_name=self.base_config.optuna_study_name)
        self.study.optimize(self.objective, n_trials=self.base_config.optuna_n_trials, 
                            timeout=self.base_config.optuna_timeout_hours * 3600,
                            gc_after_trial=True) # Enable garbage collection
        
        study_path = self.output_dir / 'optuna_study.pkl'
        joblib.dump(self.study, study_path)
        self.monitor.log_progress("OPTUNA", f"Optuna study saved to {study_path}")
        if wandb.run: wandb.save(str(study_path))

        best_params_dict = self.study.best_params
        self.monitor.log_progress("OPTUNA", f"Best parameters found: {best_params_dict}")
        self.monitor.log_progress("OPTUNA", f"Best validation loss: {self.study.best_value:.6f}")
        if wandb.run:
            wandb.summary['optuna_best_value'] = self.study.best_value
            wandb.summary.update({f"optuna_best_param_{k_par}": v_par for k_par, v_par in best_params_dict.items()})
        return best_params_dict


class ClinicalValidator: # Definition updated to accept norm_stats
    def __init__(self, config: ResearchConfig, monitor: PerformanceMonitor, norm_stats: Optional[Dict] = None):
        self.config = config
        self.monitor = monitor
        asl_params_sim = ASLParameters(
            T1_artery=config.T1_artery, T_tau=config.T_tau, 
            alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL,
            T2_factor=config.T2_factor, alpha_BS1=config.alpha_BS1
        )
        self.simulator = RealisticASLSimulator(params=asl_params_sim)
        self.plds_np = np.array(config.pld_values)
        self.norm_stats = norm_stats 

    def validate_clinical_scenarios(self, trained_nn_models: List[torch.nn.Module]) -> Dict:
        self.monitor.log_progress("CLINICAL_VAL", "Running clinical validation scenarios...")
        all_scenario_results = {}
        pldti = np.column_stack((self.plds_np, self.plds_np))
        num_plds_per_mod = len(self.plds_np)

        for scenario_name, params in self.config.clinical_scenario_definitions.items():
            self.monitor.log_progress("CLINICAL_VAL", f"  Validating {scenario_name}...")
            n_subjects = self.config.n_clinical_scenario_subjects
            true_cbf_vals = np.random.uniform(*params['cbf_range'], n_subjects)
            true_att_vals = np.random.uniform(*params['att_range'], n_subjects)
            current_snr = params['snr']
            
            scenario_metrics_collector = {
                'neural_network': {'cbf_preds': [], 'att_preds': [], 'cbf_uncs': [], 'att_uncs': []},
                'multiverse_ls_single_repeat': {'cbf_preds': [], 'att_preds': []},
                'multiverse_ls_multi_repeat_avg': {'cbf_preds': [], 'att_preds': []}
            }
            for i in range(n_subjects):
                true_cbf, true_att = true_cbf_vals[i], true_att_vals[i]
                
                # Generate single repeat data for NN and LS single repeat
                single_repeat_data_dict = self.simulator.generate_synthetic_data(
                    self.plds_np, np.array([true_att]), n_noise=1, tsnr=current_snr, cbf_val=true_cbf
                )
                # NN input: flat [PCASL_plds, VSASL_plds, M0_if_used]
                nn_input_signal_flat = np.concatenate([
                    single_repeat_data_dict['PCASL'][0,0,:], 
                    single_repeat_data_dict['VSASL'][0,0,:]
                ])
                if self.config.m0_input_feature_model: # Match how data is prepared for training
                    dummy_m0_val = np.array([1.0]) # Consistent dummy M0 for clinical val
                    nn_input_signal_flat = np.concatenate((nn_input_signal_flat, dummy_m0_val))
                
                # LS single repeat input: (n_plds, 2)
                ls_single_repeat_signal_arr = single_repeat_data_dict['MULTIVERSE'][0,0,:,:]

                if trained_nn_models: # NN prediction
                    cbf_nn, att_nn, cbf_std, att_std = self._ensemble_predict(trained_nn_models, nn_input_signal_flat)
                    scenario_metrics_collector['neural_network']['cbf_preds'].append(cbf_nn)
                    scenario_metrics_collector['neural_network']['att_preds'].append(att_nn)
                    scenario_metrics_collector['neural_network']['cbf_uncs'].append(cbf_std)
                    scenario_metrics_collector['neural_network']['att_uncs'].append(att_std)
                else: # Fill with NaNs if no models
                    for k_fill in scenario_metrics_collector['neural_network']: scenario_metrics_collector['neural_network'][k_fill].append(np.nan)

                # LS Single Repeat Fit
                try:
                    beta_ls_sr, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, ls_single_repeat_signal_arr, [50.0/6000.0, 1500.0],
                        self.config.T1_artery, self.config.T_tau, self.config.T2_factor,
                        self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL)
                    scenario_metrics_collector['multiverse_ls_single_repeat']['cbf_preds'].append(beta_ls_sr[0] * 6000.0)
                    scenario_metrics_collector['multiverse_ls_single_repeat']['att_preds'].append(beta_ls_sr[1])
                except Exception: 
                    scenario_metrics_collector['multiverse_ls_single_repeat']['cbf_preds'].append(np.nan)
                    scenario_metrics_collector['multiverse_ls_single_repeat']['att_preds'].append(np.nan)
                
                # LS Multi-Repeat (4x) Averaged Fit
                avg_multi_repeat_signals_collector = []
                # Simulate 4 repeats; effective SNR for averaged signal is current_snr * sqrt(4)
                # The generate_synthetic_data function already handles noise for 1 realization.
                # So, we generate 4 such realizations and average the *signals*, not average *parameters*.
                for _ in range(4): # 4 repeats
                    repeat_data_dict = self.simulator.generate_synthetic_data(
                        self.plds_np, np.array([true_att]), n_noise=1, tsnr=current_snr, cbf_val=true_cbf 
                        # Using same SNR per repeat, averaging signal reduces noise by sqrt(N_repeats)
                    )
                    avg_multi_repeat_signals_collector.append(repeat_data_dict['MULTIVERSE'][0,0,:,:])
                
                avg_multi_repeat_signal_arr = np.mean(avg_multi_repeat_signals_collector, axis=0)
                try:
                    beta_ls_mr, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, avg_multi_repeat_signal_arr, [50.0/6000.0, 1500.0],
                        self.config.T1_artery, self.config.T_tau, self.config.T2_factor,
                        self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL)
                    scenario_metrics_collector['multiverse_ls_multi_repeat_avg']['cbf_preds'].append(beta_ls_mr[0] * 6000.0)
                    scenario_metrics_collector['multiverse_ls_multi_repeat_avg']['att_preds'].append(beta_ls_mr[1])
                except Exception: 
                    scenario_metrics_collector['multiverse_ls_multi_repeat_avg']['cbf_preds'].append(np.nan)
                    scenario_metrics_collector['multiverse_ls_multi_repeat_avg']['att_preds'].append(np.nan)

            all_scenario_results[scenario_name] = {}
            # Use a temporary ComprehensiveComparison instance just for its _calculate_detailed_metrics
            temp_comparator = ComprehensiveComparison() # Default init is fine
            for method_key_str, data_dict_val in scenario_metrics_collector.items():
                cbf_preds_arr, att_preds_arr = np.array(data_dict_val['cbf_preds']), np.array(data_dict_val['att_preds'])
                metrics_summary_dict = temp_comparator._calculate_detailed_metrics(cbf_preds_arr, true_cbf_vals, att_preds_arr, true_att_vals)
                num_valid_fits = np.sum(~np.isnan(cbf_preds_arr) & ~np.isnan(att_preds_arr))
                metrics_summary_dict['success_rate'] = (num_valid_fits / n_subjects) * 100 if n_subjects > 0 else 0
                
                if 'cbf_uncs' in data_dict_val and data_dict_val['cbf_uncs']: # Add uncertainty stats if available
                    metrics_summary_dict['mean_cbf_uncertainty_std'] = np.nanmean(data_dict_val['cbf_uncs'])
                    metrics_summary_dict['mean_att_uncertainty_std'] = np.nanmean(data_dict_val['att_uncs'])
                
                all_scenario_results[scenario_name][method_key_str] = metrics_summary_dict
                self.monitor.log_progress("CLINICAL_VAL", f"  {scenario_name} - {method_key_str}: CBF RMSE {metrics_summary_dict.get('cbf_rmse',np.nan):.2f}, ATT RMSE {metrics_summary_dict.get('att_rmse',np.nan):.2f}, Success {metrics_summary_dict.get('success_rate',0):.1f}%")
                if wandb.run: 
                    for metric_name_val, metric_val in metrics_summary_dict.items():
                        wandb.summary[f"ClinicalVal/{scenario_name}/{method_key_str}/{metric_name_val}"] = metric_val
        return all_scenario_results

    def _ensemble_predict(self, models: List[torch.nn.Module], input_signal_flat: np.ndarray) -> Tuple[float, float, float, float]:
        if not models: return np.nan, np.nan, np.nan, np.nan
        
        num_plds_per_mod = len(self.plds_np)
        normalized_input_signal_flat = input_signal_flat
        if self.norm_stats:
            normalized_input_signal_flat = apply_normalization_to_input_flat(
                input_signal_flat, self.norm_stats, 
                num_plds_per_mod, 
                self.config.m0_input_feature_model 
            )

        input_tensor = torch.FloatTensor(normalized_input_signal_flat).unsqueeze(0).to(next(models[0].parameters()).device)
        
        cbf_means_list_vals, att_means_list_vals = [], []
        cbf_aleatoric_vars_list_vals, att_aleatoric_vars_list_vals = [], []

        for model_item in models:
            model_item.eval()
            with torch.no_grad():
                cbf_m_val, att_m_val, cbf_lv_val, att_lv_val = model_item(input_tensor)
                cbf_means_list_vals.append(cbf_m_val.item()); att_means_list_vals.append(att_m_val.item())
                cbf_aleatoric_vars_list_vals.append(torch.exp(cbf_lv_val).item())
                att_aleatoric_vars_list_vals.append(torch.exp(att_lv_val).item())
        
        ensemble_cbf_m_val = np.mean(cbf_means_list_vals) if cbf_means_list_vals else np.nan
        ensemble_att_m_val = np.mean(att_means_list_vals) if att_means_list_vals else np.nan
        
        mean_aleatoric_cbf_var_val = np.mean(cbf_aleatoric_vars_list_vals) if cbf_aleatoric_vars_list_vals else np.nan
        mean_aleatoric_att_var_val = np.mean(att_aleatoric_vars_list_vals) if att_aleatoric_vars_list_vals else np.nan
        
        epistemic_cbf_var_val = np.var(cbf_means_list_vals) if len(cbf_means_list_vals) > 1 else 0.0
        epistemic_att_var_val = np.var(att_means_list_vals) if len(att_means_list_vals) > 1 else 0.0
        
        total_cbf_var_val = mean_aleatoric_cbf_var_val + epistemic_cbf_var_val
        total_att_var_val = mean_aleatoric_att_var_val + epistemic_att_var_val
        
        total_cbf_std_val = np.sqrt(max(0, total_cbf_var_val)) if not np.isnan(total_cbf_var_val) else np.nan
        total_att_std_val = np.sqrt(max(0, total_att_var_val)) if not np.isnan(total_att_var_val) else np.nan
        
        return ensemble_cbf_m_val, ensemble_att_m_val, total_cbf_std_val, total_att_std_val


class PublicationGenerator: # No major changes needed here based on plan, already creates CSVs
    def __init__(self, config: ResearchConfig, output_dir: Path, monitor: PerformanceMonitor):
        self.config = config; self.output_dir = output_dir; self.monitor = monitor

    def generate_publication_package(self, clinical_results: Dict, comparison_df: pd.DataFrame, single_repeat_val_metrics: Optional[Dict] = None) -> Dict:
        self.monitor.log_progress("PUB_GEN", "Generating publication tables...")
        package = {'tables': {}, 'statistical_analysis': {}} # No figures section anymore

        package['tables']['performance_summary_csv'] = self._generate_performance_table_csv(comparison_df)
        package['tables']['clinical_validation_summary_csv'] = self._generate_clinical_table_csv(clinical_results)
        if single_repeat_val_metrics:
             package['tables']['single_repeat_validation_csv'] = self._generate_single_repeat_table_csv(single_repeat_val_metrics)

        self._save_publication_materials(package) # Save summary of what was generated
        return package

    def _generate_single_repeat_table_csv(self, single_repeat_metrics: Dict) -> str:
        if not single_repeat_metrics:
            self.monitor.log_progress("PUB_GEN", "No single-repeat validation results for table.", logging.WARNING)
            return ""
        
        rows = []
        for method_name, metrics in single_repeat_metrics.items():
            row = {'method': method_name}
            row.update(metrics) # metrics is already a flat dict here from SingleRepeatValidator
            rows.append(row)
        
        df = pd.DataFrame(rows)
        cols_ordered = ['method', 'cbf_rmse', 'att_rmse', 'cbf_bias', 'att_bias', 'cbf_cov', 'att_cov', 
                        'scan_time_minutes', 'efficiency_score', 'num_valid_fits']
        existing_cols = [c for c in cols_ordered if c in df.columns]
        df = df[existing_cols]
        
        table_path = self.output_dir / 'single_repeat_validation_summary.csv'
        df.to_csv(table_path, index=False, float_format='%.3f')
        self.monitor.log_progress("PUB_GEN", f"Saved single-repeat validation summary CSV to {table_path}")
        if wandb.run: wandb.save(str(table_path))
        return str(table_path)

    def _generate_clinical_table_csv(self, clinical_results: Dict) -> str: # Existing
        if not clinical_results: return ""
        rows = []
        for scenario_name, scenario_data in clinical_results.items():
            for method_name, metrics in scenario_data.items():
                row = {'scenario': scenario_name, 'method': method_name}; row.update(metrics)
                rows.append(row)
        df = pd.DataFrame(rows)
        cols_ordered = ['scenario', 'method', 'cbf_rmse', 'att_rmse', 'cbf_bias', 'att_bias', 
                        'cbf_cov', 'att_cov', 'success_rate', 
                        'mean_cbf_uncertainty_std', 'mean_att_uncertainty_std']
        existing_cols = [c for c in cols_ordered if c in df.columns]
        df = df[existing_cols]
        table_path = self.output_dir / 'clinical_validation_summary.csv'
        df.to_csv(table_path, index=False, float_format='%.2f')
        self.monitor.log_progress("PUB_GEN", f"Saved clinical validation summary CSV to {table_path}")
        if wandb.run: wandb.save(str(table_path))
        return str(table_path)

    def _generate_performance_table_csv(self, comparison_df: pd.DataFrame) -> str: # Existing
        if comparison_df.empty: return ""
        cols_to_show = ['method', 'att_range_name', 'cbf_nbias_perc', 'cbf_cov', 'cbf_nrmse_perc', 
                        'att_nbias_perc', 'att_cov', 'att_nrmse_perc', 'success_rate', 'computation_time'
        ]
        existing_cols = [col for col in cols_to_show if col in comparison_df.columns]
        summary_df = comparison_df[existing_cols]
        table_path = self.output_dir / 'benchmark_performance_summary.csv'
        summary_df.to_csv(table_path, index=False, float_format='%.2f')
        self.monitor.log_progress("PUB_GEN", f"Saved benchmark performance summary CSV to {table_path}")
        if wandb.run: wandb.save(str(table_path))
        return str(table_path)

    def _save_publication_materials(self, package: Dict): # Existing
        def make_serializable(obj): 
            if isinstance(obj, Path): return str(obj)
            if isinstance(obj, (np.ndarray, np.generic)): return obj.tolist()
            if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list): return [make_serializable(i) for i in obj]
            return obj
        serializable_package = make_serializable(package)
        fpath = self.output_dir / 'publication_package_summary.json'
        with open(fpath, 'w') as f: json.dump(serializable_package, f, indent=2)
        self.monitor.log_progress("PUB_GEN", f"Saved publication package summary to {fpath}")
        if wandb.run: wandb.save(str(fpath))


def run_comprehensive_asl_research(config: ResearchConfig, output_parent_dir: str = 'comprehensive_results') -> Dict:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_parent_dir) / f'asl_research_{timestamp}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    wandb_run = wandb.init(
        project=config.wandb_project, entity=config.wandb_entity,
        config=asdict(config), name=f"run_{timestamp}", job_type="research_pipeline"
    )
    if wandb_run: script_logger.info(f"W&B Run URL: {wandb_run.url}")

    monitor = PerformanceMonitor(config, output_path) # Monitor now uses this run's output_path
    monitor.log_progress("SETUP", f"Initializing. Output: {output_path}")
    config_save_path = output_path / 'research_config.json'
    with open(config_save_path, 'w') as f: json.dump(asdict(config), f, indent=2)
    if wandb_run: wandb.save(str(config_save_path))

    plds_np = np.array(config.pld_values)
    asl_params_sim = ASLParameters(
        T1_artery=config.T1_artery, T_tau=config.T_tau, 
        alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL,
        T2_factor=config.T2_factor, alpha_BS1=config.alpha_BS1
    )
    simulator = RealisticASLSimulator(params=asl_params_sim)
    norm_stats = None # Initialize norm_stats

    best_optuna_params = {}
    if config.optuna_n_trials > 0:
        monitor.log_progress("PHASE1", "Starting hyperparameter optimization")
        optimizer = HyperparameterOptimizer(config, monitor, output_path) # Pass output_path for saving study
        best_optuna_params = optimizer.optimize()
        if best_optuna_params:
            config.hidden_sizes = [best_optuna_params.get('hidden_size_1', config.hidden_sizes[0]),
                                   best_optuna_params.get('hidden_size_2', config.hidden_sizes[1]),
                                   best_optuna_params.get('hidden_size_3', config.hidden_sizes[2])]
            config.learning_rate = best_optuna_params.get('learning_rate', config.learning_rate)
            config.dropout_rate = best_optuna_params.get('dropout_rate', config.dropout_rate)
            config.batch_size = best_optuna_params.get('batch_size', config.batch_size)
            # Update other optimizable params if added to HPO
            monitor.log_progress("PHASE1", f"Updated config with Optuna best_params: {best_optuna_params}")
            if wandb_run: wandb.config.update(best_optuna_params, allow_val_change=True)
    else: monitor.log_progress("PHASE1", "Skipping hyperparameter optimization.")

    monitor.log_progress("PHASE2", "Starting ensemble training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    monitor.log_progress("PHASE2", f"Using device: {device}")
    
    base_input_size_nn = len(plds_np) * 2 
    model_actual_input_size = base_input_size_nn + (1 if config.m0_input_feature_model else 0)

    model_creation_config = {
        'hidden_sizes': config.hidden_sizes, 'dropout_rate': config.dropout_rate, 'norm_type': config.norm_type,
        'use_transformer_temporal': config.use_transformer_temporal_model,
        'use_focused_transformer': config.use_focused_transformer_model,
        'transformer_d_model': config.transformer_d_model,
        'transformer_d_model_focused': config.transformer_d_model_focused,
        'transformer_nhead': config.transformer_nhead_model,
        'transformer_nlayers': config.transformer_nlayers_model,
        'm0_input_feature': config.m0_input_feature_model,
        'log_var_cbf_min': config.log_var_cbf_min, 'log_var_cbf_max': config.log_var_cbf_max,
        'log_var_att_min': config.log_var_att_min, 'log_var_att_max': config.log_var_att_max,
        'loss_weight_cbf': config.loss_weight_cbf, 
        'loss_weight_att': config.loss_weight_att,
        'loss_log_var_reg_lambda': config.loss_log_var_reg_lambda,
        'n_plds': len(plds_np) # Add n_plds to model_creation_config
    }

    def create_main_model_closure(**kwargs_from_trainer):
        # Map config keys to EnhancedASLNet constructor parameter names
        # and select only valid parameters for the model.
        net_params_map = {
            'use_transformer_temporal_model': 'use_transformer_temporal',
            'use_focused_transformer_model': 'use_focused_transformer',
            'transformer_nhead_model': 'transformer_nhead',
            'transformer_nlayers_model': 'transformer_nlayers',
            'm0_input_feature_model': 'm0_input_feature',
        }
        # Parameters that have direct name match or are always needed from config
        ENHANCED_ASL_NET_DIRECT_PARAMS = [
            'hidden_sizes', 'n_plds', 'dropout_rate', 'norm_type',
            'transformer_d_model', 'transformer_d_model_focused',
            'log_var_cbf_min', 'log_var_cbf_max',
            'log_var_att_min', 'log_var_att_max'
        ]
        
        model_specific_kwargs = {}
        # Apply mapping for keys that differ
        for config_key, net_key in net_params_map.items():
            if config_key in kwargs_from_trainer:
                model_specific_kwargs[net_key] = kwargs_from_trainer[config_key]
        
        # Add directly named parameters
        for param_key in ENHANCED_ASL_NET_DIRECT_PARAMS:
            if param_key in kwargs_from_trainer:
                model_specific_kwargs[param_key] = kwargs_from_trainer[param_key]

        return EnhancedASLNet(input_size=base_input_size_nn, **model_specific_kwargs).to(device)

    trainer = EnhancedASLTrainer(model_config=model_creation_config, 
                                 model_class=create_main_model_closure, 
                                 input_size=model_actual_input_size,
                                 learning_rate=config.learning_rate,
                                 batch_size=config.batch_size, n_ensembles=config.n_ensembles, device=device,
                                 n_plds_for_model=len(plds_np), 
                                 m0_input_feature_model=config.m0_input_feature_model)
    
    monitor.log_progress("PHASE2", "Preparing curriculum training datasets...")
    train_loaders, val_loader, norm_stats = trainer.prepare_curriculum_data(
        simulator, n_training_subjects=config.n_training_subjects, plds=plds_np,
        curriculum_att_ranges_config=config.att_ranges_config,
        training_conditions_config=config.training_conditions,
        training_noise_levels_config=config.training_noise_levels,
        n_epochs_for_scheduler=config.training_n_epochs,
        include_m0_in_data=config.include_m0_in_training_data
    )
    if norm_stats: # Save norm_stats if generated
        norm_stats_path = output_path / 'norm_stats.json'
        serializable_norm_stats = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm_stats.items()}
        with open(norm_stats_path, 'w') as f: json.dump(serializable_norm_stats, f, indent=2)
        monitor.log_progress("PHASE2", f"Normalization stats saved to {norm_stats_path}")
        if wandb_run: wandb.save(str(norm_stats_path))
    else:
        monitor.log_progress("PHASE2", "No normalization stats (e.g. empty training set for stats calc).", logging.WARNING)


    if not train_loaders:
        monitor.log_progress("PHASE2", "Failed to create training loaders. Aborting.", logging.CRITICAL)
        if wandb_run: wandb_run.finish(exit_code=1)
        return {"error": "Training data preparation failed."}
    
    monitor.log_progress("PHASE2", f"Training {config.n_ensembles}-model ensemble for {config.training_n_epochs} epochs per stage...")
    training_start_time = time.time()
    training_histories_dict = trainer.train_ensemble(train_loaders, val_loader, n_epochs=config.training_n_epochs)
    training_duration_hours = (time.time() - training_start_time) / 3600
    monitor.log_progress("PHASE2", f"Training completed in {training_duration_hours:.2f} hours.")
    if wandb_run: wandb.summary['training_duration_hours'] = training_duration_hours

    monitor.log_progress("PHASE3", "Starting clinical validation")
    clinical_validator = ClinicalValidator(config, monitor, norm_stats=norm_stats)
    clinical_validation_results = clinical_validator.validate_clinical_scenarios(trainer.models if trainer.models else [])

    monitor.log_progress("PHASE3.5", "Starting single-repeat vs multi-repeat validation") # Added phase
    # Prepare nn_config_for_model_load for SingleRepeatValidator
    nn_config_srs = model_creation_config.copy() # Start with the full model config
    # SingleRepeatValidator uses this to instantiate the model if path is just state_dict
    
    # Assuming the first model of the ensemble is representative for single-repeat validation
    srs_model_path = None
    if trainer.models:
        temp_srs_model_path = output_path / 'temp_srs_model_0.pt'
        model_state_srs = trainer.best_states[0] if hasattr(trainer, 'best_states') and trainer.best_states and trainer.best_states[0] else trainer.models[0].state_dict()
        if model_state_srs: torch.save(model_state_srs, temp_srs_model_path); srs_model_path = str(temp_srs_model_path)

    _, single_repeat_val_metrics = run_single_repeat_validation_main(
        model_path=srs_model_path,
        base_nn_input_size_for_model_load=base_input_size_nn, # Added
        nn_arch_config_for_model_load=nn_config_srs, # Renamed for clarity
        # num_plds_per_modality_for_norm and m0_feature_for_norm can be derived from nn_config_srs inside
        norm_stats_for_nn=norm_stats # Pass norm_stats
    )
    if srs_model_path and temp_srs_model_path.exists(): temp_srs_model_path.unlink()


    monitor.log_progress("PHASE4", "Benchmarking NN against conventional LS methods")
    # ... (benchmark_test_dataset_raw generation) ...
    total_test_subjects = config.n_test_subjects_per_att_range * len(config.att_ranges_config)
    total_test_subjects = min(total_test_subjects, 2000) # Cap for practical test set size
    
    # Make sure n_subjects for diverse_dataset is reasonable based on conditions/noise_levels
    n_base_subjects_for_benchmark = max(1, total_test_subjects // (len(config.test_conditions) * len(config.test_snr_levels) * 3 if config.test_conditions and config.test_snr_levels else 1))

    benchmark_test_dataset_raw = simulator.generate_diverse_dataset(
        plds=plds_np, n_subjects=n_base_subjects_for_benchmark,
        conditions=config.test_conditions, noise_levels=config.test_snr_levels
    )
    benchmark_X_all_asl, benchmark_y_all = benchmark_test_dataset_raw['signals'], benchmark_test_dataset_raw['parameters']
    
    benchmark_X_all_nn_input = benchmark_X_all_asl # Default if no M0
    if config.m0_input_feature_model: # If NN uses M0, data pipeline for benchmark must also provide it
        # This dummy M0 should ideally be consistent with how training data M0 was handled (if real M0 sim isn't in place)
        m0_for_benchmark = np.random.normal(1.0, 0.1, size=(benchmark_X_all_asl.shape[0], 1)) 
        benchmark_X_all_nn_input = np.concatenate((benchmark_X_all_asl, m0_for_benchmark), axis=1)
    
    benchmark_test_data_for_comp = {
        'PCASL': benchmark_X_all_asl[:, :len(plds_np)], 
        'VSASL': benchmark_X_all_asl[:, len(plds_np):len(plds_np)*2], # Corrected slicing
        'MULTIVERSE_LS_FORMAT': benchmark_X_all_asl.reshape(-1, len(plds_np), 2),
        'NN_INPUT_FORMAT': benchmark_X_all_nn_input
    }

    nn_model_for_comp_path = None
    temp_comp_model_save_path = output_path / 'temp_comp_model_0.pt'
    if trainer.models: # Save the first model of the ensemble for ComprehensiveComparison
        model_state_to_save_comp = trainer.best_states[0] if hasattr(trainer, 'best_states') and trainer.best_states and trainer.best_states[0] else trainer.models[0].state_dict()
        if model_state_to_save_comp: torch.save(model_state_to_save_comp, temp_comp_model_save_path)
        nn_model_for_comp_path = str(temp_comp_model_save_path)

    comp_framework_output_dir = output_path / "comparison_framework_outputs"
    comp_framework = ComprehensiveComparison(
        nn_model_path=nn_model_for_comp_path,
        output_dir=str(comp_framework_output_dir), # Ensure it's string
        base_nn_input_size=base_input_size_nn, # Pass base_nn_input_size
        nn_n_plds=len(plds_np),
        nn_m0_input_feature=config.m0_input_feature_model,
        nn_model_arch_config=model_creation_config, 
        norm_stats=norm_stats
    )
    comparison_results_df = comp_framework.compare_methods(benchmark_test_data_for_comp, benchmark_y_all, plds_np, config.att_ranges_config)
    if nn_model_for_comp_path and temp_comp_model_save_path.exists(): temp_comp_model_save_path.unlink() # Clean up temp model
    
    if not comparison_results_df.empty and wandb_run:
        benchmark_table_path_wandb = comp_framework_output_dir / 'comparison_results_detailed.csv'
        if benchmark_table_path_wandb.exists(): wandb.save(str(benchmark_table_path_wandb))

    nn_benchmark_metrics_for_monitor, baseline_ls_metrics_for_monitor = {}, {}
    if not comparison_results_df.empty:
        for att_cfg_item in config.att_ranges_config: # Iterate using config for consistency
            range_name_str_cfg = att_cfg_item[2]
            nn_row_df = comparison_results_df[(comparison_results_df['method'] == 'Neural Network') & (comparison_results_df['att_range_name'] == range_name_str_cfg)]
            if not nn_row_df.empty: nn_benchmark_metrics_for_monitor[range_name_str_cfg] = nn_row_df.iloc[0].to_dict()
            
            ls_row_df = comparison_results_df[(comparison_results_df['method'] == 'MULTIVERSE-LS') & (comparison_results_df['att_range_name'] == range_name_str_cfg)]
            if not ls_row_df.empty: baseline_ls_metrics_for_monitor[range_name_str_cfg] = ls_row_df.iloc[0].to_dict()
        
        monitor.check_target_achievement(nn_benchmark_metrics_for_monitor, baseline_ls_metrics_for_monitor)

    monitor.log_progress("PHASE5", "Generating publication-ready materials (tables only)")
    pub_gen = PublicationGenerator(config, output_path, monitor)
    publication_package = pub_gen.generate_publication_package(
        clinical_validation_results, comparison_results_df, single_repeat_val_metrics
    )

    monitor.log_progress("PHASE6", "Generating comprehensive research summary")
    models_dir = output_path / 'trained_models'; models_dir.mkdir(exist_ok=True)
    if trainer.models:
        for i, model_state_val in enumerate(trainer.best_states if hasattr(trainer, 'best_states') and trainer.best_states and any(trainer.best_states) else [m.state_dict() for m in trainer.models]):
            model_file_path_val = models_dir / f'ensemble_model_{i}_best.pt' # Prefer saving best
            if model_state_val: 
                torch.save(model_state_val, model_file_path_val)
            elif trainer.models and trainer.models[i]: # Fallback to final if no best_state
                model_file_path_val = models_dir / f'ensemble_model_{i}_final.pt'
                torch.save(trainer.models[i].state_dict(), model_file_path_val)
            
            if wandb_run and model_file_path_val.exists(): wandb.save(str(model_file_path_val))

    final_results_summary = {
        'config': asdict(config), 'optuna_best_params': best_optuna_params,
        'optuna_study_path': str(output_path / 'optuna_study.pkl') if config.optuna_n_trials > 0 and (output_path / 'optuna_study.pkl').exists() else None,
        'norm_stats_path': str(norm_stats_path) if norm_stats and norm_stats_path.exists() else None,
        'training_duration_hours': training_duration_hours, 
        'training_histories_metrics': training_histories_dict.get('all_histories', None),
        'clinical_validation_results': clinical_validation_results,
        'single_repeat_validation_metrics': single_repeat_val_metrics,
        'benchmark_comparison_results_csv_path': str(comp_framework_output_dir / 'comparison_results_detailed.csv') if not comparison_results_df.empty and (comp_framework_output_dir / 'comparison_results_detailed.csv').exists() else None,
        'publication_package_summary_path': str(output_path / 'publication_package_summary.json') if (output_path / 'publication_package_summary.json').exists() else None,
        'trained_models_dir': str(models_dir),
        'wandb_run_url': wandb_run.url if wandb_run else None
    }
    with open(output_path / 'final_research_results.json', 'w') as f: 
        json.dump(final_results_summary, f, indent=2, default=lambda o: f"<not_serializable_{type(o).__name__}>")
    if wandb_run: wandb.save(str(output_path / 'final_research_results.json'))
    
    summary_report_path_txt = output_path / 'RESEARCH_SUMMARY.txt'
    with open(summary_report_path_txt, 'w') as f:
        f.write(f"Research pipeline completed. Full summary in final_research_results.json and log file at {output_path / 'research.log'}\n")
        f.write(f"Output directory: {output_path}\n")
        if best_optuna_params: f.write(f"Optuna Best Params: {best_optuna_params}\n")
        if norm_stats: f.write(f"Normalization stats saved to: {norm_stats_path}\n")
        f.write(f"Training Duration: {training_duration_hours:.2f} hours\n")
        if wandb_run: f.write(f"W&B Run: {wandb_run.url}\n")
    if wandb_run: wandb.save(str(summary_report_path_txt))

    monitor.log_progress("COMPLETE", f"Research pipeline finished. Results in {output_path}")
    if wandb_run: wandb_run.finish()
    return final_results_summary

if __name__ == "__main__":
    # Setup root logger once
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        handlers=[logging.StreamHandler(sys.stdout)], force=True) # force=True if re-running in same session
    
    script_logger.info("=" * 80 + "\nASL NEURAL NETWORK COMPREHENSIVE RESEARCH PIPELINE - Enhanced\n" + "=" * 80)
    
    config_file_path_arg = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    loaded_config_obj = ResearchConfig() # Start with defaults

    if Path(config_file_path_arg).exists():
        script_logger.info(f"Loading configuration from {config_file_path_arg}")
        try:
            with open(config_file_path_arg, 'r') as f_yaml:
                config_dict_from_yaml = yaml.safe_load(f_yaml)
            
            # Smartly update dataclass fields from YAML
            # Iterate over keys in the loaded YAML to update dataclass fields
            for key_yaml, val_yaml in config_dict_from_yaml.items():
                if hasattr(loaded_config_obj, key_yaml): # If it's a direct attribute
                    setattr(loaded_config_obj, key_yaml, val_yaml)
                else: # Check for nested structures (e.g., training.batch_size in YAML)
                      # The current ResearchConfig is flat for most of these, so direct assignment often works.
                      # This part is for if YAML has sections like 'training:' that map to multiple flat attrs.
                    if isinstance(val_yaml, dict): # If YAML has a section e.g. training: batch_size: ...
                        for sub_key_yaml, sub_val_yaml in val_yaml.items():
                            if hasattr(loaded_config_obj, sub_key_yaml): # If sub_key is a direct attr
                                setattr(loaded_config_obj, sub_key_yaml, sub_val_yaml)
            script_logger.info(f"Successfully loaded and merged config from {config_file_path_arg}")

        except Exception as e_cfg: 
            script_logger.error(f"Error loading config {config_file_path_arg}: {e_cfg}. Using defaults or partially loaded config.")
    else: 
        script_logger.info(f"Config file {config_file_path_arg} not found. Using default ResearchConfig.")
    
    script_logger.info("\nResearch Configuration:\n" + "-" * 30 + "\n" + "\n".join([f"{k_cfg_disp}: {v_cfg_disp}" for k_cfg_disp,v_cfg_disp in asdict(loaded_config_obj).items()]) + "\n" + "-" * 30)
    script_logger.info("\nStarting comprehensive ASL research pipeline...")
    
    pipeline_results_dict = run_comprehensive_asl_research(config=loaded_config_obj)
    
    script_logger.info("\n" + "=" * 80 + "\nRESEARCH PIPELINE COMPLETED!\n" + "=" * 80)
    if "error" not in pipeline_results_dict:
        script_logger.info(f"Results saved in: {pipeline_results_dict.get('trained_models_dir', 'Specified output directory')}")
        script_logger.info("Check RESEARCH_SUMMARY.txt and final_research_results.json for detailed findings.")
        if pipeline_results_dict.get('wandb_run_url'):
            script_logger.info(f"W&B Run: {pipeline_results_dict['wandb_run_url']}")
    else: 
        script_logger.error(f"Pipeline failed: {pipeline_results_dict.get('error')}")
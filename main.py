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
from tqdm import tqdm
import optuna
from dataclasses import dataclass, asdict, field
import sys
import warnings
import wandb 
import joblib 
import math
import inspect

warnings.filterwarnings('ignore', category=UserWarning)

from enhanced_asl_network import EnhancedASLNet, CustomLoss
from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer, EnhancedASLDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from comparison_framework import ComprehensiveComparison, ComparisonResult
from performance_metrics import ProposalEvaluator
from single_repeat_validation import SingleRepeatValidator, run_single_repeat_validation_main

from vsasl_functions import fit_VSASL_vectInit_pep
from pcasl_functions import fit_PCASL_vectInit_pep
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

script_logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    # Training parameters
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 256
    val_split: float = 0.2
    
    # NEW: Two-stage curriculum parameters
    n_subjects_stage1: int = 5000
    n_subjects_stage2: int = 10000
    n_epochs_stage1: int = 140
    n_epochs_stage2: int = 60
    loss_pinn_weight_stage1: float = 10.0
    loss_pinn_weight_stage2: float = 0.1
    learning_rate_stage2: float = 0.0002
    
    n_ensembles: int = 5
    dropout_rate: float = 0.1
    norm_type: str = 'batch'
    
    m0_input_feature_model: bool = False

    use_transformer_temporal_model: bool = True
    use_focused_transformer_model: bool = True
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

def engineer_signal_features(raw_signal: np.ndarray, num_plds: int) -> np.ndarray:
    """
    Engineers explicit shape-based features from raw ASL signal curves.
    """
    num_samples = raw_signal.shape[0]
    engineered_features = np.zeros((num_samples, 4))
    plds_indices = np.arange(num_plds)

    for i in range(num_samples):
        pcasl_curve = raw_signal[i, :num_plds]
        vsasl_curve = raw_signal[i, num_plds:]

        engineered_features[i, 0] = np.argmax(pcasl_curve)
        engineered_features[i, 1] = np.argmax(vsasl_curve)

        pcasl_sum = np.sum(pcasl_curve) + 1e-6
        vsasl_sum = np.sum(vsasl_curve) + 1e-6
        engineered_features[i, 2] = np.sum(pcasl_curve * plds_indices) / pcasl_sum
        engineered_features[i, 3] = np.sum(vsasl_curve * plds_indices) / vsasl_sum

    return engineered_features

def apply_normalization_to_input_flat(flat_signal: np.ndarray, 
                                      norm_stats: Dict, 
                                      num_plds_per_modality: int, 
                                      has_m0: bool) -> np.ndarray:
    if not norm_stats or not isinstance(norm_stats, dict): return flat_signal
    raw_signal_len = num_plds_per_modality * 2
    signal_part = flat_signal[:raw_signal_len]
    other_features_part = flat_signal[raw_signal_len:]

    pcasl_norm = (signal_part[:num_plds_per_modality] - norm_stats.get('pcasl_mean', 0)) / np.clip(norm_stats.get('pcasl_std', 1), a_min=1e-6, a_max=None)
    vsasl_norm = (signal_part[num_plds_per_modality:] - norm_stats.get('vsasl_mean', 0)) / np.clip(norm_stats.get('vsasl_std', 1), a_min=1e-6, a_max=None)

    return np.concatenate([pcasl_norm, vsasl_norm, other_features_part])


class PerformanceMonitor:
    def __init__(self, config: ResearchConfig, output_dir: Path):
        self.config = config; self.output_dir = output_dir
        self.logger = logging.getLogger("ASLResearchPipeline")
        for handler in self.logger.handlers[:]: self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(output_dir / 'research.log', mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(sh)
        self.logger.propagate = False

    def log_progress(self, phase: str, message: str, level: int = logging.INFO):
        self.logger.log(level, f"[{phase}] {message}")
        if wandb.run: wandb.log({f"Progress/{phase.replace(' ', '_')}": message}, step=wandb.run.step if wandb.run.step is not None else 0)

    def check_target_achievement(self, nn_att_range_results: Dict, baseline_att_range_results: Dict) -> Dict:
        achievements = {}
        for att_range_name_key in nn_att_range_results.keys():
            if att_range_name_key not in baseline_att_range_results:
                self.log_progress("TARGET_CHECK", f"Baseline results missing for {att_range_name_key}", logging.WARNING)
                continue
            nn_metrics_dict, baseline_metrics_dict = nn_att_range_results[att_range_name_key], baseline_att_range_results[att_range_name_key]
            current_cbf_cv_val = nn_metrics_dict.get('cbf_cov', float('inf')); baseline_cbf_cv_val = baseline_metrics_dict.get('cbf_cov', float('inf'))
            cbf_improvement_val = ((baseline_cbf_cv_val - current_cbf_cv_val) / baseline_cbf_cv_val) * 100 if baseline_cbf_cv_val > 0 and not np.isinf(baseline_cbf_cv_val) and not np.isinf(current_cbf_cv_val) else 0.0
            current_att_cv_val = nn_metrics_dict.get('att_cov', float('inf')); baseline_att_cv_val = baseline_metrics_dict.get('att_cov', float('inf'))
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
        self.base_config = base_config; self.monitor = monitor; self.output_dir = output_dir; self.study = None
        self.main_wandb_run_id = wandb.run.id if wandb.run else None
        self.main_wandb_project = wandb.run.project if wandb.run else self.base_config.wandb_project
        self.main_wandb_entity = wandb.run.entity if wandb.run else self.base_config.wandb_entity

    def objective(self, trial: optuna.Trial) -> float:
        hidden_size_1 = trial.suggest_categorical('hidden_size_1', [128, 256, 512]); hidden_size_2 = trial.suggest_categorical('hidden_size_2', [64, 128, 256])
        hidden_size_3 = trial.suggest_categorical('hidden_size_3', [32, 64, 128]); learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3); batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        trial_config_dict = asdict(self.base_config) 
        trial_optuna_params = {'hidden_sizes': [hidden_size_1, hidden_size_2, hidden_size_3], 'learning_rate': learning_rate, 'dropout_rate': dropout_rate, 'batch_size': batch_size, 'weight_decay': weight_decay}
        trial_config_dict.update(trial_optuna_params)
        trial_config_dict.update({'n_subjects_stage2': self.base_config.optuna_n_subjects, 'n_epochs_stage1': self.base_config.optuna_n_epochs, 'n_ensembles': 1})
        trial_run_config = ResearchConfig(**trial_config_dict)
        self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number}: Params {trial.params}")
        if self.main_wandb_run_id and wandb.run and wandb.run.id == self.main_wandb_run_id: wandb.finish(quiet=True) 
        trial_wandb_run = wandb.init(project=self.main_wandb_project, entity=self.main_wandb_entity, group=self.base_config.optuna_study_name, name=f"trial_{trial.number}_{wandb.util.generate_id()[:4]}", config=trial_optuna_params, reinit=True, job_type="hpo_trial")
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
            return float('inf') 
        finally: 
            if self.main_wandb_run_id and (wandb.run is None or wandb.run.id != self.main_wandb_run_id):
                wandb.init(project=self.main_wandb_project, entity=self.main_wandb_entity, id=self.main_wandb_run_id, resume="must") 

    def _quick_training_run(self, config_obj: ResearchConfig) -> Tuple[Any, Any, Dict[str, float]]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        asl_params_for_sim = ASLParameters(T1_artery=config_obj.T1_artery, T_tau=config_obj.T_tau, alpha_PCASL=config_obj.alpha_PCASL, alpha_VSASL=config_obj.alpha_VSASL)
        simulator_obj = RealisticASLSimulator(params=asl_params_for_sim)
        plds_numpy_arr = np.array(config_obj.pld_values); num_plds = len(plds_numpy_arr)
        precomputed_hpo_data = simulator_obj.generate_balanced_dataset(plds=plds_numpy_arr, total_subjects=config_obj.n_training_subjects, noise_levels=config_obj.training_noise_levels[:1])
        engineered_features_hpo = engineer_signal_features(precomputed_hpo_data['signals'], num_plds)
        precomputed_hpo_data['signals'] = np.concatenate([precomputed_hpo_data['signals'], engineered_features_hpo], axis=1)
        base_nn_input_size = precomputed_hpo_data['signals'].shape[1]

        trial_model_config = {k: v for k, v in asdict(config_obj).items() if hasattr(EnhancedASLNet, k) or hasattr(CustomLoss, k) or k in ['pld_values', 'T1_artery', 'T_tau', 'alpha_PCASL', 'alpha_VSASL']}
        
        def create_hpo_model(**kwargs):
            model_param_keys = inspect.signature(EnhancedASLNet).parameters.keys()
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in model_param_keys}
            return EnhancedASLNet(input_size=base_nn_input_size, **filtered_kwargs)

        trainer_obj = EnhancedASLTrainer(model_config=trial_model_config, model_class=create_hpo_model, input_size=base_nn_input_size, learning_rate=config_obj.learning_rate, weight_decay=config_obj.weight_decay, batch_size=config_obj.batch_size, n_ensembles=config_obj.n_ensembles, device=device, n_plds_for_model=num_plds, m0_input_feature_model=config_obj.m0_input_feature_model)
        train_loaders_list, val_loaders_list, _ = trainer_obj.prepare_curriculum_data(simulator_obj, plds=plds_numpy_arr, precomputed_dataset=precomputed_hpo_data, curriculum_att_ranges_config=config_obj.att_ranges_config, n_epochs_for_scheduler=config_obj.optuna_n_epochs)
        if not train_loaders_list: self.monitor.log_progress("OPTUNA_RUN", "No training data for HPO trial.", logging.ERROR); return None, None, {'val_loss': float('inf')}
        first_val_loader_for_hpo = next((vl for vl in val_loaders_list if vl and len(vl) > 0), val_loaders_list[0] if val_loaders_list else None)
        history = trainer_obj.train_ensemble([train_loaders_list[0]] if train_loaders_list else [], [first_val_loader_for_hpo] if first_val_loader_for_hpo else [], epoch_schedule=[config_obj.optuna_n_epochs], early_stopping_patience=5)
        final_val_loss = history.get('final_mean_val_loss', float('inf'))
        if np.isnan(final_val_loss): final_val_loss = float('inf')
        return trainer_obj, simulator_obj, {'val_loss': final_val_loss}

    def optimize(self) -> Dict:
        self.monitor.log_progress("OPTUNA", f"Starting HPO: {self.base_config.optuna_n_trials} trials, timeout {self.base_config.optuna_timeout_hours}h.")
        self.study = optuna.create_study(direction='minimize', study_name=self.base_config.optuna_study_name)
        self.study.optimize(self.objective, n_trials=self.base_config.optuna_n_trials, timeout=self.base_config.optuna_timeout_hours * 3600, gc_after_trial=True) 
        study_path = self.output_dir / 'optuna_study.pkl'; joblib.dump(self.study, study_path)
        self.monitor.log_progress("OPTUNA", f"Optuna study saved to {study_path}")
        if wandb.run: wandb.save(str(study_path))
        best_params_dict = self.study.best_params
        self.monitor.log_progress("OPTUNA", f"Best parameters found: {best_params_dict}")
        self.monitor.log_progress("OPTUNA", f"Best validation loss: {self.study.best_value:.6f}")
        if wandb.run:
            wandb.summary['optuna_best_value'] = self.study.best_value
            wandb.summary.update({f"optuna_best_param_{k_par}": v_par for k_par, v_par in best_params_dict.items()})
        return best_params_dict


class ClinicalValidator: 
    def __init__(self, config: ResearchConfig, monitor: PerformanceMonitor, norm_stats: Optional[Dict] = None):
        self.config = config; self.monitor = monitor
        asl_params_sim = ASLParameters(T1_artery=config.T1_artery, T_tau=config.T_tau, alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL, T2_factor=config.T2_factor, alpha_BS1=config.alpha_BS1)
        self.simulator = RealisticASLSimulator(params=asl_params_sim)
        self.plds_np = np.array(config.pld_values)
        self.norm_stats = norm_stats 

    def validate_clinical_scenarios(self, trained_nn_models: List[torch.nn.Module]) -> Dict:
        self.monitor.log_progress("CLINICAL_VAL", "Running clinical validation scenarios...")
        all_scenario_results = {}; pldti = np.column_stack((self.plds_np, self.plds_np)); num_plds_per_mod = len(self.plds_np)
        for scenario_name, params in self.config.clinical_scenario_definitions.items():
            self.monitor.log_progress("CLINICAL_VAL", f"  Validating {scenario_name}...")
            n_subjects = self.config.n_clinical_scenario_subjects
            true_cbf_vals = np.random.uniform(*params['cbf_range'], n_subjects); true_att_vals = np.random.uniform(*params['att_range'], n_subjects); current_snr = params['snr']
            
            # MODIFICATION START: Add a key for the new method
            scenario_metrics_collector = {
                'neural_network': {'cbf_preds': [], 'att_preds': [], 'cbf_uncs': [], 'att_uncs': []},
                'multiverse_ls_single_repeat': {'cbf_preds': [], 'att_preds': []},
                'multiverse_ls_multi_repeat_avg': {'cbf_preds': [], 'att_preds': []},
                'neural_network_multi_repeat_avg': {'cbf_preds': [], 'att_preds': [], 'cbf_uncs': [], 'att_uncs': []}
            }
            # MODIFICATION END

            for i in range(n_subjects):
                true_cbf, true_att = true_cbf_vals[i], true_att_vals[i]
                single_repeat_data_dict = self.simulator.generate_synthetic_data(self.plds_np, np.array([true_att]), n_noise=1, tsnr=current_snr, cbf_val=true_cbf)
                raw_nn_signal = np.concatenate([single_repeat_data_dict['PCASL'][0,0,:], single_repeat_data_dict['VSASL'][0,0,:]])
                engineered_features = engineer_signal_features(raw_nn_signal.reshape(1,-1), num_plds_per_mod)
                nn_input_signal_flat = np.concatenate([raw_nn_signal, engineered_features.flatten()])
                ls_single_repeat_signal_arr = single_repeat_data_dict['MULTIVERSE'][0,0,:,:]
                if trained_nn_models: 
                    cbf_nn, att_nn, cbf_std, att_std = self._ensemble_predict(trained_nn_models, nn_input_signal_flat)
                    scenario_metrics_collector['neural_network']['cbf_preds'].append(cbf_nn); scenario_metrics_collector['neural_network']['att_preds'].append(att_nn)
                    scenario_metrics_collector['neural_network']['cbf_uncs'].append(cbf_std); scenario_metrics_collector['neural_network']['att_uncs'].append(att_std)
                else: 
                    for k_fill in scenario_metrics_collector['neural_network']: scenario_metrics_collector['neural_network'][k_fill].append(np.nan)
                try:
                    beta_ls_sr, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, ls_single_repeat_signal_arr, [50.0/6000.0, 1500.0], self.config.T1_artery, self.config.T_tau, self.config.T2_factor, self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL)
                    scenario_metrics_collector['multiverse_ls_single_repeat']['cbf_preds'].append(beta_ls_sr[0] * 6000.0); scenario_metrics_collector['multiverse_ls_single_repeat']['att_preds'].append(beta_ls_sr[1])
                except Exception: scenario_metrics_collector['multiverse_ls_single_repeat']['cbf_preds'].append(np.nan); scenario_metrics_collector['multiverse_ls_single_repeat']['att_preds'].append(np.nan)
                avg_multi_repeat_signals_collector = []
                for _ in range(4): 
                    repeat_data_dict = self.simulator.generate_synthetic_data(self.plds_np, np.array([true_att]), n_noise=1, tsnr=current_snr, cbf_val=true_cbf)
                    avg_multi_repeat_signals_collector.append(repeat_data_dict['MULTIVERSE'][0,0,:,:])
                avg_multi_repeat_signal_arr = np.mean(avg_multi_repeat_signals_collector, axis=0)
                try:
                    beta_ls_mr, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, avg_multi_repeat_signal_arr, [50.0/6000.0, 1500.0], self.config.T1_artery, self.config.T_tau, self.config.T2_factor, self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL)
                    scenario_metrics_collector['multiverse_ls_multi_repeat_avg']['cbf_preds'].append(beta_ls_mr[0] * 6000.0); scenario_metrics_collector['multiverse_ls_multi_repeat_avg']['att_preds'].append(beta_ls_mr[1])
                except Exception: scenario_metrics_collector['multiverse_ls_multi_repeat_avg']['cbf_preds'].append(np.nan); scenario_metrics_collector['multiverse_ls_multi_repeat_avg']['att_preds'].append(np.nan)

                # MODIFICATION START: Add logic for averaged NN prediction
                if trained_nn_models:
                    # Average the PCASL and VSASL components separately to create the flat input vector for the NN
                    avg_pcasl_signals = np.mean([sig[:, 0] for sig in avg_multi_repeat_signals_collector], axis=0)
                    avg_vsasl_signals = np.mean([sig[:, 1] for sig in avg_multi_repeat_signals_collector], axis=0)
                    
                    raw_nn_signal_avg = np.concatenate([avg_pcasl_signals, avg_vsasl_signals])
                    engineered_features_avg = engineer_signal_features(raw_nn_signal_avg.reshape(1,-1), num_plds_per_mod)
                    nn_input_signal_flat_avg = np.concatenate([raw_nn_signal_avg, engineered_features_avg.flatten()])
                
                    # Predict with the NN using the averaged signal
                    cbf_nn_avg, att_nn_avg, cbf_std_avg, att_std_avg = self._ensemble_predict(trained_nn_models, nn_input_signal_flat_avg)
                    scenario_metrics_collector['neural_network_multi_repeat_avg']['cbf_preds'].append(cbf_nn_avg)
                    scenario_metrics_collector['neural_network_multi_repeat_avg']['att_preds'].append(att_nn_avg)
                    scenario_metrics_collector['neural_network_multi_repeat_avg']['cbf_uncs'].append(cbf_std_avg)
                    scenario_metrics_collector['neural_network_multi_repeat_avg']['att_uncs'].append(att_std_avg)
                else:
                    # Append NaN if no model is available
                    for k_fill in scenario_metrics_collector['neural_network_multi_repeat_avg']: 
                        scenario_metrics_collector['neural_network_multi_repeat_avg'][k_fill].append(np.nan)
                # MODIFICATION END

            all_scenario_results[scenario_name] = {}
            temp_comparator = ComprehensiveComparison() 
            for method_key_str, data_dict_val in scenario_metrics_collector.items():
                cbf_preds_arr, att_preds_arr = np.array(data_dict_val['cbf_preds']), np.array(data_dict_val['att_preds'])
                metrics_summary_dict = temp_comparator._calculate_detailed_metrics(cbf_preds_arr, true_cbf_vals, att_preds_arr, true_att_vals)
                num_valid_fits = np.sum(~np.isnan(cbf_preds_arr) & ~np.isnan(att_preds_arr))
                metrics_summary_dict['success_rate'] = (num_valid_fits / n_subjects) * 100 if n_subjects > 0 else 0
                if 'cbf_uncs' in data_dict_val and data_dict_val['cbf_uncs']: 
                    metrics_summary_dict['mean_cbf_uncertainty_std'] = np.nanmean(data_dict_val['cbf_uncs']); metrics_summary_dict['mean_att_uncertainty_std'] = np.nanmean(data_dict_val['att_uncs'])
                all_scenario_results[scenario_name][method_key_str] = metrics_summary_dict
                self.monitor.log_progress("CLINICAL_VAL", f"  {scenario_name} - {method_key_str}: CBF RMSE {metrics_summary_dict.get('cbf_rmse',np.nan):.2f}, ATT RMSE {metrics_summary_dict.get('att_rmse',np.nan):.2f}, Success {metrics_summary_dict.get('success_rate',0):.1f}%")
                if wandb.run: 
                    for metric_name_val, metric_val in metrics_summary_dict.items(): wandb.summary[f"ClinicalVal/{scenario_name}/{method_key_str}/{metric_name_val}"] = metric_val
        return all_scenario_results

    def _ensemble_predict(self, models: List[torch.nn.Module], input_signal_flat: np.ndarray) -> Tuple[float, float, float, float]:
        if not models: return np.nan, np.nan, np.nan, np.nan
        normalized_input_signal = apply_normalization_to_input_flat(input_signal_flat, self.norm_stats, len(self.plds_np), False)
        input_tensor = torch.FloatTensor(normalized_input_signal).unsqueeze(0).to(next(models[0].parameters()).device)
        cbf_means_norm_list, att_means_norm_list, cbf_aleatoric_vars_norm_list, att_aleatoric_vars_norm_list = [], [], [], []
        for model_item in models:
            model_item.eval()
            with torch.no_grad():
                cbf_m_norm, att_m_norm, cbf_lv_norm, att_lv_norm = model_item(input_tensor)
                cbf_means_norm_list.append(cbf_m_norm.item()); att_means_norm_list.append(att_m_norm.item())
                cbf_aleatoric_vars_norm_list.append(torch.exp(cbf_lv_norm).item()); att_aleatoric_vars_norm_list.append(torch.exp(att_lv_norm).item())
        ensemble_cbf_m_norm = np.mean(cbf_means_norm_list) if cbf_means_norm_list else np.nan; ensemble_att_m_norm = np.mean(att_means_norm_list) if att_means_norm_list else np.nan
        mean_aleatoric_cbf_var_norm = np.mean(cbf_aleatoric_vars_norm_list) if cbf_aleatoric_vars_norm_list else np.nan; mean_aleatoric_att_var_norm = np.mean(att_aleatoric_vars_norm_list) if att_aleatoric_vars_norm_list else np.nan
        epistemic_cbf_var_norm = np.var(cbf_means_norm_list) if len(cbf_means_norm_list) > 1 else 0.0; epistemic_att_var_norm = np.var(att_means_norm_list) if len(att_means_norm_list) > 1 else 0.0
        y_mean_cbf, y_std_cbf, y_mean_att, y_std_att = 0.0, 1.0, 0.0, 1.0
        if self.norm_stats:
            y_mean_cbf = self.norm_stats.get('y_mean_cbf', 0.0); y_std_cbf = self.norm_stats.get('y_std_cbf', 1.0) if self.norm_stats.get('y_std_cbf', 1.0) > 1e-6 else 1.0
            y_mean_att = self.norm_stats.get('y_mean_att', 0.0); y_std_att = self.norm_stats.get('y_std_att', 1.0) if self.norm_stats.get('y_std_att', 1.0) > 1e-6 else 1.0
        ensemble_cbf_m_denorm = ensemble_cbf_m_norm * y_std_cbf + y_mean_cbf; ensemble_att_m_denorm = ensemble_att_m_norm * y_std_att + y_mean_att
        total_cbf_var_norm = mean_aleatoric_cbf_var_norm + epistemic_cbf_var_norm; total_att_var_norm = mean_aleatoric_att_var_norm + epistemic_att_var_norm
        total_cbf_var_denorm = total_cbf_var_norm * (y_std_cbf**2); total_att_var_denorm = total_att_var_norm * (y_std_att**2)
        total_cbf_std_denorm = np.sqrt(max(0, total_cbf_var_denorm)) if not np.isnan(total_cbf_var_denorm) else np.nan; total_att_std_denorm = np.sqrt(max(0, total_att_var_denorm)) if not np.isnan(total_att_var_denorm) else np.nan
        return ensemble_cbf_m_denorm, ensemble_att_m_denorm, total_cbf_std_denorm, total_att_std_denorm


class PublicationGenerator: 
    def __init__(self, config: ResearchConfig, output_dir: Path, monitor: PerformanceMonitor):
        self.config = config; self.output_dir = output_dir; self.monitor = monitor

    def generate_publication_package(self, clinical_results: Dict, comparison_df: pd.DataFrame, single_repeat_val_metrics: Optional[Dict] = None) -> Dict:
        self.monitor.log_progress("PUB_GEN", "Generating publication tables...")
        package = {'tables': {}, 'statistical_analysis': {}} 
        package['tables']['performance_summary_csv'] = self._generate_performance_table_csv(comparison_df)
        package['tables']['clinical_validation_summary_csv'] = self._generate_clinical_table_csv(clinical_results)
        if single_repeat_val_metrics: package['tables']['single_repeat_validation_csv'] = self._generate_single_repeat_table_csv(single_repeat_val_metrics)
        self._save_publication_materials(package); return package

    def _generate_single_repeat_table_csv(self, single_repeat_metrics: Dict) -> str:
        if not single_repeat_metrics: self.monitor.log_progress("PUB_GEN", "No single-repeat validation results for table.", logging.WARNING); return ""
        rows = []; [row.update(metrics) for method_name, metrics in single_repeat_metrics.items() if (row := {'method': method_name}) and rows.append(row)]
        df = pd.DataFrame(rows); cols_ordered = ['method', 'cbf_rmse', 'att_rmse', 'cbf_bias', 'att_bias', 'cbf_cov', 'att_cov', 'scan_time_minutes', 'efficiency_score', 'num_valid_fits']
        existing_cols = [c for c in cols_ordered if c in df.columns]; df = df[existing_cols]
        table_path = self.output_dir / 'single_repeat_validation_summary.csv'
        df.to_csv(table_path, index=False, float_format='%.3f')
        self.monitor.log_progress("PUB_GEN", f"Saved single-repeat validation summary CSV to {table_path}")
        if wandb.run: wandb.save(str(table_path)); return str(table_path)

    def _generate_clinical_table_csv(self, clinical_results: Dict) -> str: 
        if not clinical_results: return ""
        rows = []; [row.update(metrics) for scenario_name, scenario_data in clinical_results.items() for method_name, metrics in scenario_data.items() if (row := {'scenario': scenario_name, 'method': method_name}) and rows.append(row)]
        df = pd.DataFrame(rows); cols_ordered = ['scenario', 'method', 'cbf_rmse', 'att_rmse', 'cbf_bias', 'att_bias', 'cbf_cov', 'att_cov', 'success_rate', 'mean_cbf_uncertainty_std', 'mean_att_uncertainty_std']
        existing_cols = [c for c in cols_ordered if c in df.columns]; df = df[existing_cols]
        table_path = self.output_dir / 'clinical_validation_summary.csv'
        df.to_csv(table_path, index=False, float_format='%.2f')
        self.monitor.log_progress("PUB_GEN", f"Saved clinical validation summary CSV to {table_path}")
        if wandb.run: wandb.save(str(table_path)); return str(table_path)

    def _generate_performance_table_csv(self, comparison_df: pd.DataFrame) -> str: 
        if comparison_df.empty: return ""
        cols_to_show = ['method', 'att_range_name', 'cbf_nbias_perc', 'cbf_cov', 'cbf_nrmse_perc', 'att_nbias_perc', 'att_cov', 'att_nrmse_perc', 'success_rate', 'computation_time']
        existing_cols = [col for col in cols_to_show if col in comparison_df.columns]; summary_df = comparison_df[existing_cols]
        table_path = self.output_dir / 'benchmark_performance_summary.csv'
        summary_df.to_csv(table_path, index=False, float_format='%.2f')
        self.monitor.log_progress("PUB_GEN", f"Saved benchmark performance summary CSV to {table_path}")
        if wandb.run: wandb.save(str(table_path)); return str(table_path)

    def _save_publication_materials(self, package: Dict): 
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
    
    wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=asdict(config), name=f"run_{timestamp}", job_type="research_pipeline")
    if wandb_run: script_logger.info(f"W&B Run URL: {wandb_run.url}")

    monitor = PerformanceMonitor(config, output_path) 
    monitor.log_progress("SETUP", f"Initializing. Output: {output_path}")
    config_save_path = output_path / 'research_config.json'
    with open(config_save_path, 'w') as f: json.dump(asdict(config), f, indent=2)
    if wandb_run: wandb.save(str(config_save_path))

    plds_np = np.array(config.pld_values); num_plds = len(plds_np)
    asl_params_sim = ASLParameters(T1_artery=config.T1_artery, T_tau=config.T_tau, alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL, T2_factor=config.T2_factor, alpha_BS1=config.alpha_BS1)
    simulator = RealisticASLSimulator(params=asl_params_sim); norm_stats_final = None 

    best_optuna_params = {}
    if config.optuna_n_trials > 0:
        monitor.log_progress("PHASE1", "Starting hyperparameter optimization")
        optimizer = HyperparameterOptimizer(config, monitor, output_path) 
        best_optuna_params = optimizer.optimize()
        if best_optuna_params:
            config.hidden_sizes = [best_optuna_params.get('hidden_size_1', config.hidden_sizes[0]), best_optuna_params.get('hidden_size_2', config.hidden_sizes[1]), best_optuna_params.get('hidden_size_3', config.hidden_sizes[2])]
            config.learning_rate = best_optuna_params.get('learning_rate', config.learning_rate); config.dropout_rate = best_optuna_params.get('dropout_rate', config.dropout_rate)
            config.batch_size = best_optuna_params.get('batch_size', config.batch_size); config.weight_decay = best_optuna_params.get('weight_decay', config.weight_decay)
            monitor.log_progress("PHASE1", f"Updated config with Optuna best_params: {best_optuna_params}")
            if wandb_run: wandb.config.update(best_optuna_params, allow_val_change=True)
    else: monitor.log_progress("PHASE1", "Skipping hyperparameter optimization.")

    monitor.log_progress("PHASE2", "Starting two-stage ensemble training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); monitor.log_progress("PHASE2", f"Using device: {device}")
    
    # --- STAGE 1: Foundational Pre-training Dataset ---
    monitor.log_progress("PHASE2", "Generating Dataset A (Foundational: healthy, high-SNR)")
    dataset_A = simulator.generate_balanced_dataset(plds=plds_np, total_subjects=config.n_subjects_stage1, noise_levels=[10.0, 15.0])
    
    # --- STAGE 2: Full-Spectrum Fine-tuning Dataset ---
    monitor.log_progress("PHASE2", "Generating Dataset B (Full-Spectrum: all conditions, all noises)")
    dataset_B = simulator.generate_balanced_dataset(plds=plds_np, total_subjects=config.n_subjects_stage2, noise_levels=config.training_noise_levels)

    # --- Feature Engineering for both datasets ---
    monitor.log_progress("PHASE2", "Applying signal feature engineering")
    engineered_features_A = engineer_signal_features(dataset_A['signals'], num_plds)
    dataset_A['signals'] = np.concatenate([dataset_A['signals'], engineered_features_A], axis=1)
    engineered_features_B = engineer_signal_features(dataset_B['signals'], num_plds)
    dataset_B['signals'] = np.concatenate([dataset_B['signals'], engineered_features_B], axis=1)

    # --- Normalization based on Full-Spectrum dataset ---
    monitor.log_progress("PHASE2", "Calculating normalization statistics from full-spectrum data")
    X_B_raw, y_B_raw = dataset_B['signals'], dataset_B['parameters']
    n_val_B = int(X_B_raw.shape[0] * config.val_split)
    perm_B = np.random.permutation(X_B_raw.shape[0])
    train_idx_B, val_idx_B = perm_B[:-n_val_B], perm_B[-n_val_B:]
    X_train_B_raw, y_train_B_raw = X_B_raw[train_idx_B], y_B_raw[train_idx_B]

    num_raw_signal_features = num_plds * 2
    pcasl_train_signals = X_train_B_raw[:, :num_plds]
    vsasl_train_signals = X_train_B_raw[:, num_plds:num_raw_signal_features]
    norm_stats_final = {
        'pcasl_mean': np.mean(pcasl_train_signals, axis=0), 'pcasl_std': np.std(pcasl_train_signals, axis=0),
        'vsasl_mean': np.mean(vsasl_train_signals, axis=0), 'vsasl_std': np.std(vsasl_train_signals, axis=0),
        'y_mean_cbf': np.mean(y_train_B_raw[:, 0]), 'y_std_cbf': np.std(y_train_B_raw[:, 0]),
        'y_mean_att': np.mean(y_train_B_raw[:, 1]), 'y_std_att': np.std(y_train_B_raw[:, 1])
    }
    
    # --- FIX START: Handle arrays and scalars separately ---
    # Handle array-based stats (for signals)
    for key in ['pcasl_std', 'vsasl_std']:
        norm_stats_final[key][norm_stats_final[key] < 1e-6] = 1.0

    # Handle scalar stats (for parameters)
    if norm_stats_final['y_std_cbf'] < 1e-6:
        norm_stats_final['y_std_cbf'] = 1.0
    if norm_stats_final['y_std_att'] < 1e-6:
        norm_stats_final['y_std_att'] = 1.0
    # --- FIX END ---

    # --- Function to create dataloaders for a given dataset ---
    def create_dataloaders(dataset_dict, norm_stats, val_split, batch_size):
        X_all, y_all = dataset_dict['signals'], dataset_dict['parameters']
        X_all_norm = np.array([apply_normalization_to_input_flat(s, norm_stats, num_plds, False) for s in X_all])
        y_all_norm = np.column_stack([
            (y_all[:, 0] - norm_stats['y_mean_cbf']) / norm_stats['y_std_cbf'],
            (y_all[:, 1] - norm_stats['y_mean_att']) / norm_stats['y_std_att']
        ])
        perm = np.random.permutation(len(X_all))
        n_val = int(len(X_all) * val_split)
        train_idx, val_idx = perm[:-n_val], perm[-n_val:]
        X_train, y_train_norm, y_train_raw = X_all_norm[train_idx], y_all_norm[train_idx], y_all[train_idx]
        X_val, y_val_norm = X_all_norm[val_idx], y_all_norm[val_idx]
        
        train_att_weights = np.exp(-np.clip(y_train_raw[:, 1], 100.0, None) / 2000.0)
        train_sampler = WeightedRandomSampler(train_att_weights, len(train_att_weights), replacement=True)
        
        train_dataset = EnhancedASLDataset(X_train, y_train_norm)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader = DataLoader(EnhancedASLDataset(X_val, y_val_norm), batch_size=batch_size) if len(X_val) > 0 else None
        return train_loader, val_loader
    
    monitor.log_progress("PHASE2", "Creating DataLoaders for both training stages")
    train_loader_A, val_loader_A = create_dataloaders(dataset_A, norm_stats_final, config.val_split, config.batch_size)
    train_loader_B, val_loader_B = create_dataloaders(dataset_B, norm_stats_final, config.val_split, config.batch_size)
    
    base_input_size_nn = dataset_A['signals'].shape[1]
    model_creation_config = {k: v for k, v in asdict(config).items()}

    def create_main_model_closure(**kwargs):
        model_param_keys = inspect.signature(EnhancedASLNet).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in model_param_keys}
        return EnhancedASLNet(input_size=base_input_size_nn, **filtered_kwargs)

    trainer = EnhancedASLTrainer(model_config=model_creation_config, model_class=create_main_model_closure, input_size=base_input_size_nn, learning_rate=config.learning_rate, weight_decay=config.weight_decay, batch_size=config.batch_size, n_ensembles=config.n_ensembles, device=device, n_plds_for_model=num_plds, m0_input_feature_model=False)

    trainer.norm_stats = norm_stats_final
    trainer.custom_loss_fn.norm_stats = norm_stats_final
    
    total_steps = len(train_loader_A) * config.n_epochs_stage1 + len(train_loader_B) * config.n_epochs_stage2
    for opt in trainer.optimizers:
        trainer.schedulers.append(torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.learning_rate, total_steps=total_steps))
    
    if norm_stats_final: 
        norm_stats_path = output_path / 'norm_stats.json'
        serializable_norm_stats = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm_stats_final.items()}
        with open(norm_stats_path, 'w') as f: json.dump(serializable_norm_stats, f, indent=2)
        if wandb_run: wandb.save(str(norm_stats_path))
    else: monitor.log_progress("PHASE2", "No normalization stats generated.", logging.WARNING)

    monitor.log_progress("PHASE2", f"Training {config.n_ensembles}-model ensemble using 2-stage curriculum...")
    training_start_time = time.time()
    training_histories_dict = trainer.train_ensemble(
        train_loaders=[train_loader_A, train_loader_B], 
        val_loaders=[val_loader_A, val_loader_B], 
        epoch_schedule=[config.n_epochs_stage1, config.n_epochs_stage2]
    )
    training_duration_hours = (time.time() - training_start_time) / 3600
    monitor.log_progress("PHASE2", f"Training completed in {training_duration_hours:.2f} hours.")
    if wandb_run: wandb.summary['training_duration_hours'] = training_duration_hours

    monitor.log_progress("PHASE3", "Starting clinical validation")
    clinical_validator = ClinicalValidator(config, monitor, norm_stats=norm_stats_final)
    clinical_validation_results = clinical_validator.validate_clinical_scenarios(trainer.models if trainer.models else [])

    monitor.log_progress("PHASE3.5", "Starting single-repeat vs multi-repeat validation") 
    srs_model_path = None
    if trainer.models:
        temp_srs_model_path = output_path / 'temp_srs_model_0.pt'
        model_state_srs = trainer.best_states[0] if hasattr(trainer, 'best_states') and trainer.best_states and trainer.best_states[0] else trainer.models[0].state_dict()
        if model_state_srs: torch.save(model_state_srs, temp_srs_model_path); srs_model_path = str(temp_srs_model_path)

    _, single_repeat_val_metrics = run_single_repeat_validation_main(model_path=srs_model_path, base_nn_input_size_for_model_load=base_input_size_nn, nn_arch_config_for_model_load=model_creation_config, norm_stats_for_nn=norm_stats_final)
    if srs_model_path and temp_srs_model_path.exists(): temp_srs_model_path.unlink()

    monitor.log_progress("PHASE4", "Benchmarking NN against conventional LS methods")
    total_test_subjects = config.n_test_subjects_per_att_range * len(config.att_ranges_config)
    n_base_subjects_for_benchmark = max(1, min(total_test_subjects, 2000) // (len(config.test_conditions) * len(config.test_snr_levels) * 3 if config.test_conditions and config.test_snr_levels else 1))
    benchmark_test_dataset_raw = simulator.generate_diverse_dataset(plds=plds_np, n_subjects=n_base_subjects_for_benchmark, conditions=config.test_conditions, noise_levels=config.test_snr_levels)
    benchmark_y_all = benchmark_test_dataset_raw['parameters']
    
    raw_benchmark_signals = benchmark_test_dataset_raw['signals']
    engineered_benchmark_features = engineer_signal_features(raw_benchmark_signals, num_plds)
    benchmark_X_all_nn_input = np.concatenate([raw_benchmark_signals, engineered_benchmark_features], axis=1)

    benchmark_test_data_for_comp = {'PCASL_LS': raw_benchmark_signals[:, :num_plds], 'VSASL_LS': raw_benchmark_signals[:, num_plds:], 'MULTIVERSE_LS_FORMAT': raw_benchmark_signals.reshape(-1, num_plds, 2), 'NN_INPUT_FORMAT': benchmark_X_all_nn_input}

    nn_model_for_comp_path = None; temp_comp_model_save_path = output_path / 'temp_comp_model_0.pt'
    if trainer.models: 
        model_state_to_save_comp = trainer.best_states[0] if hasattr(trainer, 'best_states') and trainer.best_states and trainer.best_states[0] else trainer.models[0].state_dict()
        if model_state_to_save_comp: torch.save(model_state_to_save_comp, temp_comp_model_save_path); nn_model_for_comp_path = str(temp_comp_model_save_path)

    comp_framework_output_dir = output_path / "comparison_framework_outputs"
    comp_framework = ComprehensiveComparison(nn_model_path=nn_model_for_comp_path, output_dir=str(comp_framework_output_dir), base_nn_input_size=base_input_size_nn, nn_n_plds=num_plds, nn_model_arch_config=model_creation_config, norm_stats=norm_stats_final)
    comparison_results_df = comp_framework.compare_methods(benchmark_test_data_for_comp, benchmark_y_all, plds_np, config.att_ranges_config)
    if nn_model_for_comp_path and temp_comp_model_save_path.exists(): temp_comp_model_save_path.unlink() 
    if not comparison_results_df.empty and wandb_run:
        if (comp_framework_output_dir / 'comparison_results_detailed.csv').exists(): wandb.save(str(comp_framework_output_dir / 'comparison_results_detailed.csv'))

    nn_benchmark_metrics, baseline_ls_metrics = {}, {}
    if not comparison_results_df.empty:
        for att_cfg_item in config.att_ranges_config: 
            range_name = att_cfg_item[2]
            nn_row_df = comparison_results_df[(comparison_results_df['method'] == 'Neural Network') & (comparison_results_df['att_range_name'] == range_name)]
            if not nn_row_df.empty: nn_benchmark_metrics[range_name] = nn_row_df.iloc[0].to_dict()
            ls_row_df = comparison_results_df[(comparison_results_df['method'] == 'MULTIVERSE-LS') & (comparison_results_df['att_range_name'] == range_name)]
            if not ls_row_df.empty: baseline_ls_metrics[range_name] = ls_row_df.iloc[0].to_dict()
        monitor.check_target_achievement(nn_benchmark_metrics, baseline_ls_metrics)

    monitor.log_progress("PHASE5", "Generating publication-ready materials (tables only)")
    pub_gen = PublicationGenerator(config, output_path, monitor)
    publication_package = pub_gen.generate_publication_package(clinical_validation_results, comparison_results_df, single_repeat_val_metrics)

    monitor.log_progress("PHASE6", "Saving final models and summarizing results")
    models_dir = output_path / 'trained_models'; models_dir.mkdir(exist_ok=True)
    if trainer.models:
        for i, model_state_val in enumerate(trainer.best_states if hasattr(trainer, 'best_states') and any(trainer.best_states) else [m.state_dict() for m in trainer.models]):
            model_file_path_val = models_dir / f'ensemble_model_{i}.pt' 
            if model_state_val: torch.save(model_state_val, model_file_path_val)
            if wandb_run and model_file_path_val.exists(): wandb.save(str(model_file_path_val))

    final_results_summary = {'config': asdict(config), 'optuna_best_params': best_optuna_params, 'training_duration_hours': training_duration_hours, 'clinical_validation_results': clinical_validation_results, 'single_repeat_validation_metrics': single_repeat_val_metrics, 'trained_models_dir': str(models_dir), 'wandb_run_url': wandb_run.url if wandb_run else None}
    with open(output_path / 'final_research_results.json', 'w') as f: json.dump(final_results_summary, f, indent=2, default=lambda o: f"<not_serializable_{type(o).__name__}>")
    if wandb_run: wandb.save(str(output_path / 'final_research_results.json'))
    
    monitor.log_progress("COMPLETE", f"Research pipeline finished. Results in {output_path}")
    if wandb_run: wandb_run.finish()
    return final_results_summary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True) 
    config_file_path_arg = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    loaded_config_obj = ResearchConfig() 

    if Path(config_file_path_arg).exists():
        script_logger.info(f"Loading configuration from {config_file_path_arg}")
        with open(config_file_path_arg, 'r') as f_yaml: config_from_yaml = yaml.safe_load(f_yaml) or {}
        all_yaml_params = {}
        for key, value in config_from_yaml.items():
            if isinstance(value, dict): all_yaml_params.update(value)
            else: all_yaml_params[key] = value
        for key, value in all_yaml_params.items():
            if hasattr(loaded_config_obj, key): setattr(loaded_config_obj, key, value)
            else: script_logger.warning(f"YAML key '{key}' not found in ResearchConfig, ignoring.")
    else: 
        script_logger.info(f"Config file {config_file_path_arg} not found. Using default ResearchConfig.")

    script_logger.info("\nStarting comprehensive ASL research pipeline with configuration:")
    pipeline_results_dict = run_comprehensive_asl_research(config=loaded_config_obj)
    script_logger.info("\n" + "=" * 80 + "\nRESEARCH PIPELINE COMPLETED!\n" + "=" * 80)
    if "error" not in pipeline_results_dict:
        script_logger.info(f"Results saved in: {pipeline_results_dict.get('trained_models_dir', 'Specified output directory')}")
        if pipeline_results_dict.get('wandb_run_url'): script_logger.info(f"W&B Run: {pipeline_results_dict['wandb_run_url']}")
    else: 
        script_logger.error(f"Pipeline failed: {pipeline_results_dict.get('error')}")
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
import wandb # Added for Weights & Biases
import joblib # Added for saving Optuna study

warnings.filterwarnings('ignore', category=UserWarning)

from enhanced_asl_network import EnhancedASLNet, CustomLoss
from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer
from comparison_framework import ComprehensiveComparison, ComparisonResult
from performance_metrics import ProposalEvaluator
from single_repeat_validation import SingleRepeatValidator

from vsasl_functions import fit_VSASL_vectInit_pep
from pcasl_functions import fit_PCASL_vectInit_pep
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

script_logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    # Training parameters
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    batch_size: int = 256
    n_training_subjects: int = 10000
    n_epochs: int = 200
    n_ensembles: int = 5
    dropout_rate: float = 0.1
    norm_type: str = 'batch'
    use_transformer_temporal_model: bool = True
    transformer_nhead_model: int = 4
    transformer_nlayers_model: int = 2
    m0_input_feature_model: bool = False

    # Hyperparameter optimization
    optuna_n_trials: int = 20
    optuna_timeout_hours: float = 0.5
    optuna_n_subjects: int = 500
    optuna_n_epochs: int = 20
    optuna_study_name: str = "asl_multiverse_hpo" # For W&B grouping if desired

    # Data generation
    pld_values: List[int] = field(default_factory=lambda: list(range(500, 3001, 500)))
    att_ranges_config: List[Tuple[float, float, str]] = field(default_factory=lambda: [
        (500.0, 1500.0, "Short ATT"),
        (1500.0, 2500.0, "Medium ATT"),
        (2500.0, 4000.0, "Long ATT")
    ])
    include_m0_in_training_data: bool = False

    # Simulation parameters
    T1_artery: float = 1850.0; T2_factor: float = 1.0; alpha_BS1: float = 1.0
    alpha_PCASL: float = 0.85; alpha_VSASL: float = 0.56; T_tau: float = 1800.0
    reference_CBF: float = 60.0

    # Clinical validation
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

    # Performance targets
    target_cbf_cv_improvement_perc: float = 50.0
    target_att_cv_improvement_perc: float = 50.0

    # W&B configuration
    wandb_project: str = "asl-multiverse-project" # Your W&B project name
    wandb_entity: Optional[str] = None # Your W&B entity (username or team), or set via env


class PerformanceMonitor:
    def __init__(self, config: ResearchConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger("ASLResearchPipeline")
        self.logger.setLevel(logging.INFO)
        for handler in self.logger.handlers[:]: self.logger.removeHandler(handler)
        fh = logging.FileHandler(output_dir / 'research.log', mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(sh)
        self.logger.propagate = False

    def log_progress(self, phase: str, message: str, level: int = logging.INFO):
        self.logger.log(level, f"[{phase}] {message}")
        if wandb.run: # Log to W&B if a run is active
             wandb.log({f"Progress/{phase}": message, "global_step": wandb.run.step if wandb.run.step else 0})


    def check_target_achievement(self, nn_att_range_results: Dict, baseline_att_range_results: Dict) -> Dict:
        achievements = {}
        for att_range_name in nn_att_range_results.keys():
            if att_range_name not in baseline_att_range_results:
                self.log_progress("TARGET_CHECK", f"Baseline results missing for {att_range_name}", logging.WARNING)
                continue
            nn_metrics, baseline_metrics = nn_att_range_results[att_range_name], baseline_att_range_results[att_range_name]
            current_cbf_cv = nn_metrics.get('cbf_cov', float('inf'))
            baseline_cbf_cv = baseline_metrics.get('cbf_cov', float('inf'))
            cbf_improvement_perc = ((baseline_cbf_cv - current_cbf_cv) / baseline_cbf_cv) * 100 if baseline_cbf_cv > 0 and baseline_cbf_cv != float('inf') and current_cbf_cv != float('inf') else 0.0
            current_att_cv = nn_metrics.get('att_cov', float('inf'))
            baseline_att_cv = baseline_metrics.get('att_cov', float('inf'))
            att_improvement_perc = ((baseline_att_cv - current_att_cv) / baseline_att_cv) * 100 if baseline_att_cv > 0 and baseline_att_cv != float('inf') and current_att_cv != float('inf') else 0.0
            cbf_target_met, att_target_met = cbf_improvement_perc >= self.config.target_cbf_cv_improvement_perc, att_improvement_perc >= self.config.target_att_cv_improvement_perc
            achievements[att_range_name] = {'cbf_cv_improvement_perc': cbf_improvement_perc, 'att_cv_improvement_perc': att_improvement_perc, 'cbf_target_met': cbf_target_met, 'att_target_met': att_target_met}
            self.log_progress("TARGET_CHECK", f"{att_range_name} - CBF CV Improv: {cbf_improvement_perc:.1f}% ({'MET' if cbf_target_met else 'NOT MET'})")
            self.log_progress("TARGET_CHECK", f"{att_range_name} - ATT CV Improv: {att_improvement_perc:.1f}% ({'MET' if att_target_met else 'NOT MET'})")
            if wandb.run:
                wandb.summary[f'TargetAchieved/{att_range_name}/CBF_CV_Improvement'] = cbf_improvement_perc
                wandb.summary[f'TargetAchieved/{att_range_name}/ATT_CV_Improvement'] = att_improvement_perc
        return achievements

class HyperparameterOptimizer:
    def __init__(self, base_config: ResearchConfig, monitor: PerformanceMonitor, output_dir: Path):
        self.base_config = base_config
        self.monitor = monitor
        self.output_dir = output_dir # To save the study object
        self.study = None

    def objective(self, trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        hidden_size_1 = trial.suggest_categorical('hidden_size_1', [128, 256, 512])
        hidden_size_2 = trial.suggest_categorical('hidden_size_2', [64, 128, 256])
        hidden_size_3 = trial.suggest_categorical('hidden_size_3', [32, 64, 128])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        transformer_nhead = trial.suggest_categorical('transformer_nhead', [2, 4, 8]) if self.base_config.use_transformer_temporal_model else self.base_config.transformer_nhead_model
        transformer_nlayers = trial.suggest_categorical('transformer_nlayers', [1, 2, 3]) if self.base_config.use_transformer_temporal_model else self.base_config.transformer_nlayers_model

        trial_config_dict = asdict(self.base_config)
        trial_params_for_wandb = {
            'hidden_sizes': [hidden_size_1, hidden_size_2, hidden_size_3],
            'learning_rate': learning_rate, 'dropout_rate': dropout_rate, 'batch_size': batch_size,
            'transformer_nhead_model': transformer_nhead, 'transformer_nlayers_model': transformer_nlayers,
        }
        trial_config_dict.update(trial_params_for_wandb)
        trial_config_dict.update({ # Optuna specific overrides
            'n_training_subjects': self.base_config.optuna_n_subjects,
            'n_epochs': self.base_config.optuna_n_epochs, 'n_ensembles': 1,
        })
        trial_config = ResearchConfig(**trial_config_dict)
        self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number}: {trial.params}")

        # Initialize a new W&B run for each Optuna trial if main W&B run is active
        # This allows grouping Optuna trials under the main experiment.
        if wandb.run:
            current_wandb_run_id = wandb.run.id
            current_wandb_project = wandb.run.project
            current_wandb_entity = wandb.run.entity
            wandb.finish() # Finish the main run temporarily
            
            trial_wandb_run = wandb.init(
                project=current_wandb_project,
                entity=current_wandb_entity,
                group=self.base_config.optuna_study_name, # Group Optuna trials
                name=f"trial_{trial.number}",
                config=trial_params_for_wandb, # Log Optuna trial's specific params
                reinit=True,
                job_type="hpo_trial"
            )
        else: # If no main W&B run, init one for the trial
             trial_wandb_run = wandb.init(
                project=self.base_config.wandb_project,
                entity=self.base_config.wandb_entity,
                group=self.base_config.optuna_study_name,
                name=f"trial_{trial.number}",
                config=trial_params_for_wandb,
                reinit=True,
                job_type="hpo_trial"
             )


        try:
            _, _, validation_metrics = self._quick_training_run(trial_config)
            validation_loss = validation_metrics.get('val_loss', float('inf'))
            self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number} Val Loss: {validation_loss:.6f}")
            if trial_wandb_run:
                trial_wandb_run.summary['final_validation_loss'] = validation_loss
                trial_wandb_run.finish()
            
            # Re-initialize the main W&B run if it was active
            if current_wandb_run_id and wandb.run is None : # Check if no active run
                wandb.init(project=current_wandb_project, entity=current_wandb_entity, id=current_wandb_run_id, resume="must")


            return validation_loss
        except Exception as e:
            self.monitor.log_progress("OPTUNA_TRIAL", f"Trial {trial.number} FAILED: {e}", logging.ERROR)
            if trial_wandb_run:
                trial_wandb_run.summary['status'] = 'failed'
                trial_wandb_run.finish()
            # Re-initialize the main W&B run
            if current_wandb_run_id and wandb.run is None:
                 wandb.init(project=current_wandb_project, entity=current_wandb_entity, id=current_wandb_run_id, resume="must")
            return float('inf')

    def _quick_training_run(self, config: ResearchConfig) -> Tuple[Any, Any, Dict[str, float]]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        asl_params_sim = ASLParameters(T1_artery=config.T1_artery, T_tau=config.T_tau, alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL)
        simulator = RealisticASLSimulator(params=asl_params_sim)
        plds_np = np.array(config.pld_values)
        base_input_size = len(plds_np) * 2
        model_input_size = base_input_size + 1 if config.m0_input_feature_model else base_input_size

        def create_trial_model():
            return EnhancedASLNet(input_size=base_input_size, hidden_sizes=config.hidden_sizes, n_plds=len(plds_np),
                                  dropout_rate=config.dropout_rate, norm_type=config.norm_type,
                                  use_transformer_temporal=config.use_transformer_temporal_model,
                                  transformer_nhead=config.transformer_nhead_model,
                                  transformer_nlayers=config.transformer_nlayers_model,
                                  m0_input_feature=config.m0_input_feature_model).to(device)

        trainer = EnhancedASLTrainer(model_class=create_trial_model, input_size=model_input_size,
                                     hidden_sizes=config.hidden_sizes, learning_rate=config.learning_rate,
                                     batch_size=config.batch_size, n_ensembles=config.n_ensembles, device=device,
                                     n_plds_for_model=len(plds_np), m0_input_feature_model=config.m0_input_feature_model)
        train_loaders, val_loader = trainer.prepare_curriculum_data(
            simulator, n_training_subjects=config.n_training_subjects, plds=plds_np,
            curriculum_att_ranges_config=config.att_ranges_config,
            training_conditions_config=config.training_conditions[:1],
            training_noise_levels_config=config.training_noise_levels[:1],
            n_epochs_for_scheduler=config.n_epochs,
            include_m0_in_data=config.include_m0_in_training_data
        )
        if not train_loaders:
            self.monitor.log_progress("OPTUNA_RUN", "No training data for Optuna trial.", logging.ERROR)
            return None, None, {'val_loss': float('inf')}
        
        trainer.train_ensemble(train_loaders, val_loader, n_epochs=config.n_epochs, early_stopping_patience=5)
        
        final_val_metrics = {'val_loss': float('inf')}
        if val_loader and trainer.models and trainer.val_metrics[0]: # Check if val_metrics has data for model 0
            final_val_metrics = trainer.val_metrics[0][-1] # Get last validation metrics for the first model
        elif val_loader and trainer.models: # Fallback if val_metrics structure is different or empty
            # Try to re-validate to get metrics if val_metrics not populated as expected
             try:
                final_val_metrics = trainer._validate(trainer.models[0], val_loader, config.n_epochs-1, len(train_loaders)-1, config.n_epochs)
             except Exception:
                pass # Keep default float('inf')

        return trainer, simulator, final_val_metrics


    def optimize(self) -> Dict:
        self.monitor.log_progress("OPTUNA", f"Starting HPO: {self.base_config.optuna_n_trials} trials, timeout {self.base_config.optuna_timeout_hours}h.")
        self.study = optuna.create_study(direction='minimize', study_name=self.base_config.optuna_study_name)
        self.study.optimize(self.objective, n_trials=self.base_config.optuna_n_trials, timeout=self.base_config.optuna_timeout_hours * 3600)
        
        # Save the study object
        study_path = self.output_dir / 'optuna_study.pkl'
        joblib.dump(self.study, study_path)
        self.monitor.log_progress("OPTUNA", f"Optuna study saved to {study_path}")
        if wandb.run: # Log study path to main W&B run if active
            wandb.save(str(study_path))

        best_params = self.study.best_params
        self.monitor.log_progress("OPTUNA", f"Best parameters found: {best_params}")
        self.monitor.log_progress("OPTUNA", f"Best validation loss: {self.study.best_value:.6f}")
        if wandb.run:
            wandb.summary['optuna_best_value'] = self.study.best_value
            wandb.summary.update({f"optuna_best_param_{k}": v for k, v in best_params.items()})
        return best_params


class ClinicalValidator:
    def __init__(self, config: ResearchConfig, monitor: PerformanceMonitor):
        self.config = config
        self.monitor = monitor
        asl_params_sim = ASLParameters(T1_artery=config.T1_artery, T_tau=config.T_tau, alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL)
        self.simulator = RealisticASLSimulator(params=asl_params_sim)
        self.plds_np = np.array(config.pld_values)

    def validate_clinical_scenarios(self, trained_nn_models: List[torch.nn.Module]) -> Dict:
        self.monitor.log_progress("CLINICAL_VAL", "Running clinical validation scenarios...")
        all_scenario_results = {}
        pldti = np.column_stack((self.plds_np, self.plds_np))
        for scenario_name, params in self.config.clinical_scenario_definitions.items():
            self.monitor.log_progress("CLINICAL_VAL", f"  Validating {scenario_name}...")
            n_subjects = self.config.n_clinical_scenario_subjects
            true_cbf_vals, true_att_vals = np.random.uniform(*params['cbf_range'], n_subjects), np.random.uniform(*params['att_range'], n_subjects)
            current_snr = params['snr']
            scenario_metrics = {
                'neural_network': {'cbf_preds': [], 'att_preds': [], 'cbf_uncs': [], 'att_uncs': []},
                'multiverse_ls_single_repeat': {'cbf_preds': [], 'att_preds': []},
                'multiverse_ls_multi_repeat_avg': {'cbf_preds': [], 'att_preds': []}
            }
            for i in range(n_subjects):
                true_cbf, true_att = true_cbf_vals[i], true_att_vals[i]
                single_repeat_data = self.simulator.generate_synthetic_data(
                    self.plds_np, np.array([true_att]), n_noise=1, tsnr=current_snr, cbf_val=true_cbf
                )
                nn_input_signal = np.concatenate([single_repeat_data['PCASL'][0,0,:], single_repeat_data['VSASL'][0,0,:]])
                ls_single_repeat_signal = single_repeat_data['MULTIVERSE'][0,0,:,:]
                if self.config.m0_input_feature_model:
                    dummy_m0 = np.array([1.0]); nn_input_signal = np.concatenate((nn_input_signal, dummy_m0))

                if trained_nn_models:
                    cbf_nn, att_nn, cbf_std, att_std = self._ensemble_predict(trained_nn_models, nn_input_signal)
                    scenario_metrics['neural_network']['cbf_preds'].append(cbf_nn)
                    scenario_metrics['neural_network']['att_preds'].append(att_nn)
                    scenario_metrics['neural_network']['cbf_uncs'].append(cbf_std)
                    scenario_metrics['neural_network']['att_uncs'].append(att_std)
                else:
                    for k_ in scenario_metrics['neural_network']: scenario_metrics['neural_network'][k_].append(np.nan)
                try:
                    beta_ls_sr, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, ls_single_repeat_signal, [50.0/6000.0, 1500.0],
                        self.config.T1_artery, self.config.T_tau, self.config.T2_factor,
                        self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL)
                    scenario_metrics['multiverse_ls_single_repeat']['cbf_preds'].append(beta_ls_sr[0] * 6000.0)
                    scenario_metrics['multiverse_ls_single_repeat']['att_preds'].append(beta_ls_sr[1])
                except Exception: scenario_metrics['multiverse_ls_single_repeat']['cbf_preds'].append(np.nan); scenario_metrics['multiverse_ls_single_repeat']['att_preds'].append(np.nan)
                
                avg_multi_repeat_signal_collector = []
                for _ in range(4):
                    repeat_data = self.simulator.generate_synthetic_data(
                        self.plds_np, np.array([true_att]), n_noise=1, tsnr=current_snr * np.sqrt(4), cbf_val=true_cbf
                    )
                    avg_multi_repeat_signal_collector.append(repeat_data['MULTIVERSE'][0,0,:,:])
                avg_multi_repeat_signal = np.mean(avg_multi_repeat_signal_collector, axis=0)
                try:
                    beta_ls_mr, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, avg_multi_repeat_signal, [50.0/6000.0, 1500.0],
                        self.config.T1_artery, self.config.T_tau, self.config.T2_factor,
                        self.config.alpha_BS1, self.config.alpha_PCASL, self.config.alpha_VSASL)
                    scenario_metrics['multiverse_ls_multi_repeat_avg']['cbf_preds'].append(beta_ls_mr[0] * 6000.0)
                    scenario_metrics['multiverse_ls_multi_repeat_avg']['att_preds'].append(beta_ls_mr[1])
                except Exception: scenario_metrics['multiverse_ls_multi_repeat_avg']['cbf_preds'].append(np.nan); scenario_metrics['multiverse_ls_multi_repeat_avg']['att_preds'].append(np.nan)

            all_scenario_results[scenario_name] = {}
            temp_comparator = ComprehensiveComparison(nn_input_size=len(self.plds_np)*2 + (1 if self.config.m0_input_feature_model else 0), nn_n_plds=len(self.plds_np), nn_m0_input_feature=self.config.m0_input_feature_model)
            for method_key, data_dict in scenario_metrics.items():
                c_preds, a_preds = np.array(data_dict['cbf_preds']), np.array(data_dict['att_preds'])
                metrics_summary = temp_comparator._calculate_detailed_metrics(c_preds, true_cbf_vals, a_preds, true_att_vals)
                num_valid = np.sum(~np.isnan(c_preds) & ~np.isnan(a_preds))
                metrics_summary['success_rate'] = (num_valid / n_subjects) * 100 if n_subjects > 0 else 0
                if 'cbf_uncs' in data_dict and data_dict['cbf_uncs']:
                    metrics_summary['mean_cbf_uncertainty_std'] = np.nanmean(data_dict['cbf_uncs'])
                    metrics_summary['mean_att_uncertainty_std'] = np.nanmean(data_dict['att_uncs'])
                all_scenario_results[scenario_name][method_key] = metrics_summary
                self.monitor.log_progress("CLINICAL_VAL", f"  {scenario_name} - {method_key}: CBF RMSE {metrics_summary.get('cbf_rmse',np.nan):.2f}, ATT RMSE {metrics_summary.get('att_rmse',np.nan):.2f}, Success {metrics_summary.get('success_rate',0):.1f}%")
                if wandb.run: # Log detailed clinical scenario metrics
                    for m_name, m_val in metrics_summary.items():
                        wandb.summary[f"ClinicalVal/{scenario_name}/{method_key}/{m_name}"] = m_val

        return all_scenario_results

    def _ensemble_predict(self, models: List[torch.nn.Module], input_signal_flat: np.ndarray) -> Tuple[float, float, float, float]:
        if not models: return np.nan, np.nan, np.nan, np.nan
        input_tensor = torch.FloatTensor(input_signal_flat).unsqueeze(0).to(next(models[0].parameters()).device)
        cbf_means_l, att_means_l, cbf_vars_l, att_vars_l = [], [], [], []
        for model in models:
            model.eval()
            with torch.no_grad():
                cbf_m, att_m, cbf_lv, att_lv = model(input_tensor)
                cbf_means_l.append(cbf_m.item()); att_means_l.append(att_m.item())
                cbf_vars_l.append(torch.exp(cbf_lv).item()); att_vars_l.append(torch.exp(att_lv).item())
        ens_cbf_m, ens_att_m = np.mean(cbf_means_l) if cbf_means_l else np.nan, np.mean(att_means_l) if att_means_l else np.nan
        al_cbf_v, al_att_v = np.mean(cbf_vars_l) if cbf_vars_l else np.nan, np.mean(att_vars_l) if att_vars_l else np.nan
        ep_cbf_v, ep_att_v = np.var(cbf_means_l) if len(cbf_means_l) > 1 else 0.0, np.var(att_means_l) if len(att_means_l) > 1 else 0.0
        tot_cbf_std = np.sqrt(al_cbf_v + ep_cbf_v) if not (np.isnan(al_cbf_v) or np.isnan(ep_cbf_v)) else np.nan
        tot_att_std = np.sqrt(al_att_v + ep_att_v) if not (np.isnan(al_att_v) or np.isnan(ep_att_v)) else np.nan
        return ens_cbf_m, ens_att_m, tot_cbf_std, tot_att_std

class PublicationGenerator:
    def __init__(self, config: ResearchConfig, output_dir: Path, monitor: PerformanceMonitor):
        self.config = config; self.output_dir = output_dir; self.monitor = monitor

    def generate_publication_package(self, clinical_results: Dict, comparison_df: pd.DataFrame) -> Dict:
        self.monitor.log_progress("PUB_GEN", "Generating publication tables...")
        package = {'tables': {}, 'statistical_analysis': {}} # No figures section anymore

        package['tables']['performance_summary_csv'] = self._generate_performance_table_csv(comparison_df)
        package['tables']['clinical_validation_summary_csv'] = self._generate_clinical_table_csv(clinical_results)
        
        self._save_publication_materials(package) # Save summary of what was generated
        return package

    def _generate_clinical_table_csv(self, clinical_results: Dict) -> str:
        if not clinical_results:
            self.monitor.log_progress("PUB_GEN", "No clinical results to create table from.", logging.WARNING)
            return ""
        
        rows = []
        for scenario_name, scenario_data in clinical_results.items():
            for method_name, metrics in scenario_data.items():
                row = {'scenario': scenario_name, 'method': method_name}
                row.update(metrics)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        # Select and reorder columns for clarity
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

    def _generate_performance_table_csv(self, comparison_df: pd.DataFrame) -> str:
        if comparison_df.empty:
            self.monitor.log_progress("PUB_GEN", "Comparison DataFrame empty, cannot generate table.", logging.WARNING)
            return ""
        cols_to_show = ['method', 'att_range_name', 'cbf_nbias_perc', 'cbf_cov', 'cbf_nrmse_perc', 
                        'att_nbias_perc', 'att_cov', 'att_nrmse_perc', 'success_rate', 'computation_time']
        existing_cols = [col for col in cols_to_show if col in comparison_df.columns]
        summary_df = comparison_df[existing_cols]
        table_path = self.output_dir / 'benchmark_performance_summary.csv'
        summary_df.to_csv(table_path, index=False, float_format='%.2f')
        self.monitor.log_progress("PUB_GEN", f"Saved benchmark performance summary CSV to {table_path}")
        if wandb.run: wandb.save(str(table_path))
        return str(table_path)

    def _save_publication_materials(self, package: Dict):
        def make_serializable(obj): # Helper for JSON serialization
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
    
    # Initialize W&B Run
    wandb_run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity, # Can be None, W&B will use default
        config=asdict(config), # Log the entire configuration
        name=f"run_{timestamp}",
        job_type="research_pipeline"
    )
    if wandb_run:
        script_logger.info(f"W&B Run URL: {wandb_run.url}")


    monitor = PerformanceMonitor(config, output_path)
    monitor.log_progress("SETUP", f"Initializing. Output: {output_path}")
    with open(output_path / 'research_config.json', 'w') as f: json.dump(asdict(config), f, indent=2)
    if wandb_run: wandb.save(str(output_path / 'research_config.json')) # Save config to W&B artifacts

    plds_np = np.array(config.pld_values)
    asl_params_sim = ASLParameters(T1_artery=config.T1_artery, T_tau=config.T_tau, alpha_PCASL=config.alpha_PCASL, alpha_VSASL=config.alpha_VSASL)
    simulator = RealisticASLSimulator(params=asl_params_sim)

    best_optuna_params = {}
    if config.optuna_n_trials > 0:
        monitor.log_progress("PHASE1", "Starting hyperparameter optimization")
        optimizer = HyperparameterOptimizer(config, monitor, output_path)
        best_optuna_params = optimizer.optimize()
        if best_optuna_params:
            config.hidden_sizes = [best_optuna_params['hidden_size_1'], best_optuna_params['hidden_size_2'], best_optuna_params['hidden_size_3']]
            config.learning_rate = best_optuna_params['learning_rate']; config.dropout_rate = best_optuna_params['dropout_rate']
            config.batch_size = best_optuna_params['batch_size']
            config.transformer_nhead_model = best_optuna_params.get('transformer_nhead', config.transformer_nhead_model)
            config.transformer_nlayers_model = best_optuna_params.get('transformer_nlayers', config.transformer_nlayers_model)
            monitor.log_progress("PHASE1", f"Updated config with Optuna best_params: {best_optuna_params}")
            if wandb_run: wandb.config.update(best_optuna_params, allow_val_change=True) # Update W&B config
    else: monitor.log_progress("PHASE1", "Skipping hyperparameter optimization.")

    monitor.log_progress("PHASE2", "Starting multi-objective ensemble training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    monitor.log_progress("PHASE2", f"Using device: {device}")
    base_input_size_nn = len(plds_np) * 2
    model_actual_input_size = base_input_size_nn + 1 if config.m0_input_feature_model else base_input_size_nn

    def create_main_model():
        return EnhancedASLNet(input_size=base_input_size_nn, hidden_sizes=config.hidden_sizes,
                              n_plds=len(plds_np), dropout_rate=config.dropout_rate, norm_type=config.norm_type,
                              use_transformer_temporal=config.use_transformer_temporal_model,
                              transformer_nhead=config.transformer_nhead_model,
                              transformer_nlayers=config.transformer_nlayers_model,
                              m0_input_feature=config.m0_input_feature_model).to(device)
    trainer = EnhancedASLTrainer(model_class=create_main_model, input_size=model_actual_input_size,
                                 hidden_sizes=config.hidden_sizes, learning_rate=config.learning_rate,
                                 batch_size=config.batch_size, n_ensembles=config.n_ensembles, device=device,
                                 n_plds_for_model=len(plds_np), m0_input_feature_model=config.m0_input_feature_model)
    monitor.log_progress("PHASE2", "Preparing curriculum training datasets...")
    train_loaders, val_loader = trainer.prepare_curriculum_data(
        simulator, n_training_subjects=config.n_training_subjects, plds=plds_np,
        curriculum_att_ranges_config=config.att_ranges_config,
        training_conditions_config=config.training_conditions,
        training_noise_levels_config=config.training_noise_levels,
        n_epochs_for_scheduler=config.n_epochs,
        include_m0_in_data=config.include_m0_in_training_data
    )
    if not train_loaders:
        monitor.log_progress("PHASE2", "Failed to create training loaders. Aborting.", logging.CRITICAL)
        if wandb_run: wandb_run.finish(exit_code=1);
        return {"error": "Training data preparation failed."}
    monitor.log_progress("PHASE2", f"Training {config.n_ensembles}-model ensemble for {config.n_epochs} epochs...")
    training_start_time = time.time()
    training_histories_dict = trainer.train_ensemble(train_loaders, val_loader, n_epochs=config.n_epochs) # Now contains more metrics
    training_duration_hours = (time.time() - training_start_time) / 3600
    monitor.log_progress("PHASE2", f"Training completed in {training_duration_hours:.2f} hours.")
    if wandb_run: wandb.summary['training_duration_hours'] = training_duration_hours

    monitor.log_progress("PHASE3", "Starting clinical validation")
    clinical_validator = ClinicalValidator(config, monitor)
    clinical_validation_results = clinical_validator.validate_clinical_scenarios(trainer.models if trainer.models else [])

    monitor.log_progress("PHASE4", "Benchmarking NN against conventional LS methods")
    total_test_subjects = config.n_test_subjects_per_att_range * len(config.att_ranges_config)
    total_test_subjects = min(total_test_subjects, 1000)
    benchmark_test_dataset_raw = simulator.generate_diverse_dataset(
        plds=plds_np, n_subjects=max(1, total_test_subjects // (len(config.test_conditions) * len(config.test_snr_levels)*3)),
        conditions=config.test_conditions, noise_levels=config.test_snr_levels
    )
    benchmark_X_all_asl, benchmark_y_all = benchmark_test_dataset_raw['signals'], benchmark_test_dataset_raw['parameters']
    if config.m0_input_feature_model:
        m0_for_benchmark = np.random.normal(1.0, 0.1, size=(benchmark_X_all_asl.shape[0], 1))
        benchmark_X_all_nn_input = np.concatenate((benchmark_X_all_asl, m0_for_benchmark), axis=1)
    else: benchmark_X_all_nn_input = benchmark_X_all_asl
    benchmark_test_data_for_comp = {
        'PCASL': benchmark_X_all_asl[:, :len(plds_np)], 'VSASL': benchmark_X_all_asl[:, len(plds_np):],
        'MULTIVERSE_LS_FORMAT': benchmark_X_all_asl.reshape(-1, len(plds_np), 2),
        'NN_INPUT_FORMAT': benchmark_X_all_nn_input
    }
    nn_model_for_comp_path = None
    if trainer.models:
        temp_model_save_path = output_path / 'temp_ensemble_model_0_for_comp.pt'
        model_state_to_save = trainer.best_states[0] if hasattr(trainer, 'best_states') and trainer.best_states and trainer.best_states[0] else trainer.models[0].state_dict()
        torch.save(model_state_to_save, temp_model_save_path)
        nn_model_for_comp_path = str(temp_model_save_path)

    comp_framework_output_dir = output_path / "comparison_framework_outputs"
    comp_framework = ComprehensiveComparison(
        nn_model_path=nn_model_for_comp_path,
        output_dir=comp_framework_output_dir,
        nn_input_size=base_input_size_nn,
        nn_hidden_sizes=config.hidden_sizes,
        nn_n_plds=len(plds_np),
        nn_m0_input_feature=config.m0_input_feature_model,
        nn_use_transformer_temporal=config.use_transformer_temporal_model,
        nn_transformer_nlayers=config.transformer_nlayers_model,
        nn_transformer_nhead=config.transformer_nhead_model,
    )
    comparison_results_df = comp_framework.compare_methods(benchmark_test_data_for_comp, benchmark_y_all, plds_np, config.att_ranges_config)
    # The visualize_results in comp_framework no longer saves PNGs.
    # If interactive viewing is needed, it can be called: comp_framework.visualize_results(comparison_results_df)
    if nn_model_for_comp_path and temp_model_save_path.exists(): temp_model_save_path.unlink()
    
    if not comparison_results_df.empty and wandb_run:
        benchmark_table_path = comp_framework_output_dir / 'comparison_results_detailed.csv'
        wandb.save(str(benchmark_table_path)) # Save the CSV to W&B

    nn_benchmark_metrics_for_monitor, baseline_ls_metrics_for_monitor = {}, {}
    if not comparison_results_df.empty:
        for _, _, range_name in config.att_ranges_config:
            nn_row = comparison_results_df[(comparison_results_df['method'] == 'Neural Network') & (comparison_results_df['att_range_name'] == range_name)]
            if not nn_row.empty: nn_benchmark_metrics_for_monitor[range_name] = nn_row.iloc[0].to_dict()
            ls_row = comparison_results_df[(comparison_results_df['method'] == 'MULTIVERSE-LS') & (comparison_results_df['att_range_name'] == range_name)]
            if not ls_row.empty: baseline_ls_metrics_for_monitor[range_name] = ls_row.iloc[0].to_dict()
        monitor.check_target_achievement(nn_benchmark_metrics_for_monitor, baseline_ls_metrics_for_monitor)

    monitor.log_progress("PHASE5", "Generating publication-ready materials (tables only)")
    pub_gen = PublicationGenerator(config, output_path, monitor)
    publication_package = pub_gen.generate_publication_package(clinical_validation_results, comparison_results_df)

    monitor.log_progress("PHASE6", "Generating comprehensive research summary")
    models_dir = output_path / 'trained_models'; models_dir.mkdir(exist_ok=True)
    if trainer.models:
        for i, model_state in enumerate(trainer.best_states if hasattr(trainer, 'best_states') and trainer.best_states else [m.state_dict() for m in trainer.models]):
            model_file_path = models_dir / f'ensemble_model_{i}_best.pt'
            if model_state: torch.save(model_state, model_file_path)
            elif trainer.models and trainer.models[i]:
                model_file_path = models_dir / f'ensemble_model_{i}_final.pt'
                torch.save(trainer.models[i].state_dict(), model_file_path)
            if wandb_run and model_file_path.exists(): wandb.save(str(model_file_path)) # Save models to W&B

    final_results_summary = {
        'config': asdict(config), 'optuna_best_params': best_optuna_params,
        'optuna_study_path': str(output_path / 'optuna_study.pkl') if config.optuna_n_trials > 0 else None,
        'training_duration_hours': training_duration_hours, 
        'training_histories_metrics': training_histories_dict['all_histories'] if 'all_histories' in training_histories_dict else None,
        'clinical_validation_results': clinical_validation_results,
        'benchmark_comparison_results_csv_path': str(comp_framework_output_dir / 'comparison_results_detailed.csv') if not comparison_results_df.empty else None,
        'publication_package_summary_path': str(output_path / 'publication_package_summary.json'),
        'trained_models_dir': str(models_dir),
        'wandb_run_url': wandb_run.url if wandb_run else None
    }
    with open(output_path / 'final_research_results.json', 'w') as f: json.dump(final_results_summary, f, indent=2, default=lambda o: f"<not_serializable_{type(o).__name__}>")
    if wandb_run: wandb.save(str(output_path / 'final_research_results.json'))
    
    summary_report_path = output_path / 'RESEARCH_SUMMARY.txt'
    with open(summary_report_path, 'w') as f:
        f.write(f"Research pipeline completed. Full summary in final_research_results.json and log file at {output_path / 'research.log'}\n")
        f.write(f"Output directory: {output_path}\n")
        if best_optuna_params: f.write(f"Optuna Best Params: {best_optuna_params}\n")
        f.write(f"Training Duration: {training_duration_hours:.2f} hours\n")
        if wandb_run: f.write(f"W&B Run: {wandb_run.url}\n")
    if wandb_run: wandb.save(str(summary_report_path))

    monitor.log_progress("COMPLETE", f"Research pipeline finished. Results in {output_path}")
    if wandb_run: wandb_run.finish()
    return final_results_summary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    script_logger.info("=" * 80 + "\nASL NEURAL NETWORK COMPREHENSIVE RESEARCH PIPELINE - Enhanced\n" + "=" * 80)
    config_file_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml" # Default to default.yaml
    loaded_config = ResearchConfig()
    if Path(config_file_path).exists():
        script_logger.info(f"Loading configuration from {config_file_path}")
        try:
            with open(config_file_path, 'r') as f:
                config_dict_loaded = yaml.safe_load(f)
            # Smart merging of config:
            # Iterate over top-level keys in loaded_config (dataclass)
            for key_dataclass in asdict(loaded_config).keys():
                if key_dataclass in config_dict_loaded: # If key exists in YAML
                    # Handle nested structures (like 'training', 'data', 'simulation', 'optuna')
                    if isinstance(getattr(loaded_config, key_dataclass), dict) and isinstance(config_dict_loaded[key_dataclass], dict):
                        getattr(loaded_config, key_dataclass).update(config_dict_loaded[key_dataclass])
                    elif isinstance(config_dict_loaded[key_dataclass], (list, dict)) and key_dataclass in ['training', 'data', 'model', 'simulation', 'optuna', 'logging', 'paths', 'wandb']:
                         # For cases where the dataclass attribute is a simple type but YAML has a dict
                         # This logic assumes flat structure in dataclass for these sections, 
                         # and YAML provides specific overrides for those attributes
                         sub_config_dict = config_dict_loaded[key_dataclass]
                         if isinstance(sub_config_dict, dict):
                             for sub_k, sub_v in sub_config_dict.items():
                                 if hasattr(loaded_config, sub_k): # Check if it's a direct attribute of ResearchConfig
                                     setattr(loaded_config, sub_k, sub_v)
                                 # elif hasattr(getattr(loaded_config, key_dataclass, None), sub_k): # Check if it's an attribute of a nested dataclass/dict
                                 #     setattr(getattr(loaded_config, key_dataclass), sub_k, sub_v)
                         else: # If YAML value for the key is not a dict, assign directly
                            setattr(loaded_config, key_dataclass, config_dict_loaded[key_dataclass])

                    else: # Direct assignment for simple types
                        setattr(loaded_config, key_dataclass, config_dict_loaded[key_dataclass])
                # For keys in YAML not directly in ResearchConfig but potentially part of sub-dicts like 'training.batch_size'
                elif '.' in key_dataclass: # Simplistic check, assumes flat YAML for sub-keys
                    pass # More complex parsing would be needed here, current structure is better
                
            # After initial loading based on dataclass fields, explicitly load known dict-like sections if they exist in YAML
            for section_name in ['training', 'data', 'simulation', 'optuna', 'wandb']:
                 if section_name in config_dict_loaded and isinstance(config_dict_loaded[section_name], dict):
                     for k,v in config_dict_loaded[section_name].items():
                         if hasattr(loaded_config, k):
                             setattr(loaded_config, k, v)


        except Exception as e: script_logger.error(f"Error loading config {config_file_path}: {e}. Using defaults or partially loaded config.")
    else: script_logger.info(f"Config file {config_file_path} not found. Using default ResearchConfig.")
    
    script_logger.info("\nResearch Configuration:\n" + "-" * 30 + "\n" + "\n".join([f"{k}: {v}" for k,v in asdict(loaded_config).items()]) + "\n" + "-" * 30)
    script_logger.info("\nStarting comprehensive ASL research pipeline...")
    pipeline_results = run_comprehensive_asl_research(config=loaded_config)
    script_logger.info("\n" + "=" * 80 + "\nRESEARCH PIPELINE COMPLETED!\n" + "=" * 80)
    if "error" not in pipeline_results:
        script_logger.info(f"Results saved in: {pipeline_results.get('trained_models_dir', 'Specified output directory')}")
        script_logger.info("Check RESEARCH_SUMMARY.txt and final_research_results.json for detailed findings.")
        if pipeline_results.get('wandb_run_url'):
            script_logger.info(f"W&B Run: {pipeline_results['wandb_run_url']}")
    else: script_logger.error(f"Pipeline failed: {pipeline_results.get('error')}")

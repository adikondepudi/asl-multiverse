# main.py

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
import os

warnings.filterwarnings('ignore', category=UserWarning)

from enhanced_asl_network import EnhancedASLNet
from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer, ASLIterableDataset, EnhancedASLDataset
from torch.utils.data import DataLoader
from comparison_framework import ComprehensiveComparison
from performance_metrics import ProposalEvaluator
from single_repeat_validation import SingleRepeatValidator, run_single_repeat_validation_main

script_logger = logging.getLogger(__name__)

@dataclass
class ResearchConfig:
    # Training parameters
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 256
    val_split: float = 0.2
    
    # --- MODIFIED: Switched from n_subjects to explicit steps for IterableDataset ---
    steps_per_epoch_stage1: int = 20
    steps_per_epoch_stage2: int = 40
    n_epochs_stage1: int = 140
    n_epochs_stage2: int = 60
    
    loss_pinn_weight_stage1: float = 1.0
    loss_pinn_weight_stage2: float = 0.1
    pre_estimator_loss_weight_stage1: float = 1.0 
    pre_estimator_loss_weight_stage2: float = 0.0 
    
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

    T1_artery: float = 1850.0; T2_factor: float = 1.0; alpha_BS1: float = 1.0
    alpha_PCASL: float = 0.85; alpha_VSASL: float = 0.56; T_tau: float = 1800.0
    
    training_noise_levels_stage1: List[float] = field(default_factory=lambda: [3.0, 5.0, 10.0, 15.0])
    training_noise_levels_stage2: List[float] = field(default_factory=lambda: [10.0, 15.0, 20.0])
    
    # --- MODIFIED: Increased validation subjects for robustness ---
    n_validation_subjects: int = 10000

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

    wandb_project: str = "asl-multiverse-project"
    wandb_entity: Optional[str] = None 

def engineer_signal_features(raw_signal: np.ndarray, num_plds: int) -> np.ndarray:
    if raw_signal.ndim == 1: raw_signal = raw_signal[np.newaxis, :]
    num_samples = raw_signal.shape[0]
    engineered_features = np.zeros((num_samples, 4))
    plds_indices = np.arange(num_plds)
    for i in range(num_samples):
        pcasl_curve, vsasl_curve = raw_signal[i, :num_plds], raw_signal[i, num_plds:]
        engineered_features[i, 0] = np.argmax(pcasl_curve)
        engineered_features[i, 1] = np.argmax(vsasl_curve)
        pcasl_sum = np.sum(pcasl_curve) + 1e-6
        vsasl_sum = np.sum(vsasl_curve) + 1e-6
        engineered_features[i, 2] = np.sum(pcasl_curve * plds_indices) / pcasl_sum
        engineered_features[i, 3] = np.sum(vsasl_curve * plds_indices) / vsasl_sum
    return engineered_features.astype(np.float32)

class CollateFn:
    """A collate function to process batches from the IterableDataset."""
    def __init__(self, norm_stats, num_plds):
        self.norm_stats = norm_stats
        self.num_plds = num_plds
        self.num_raw_signal_features = num_plds * 2

    def __call__(self, batch):
        raw_signals, raw_params = zip(*batch)
        raw_signals = np.array(raw_signals)
        raw_params = np.array(raw_params)

        # 1. Engineer features from raw signals
        eng_features = engineer_signal_features(raw_signals, self.num_plds)
        
        # 2. Normalize raw signals
        pcasl_norm = (raw_signals[:, :self.num_plds] - self.norm_stats['pcasl_mean']) / (self.norm_stats['pcasl_std'] + 1e-6)
        vsasl_norm = (raw_signals[:, self.num_plds:] - self.norm_stats['vsasl_mean']) / (self.norm_stats['vsasl_std'] + 1e-6)
        
        # 3. Concatenate to form the final model input
        final_input = np.concatenate([pcasl_norm, vsasl_norm, eng_features], axis=1)

        # 4. Normalize the target parameters
        params_norm = np.column_stack([
            (raw_params[:, 0] - self.norm_stats['y_mean_cbf']) / self.norm_stats['y_std_cbf'],
            (raw_params[:, 1] - self.norm_stats['y_mean_att']) / self.norm_stats['y_std_att']
        ])
        
        return torch.from_numpy(final_input.astype(np.float32)), torch.from_numpy(params_norm.astype(np.float32))

class PerformanceMonitor: # Placeholder
    def __init__(self, *args, **kwargs): pass
    def log_progress(self, *args, **kwargs): script_logger.info(f"{args} {kwargs}")
    def check_target_achievement(self, *args, **kwargs): return {}
class HyperparameterOptimizer: # Placeholder
    def __init__(self, *args, **kwargs): pass
    def optimize(self): return {}
class ClinicalValidator: # Placeholder
    def __init__(self, *args, **kwargs): pass
    def validate_clinical_scenarios(self, *args, **kwargs): return {}
class PublicationGenerator: # Placeholder
    def __init__(self, *args, **kwargs): pass
    def generate_publication_package(self, *args, **kwargs): return {}


def run_comprehensive_asl_research(config: ResearchConfig, output_parent_dir: Optional[str] = None) -> Dict:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_parent_dir or f'comprehensive_results/asl_research_{timestamp}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.backends.cudnn.benchmark = True

    wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=asdict(config), name=f"run_{timestamp}", job_type="research_pipeline")
    if wandb_run: script_logger.info(f"W&B Run URL: {wandb_run.url}")

    monitor = PerformanceMonitor(config, output_path) 
    monitor.log_progress("SETUP", f"Initializing. Output: {output_path}")

    with open(output_path / 'research_config.json', 'w') as f: json.dump(asdict(config), f, indent=2)
    
    plds_np = np.array(config.pld_values)
    num_plds = len(plds_np)
    asl_params_sim = ASLParameters(**{k:v for k,v in asdict(config).items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params_sim)

    monitor.log_progress("PHASE2", "Starting two-stage ensemble training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    monitor.log_progress("PHASE2", f"Using device: {device}")

    monitor.log_progress("PHASE2", "Generating a fixed dataset for normalization stats...")
    norm_dataset = simulator.generate_diverse_dataset(plds=plds_np, n_subjects=5000, conditions=['healthy'], noise_levels=config.training_noise_levels_stage1)
    
    raw_signals_norm, raw_params_norm = norm_dataset['signals'], norm_dataset['parameters']
    norm_stats_final = {
        'pcasl_mean': np.mean(raw_signals_norm[:, :num_plds], axis=0).tolist(),
        'pcasl_std': np.std(raw_signals_norm[:, :num_plds], axis=0).tolist(),
        'vsasl_mean': np.mean(raw_signals_norm[:, num_plds:], axis=0).tolist(),
        'vsasl_std': np.std(raw_signals_norm[:, num_plds:], axis=0).tolist(),
        'y_mean_cbf': np.mean(raw_params_norm[:, 0]), 'y_std_cbf': np.std(raw_params_norm[:, 0]),
        'y_mean_att': np.mean(raw_params_norm[:, 1]), 'y_std_att': np.std(raw_params_norm[:, 1]),
    }
    
    # Clip small standard deviations to 1.0 to prevent division by zero
    for key in ['pcasl_std', 'vsasl_std']:
        norm_stats_final[key] = np.clip(norm_stats_final[key], a_min=1e-6, a_max=None).tolist()
    for key in ['y_std_cbf', 'y_std_att']:
        norm_stats_final[key] = max(norm_stats_final[key], 1e-6)

    amplitudes = np.linalg.norm(raw_signals_norm, axis=1)
    norm_stats_final['amplitude_mean'] = np.mean(amplitudes)
    norm_stats_final['amplitude_std'] = max(np.std(amplitudes), 1e-6)

    train_dataset_s1 = ASLIterableDataset(simulator, plds_np, config.training_noise_levels_stage1)
    train_dataset_s2 = ASLIterableDataset(simulator, plds_np, config.training_noise_levels_stage2)
    
    collate_fn = CollateFn(norm_stats_final, num_plds)
    
    # --- MODIFIED: Use SLURM environment variables for optimal worker count on HPC ---
    try:
        # Best for SLURM: Use the number of CPUs allocated to the task
        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    except (ValueError, TypeError):
        # Fallback if the env var is not set or not a number
        num_workers = os.cpu_count()
    script_logger.info(f"Using {num_workers} DataLoader workers.")

    train_loader_s1 = DataLoader(train_dataset_s1, batch_size=config.batch_size, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=(num_workers > 0))
    train_loader_s2 = DataLoader(train_dataset_s2, batch_size=config.batch_size, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=(num_workers > 0))

    monitor.log_progress("PHASE2", f"Generating a fixed validation set with {config.n_validation_subjects} subjects...")
    val_dataset_fixed = simulator.generate_diverse_dataset(plds=plds_np, n_subjects=config.n_validation_subjects, conditions=['healthy'], noise_levels=config.training_noise_levels_stage1)
    
    val_raw_signals, val_raw_params = val_dataset_fixed['signals'], val_dataset_fixed['parameters']
    val_eng_features = engineer_signal_features(val_raw_signals, num_plds)
    val_pcasl_norm = (val_raw_signals[:, :num_plds] - norm_stats_final['pcasl_mean']) / (np.array(norm_stats_final['pcasl_std']) + 1e-6)
    val_vsasl_norm = (val_raw_signals[:, num_plds:] - norm_stats_final['vsasl_mean']) / (np.array(norm_stats_final['vsasl_std']) + 1e-6)
    val_input_final = np.concatenate([val_pcasl_norm, val_vsasl_norm, val_eng_features], axis=1)
    val_params_norm = np.column_stack([
        (val_raw_params[:, 0] - norm_stats_final['y_mean_cbf']) / norm_stats_final['y_std_cbf'],
        (val_raw_params[:, 1] - norm_stats_final['y_mean_att']) / norm_stats_final['y_std_att']
    ])
    val_dataset = EnhancedASLDataset(val_input_final, val_params_norm)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, num_workers=num_workers, pin_memory=True, drop_last=False, persistent_workers=(num_workers > 0))

    base_input_size_nn = num_plds * 2 + 4
    model_creation_config = asdict(config)
    def create_main_model_closure(**kwargs): return EnhancedASLNet(input_size=base_input_size_nn, **kwargs)

    trainer = EnhancedASLTrainer(model_config=model_creation_config, model_class=create_main_model_closure, input_size=base_input_size_nn, learning_rate=config.learning_rate, weight_decay=config.weight_decay, batch_size=config.batch_size, n_ensembles=config.n_ensembles, device=device, n_plds_for_model=num_plds)
    trainer.norm_stats = norm_stats_final
    trainer.custom_loss_fn.norm_stats = norm_stats_final
    for model in trainer.models: model.set_norm_stats(norm_stats_final)
    
    # --- MODIFIED: Use explicit steps from config, not calculated from n_subjects ---
    steps_per_epoch_s1 = config.steps_per_epoch_stage1
    steps_per_epoch_s2 = config.steps_per_epoch_stage2
    total_steps = steps_per_epoch_s1 * config.n_epochs_stage1 + steps_per_epoch_s2 * config.n_epochs_stage2

    for opt in trainer.optimizers:
        trainer.schedulers.append(torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.learning_rate, total_steps=total_steps))
    
    monitor.log_progress("PHASE2", f"Training {config.n_ensembles}-model ensemble...")
    training_start_time = time.time()
    
    trainer.train_ensemble(
        train_loaders=[train_loader_s1, train_loader_s2],
        val_loaders=[val_loader, val_loader],
        epoch_schedule=[config.n_epochs_stage1, config.n_epochs_stage2],
        steps_per_epoch_schedule=[steps_per_epoch_s1, steps_per_epoch_s2],
        early_stopping_patience=25 
    )

    training_duration_hours = (time.time() - training_start_time) / 3600
    monitor.log_progress("PHASE2", f"Training completed in {training_duration_hours:.2f} hours.")
    
    # The rest of the pipeline remains the same
    # ...
    
    return {} # Return dummy dict

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True) 
    config_file_path_arg = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    output_dir_arg = sys.argv[2] if len(sys.argv) > 2 else None
    loaded_config_obj = ResearchConfig()
    if Path(config_file_path_arg).exists():
        with open(config_file_path_arg, 'r') as f_yaml: config_from_yaml = yaml.safe_load(f_yaml) or {}
        all_yaml_params = {}
        for key, value in config_from_yaml.items():
            if isinstance(value, dict): all_yaml_params.update(value)
            else: all_yaml_params[key] = value
        for key, value in all_yaml_params.items():
            if hasattr(loaded_config_obj, key): setattr(loaded_config_obj, key, value)
    pipeline_results_dict = run_comprehensive_asl_research(config=loaded_config_obj, output_parent_dir=output_dir_arg)
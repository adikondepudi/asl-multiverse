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
import argparse
import copy

warnings.filterwarnings('ignore', category=UserWarning)

from enhanced_asl_network import EnhancedASLNet
from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer, ASLIterableDataset
from torch.utils.data import DataLoader
from comparison_framework import ComprehensiveComparison
from performance_metrics import ProposalEvaluator
from single_repeat_validation import SingleRepeatValidator, run_single_repeat_validation_main
from utils import engineer_signal_features, ParallelStreamingStatsCalculator

script_logger = logging.getLogger(__name__)

@dataclass
class ResearchConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 256
    steps_per_epoch_stage1: int = 20
    steps_per_epoch_stage2: int = 40
    n_epochs_stage1: int = 140
    n_epochs_stage2: int = 60
    validation_steps_per_epoch: int = 50
    learning_rate_stage2: float = 0.0001
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
    optuna_n_trials: int = 50
    optuna_timeout_hours: float = 4.0
    optuna_n_subjects_for_norm: int = 2000
    optuna_n_epochs: int = 25
    optuna_steps_per_epoch: int = 15
    optuna_study_name: str = "asl_multiverse_hpo"
    pld_values: List[int] = field(default_factory=lambda: list(range(500, 3001, 500)))
    att_ranges_config: List[Tuple[float, float, str]] = field(default_factory=lambda: [(500.0, 1500.0, "Short ATT"),(1500.0, 2500.0, "Medium ATT"),(2500.0, 4000.0, "Long ATT")])
    T1_artery: float = 1850.0; T2_factor: float = 1.0; alpha_BS1: float = 1.0
    alpha_PCASL: float = 0.85; alpha_VSASL: float = 0.56; T_tau: float = 1800.0
    training_noise_levels_stage1: List[float] = field(default_factory=lambda: [3.0, 5.0, 10.0, 15.0])
    training_noise_levels_stage2: List[float] = field(default_factory=lambda: [10.0, 15.0, 20.0])
    n_test_subjects_per_att_range: int = 200 
    test_snr_levels: List[float] = field(default_factory=lambda: [5.0, 10.0])
    test_conditions: List[str] = field(default_factory=lambda: ['healthy', 'stroke'])
    n_clinical_scenario_subjects: int = 100 
    clinical_scenario_definitions: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {'healthy_adult': {'cbf_range': (50.0, 80.0), 'att_range': (800.0, 1800.0), 'snr': 8.0},'elderly_patient': {'cbf_range': (30.0, 60.0), 'att_range': (1500.0, 3000.0), 'snr': 5.0},'stroke_patient': {'cbf_range': (10.0, 40.0), 'att_range': (2000.0, 4000.0), 'snr': 3.0},'tumor_patient': {'cbf_range': (20.0, 120.0), 'att_range': (1000.0, 3000.0), 'snr': 6.0}})
    wandb_project: str = "asl-multiverse-project"
    wandb_entity: Optional[str] = None

### MODIFIED ### - Function now accepts pre-computed norm_stats
def objective(trial: optuna.Trial, base_config: ResearchConfig, output_dir: Path, norm_stats: Dict) -> float:
    """
    The objective function for Optuna hyperparameter optimization.
    Trains a model with a given set of hyperparameters and returns the validation loss.
    """
    trial_config = copy.deepcopy(base_config)
    
    trial_config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    trial_config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    trial_config.dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.4)
    hidden_size_1 = trial.suggest_categorical("hidden_size_1", [128, 256, 512])
    hidden_size_2 = trial.suggest_categorical("hidden_size_2", [64, 128, 256])
    hidden_size_3 = trial.suggest_categorical("hidden_size_3", [32, 64, 128])
    trial_config.hidden_sizes = [hidden_size_1, hidden_size_2, hidden_size_3]
    trial_config.loss_pinn_weight_stage1 = trial.suggest_float("loss_pinn_weight_stage1", 0.1, 10.0, log=True)
    trial_config.loss_pinn_weight_stage2 = trial.suggest_float("loss_pinn_weight_stage2", 0.0, 1.0)
    
    script_logger.info(f"--- Optuna Trial {trial.number} ---")
    script_logger.info(f"  Params: lr={trial_config.learning_rate:.5f}, wd={trial_config.weight_decay:.6f}, dropout={trial_config.dropout_rate:.3f}")
    
    trial_config.n_ensembles = 1
    trial_config.n_epochs_stage1 = base_config.optuna_n_epochs
    trial_config.n_epochs_stage2 = 0
    trial_config.steps_per_epoch_stage1 = base_config.optuna_steps_per_epoch
    
    plds_np = np.array(trial_config.pld_values)
    num_plds = len(plds_np)
    asl_params_sim = ASLParameters(**{k:v for k,v in asdict(trial_config).items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params_sim)
    
    ### MODIFIED ### - The per-trial stats calculation is REMOVED. norm_stats is now passed in.
    
    train_dataset = ASLIterableDataset(simulator, plds_np, trial_config.training_noise_levels_stage1, norm_stats=norm_stats)
    val_dataset = ASLIterableDataset(simulator, plds_np, trial_config.training_noise_levels_stage1, norm_stats=norm_stats)
    
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=trial_config.batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=trial_config.batch_size * 2, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_input_size_nn = num_plds * 2 + 4
    
    model_config = asdict(trial_config)
    trainer = EnhancedASLTrainer(model_config=model_config, model_class=lambda **kwargs: EnhancedASLNet(input_size=base_input_size_nn, **kwargs),
                                 input_size=base_input_size_nn, learning_rate=trial_config.learning_rate, weight_decay=trial_config.weight_decay, 
                                 batch_size=trial_config.batch_size, n_ensembles=trial_config.n_ensembles, device=device)
    
    trainer.norm_stats = norm_stats
    trainer.custom_loss_fn.norm_stats = norm_stats
    for model in trainer.models: model.set_norm_stats(norm_stats)

    try:
        train_results = trainer.train_ensemble(
            train_loaders=[train_loader], val_loaders=[val_loader],
            epoch_schedule=[trial_config.n_epochs_stage1],
            steps_per_epoch_schedule=[trial_config.steps_per_epoch_stage1],
            early_stopping_patience=5,
            optuna_trial=trial
        )
        return train_results.get('final_mean_val_loss', float('inf'))

    except optuna.exceptions.TrialPruned:
        script_logger.info(f"Trial {trial.number} pruned.")
        return float('inf')

### MODIFIED ### - Function now accepts pre-computed norm_stats
def run_hyperparameter_optimization(config: ResearchConfig, output_dir: Path, norm_stats: Dict):
    """Manages the Optuna HPO study, including saving and resuming."""
    study_name = config.optuna_study_name
    storage_path = output_dir / f"{study_name}.db"
    study_journal_path = output_dir / f"{study_name}_study.pkl"

    script_logger.info(f"--- Starting Hyperparameter Optimization ---")
    script_logger.info(f"Study Name: {study_name}")
    script_logger.info(f"Output Directory: {output_dir}")
    
    if study_journal_path.exists():
        script_logger.info(f"Resuming study from {study_journal_path}")
        study = joblib.load(study_journal_path)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name)
        script_logger.info("Created a new Optuna study.")

    try:
        ### MODIFIED ### - Pass norm_stats into the objective function via lambda
        study.optimize(
            lambda trial: objective(trial, config, output_dir, norm_stats),
            n_trials=config.optuna_n_trials,
            timeout=config.optuna_timeout_hours * 3600,
            callbacks=[lambda s, t: joblib.dump(s, study_journal_path)]
        )
    except KeyboardInterrupt:
        script_logger.warning("HPO interrupted. Saving current state.")
    
    joblib.dump(study, study_journal_path)
    script_logger.info(f"HPO finished. Study saved to {study_journal_path}")
    
    best_params_path = output_dir / "best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
        
    script_logger.info(f"Best trial: {study.best_trial.number}")
    script_logger.info(f"  Value (Validation Loss): {study.best_trial.value:.4f}")
    script_logger.info("  Best Parameters:")
    for key, value in study.best_trial.params.items():
        script_logger.info(f"    {key}: {value}")
    script_logger.info(f"Best parameters saved to {best_params_path}")

### MODIFIED ### - Function now accepts pre-computed norm_stats
def run_comprehensive_asl_research(config: ResearchConfig, output_dir: Path, norm_stats: Dict) -> Dict:
    """The main research pipeline for a full training run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=asdict(config), name=f"run_{output_dir.name}", job_type="research_pipeline")
    if wandb_run: script_logger.info(f"W&B Run URL: {wandb_run.url}")

    with open(output_dir / 'research_config.json', 'w') as f: json.dump(asdict(config), f, indent=2)
    
    plds_np = np.array(config.pld_values)
    num_plds = len(plds_np)
    asl_params_sim = ASLParameters(**{k:v for k,v in asdict(config).items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params_sim)

    script_logger.info("Starting two-stage ensemble training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_logger.info(f"Using device: {device}")

    ### MODIFIED ### - Stats calculation block is REMOVED. We use the passed-in norm_stats.
    norm_stats_final = norm_stats
    
    train_dataset_s1 = ASLIterableDataset(simulator, plds_np, config.training_noise_levels_stage1, norm_stats=norm_stats_final)
    train_dataset_s2 = ASLIterableDataset(simulator, plds_np, config.training_noise_levels_stage2, norm_stats=norm_stats_final)
    
    try:
        # Use a reasonable cap, e.g., 16, to prevent excessive overhead
        cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
        num_workers = min(cpu_count, 16) 
    except (ValueError, TypeError):
        num_workers = min(os.cpu_count(), 16)
    script_logger.info(f"Using {num_workers} DataLoader workers (capped at 16).")

    train_loader_s1 = DataLoader(train_dataset_s1, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    train_loader_s2 = DataLoader(train_dataset_s2, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    script_logger.info("Creating iterable validation dataset...")
    val_dataset_iterable = ASLIterableDataset(simulator, plds_np, config.training_noise_levels_stage1, norm_stats=norm_stats_final)
    val_loader = DataLoader(val_dataset_iterable, batch_size=config.batch_size * 2, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    base_input_size_nn = num_plds * 2 + 4
    model_creation_config = asdict(config)
    def create_main_model_closure(**kwargs): return EnhancedASLNet(input_size=base_input_size_nn, **kwargs)

    trainer = EnhancedASLTrainer(model_config=model_creation_config, model_class=create_main_model_closure, input_size=base_input_size_nn, learning_rate=config.learning_rate, weight_decay=config.weight_decay, batch_size=config.batch_size, n_ensembles=config.n_ensembles, device=device, n_plds_for_model=num_plds)
    trainer.norm_stats = norm_stats_final
    trainer.custom_loss_fn.norm_stats = norm_stats_final
    for model in trainer.models: model.set_norm_stats(norm_stats_final)
    
    script_logger.info(f"Training {config.n_ensembles}-model ensemble...")
    training_start_time = time.time()
    
    trainer.train_ensemble(
        train_loaders=[train_loader_s1, train_loader_s2],
        val_loaders=[val_loader, val_loader],
        epoch_schedule=[config.n_epochs_stage1, config.n_epochs_stage2],
        steps_per_epoch_schedule=[config.steps_per_epoch_stage1, config.steps_per_epoch_stage2],
        early_stopping_patience=25 
    )

    training_duration_hours = (time.time() - training_start_time) / 3600
    script_logger.info(f"Training completed in {training_duration_hours:.2f} hours.")
    
    # ... (evaluation phases would follow here) ...
    
    return {}

### MODIFIED ### - Entire main execution block is updated for unified stats calculation.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True) 
    
    parser = argparse.ArgumentParser(description="Run ASL Multiverse Training or Hyperparameter Optimization.")
    parser.add_argument("config_file", type=str, help="Path to the base YAML configuration file.")
    parser.add_argument("output_dir", type=str, nargs='?', default=None, help="Path to the output directory. If not provided, a timestamped one will be created.")
    parser.add_argument("--optimize", action="store_true", help="Run in hyperparameter optimization (HPO) mode instead of a full training run.")
    parser.add_argument("--study-name", type=str, default=None, help="Name for the Optuna study. Used for resuming HPO runs.")

    args = parser.parse_args()

    # Load configuration
    config_obj = ResearchConfig()
    if Path(args.config_file).exists():
        with open(args.config_file, 'r') as f_yaml: config_from_yaml = yaml.safe_load(f_yaml) or {}
        for section, params in config_from_yaml.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    if hasattr(config_obj, key): setattr(config_obj, key, value)
    
    # Determine output directory and create it
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = "hpo" if args.optimize else "train"
        output_path = Path(f'comprehensive_results/asl_research_{mode}_{timestamp}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.study_name:
        config_obj.optuna_study_name = args.study_name

    # --- NEW: Centralized Normalization Statistics Calculation ---
    norm_stats_path = output_path / 'norm_stats.json'
    if norm_stats_path.exists():
        script_logger.info(f"Loading cached normalization stats from {norm_stats_path}")
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
    else:
        script_logger.info("Normalization stats not found. Calculating now...")
        plds_np = np.array(config_obj.pld_values)
        asl_params_sim = ASLParameters(**{k:v for k,v in asdict(config_obj).items() if k in ASLParameters.__annotations__})
        simulator = RealisticASLSimulator(params=asl_params_sim)
        try:
            num_stat_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
        except (ValueError, TypeError):
            num_stat_workers = os.cpu_count()

        stats_calculator = ParallelStreamingStatsCalculator(
            simulator=simulator, plds=plds_np, num_samples=20000, num_workers=num_stat_workers
        )
        norm_stats = stats_calculator.calculate()
        
        script_logger.info(f"Saving unified normalization stats to {norm_stats_path}")
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)

    # --- Execute requested mode with unified norm_stats ---
    if args.optimize:
        run_hyperparameter_optimization(config=config_obj, output_dir=output_path, norm_stats=norm_stats)
    else:
        pipeline_results_dict = run_comprehensive_asl_research(config=config_obj, output_dir=output_path, norm_stats=norm_stats)
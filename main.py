# FILE: main.py
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

from enhanced_asl_network import DisentangledASLNet, CustomLoss
from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer, ASLIterableDataset, ASLInMemoryDataset
from torch.utils.data import DataLoader
from utils import ParallelStreamingStatsCalculator

script_logger = logging.getLogger(__name__)

@dataclass
class ResearchConfig:
    model_class_name: str = "DisentangledASLNet"
    encoder_type: str = 'conv1d'
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    use_offline_dataset: bool = True
    offline_dataset_path: Optional[str] = None
    n_epochs: int = 100
    steps_per_epoch: int = 1000
    weight_decay: float = 1e-5
    batch_size: int = 256
    validation_steps_per_epoch: int = 50
    n_ensembles: int = 5
    dropout_rate: float = 0.1
    norm_type: str = 'batch'
    log_var_cbf_min: float = -6.0
    log_var_cbf_max: float = 7.0
    log_var_att_min: float = -2.0
    log_var_att_max: float = 14.0
    loss_weight_cbf: float = 1.0
    loss_weight_att: float = 1.0
    loss_log_var_reg_lambda: float = 0.0
    pld_values: List[int] = field(default_factory=lambda: list(range(500, 3001, 500)))
    T1_artery: float = 1850.0; T2_factor: float = 1.0; alpha_BS1: float = 1.0
    alpha_PCASL: float = 0.85; alpha_VSASL: float = 0.56; T_tau: float = 1800.0
    training_noise_levels: List[float] = field(default_factory=lambda: [3.0, 5.0, 10.0, 15.0, 20.0])
    wandb_project: str = "asl-multiverse-project"
    wandb_entity: Optional[str] = None
    num_samples: int = 1000000
    pretrained_encoder_path: Optional[str] = None
    moe: Optional[Dict[str, Any]] = None
    transformer_d_model_focused: int = 32      
    transformer_nhead_model: int = 4
    transformer_nlayers_model: int = 2

def run_comprehensive_asl_research(config: ResearchConfig, stage: int, output_dir: Path, norm_stats: Dict) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    run_name = wandb.run.name if wandb.run and wandb.run.name else f"run_{output_dir.name}"
    wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=asdict(config), name=run_name, job_type=f"stage_{stage}", reinit=True)
    if wandb_run: script_logger.info(f"W&B Run URL: {wandb_run.url}")

    with open(output_dir / 'research_config.json', 'w') as f: json.dump(asdict(config), f, indent=2)
    
    plds_np = np.array(config.pld_values)
    num_plds = len(plds_np)
    asl_params_sim = ASLParameters(**{k:v for k,v in asdict(config).items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params_sim)

    script_logger.info(f"--- Starting Stage {stage} Run ---")
    if stage == 1:
        script_logger.info("Mode: Self-supervised denoising autoencoder training.")
    elif stage == 2:
        script_logger.info("Mode: Supervised regression head fine-tuning with frozen encoder.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_logger.info(f"Using device: {device}")

    base_input_size_nn = num_plds * 2 + 4 + 1
    
    use_offline = config.use_offline_dataset
    offline_path = config.offline_dataset_path
    num_workers = min(os.cpu_count() or 1, 32)
    script_logger.info(f"Using a capped number of {num_workers} CPU cores for data loading.")

    if use_offline and offline_path:
        script_logger.info(f"Using OFFLINE In-Memory dataset from: {offline_path}")
        train_dataset = ASLInMemoryDataset(data_dir=offline_path, norm_stats=norm_stats, stage=stage, disentangled_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=num_workers,
                                     pin_memory=True, persistent_workers=(num_workers > 0), shuffle=True)
    else:
        script_logger.info("Using ON-THE-FLY IterableDataset for training.")
        train_dataset = ASLInMemoryDataset(simulator, plds_np, config.training_noise_levels, norm_stats, stage=stage, disentangled_mode=True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    
    val_dataset = ASLIterableDataset(simulator, plds_np, config.training_noise_levels, norm_stats, stage=stage, disentangled_mode=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    
    model_creation_config = asdict(config)
    model_mode = 'denoising' if stage == 1 else 'regression'
    def create_model_closure(**kwargs): return DisentangledASLNet(mode=model_mode, input_size=base_input_size_nn, **kwargs)

    trainer = EnhancedASLTrainer(stage=stage, model_config=model_creation_config, model_class=create_model_closure, 
                                 weight_decay=config.weight_decay, batch_size=config.batch_size, 
                                 n_ensembles=config.n_ensembles, device=device)
    
    fine_tuning_cfg = None
    if stage == 2:
        script_logger.info(f"Loading pre-trained encoder weights from: {config.pretrained_encoder_path}")
        encoder_state_dict = torch.load(config.pretrained_encoder_path, map_location=device)
        num_loaded = 0
        for model in trainer.models:
            if hasattr(model, 'encoder'):
                model.encoder.load_state_dict(encoder_state_dict, strict=True)
                num_loaded += 1
        script_logger.info(f"Successfully loaded pre-trained weights into {num_loaded} model encoders.")
        fine_tuning_cfg = {'enabled': True}

    script_logger.info(f"Training {config.n_ensembles}-model ensemble for {config.n_epochs} epochs...")
    training_start_time = time.time()
    
    trainer.train_ensemble(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        early_stopping_patience=25,
        fine_tuning_config=fine_tuning_cfg
    )

    training_duration_hours = (time.time() - training_start_time) / 3600
    script_logger.info(f"Training completed in {training_duration_hours:.2f} hours.")
    
    if stage == 1:
        script_logger.info("Saving pre-trained encoder...")
        encoder_path = output_dir / 'encoder_pretrained.pt'
        unwrapped_model = getattr(trainer.models[0], '_orig_mod', trainer.models[0])
        torch.save(unwrapped_model.encoder.state_dict(), encoder_path)
        script_logger.info(f"Saved pre-trained encoder from model 0 to {encoder_path}")
    else: # stage 2
        script_logger.info("Saving final trained ensemble models...")
        models_dir = output_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        num_saved = 0
        for i, state_dict in enumerate(trainer.best_states):
            if state_dict:
                model_path = models_dir / f'ensemble_model_{i}.pt'
                torch.save(state_dict, model_path)
                num_saved += 1
        script_logger.info(f"Successfully saved {num_saved} models.")
        if wandb.run and num_saved > 0:
            model_artifact = wandb.Artifact(name=f"asl_ensemble_{wandb_run.id}", type="model")
            model_artifact.add_dir(str(models_dir))
            wandb_run.log_artifact(model_artifact)

    if wandb.run:
        wandb.finish()
        
    return {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True) 
    
    parser = argparse.ArgumentParser(description="Run ASL Multiverse v5 Training Pipeline.")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file for the desired stage.")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True, help="Specify the training stage (1 for pre-training, 2 for fine-tuning).")
    parser.add_argument("--load-weights-from", type=str, help="Path to the Stage 1 output directory containing the 'encoder_pretrained.pt' file. Required for Stage 2.")
    parser.add_argument("--run-name", type=str, help="Optional specific name for the W&B run.")
    parser.add_argument("--output-dir", type=str, help="Optional path to the output directory. If not provided, a timestamped one will be created.")

    args = parser.parse_args()
    
    if args.stage == 2 and not args.load_weights_from:
        parser.error("--load-weights-from is required for --stage 2.")

    config_obj = ResearchConfig()
    if Path(args.config_file).exists():
        with open(args.config_file, 'r') as f_yaml:
            config_from_yaml = yaml.safe_load(f_yaml) or {}
        
        flat_config = {}
        for section, params in config_from_yaml.items():
            if isinstance(params, dict):
                flat_config.update(params)
        
        if 'moe' in config_from_yaml:
            flat_config['moe'] = config_from_yaml['moe']

        for key, value in flat_config.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)

    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name_prefix = args.run_name if args.run_name else f"asl_research_v5_stage{args.stage}"
        output_path = Path(f'comprehensive_results/{run_name_prefix}_{timestamp}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.stage == 2:
        encoder_path = Path(args.load_weights_from) / 'encoder_pretrained.pt'
        if not encoder_path.exists():
            script_logger.error(f"FATAL: Encoder file not found at expected path: {encoder_path}")
            sys.exit(1)
        config_obj.pretrained_encoder_path = str(encoder_path)

    if args.run_name:
        os.environ['WANDB_RUN_NAME'] = args.run_name

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
        num_stat_workers = os.cpu_count() or 1
        stats_calculator = ParallelStreamingStatsCalculator(
            simulator=simulator, plds=plds_np, num_samples=config_obj.num_samples, num_workers=num_stat_workers
        )
        norm_stats = stats_calculator.calculate()
        script_logger.info(f"Saving unified normalization stats to {norm_stats_path}")
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)

    run_comprehensive_asl_research(config=config_obj, stage=args.stage, output_dir=output_path, norm_stats=norm_stats)
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

from enhanced_asl_network import DisentangledASLNet, CustomLoss, PhysicsInformedASLProcessor
from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator, PhysiologicalVariation
from asl_trainer import EnhancedASLTrainer, ASLInMemoryDataset
from torch.utils.data import DataLoader
from utils import ParallelStreamingStatsCalculator, engineer_signal_features

script_logger = logging.getLogger(__name__)

@dataclass
class ResearchConfig:
    # ... (rest of the dataclass is unchanged) ...
    model_class_name: str = "DisentangledASLNet"
    encoder_type: str = 'physics_processor'
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    use_offline_dataset: bool = True
    offline_dataset_path: Optional[str] = None
    num_samples_to_load: Optional[int] = None
    n_epochs: int = 100
    steps_per_epoch: int = 1000
    weight_decay: float = 1e-5
    batch_size: int = 256
    validation_steps_per_epoch: int = 50
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
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
    fine_tuning: Optional[Dict[str, Any]] = None
    transformer_d_model_focused: int = 32      
    transformer_nhead_model: int = 4
    transformer_nlayers_model: int = 2

# --- NEW HELPER FUNCTION FOR BASELINE EXPERIMENT ---
def _generate_simple_validation_set(simulator: RealisticASLSimulator, plds: np.ndarray, n_subjects: int, conditions: List, noise_levels: List) -> Dict:
    """
    Generates a validation set using SIMPLE GAUSSIAN NOISE to match the baseline training data.
    This is a simplified, local version of the `generate_diverse_dataset` function.
    """
    dataset = {'signals': [], 'parameters': [], 'conditions': [], 'noise_levels': [], 'perturbed_params': []}
    physio_var = simulator.physio_var
    base_params = simulator.params
    num_plds = len(plds)

    condition_map = {
        'healthy': (physio_var.cbf_range, physio_var.att_range, physio_var.t1_artery_range),
        'stroke': (physio_var.stroke_cbf_range, physio_var.stroke_att_range, (physio_var.t1_artery_range[0]-100, physio_var.t1_artery_range[1]+100)),
        'tumor': (physio_var.tumor_cbf_range, physio_var.tumor_att_range, (physio_var.t1_artery_range[0]-150, physio_var.t1_artery_range[1]+150)),
        'elderly': (physio_var.elderly_cbf_range, physio_var.elderly_att_range, (physio_var.t1_artery_range[0]+50, physio_var.t1_artery_range[1]+150))
    }

    for _ in range(n_subjects):
        condition = np.random.choice(conditions)
        cbf_range, att_range, t1_range = condition_map.get(condition, (physio_var.cbf_range, physio_var.att_range, physio_var.t1_artery_range))

        cbf = np.random.uniform(*cbf_range)
        att = np.random.uniform(*att_range)
        t1_a = np.random.uniform(*t1_range)
        
        perturbed_t_tau = base_params.T_tau * (1 + np.random.uniform(*physio_var.t_tau_perturb_range))
        perturbed_alpha_pcasl = np.clip(base_params.alpha_PCASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.1)
        perturbed_alpha_vsasl = np.clip(base_params.alpha_VSASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.0)
        
        vsasl_clean = simulator._generate_vsasl_signal(plds, att, cbf, t1_a, perturbed_alpha_vsasl)
        pcasl_clean = simulator._generate_pcasl_signal(plds, att, cbf, t1_a, perturbed_t_tau, perturbed_alpha_pcasl)
        
        for snr in noise_levels:
            ref_signal_level = simulator._compute_reference_signal()
            noise_sd = ref_signal_level / snr
            noise_scaling = simulator.compute_tr_noise_scaling(plds)
            
            pcasl_noisy = pcasl_clean + noise_sd * noise_scaling['PCASL'] * np.random.randn(num_plds)
            vsasl_noisy = vsasl_clean + noise_sd * noise_scaling['VSASL'] * np.random.randn(num_plds)
            
            multiverse_signal_flat = np.concatenate([pcasl_noisy, vsasl_noisy])
            
            dataset['signals'].append(multiverse_signal_flat)
            dataset['parameters'].append([cbf, att])
            dataset['conditions'].append(condition)
            dataset['noise_levels'].append(snr)
            dataset['perturbed_params'].append({
                't1_artery': t1_a, 't_tau': perturbed_t_tau, 
                'alpha_pcasl': perturbed_alpha_pcasl, 'alpha_vsasl': perturbed_alpha_vsasl
            })

    dataset['signals'] = np.array(dataset['signals'])
    dataset['parameters'] = np.array(dataset['parameters'])
    return dataset
# --- END OF NEW HELPER FUNCTION ---

def run_comprehensive_asl_research(config: ResearchConfig, stage: int, output_dir: Path, norm_stats: Dict) -> Dict:
    # ... (function start is unchanged) ...
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
        script_logger.info("Mode: Supervised regression head fine-tuning.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_logger.info(f"Using device: {device}")

    # V6: Input size is 1 shape curve (num_plds*2) + 8 scalar features
    base_input_size_nn = num_plds * 2 + 8
    
    if not (config.use_offline_dataset and config.offline_dataset_path):
        script_logger.error("FATAL: Offline dataset is required. Please set `use_offline_dataset` to true and provide `offline_dataset_path`.")
        sys.exit(1)

    num_workers = min(os.cpu_count() or 1, 32)
    script_logger.info(f"Using a capped number of {num_workers} CPU cores for data loading.")

    script_logger.info(f"Using OFFLINE In-Memory dataset from: {config.offline_dataset_path}")
    train_dataset = ASLInMemoryDataset(
        data_dir=config.offline_dataset_path, norm_stats=norm_stats, stage=stage, 
        num_samples_to_load=config.num_samples_to_load
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=num_workers,
                                 pin_memory=True, persistent_workers=(num_workers > 0), shuffle=True)
    
    # --- CHANGED FOR BASELINE EXPERIMENT ---
    # The validation set is now generated using the new helper function that applies
    # only simple Gaussian noise, ensuring consistency with the training data.
    script_logger.info("Generating a fixed validation dataset with SIMPLE GAUSSIAN NOISE...")
    validation_data_dict = _generate_simple_validation_set(
        simulator=simulator, plds=plds_np, n_subjects=200, 
        conditions=['healthy', 'stroke', 'tumor', 'elderly'], 
        noise_levels=[5.0, 10.0, 15.0]
    )
    # --- END OF CHANGE ---

    val_signals_noisy_raw = validation_data_dict['signals']
    # NOTE: For validation, engineered features are calculated from noisy curves to simulate real-world usage
    val_features_eng = engineer_signal_features(val_signals_noisy_raw, num_plds)
    val_signals_for_processing = np.concatenate([val_signals_noisy_raw, val_features_eng], axis=1)

    val_dataset = ASLInMemoryDataset(data_dir=None, norm_stats=norm_stats, stage=stage)
    val_dataset.signals_noisy_unprocessed = val_signals_for_processing
    val_dataset.params_unnormalized = validation_data_dict['parameters']
    
    val_dataset.signals_processed = val_dataset._process_signals(val_dataset.signals_noisy_unprocessed)
    val_dataset.signals_tensor = torch.from_numpy(val_dataset.signals_processed.astype(np.float32))

    if stage == 1:
        clean_signals_for_val = []
        for idx, params in enumerate(tqdm(validation_data_dict['perturbed_params'], desc="Generating clean validation signals", leave=False)):
            true_cbf_val = validation_data_dict['parameters'][idx, 0]
            true_att_val = validation_data_dict['parameters'][idx, 1]
            vsasl_c = simulator._generate_vsasl_signal(plds_np, true_att_val, true_cbf_val, params['t1_artery'], params['alpha_vsasl'])
            pcasl_c = simulator._generate_pcasl_signal(plds_np, true_att_val, true_cbf_val, params['t1_artery'], params['t_tau'], params['alpha_pcasl'])
            clean_signals_for_val.append(np.concatenate([pcasl_c, vsasl_c]))
        val_dataset.targets_tensor = torch.from_numpy(np.array(clean_signals_for_val).astype(np.float32))
    else: # stage 2
        val_dataset.params_normalized = val_dataset._normalize_params(val_dataset.params_unnormalized)
        val_dataset.targets_tensor = torch.from_numpy(val_dataset.params_normalized.astype(np.float32))

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    
    # ... (rest of the run_comprehensive_asl_research function is unchanged) ...
    model_creation_config = asdict(config)
    model_mode = 'denoising' if stage == 1 else 'regression'
    def create_model_closure(**kwargs): return DisentangledASLNet(mode=model_mode, input_size=base_input_size_nn, **kwargs)

    trainer = EnhancedASLTrainer(stage=stage, model_config=model_creation_config, model_class=create_model_closure, 
                                 weight_decay=config.weight_decay, batch_size=config.batch_size, 
                                 n_ensembles=config.n_ensembles, device=device)
    
    fine_tuning_cfg = config.fine_tuning if stage == 2 else None
    
    if stage == 2:
        script_logger.info(f"Loading pre-trained encoder weights from: {config.pretrained_encoder_path}")
        encoder_state_dict = torch.load(config.pretrained_encoder_path, map_location=device)
        num_loaded = 0
        for model in trainer.models:
            if hasattr(model, 'encoder'):
                model.encoder.load_state_dict(encoder_state_dict, strict=True)
                num_loaded += 1
        script_logger.info(f"Successfully loaded pre-trained weights into {num_loaded} model encoders.")
        if fine_tuning_cfg is None:
            fine_tuning_cfg = {'enabled': True} # Default to fine-tuning if stage 2
        else:
            fine_tuning_cfg['enabled'] = fine_tuning_cfg.get('enabled', True)


    script_logger.info(f"Training {config.n_ensembles}-model ensemble for {config.n_epochs} epochs...")
    training_start_time = time.time()
    
    trainer.train_ensemble(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        output_dir=output_dir,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        fine_tuning_config=fine_tuning_cfg
    )

    training_duration_hours = (time.time() - training_start_time) / 3600
    script_logger.info(f"Training completed in {training_duration_hours:.2f} hours.")
    
    if stage == 1:
        encoder_path = output_dir / 'encoder_pretrained.pt'
        unwrapped_model = getattr(trainer.models[0], '_orig_mod', trainer.models[0])
        torch.save(unwrapped_model.encoder.state_dict(), encoder_path)
        script_logger.info(f"Saved final pre-trained encoder from model 0 to {encoder_path}")
    else: # stage 2
        script_logger.info(f"Best models were saved during training to {output_dir / 'trained_models'}")
        if wandb.run:
            model_artifact = wandb.Artifact(name=f"asl_ensemble_{wandb_run.id}", type="model")
            model_artifact.add_dir(str(output_dir / 'trained_models'))
            wandb_run.log_artifact(model_artifact)

    if wandb.run:
        wandb.finish()
        
    return {}
    
# ... (rest of the main.py file, including the __main__ block, is unchanged) ...
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
    
    config_path = Path(args.config_file)
    if not config_path.exists():
        script_logger.error(f"FATAL: Configuration file not found at: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f_yaml:
        config_from_yaml = yaml.safe_load(f_yaml) or {}

    if not config_from_yaml:
        script_logger.error(f"FATAL: Configuration file {config_path} is empty or invalid.")
        sys.exit(1)
    
    def apply_yaml_to_dataclass(data: Any, config_object: ResearchConfig):
        if isinstance(data, dict):
            for key, value in data.items():
                if hasattr(config_object, key) and isinstance(getattr(config_object, key), dict) and isinstance(value, dict):
                    getattr(config_object, key).update(value)
                elif hasattr(config_object, key):
                    setattr(config_object, key, value)
                
                if isinstance(value, (dict, list)):
                    apply_yaml_to_dataclass(value, config_object)
        
        elif isinstance(data, list):
            for item in data:
                apply_yaml_to_dataclass(item, config_object)

    apply_yaml_to_dataclass(config_from_yaml, config_obj)
    
    script_logger.info(f"Successfully loaded and applied configuration from {config_path}")
    
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
        num_stat_workers = min(os.cpu_count() or 1, 16)
        stats_calculator = ParallelStreamingStatsCalculator(
            simulator=simulator, plds=plds_np, num_samples=config_obj.num_samples, num_workers=num_stat_workers
        )
        norm_stats = stats_calculator.calculate()
        script_logger.info(f"Saving unified normalization stats to {norm_stats_path}")
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)

    run_comprehensive_asl_research(config=config_obj, stage=args.stage, output_dir=output_path, norm_stats=norm_stats)
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
import sys
import warnings
import wandb 
import os
import argparse
from dataclasses import dataclass, asdict, field

warnings.filterwarnings('ignore', category=UserWarning)

from enhanced_asl_network import DisentangledASLNet, PhysicsInformedASLProcessor
from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer, FastTensorDataLoader
from utils import ParallelStreamingStatsCalculator, engineer_signal_features

script_logger = logging.getLogger(__name__)

@dataclass
class ResearchConfig:
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
    log_var_cbf_min: float = -3.0 # Relaxed from -5.0
    log_var_cbf_max: float = 7.0
    log_var_att_min: float = -3.0 # Relaxed from -5.0
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

def _generate_simple_validation_set(simulator: RealisticASLSimulator, plds: np.ndarray, n_subjects: int, conditions: List, noise_levels: List) -> Dict:
    """Generates a fixed validation set."""
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
        cbf = np.random.uniform(*cbf_range); att = np.random.uniform(*att_range); t1_a = np.random.uniform(*t1_range)
        
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

def process_signals_cpu(signals_unnorm: np.ndarray, norm_stats: dict, num_plds: int) -> np.ndarray:
    """CPU version of preprocessing for validation data."""
    raw_curves = signals_unnorm[:, :num_plds * 2]
    eng_ttp_com = signals_unnorm[:, num_plds * 2:]

    pcasl_raw = raw_curves[:, :num_plds]
    vsasl_raw = raw_curves[:, num_plds:]

    pcasl_mu = np.mean(pcasl_raw, axis=1, keepdims=True)
    pcasl_sigma = np.std(pcasl_raw, axis=1, keepdims=True)
    pcasl_shape = (pcasl_raw - pcasl_mu) / (pcasl_sigma + 1e-6)

    vsasl_mu = np.mean(vsasl_raw, axis=1, keepdims=True)
    vsasl_sigma = np.std(vsasl_raw, axis=1, keepdims=True)
    vsasl_shape = (vsasl_raw - vsasl_mu) / (vsasl_sigma + 1e-6)

    shape_vector = np.concatenate([pcasl_shape, vsasl_shape], axis=1)
    scalar_features_unnorm = np.concatenate([pcasl_mu, pcasl_sigma, vsasl_mu, vsasl_sigma, eng_ttp_com], axis=1)
    
    s_mean = np.array(norm_stats['scalar_features_mean'])
    s_std = np.array(norm_stats['scalar_features_std']) + 1e-6
    scalar_features_norm = (scalar_features_unnorm - s_mean) / s_std

    return np.concatenate([shape_vector, scalar_features_norm], axis=1)

def run_comprehensive_asl_research(config: ResearchConfig, stage: int, output_dir: Path, norm_stats: Dict) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    
    # T4 Optimization: TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_name = wandb.run.name if wandb.run and wandb.run.name else f"run_{output_dir.name}"
    wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=asdict(config), name=run_name, job_type=f"stage_{stage}", reinit=True)
    if wandb_run: script_logger.info(f"W&B Run URL: {wandb_run.url}")

    with open(output_dir / 'research_config.json', 'w') as f: json.dump(asdict(config), f, indent=2)
    
    plds_np = np.array(config.pld_values)
    num_plds = len(plds_np)
    asl_params_sim = ASLParameters(**{k:v for k,v in asdict(config).items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params_sim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- OPTIMIZED DATA LOADING FOR HPC ---
    # 1. Load CLEAN signals into RAM
    script_logger.info(f"Loading raw CLEAN data to RAM from {config.offline_dataset_path}...")
    files = sorted(list(Path(config.offline_dataset_path).glob('dataset_chunk_*.npz')))
    all_signals_clean = []
    all_params = []
    samples_loaded = 0
    
    for f in files:
        d = np.load(f)
        all_signals_clean.append(d['signals_clean']) # Load CLEAN signals
        all_params.append(d['params'])
        samples_loaded += len(d['signals_clean'])
        if config.num_samples_to_load and samples_loaded >= config.num_samples_to_load:
            break
    
    raw_signals_clean = np.concatenate(all_signals_clean, axis=0)[:config.num_samples_to_load]
    raw_params = np.concatenate(all_params, axis=0)[:config.num_samples_to_load]
    
    # 2. Move to GPU immediately
    script_logger.info(f"Moving {len(raw_signals_clean)} samples to GPU VRAM...")
    gpu_signals_clean = torch.from_numpy(raw_signals_clean).float().to(device)
    
    # 3. Prepare targets on GPU
    if stage == 1:
        gpu_targets = gpu_signals_clean.clone()
    else:
        cbf = raw_params[:, 0]; att = raw_params[:, 1]
        cbf_norm = (cbf - norm_stats['y_mean_cbf']) / norm_stats['y_std_cbf']
        att_norm = (att - norm_stats['y_mean_att']) / norm_stats['y_std_att']
        params_norm = np.stack([cbf_norm, att_norm], axis=1)
        gpu_targets = torch.from_numpy(params_norm).float().to(device)

    # 4. Initialize Fast GPU Loader
    train_loader = FastTensorDataLoader(gpu_signals_clean, gpu_targets, batch_size=config.batch_size, shuffle=True)
    
    # --- VALIDATION SET PREPARATION (Move to GPU) ---
    script_logger.info("Generating fixed validation dataset on CPU...")
    validation_data_dict = _generate_simple_validation_set(
        simulator=simulator, plds=plds_np, n_subjects=200, 
        conditions=['healthy', 'stroke', 'tumor', 'elderly'], 
        noise_levels=[5.0, 10.0, 15.0]
    )
    
    # Process validation data on CPU first, then move to GPU
    val_signals_noisy_raw = validation_data_dict['signals']
    val_features_eng = engineer_signal_features(val_signals_noisy_raw, num_plds)
    val_signals_concat = np.concatenate([val_signals_noisy_raw, val_features_eng], axis=1)
    val_signals_processed = process_signals_cpu(val_signals_concat, norm_stats, num_plds)
    
    val_signals_gpu = torch.from_numpy(val_signals_processed.astype(np.float32)).to(device)

    if stage == 1:
        clean_signals_for_val = []
        for idx, params in enumerate(validation_data_dict['perturbed_params']):
            true_cbf = validation_data_dict['parameters'][idx, 0]
            true_att = validation_data_dict['parameters'][idx, 1]
            vsasl_c = simulator._generate_vsasl_signal(plds_np, true_att, true_cbf, params['t1_artery'], params['alpha_vsasl'])
            pcasl_c = simulator._generate_pcasl_signal(plds_np, true_att, true_cbf, params['t1_artery'], params['t_tau'], params['alpha_pcasl'])
            clean_signals_for_val.append(np.concatenate([pcasl_c, vsasl_c]))
        
        # Normalize clean targets for Stage 1 validation to match training reconstruction target
        clean_val_np = np.array(clean_signals_for_val).astype(np.float32)
        pcasl_c_raw = clean_val_np[:, :num_plds]
        vsasl_c_raw = clean_val_np[:, num_plds:]
        
        pcasl_mu = np.mean(pcasl_c_raw, axis=1, keepdims=True)
        pcasl_sigma = np.std(pcasl_c_raw, axis=1, keepdims=True) + 1e-6
        pcasl_shape = (pcasl_c_raw - pcasl_mu) / pcasl_sigma
        
        vsasl_mu = np.mean(vsasl_c_raw, axis=1, keepdims=True)
        vsasl_sigma = np.std(vsasl_c_raw, axis=1, keepdims=True) + 1e-6
        vsasl_shape = (vsasl_c_raw - vsasl_mu) / vsasl_sigma
        
        val_targets_processed = np.concatenate([pcasl_shape, vsasl_shape], axis=1)
        val_targets_gpu = torch.from_numpy(val_targets_processed).float().to(device)

    else:
        val_params = validation_data_dict['parameters']
        cbf_v = val_params[:, 0]; att_v = val_params[:, 1]
        cbf_norm_v = (cbf_v - norm_stats['y_mean_cbf']) / norm_stats['y_std_cbf']
        att_norm_v = (att_v - norm_stats['y_mean_att']) / norm_stats['y_std_att']
        val_targets_gpu = torch.from_numpy(np.stack([cbf_norm_v, att_norm_v], axis=1).astype(np.float32)).to(device)

    val_loader = FastTensorDataLoader(val_signals_gpu, val_targets_gpu, batch_size=config.batch_size, shuffle=False)

    # --- MODEL SETUP ---
    base_input_size_nn = num_plds * 2 + 8
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
        for model in trainer.models:
            if hasattr(model, 'encoder'):
                model.encoder.load_state_dict(encoder_state_dict, strict=True)
        if fine_tuning_cfg is None: fine_tuning_cfg = {'enabled': True}
        else: fine_tuning_cfg['enabled'] = True

    script_logger.info(f"Training {config.n_ensembles}-model ensemble with GPU-resident data...")
    trainer.train_ensemble(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        output_dir=output_dir,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        fine_tuning_config=fine_tuning_cfg,
        simulator=simulator,
        pld_list=config.pld_values,
        norm_stats=norm_stats
    )

    if stage == 1:
        encoder_path = output_dir / 'encoder_pretrained.pt'
        unwrapped_model = getattr(trainer.models[0], '_orig_mod', trainer.models[0])
        torch.save(unwrapped_model.encoder.state_dict(), encoder_path)
        script_logger.info(f"Saved final pre-trained encoder from model 0 to {encoder_path}")

    if wandb.run: wandb.finish()
    return {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True) 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--load-weights-from", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()
    
    config_obj = ResearchConfig()
    config_path = Path(args.config_file)
    
    with open(config_path, 'r') as f_yaml:
        config_from_yaml = yaml.safe_load(f_yaml) or {}

    def apply_yaml_to_dataclass(data: Any, config_object: ResearchConfig):
        if isinstance(data, dict):
            for key, value in data.items():
                if hasattr(config_object, key) and isinstance(getattr(config_object, key), dict) and isinstance(value, dict):
                    getattr(config_object, key).update(value)
                elif hasattr(config_object, key):
                    setattr(config_object, key, value)
                if isinstance(value, (dict, list)):
                    apply_yaml_to_dataclass(value, config_object)
    
    apply_yaml_to_dataclass(config_from_yaml, config_obj)
    
    if args.output_dir: output_path = Path(args.output_dir)
    else: output_path = Path(f'comprehensive_results/{args.run_name if args.run_name else "run"}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.stage == 2:
        encoder_path = Path(args.load_weights_from) / 'encoder_pretrained.pt'
        config_obj.pretrained_encoder_path = str(encoder_path)

    if args.run_name: os.environ['WANDB_RUN_NAME'] = args.run_name

    norm_stats_path = output_path / 'norm_stats.json'
    if norm_stats_path.exists():
        with open(norm_stats_path, 'r') as f: norm_stats = json.load(f)
    else:
        script_logger.info("Calculating normalization stats...")
        plds_np = np.array(config_obj.pld_values)
        asl_params_sim = ASLParameters(**{k:v for k,v in asdict(config_obj).items() if k in ASLParameters.__annotations__})
        simulator = RealisticASLSimulator(params=asl_params_sim)
        stats_calculator = ParallelStreamingStatsCalculator(simulator, plds_np, num_samples=config_obj.num_samples, num_workers=min(os.cpu_count(), 16))
        norm_stats = stats_calculator.calculate()
        with open(norm_stats_path, 'w') as f: json.dump(norm_stats, f, indent=2)

    run_comprehensive_asl_research(config=config_obj, stage=args.stage, output_dir=output_path, norm_stats=norm_stats)
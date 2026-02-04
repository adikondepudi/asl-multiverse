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

from enhanced_asl_network import DisentangledASLNet, PhysicsInformedASLProcessor, CustomLoss
from spatial_asl_network import SpatialASLNet, SpatialDataset, MaskedSpatialLoss
from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator
from asl_trainer import EnhancedASLTrainer, FastTensorDataLoader
from utils import ParallelStreamingStatsCalculator, process_signals_dynamic
from feature_registry import FeatureRegistry, FeatureConfigError
from torch.utils.data import DataLoader, random_split

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
    log_var_cbf_min: float = -3.0
    log_var_cbf_max: float = 7.0
    log_var_att_min: float = -3.0
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
    # NEW: Ablation Study Configs
    active_features: List[str] = field(default_factory=lambda: ['mean', 'std', 'peak', 't1_artery'])
    data_noise_components: List[str] = field(default_factory=lambda: ['thermal'])

    # NEW: Noise and Normalization Options
    # noise_type: 'gaussian' (default) or 'rician' (correct MRI physics)
    noise_type: str = 'gaussian'
    # normalization_mode: 'per_curve' (SNR-invariant shape vectors) or 'global_scale' (preserves magnitude)
    normalization_mode: str = 'per_curve'
    # global_scale_factor: multiplier for 'global_scale' mode to get signals into ~0-1 range
    global_scale_factor: float = 10.0

    # NEW: Loss Configuration (for voxel-wise models)
    # loss_mode options: 'mae_only', 'mse_only', 'nll_only', 'mae_nll', 'mse_nll'
    # RECOMMENDED: 'mae_only' or 'mae_nll' - forces model to predict accurately
    # WARNING: 'nll_only' allows model to minimize loss by predicting high uncertainty
    loss_mode: str = 'mae_nll'  # Default to balanced MAE + NLL
    mae_weight: float = 1.0     # Weight for MAE loss component
    nll_weight: float = 0.1     # Weight for NLL loss component (when using mae_nll or mse_nll)
    mse_weight: float = 0.0     # Legacy MSE weight (for backward compatibility)

    # NEW: Spatial model loss configuration
    loss_type: str = 'l1'       # For spatial: 'l1', 'l2', 'huber'
    att_scale: float = 1.0      # Scale ATT loss (1.0 with normalized targets)
    cbf_weight: float = 1.0     # Weight for CBF loss
    att_weight: float = 1.0     # Weight for ATT loss
    dc_weight: float = 0.0      # Data consistency loss weight
    variance_weight: float = 0.1  # Anti-mean-collapse: penalize low prediction variance
    # noise_config: dict for NoiseInjector configuration (snr_range, physio, drift, spikes, etc.)
    noise_config: Optional[Dict[str, Any]] = None

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
        
        # Physics consistency: Sample aBV and Slice Index
        val_slice_idx = np.random.randint(0, 30)
        slice_delay = np.exp(-(val_slice_idx * 45.0)/1000.0)
        val_abv = np.random.uniform(0.0, 0.015) if np.random.rand() > 0.5 else 0.0

        perturbed_t_tau = base_params.T_tau * (1 + np.random.uniform(*physio_var.t_tau_perturb_range))
        perturbed_alpha_pcasl = np.clip(base_params.alpha_PCASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.1)
        perturbed_alpha_vsasl = np.clip(base_params.alpha_VSASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.0)
        
        vsasl_clean = simulator._generate_vsasl_signal(plds, att, cbf, t1_a, perturbed_alpha_vsasl * slice_delay)
        pcasl_clean = simulator._generate_pcasl_signal(plds, att, cbf, t1_a, perturbed_t_tau, perturbed_alpha_pcasl * slice_delay)
        art_sig = simulator._generate_arterial_signal(plds, att, val_abv, t1_a, perturbed_alpha_pcasl * slice_delay)
        pcasl_clean += art_sig
        
        for snr in noise_levels:
            ref_signal_level = simulator._compute_reference_signal()
            noise_sd = ref_signal_level / snr
            noise_scaling = simulator.compute_tr_noise_scaling(plds)
            
            pcasl_noisy = pcasl_clean + noise_sd * noise_scaling['PCASL'] * np.random.randn(num_plds)
            vsasl_noisy = vsasl_clean + noise_sd * noise_scaling['VSASL'] * np.random.randn(num_plds)
            
            multiverse_signal_flat = np.concatenate([pcasl_noisy, vsasl_noisy])
            
            dataset['signals'].append(multiverse_signal_flat)
            # Save parameters consistent with training data structure [CBF, ATT, T1, Z]
            dataset['parameters'].append([cbf, att, t1_a, float(val_slice_idx)])
            dataset['conditions'].append(condition)
            dataset['noise_levels'].append(snr)
            dataset['perturbed_params'].append({
                't1_artery': t1_a, 't_tau': perturbed_t_tau, 
                'alpha_pcasl': perturbed_alpha_pcasl, 'alpha_vsasl': perturbed_alpha_vsasl
            })

    dataset['signals'] = np.array(dataset['signals'])
    dataset['parameters'] = np.array(dataset['parameters'])
    return dataset



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
    
    # === 1. DATA LOADING STRATEGY ===
    is_spatial_model = config.model_class_name in ["SpatialASLNet", "AmplitudeAwareSpatialASLNet"]
    train_loader = None
    val_loader = None
    loss_function = None

    if is_spatial_model:
        script_logger.info(f"[SPATIAL MODE] initializing lazy-loading SpatialDataset from {config.offline_dataset_path}...")
        
        full_dataset = SpatialDataset(
            data_dir=config.offline_dataset_path,
            transform=True, # Augmentation
            flip_prob=0.5,
            per_pixel_norm=False  # Normalization done in trainer._process_batch_on_gpu AFTER noise
        )
        
        total_len = len(full_dataset)
        if total_len == 0:
            raise FileNotFoundError(f"No spatial chunks found in {config.offline_dataset_path}")
            
        # Limit dataset size if requested (subsetting)
        if config.num_samples_to_load and config.num_samples_to_load < total_len:
            dataset_size = config.num_samples_to_load
            full_dataset = torch.utils.data.Subset(full_dataset, range(dataset_size))
            script_logger.info(f"Subsetting dataset to {dataset_size} samples.")
        else:
            dataset_size = total_len

        # Split Train/Val (90/10 split)
        train_size = int(0.9 * dataset_size)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                                  generator=torch.Generator().manual_seed(42))
        
        script_logger.info(f"Spatial Dataset: {train_size} training, {val_size} validation.")

        # Create Standard DataLoaders (CPU -> GPU streaming)
        # num_workers > 0 helps when loading from disk; 0 is better for fully in-memory datasets
        num_workers = getattr(config, 'num_workers', 4)
        pin_memory = getattr(config, 'pin_memory', True)
        # persistent_workers keeps workers alive between epochs (faster, but uses more RAM)
        persistent = num_workers > 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        
        # Instantiate Spatial Loss with configurable parameters
        # DC loss should be opt-in (default 0.0). When enabled, use small weight (0.0001-0.001)
        # since raw DC loss is ~1000x larger than supervised losses.
        dc_weight = getattr(config, 'dc_weight', 0.0)
        loss_type = getattr(config, 'loss_type', 'l1')  # l1/mae, l2/mse, huber
        # Note: With normalized targets, att_scale should be 1.0 (both in z-score units)
        att_scale = getattr(config, 'att_scale', 1.0)
        cbf_weight = getattr(config, 'cbf_weight', 1.0)
        att_weight = getattr(config, 'att_weight', 1.0)
        # Variance penalty: penalizes if prediction variance < target variance (anti-mean-collapse)
        variance_weight = getattr(config, 'variance_weight', 0.1)

        # CRITICAL: Pass norm_stats to loss function for target normalization!
        # Without this, the model will learn to predict the dataset mean.
        loss_function = MaskedSpatialLoss(
            loss_type=loss_type,
            dc_weight=dc_weight,
            att_scale=att_scale,
            cbf_weight=cbf_weight,
            att_weight=att_weight,
            norm_stats=norm_stats,
            variance_weight=variance_weight
        )
        
        # Define input size for model creation (channels)
        base_input_size_nn = num_plds * 2 
        model_mode = 'regression'

    else:
        # === LEGACY 1D LOADING (RAM HEAVY) ===
        script_logger.info(f"Loading raw CLEAN data to RAM from {config.offline_dataset_path}...")
    
        data_path_obj = Path(config.offline_dataset_path)
        files_1d = sorted(list(data_path_obj.glob('dataset_chunk_*.npz')))
        files_2d = sorted(list(data_path_obj.glob('spatial_chunk_*.npz')))
        
        all_signals_clean = []
        all_params = []
        samples_loaded = 0
        
        if files_1d:
            script_logger.info(f"Found {len(files_1d)} 1D dataset chunks.")
            for f in files_1d:
                d = np.load(f)
                all_signals_clean.append(d['signals_clean'])
                all_params.append(d['params'])
                samples_loaded += len(d['signals_clean'])
                if config.num_samples_to_load and samples_loaded >= config.num_samples_to_load:
                    break
        elif files_2d:
            script_logger.info(f"Found {len(files_2d)} 2D spatial chunks. Flattening for 1D training...")
            for f in files_2d:
                d = np.load(f)
                sigs = d['signals']
                tgts = d['targets']
                B, C_sig, H, W = sigs.shape
                
                N = B * H * W
                sigs_flat = sigs.transpose(0, 2, 3, 1).reshape(N, -1)
                tgts_flat = tgts.transpose(0, 2, 3, 1).reshape(N, 2)
                
                t1_vals = np.full((N, 1), config.T1_artery, dtype=np.float32)
                z_vals = np.full((N, 1), 15.0, dtype=np.float32)
                
                params_flat = np.concatenate([tgts_flat, t1_vals, z_vals], axis=1)
                
                all_signals_clean.append(sigs_flat)
                all_params.append(params_flat)
                samples_loaded += N
                
                if config.num_samples_to_load and samples_loaded >= config.num_samples_to_load:
                    break
        else:
            raise FileNotFoundError(f"No valid dataset files found in {config.offline_dataset_path}")
        
        raw_signals_clean = np.concatenate(all_signals_clean, axis=0)[:config.num_samples_to_load]
        raw_params = np.concatenate(all_params, axis=0)[:config.num_samples_to_load]
        
        script_logger.info(f"Moving {len(raw_signals_clean)} samples to GPU VRAM...")
        gpu_signals_clean = torch.from_numpy(raw_signals_clean).float().to(device)
        
        if stage == 1:
            t1_values = raw_params[:, 2:3]
            z_values = raw_params[:, 3:4]
            gpu_cond = torch.from_numpy(np.concatenate([t1_values, z_values], axis=1)).float().to(device)
            gpu_targets = torch.cat([gpu_signals_clean, gpu_cond], dim=1)
        else:
            cbf = raw_params[:, 0]; att = raw_params[:, 1]; t1 = raw_params[:, 2]; z_idx = raw_params[:, 3]
            cbf_norm = (cbf - norm_stats['y_mean_cbf']) / norm_stats['y_std_cbf']
            att_norm = (att - norm_stats['y_mean_att']) / norm_stats['y_std_att']
            targets_stack = np.stack([cbf_norm, att_norm, t1, z_idx], axis=1)
            gpu_targets = torch.from_numpy(targets_stack).float().to(device)

        train_loader = FastTensorDataLoader(gpu_signals_clean, gpu_targets, batch_size=config.batch_size, shuffle=True)
        
        script_logger.info("Generating fixed validation dataset on CPU...")
        validation_data_dict = _generate_simple_validation_set(
            simulator=simulator, plds=plds_np, n_subjects=200, 
            conditions=['healthy', 'stroke', 'tumor', 'elderly'], 
            noise_levels=[5.0, 10.0, 15.0]
        )
        
        val_signals_noisy_raw = validation_data_dict['signals']
        val_t1_input_np = np.array([p['t1_artery'] for p in validation_data_dict['perturbed_params']]).reshape(-1, 1).astype(np.float32)
        val_z_input_np = validation_data_dict['parameters'][:, 3:4].astype(np.float32)

        processing_config = {'pld_values': config.pld_values, 'active_features': config.active_features}
        val_signals_processed = process_signals_dynamic(val_signals_noisy_raw, norm_stats, processing_config, t1_values=val_t1_input_np, z_values=val_z_input_np)
        val_signals_gpu = torch.from_numpy(val_signals_processed.astype(np.float32)).to(device)

        if stage == 1:
            clean_signals_for_val = []
            for idx, params in enumerate(validation_data_dict['perturbed_params']):
                true_cbf = validation_data_dict['parameters'][idx, 0]
                true_att = validation_data_dict['parameters'][idx, 1]
                vsasl_c = simulator._generate_vsasl_signal(plds_np, true_att, true_cbf, params['t1_artery'], params['alpha_vsasl'])
                pcasl_c = simulator._generate_pcasl_signal(plds_np, true_att, true_cbf, params['t1_artery'], params['t_tau'], params['alpha_pcasl'])
                clean_signals_for_val.append(np.concatenate([pcasl_c, vsasl_c]))
            clean_val_np = np.array(clean_signals_for_val).astype(np.float32)
            pcasl_c_raw = clean_val_np[:, :num_plds]; vsasl_c_raw = clean_val_np[:, num_plds:]
            pcasl_shape = (pcasl_c_raw - np.mean(pcasl_c_raw, axis=1, keepdims=True)) / (np.std(pcasl_c_raw, axis=1, keepdims=True) + 1e-6)
            vsasl_shape = (vsasl_c_raw - np.mean(vsasl_c_raw, axis=1, keepdims=True)) / (np.std(vsasl_c_raw, axis=1, keepdims=True) + 1e-6)
            val_targets_processed = np.concatenate([pcasl_shape, vsasl_shape], axis=1)
            val_targets_gpu = torch.from_numpy(np.concatenate([val_targets_processed, val_t1_input_np], axis=1)).float().to(device)
        else:
            val_params = validation_data_dict['parameters']
            cbf_v = val_params[:, 0]; att_v = val_params[:, 1]
            cbf_norm_v = (cbf_v - norm_stats['y_mean_cbf']) / norm_stats['y_std_cbf']
            att_norm_v = (att_v - norm_stats['y_mean_att']) / norm_stats['y_std_att']
            val_targets_gpu = torch.from_numpy(np.stack([cbf_norm_v, att_norm_v], axis=1).astype(np.float32)).to(device)

        val_loader = FastTensorDataLoader(val_signals_gpu, val_targets_gpu, batch_size=config.batch_size, shuffle=False)
        
        # Model config
        active_feats = config.active_features
        num_scalar_features_dynamic = FeatureRegistry.compute_scalar_dim(active_feats)
        base_input_size_nn = num_plds * 2 + num_scalar_features_dynamic
        model_mode = 'denoising' if stage == 1 else 'regression'
        
    # === MODEL & TRAINER SETUP ===
    model_creation_config = asdict(config)
    
    def create_model_closure(**kwargs):
        if config.model_class_name == "SpatialASLNet":
            # Map MLP 'hidden_sizes' to U-Net 'features'
            # 1. Sort ascending (U-Net encoders expand: 32 -> 64 -> ...)
            features = sorted(kwargs.get('hidden_sizes', [256, 128, 64]))

            # 2. PAD to 4 levels: SpatialASLNet HARDCODES 4 layers.
            # If config has only 3, prepend a smaller layer (e.g. [64,128,256] -> [32,64,128,256])
            while len(features) < 4:
                features.insert(0, max(1, features[0] // 2))

            return SpatialASLNet(
                n_plds=num_plds,
                features=features,
                **kwargs
            )
        elif config.model_class_name == "AmplitudeAwareSpatialASLNet":
            # AmplitudeAwareSpatialASLNet: preserves amplitude info for CBF estimation
            features = sorted(kwargs.get('hidden_sizes', [256, 128, 64]))
            while len(features) < 4:
                features.insert(0, max(1, features[0] // 2))

            return AmplitudeAwareSpatialASLNet(
                n_plds=num_plds,
                features=features,
                use_film_at_bottleneck=kwargs.get('use_film_at_bottleneck', True),
                use_film_at_decoder=kwargs.get('use_film_at_decoder', True),
                use_amplitude_output_modulation=kwargs.get('use_amplitude_output_modulation', True),
                **kwargs
            )
        else:
            return DisentangledASLNet(
                mode=model_mode, 
                input_size=base_input_size_nn, 
                num_scalar_features=kwargs.get('num_scalar_features', 0),
                active_features_list=kwargs.get('active_features_list', None),
                **kwargs
            )

    trainer = EnhancedASLTrainer(stage=stage, model_config=model_creation_config, model_class=create_model_closure, 
                                 weight_decay=config.weight_decay, batch_size=config.batch_size, 
                                 n_ensembles=config.n_ensembles, device=device, loss_fn=loss_function)
    
    fine_tuning_cfg = config.fine_tuning if stage == 2 else None
    
    if stage == 2:
        if config.pretrained_encoder_path:
            script_logger.info(f"Loading pre-trained encoder weights from: {config.pretrained_encoder_path}")
            encoder_state_dict = torch.load(config.pretrained_encoder_path, map_location=device)
            for model in trainer.models:
                if hasattr(model, 'encoder'):
                    model.encoder.load_state_dict(encoder_state_dict, strict=True)
            if fine_tuning_cfg is None: fine_tuning_cfg = {'enabled': True}
            else: fine_tuning_cfg['enabled'] = True
        else:
            script_logger.info("No pre-trained encoder path provided. Training from scratch (Standard Mode).")
            # Explicitly disable fine-tuning to prevent reduced encoder learning rates on random initialization
            if fine_tuning_cfg: fine_tuning_cfg['enabled'] = False

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
    
    # DEFENSIVE: Validate config before any processing
    try:
        FeatureRegistry.validate_active_features(config_obj.active_features)
        FeatureRegistry.validate_noise_components(config_obj.data_noise_components)
        FeatureRegistry.validate_plds(config_obj.pld_values)
        FeatureRegistry.validate_encoder_type(config_obj.encoder_type)
        FeatureRegistry.validate_noise_type(config_obj.noise_type)
        FeatureRegistry.validate_normalization_mode(config_obj.normalization_mode)
        script_logger.info(f"Config validated: features={config_obj.active_features}, "
                          f"noise_components={config_obj.data_noise_components}, "
                          f"encoder={config_obj.encoder_type}, "
                          f"noise_type={config_obj.noise_type}, "
                          f"norm_mode={config_obj.normalization_mode}")
    except FeatureConfigError as e:
        script_logger.error(f"CONFIG VALIDATION FAILED: {e}")
        sys.exit(1)
    
    if args.output_dir: output_path = Path(args.output_dir)
    else: output_path = Path(f'comprehensive_results/{args.run_name if args.run_name else "run"}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.stage == 2 and args.load_weights_from:
        if args.load_weights_from:
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
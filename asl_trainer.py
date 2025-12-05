# FILE: asl_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import wandb 
import time
import traceback
from pathlib import Path

from enhanced_asl_network import CustomLoss, DisentangledASLNet
from utils import engineer_signal_features_torch

import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FastTensorDataLoader:
    def __init__(self, signals, targets, batch_size, shuffle=True):
        self.signals = signals 
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_len = self.signals.shape[0]
        self.n_batches = (self.dataset_len + batch_size - 1) // batch_size
        self.device = signals.device
        
    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            self.indices = torch.arange(self.dataset_len, device=self.device)
        self.curr_idx = 0
        return self

    def __next__(self):
        if self.curr_idx >= self.dataset_len:
            raise StopIteration
        
        end_idx = min(self.curr_idx + self.batch_size, self.dataset_len)
        batch_indices = self.indices[self.curr_idx:end_idx]
        
        batch_x = self.signals[batch_indices]
        batch_y = self.targets[batch_indices]
        
        self.curr_idx = end_idx
        return batch_x, batch_y
        
    def __len__(self):
        return self.n_batches

class EnhancedASLTrainer:
    def __init__(self,
                 stage: int, model_config: Dict, model_class: callable,
                 weight_decay: float = 1e-5, batch_size: int = 256, n_ensembles: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        self.device = torch.device(device); self.model_config = model_config; self.stage = stage
        self.lr_base = float(model_config.get('learning_rate', 0.001)); self.weight_decay = float(weight_decay)
        self.n_ensembles = n_ensembles; self.validation_steps_per_epoch = model_config.get('validation_steps_per_epoch', 50)
        
        # NEW: Ablation Study Configs
        self.active_features = model_config.get('active_features', ['mean', 'std', 'peak', 't1_artery'])
        self.data_noise_components = model_config.get('data_noise_components', ['thermal'])
        logger.info(f"Ablation Config - Active Features: {self.active_features}")
        logger.info(f"Ablation Config - Noise Components: {self.data_noise_components}")
        
        logger.info("Initializing models (Float32)...")
        self.models = [model_class(**model_config).to(self.device) for _ in range(n_ensembles)]
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=True)

        if wandb.run:
            for i, model in enumerate(self.models): wandb.watch(model, log='gradients', log_freq=200, idx=i)
        
        # Extract mse_weight from config
        loss_params = {
            'training_stage': stage, 
            'w_cbf': model_config.get('loss_weight_cbf', 1.0), 
            'w_att': model_config.get('loss_weight_att', 1.0),
            'log_var_reg_lambda': model_config.get('loss_log_var_reg_lambda', 0.0),
            'mse_weight': model_config.get('mse_weight', 0.0) 
        }
        self.custom_loss_fn = CustomLoss(**loss_params)
        self.global_step = 0; self.norm_stats = None
        
        self.ref_signal_gpu = None
        self.noise_scale_vec_gpu = None
        self.norm_stats_gpu = None
        self.simulator = None  # Will be set in setup_gpu_noise_params

    def setup_gpu_noise_params(self, simulator, pld_list, norm_stats):
        if simulator is None or pld_list is None:
            return

        # Store simulator and plds for modular noise
        self.simulator = simulator
        self.pld_list = pld_list
        self.n_plds = len(pld_list)

        ref_signal = simulator._compute_reference_signal()
        self.ref_signal_gpu = torch.tensor(ref_signal, device=self.device, dtype=torch.float32)
        
        scalings = simulator.compute_tr_noise_scaling(np.array(pld_list))
        
        scale_vec = np.concatenate([
            np.full(self.n_plds, scalings['PCASL']),
            np.full(self.n_plds, scalings['VSASL'])
        ])
        self.noise_scale_vec_gpu = torch.tensor(scale_vec, device=self.device, dtype=torch.float32)
        
        self.norm_stats_gpu = {}
        for k, v in norm_stats.items():
            if isinstance(v, (list, np.ndarray)):
                self.norm_stats_gpu[k] = torch.tensor(v, device=self.device, dtype=torch.float32)
            else:
                self.norm_stats_gpu[k] = torch.tensor(v, device=self.device, dtype=torch.float32)
        
        # Register T1 stats for normalization
        self.t1_mean_gpu = torch.tensor(norm_stats.get('y_mean_t1', 1850.0), device=self.device, dtype=torch.float32)
        self.t1_std_gpu = torch.tensor(norm_stats.get('y_std_t1', 200.0), device=self.device, dtype=torch.float32)
        
        # Register Z stats
        self.z_mean_gpu = torch.tensor(norm_stats.get('y_mean_z', 15.0), device=self.device, dtype=torch.float32)
        self.z_std_gpu = torch.tensor(norm_stats.get('y_std_z', 8.0), device=self.device, dtype=torch.float32)
        
        logger.info(f"GPU Noise parameters loaded. Noise components: {self.data_noise_components}")

    def _process_batch_on_gpu(self, raw_signals, t1_values=None, z_values=None):
        """
        Process batch with CONFIGURABLE noise and DYNAMIC feature selection.
        Noise components: ['thermal', 'physio', 'drift', 'spikes']
        Active features: ['mean', 'std', 'ttp', 'com', 'peak', 't1_artery', 'z_coord']
        """
        if self.noise_scale_vec_gpu is None:
            return raw_signals 

        batch_size = raw_signals.shape[0]
        n_plds = raw_signals.shape[1] // 2
        
        # ========== 1. MODULAR NOISE GENERATION ==========
        current_snr = torch.empty(batch_size, 1, device=self.device).uniform_(1.5, 10.0)
        noise_sigma = (self.ref_signal_gpu / current_snr)
        
        # Start with thermal noise (always present as base)
        total_noise = torch.zeros_like(raw_signals)
        
        if 'thermal' in self.data_noise_components:
            thermal_noise = torch.randn_like(raw_signals) * noise_sigma * self.noise_scale_vec_gpu
            total_noise += thermal_noise
        
        if 'physio' in self.data_noise_components:
            # Simulate cardiac/respiratory oscillation (low-frequency sinusoid)
            # Amplitude ~5-15% of signal
            physio_amp = torch.empty(batch_size, 1, device=self.device).uniform_(0.05, 0.15)
            physio_freq = torch.empty(batch_size, 1, device=self.device).uniform_(0.5, 2.0)  # Hz
            t = torch.arange(n_plds * 2, device=self.device).float().unsqueeze(0)
            physio_noise = physio_amp * noise_sigma * torch.sin(2 * 3.14159 * physio_freq * t / (n_plds * 2))
            total_noise += physio_noise
        
        if 'drift' in self.data_noise_components:
            # Baseline drift (linear trend)
            drift_slope = torch.empty(batch_size, 1, device=self.device).uniform_(-0.02, 0.02)
            t = torch.arange(n_plds * 2, device=self.device).float().unsqueeze(0) / (n_plds * 2)
            drift_noise = drift_slope * noise_sigma * t
            total_noise += drift_noise
        
        if 'spikes' in self.data_noise_components:
            # Random motion spikes (affects ~5% of samples)
            spike_mask = (torch.rand(batch_size, n_plds * 2, device=self.device) < 0.05).float()
            spike_magnitude = torch.empty(batch_size, 1, device=self.device).uniform_(2.0, 5.0)
            spike_noise = spike_mask * spike_magnitude * noise_sigma
            total_noise += spike_noise
        
        noisy_signals = raw_signals + total_noise
        
        # ========== 2. COMPUTE ALL FEATURES ==========
        pcasl_raw = noisy_signals[:, :n_plds]
        vsasl_raw = noisy_signals[:, n_plds:]
        
        # Basic stats (always computed)
        pcasl_mu = torch.mean(pcasl_raw, dim=1, keepdim=True)
        pcasl_sigma = torch.std(pcasl_raw, dim=1, keepdim=True) + 1e-6
        pcasl_shape = (pcasl_raw - pcasl_mu) / pcasl_sigma
        
        vsasl_mu = torch.mean(vsasl_raw, dim=1, keepdim=True)
        vsasl_sigma = torch.std(vsasl_raw, dim=1, keepdim=True) + 1e-6
        vsasl_shape = (vsasl_raw - vsasl_mu) / vsasl_sigma
        
        shape_vector = torch.cat([pcasl_shape, vsasl_shape], dim=1)
        
        # Compute all possible engineered features
        eng_features = engineer_signal_features_torch(noisy_signals, n_plds)
        # eng_features contains: [pcasl_ttp, vsasl_ttp, pcasl_com, vsasl_com, pcasl_peak, vsasl_peak]
        
        # ========== 3. DYNAMIC FEATURE SELECTION ==========
        # Mapping feature names to indices in scalar_features_mean/std
        # Layout: [mu_p(0), sig_p(1), mu_v(2), sig_v(3), ttp_p(4), ttp_v(5), com_p(6), com_v(7), peak_p(8), peak_v(9)]
        feature_indices = {
            'mean': [0, 2],   # mu_p, mu_v
            'std': [1, 3],    # sig_p, sig_v 
            'ttp': [4, 5],    # ttp_p, ttp_v
            'com': [6, 7],    # com_p, com_v
            'peak': [8, 9],   # peak_p, peak_v
        }
        
        s_mean = self.norm_stats_gpu['scalar_features_mean']
        s_std = self.norm_stats_gpu['scalar_features_std'] + 1e-6
        
        selected_scalars = []
        
        for feat_name in self.active_features:
            if feat_name in feature_indices:
                indices = feature_indices[feat_name]
                
                # Get the unnormalized feature values
                if feat_name == 'mean':
                    feat_vals = torch.cat([pcasl_mu, vsasl_mu], dim=1)
                elif feat_name == 'std':
                    feat_vals = torch.cat([pcasl_sigma, vsasl_sigma], dim=1)
                elif feat_name == 'ttp':
                    feat_vals = eng_features[:, 0:2]  # pcasl_ttp, vsasl_ttp
                elif feat_name == 'com':
                    feat_vals = eng_features[:, 2:4]  # pcasl_com, vsasl_com
                elif feat_name == 'peak':
                    feat_vals = eng_features[:, 4:6]  # pcasl_peak, vsasl_peak
                else:
                    continue
                
                # Normalize using the correct indices from norm_stats
                mu = s_mean[indices]
                std = s_std[indices]
                feat_norm = (feat_vals - mu) / std
                selected_scalars.append(feat_norm)
        
        if len(selected_scalars) > 0:
            scalars_norm = torch.cat(selected_scalars, dim=1)
        else:
            scalars_norm = torch.zeros(batch_size, 0, device=self.device)
        
        # Add T1 if in active_features
        if 't1_artery' in self.active_features and t1_values is not None:
            t1_norm = (t1_values - self.t1_mean_gpu) / (self.t1_std_gpu + 1e-6)
            scalars_norm = torch.cat([scalars_norm, t1_norm], dim=1)
        
        # Add Z if in active_features  
        if 'z_coord' in self.active_features and z_values is not None:
            z_norm = (z_values - self.z_mean_gpu) / (self.z_std_gpu + 1e-6)
            scalars_norm = torch.cat([scalars_norm, z_norm], dim=1)
        
        final_input = torch.cat([shape_vector, scalars_norm], dim=1)
        return torch.clamp(final_input, -15.0, 15.0)

    def train_ensemble(self,
                       train_loader: FastTensorDataLoader, val_loader: FastTensorDataLoader,
                       n_epochs: int, steps_per_epoch: Optional[int], output_dir: Path,
                       early_stopping_patience: int = 10,
                       early_stopping_min_delta: float = 0.0,
                       fine_tuning_config: Optional[Dict] = None,
                       simulator=None, pld_list=None, norm_stats=None):
        
        if simulator and pld_list and norm_stats:
            self.setup_gpu_noise_params(simulator, pld_list, norm_stats)

        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)
        
        is_finetuning = fine_tuning_config is not None and fine_tuning_config.get('enabled', False)
        run_type = "Fine-Tuning" if is_finetuning else "Training"
        logger.info(f"--- Starting Stage {self.stage} {run_type} for {n_epochs} epochs ---")
        
        if is_finetuning:
            lr_factor = fine_tuning_config.get('encoder_lr_factor', 20.0)
            lr_finetune_encoder = self.lr_base / lr_factor
            logger.info(f"--- Discriminative Fine-tuning: Unfreezing encoder with LR={lr_finetune_encoder} and Head LR={self.lr_base} ---")
            self.optimizers = []
            for m in self.models:
                if hasattr(m, 'unfreeze_all'): m.unfreeze_all()
                param_groups = [
                    {'params': m.encoder.parameters(), 'lr': lr_finetune_encoder},
                    {'params': m.head.parameters(), 'lr': self.lr_base}
                ]
                self.optimizers.append(torch.optim.AdamW(param_groups, weight_decay=self.weight_decay))
        else:
            logger.info(f"--- Standard Training Mode: Creating optimizer for all parameters. ---")
            self.optimizers = [torch.optim.AdamW(m.parameters(), lr=self.lr_base, weight_decay=self.weight_decay) for m in self.models]
        
        total_steps = steps_per_epoch * n_epochs
        self.schedulers = [torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[pg['lr'] for pg in opt.param_groups], total_steps=total_steps) for opt in self.optimizers]
        
        patience_counters = [0] * self.n_ensembles; self.overall_best_val_losses = [float('inf')] * self.n_ensembles
        models_save_dir = output_dir / 'trained_models'
        models_save_dir.mkdir(exist_ok=True)

        for epoch in range(n_epochs):
            train_loss, _ = self._train_epoch(self.models, train_loader, self.optimizers, self.schedulers, epoch, steps_per_epoch)
            val_metrics = self._validate(self.models, val_loader, epoch)
            
            valid_losses = [m.get('val_loss', np.nan) for m in val_metrics]
            if np.isnan(valid_losses).all():
                mean_val_loss = float('nan')
            else:
                mean_val_loss = np.nanmean(valid_losses)

            logger.info(f"Epoch {epoch + 1}/{n_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {mean_val_loss:.6f}")
            if wandb.run: wandb.log({'epoch': epoch, 'mean_train_loss': train_loss, 'mean_val_loss': mean_val_loss}, step=self.global_step)
            
            if np.isnan(mean_val_loss): continue

            for i, vm in enumerate(val_metrics):
                val_loss = vm.get('val_loss', float('inf'))
                if np.isnan(val_loss): continue

                if val_loss < self.overall_best_val_losses[i] - early_stopping_min_delta:
                    self.overall_best_val_losses[i] = val_loss
                    patience_counters[i] = 0
                    unwrapped_model = getattr(self.models[i], '_orig_mod', self.models[i])
                    
                    if self.stage == 1:
                        if i == 0:
                            model_path = output_dir / 'encoder_pretrained.pt'
                            torch.save(unwrapped_model.encoder.state_dict(), model_path)
                            logger.info(f"Saved new best encoder from model 0 to {model_path} (Val Loss: {val_loss:.6f})")
                    else: # stage 2
                        model_path = models_save_dir / f'ensemble_model_{i}.pt'
                        torch.save(unwrapped_model.state_dict(), model_path)
                        logger.info(f"Saved new best model for ensemble {i} to {model_path} (Val Loss: {val_loss:.6f})")
                else:
                    patience_counters[i] += 1
            if all(p >= early_stopping_patience for p in patience_counters): logger.info("All models early stopped."); break
        
        return {'final_mean_val_loss': np.nanmean(self.overall_best_val_losses)}

    def _train_epoch(self, models, loader, optimizers, schedulers, epoch, steps):
        for m in models: m.train()
        total_loss = 0.0; loader_iter = iter(loader)
        
        for i in range(steps):
            try: 
                raw_signals, targets = next(loader_iter)
            except StopIteration: 
                loader_iter = iter(loader)
                raw_signals, targets = next(loader_iter)
            
            # SPLIT TARGETS
            # Targets now: [CBF, ATT, T1, Z]
            if self.stage == 2:
                training_targets = targets[:, :2] # CBF, ATT for loss
                t1_inputs = targets[:, 2:3]       # T1 for input
                z_inputs = targets[:, 3:4]        # Z for input
                processed_signals = self._process_batch_on_gpu(raw_signals, t1_values=t1_inputs, z_values=z_inputs)
                target_for_loss = training_targets
            else:
                # Stage 1 logic: Targets are [Clean Signals, T1, Z]
                # actually targets.shape[1] = 2*P + 2 (T1 + Z)
                n_plds = (targets.shape[1] - 2) // 2
                
                t1_inputs = targets[:, -2:-1]
                z_inputs = targets[:, -1:]
                
                processed_signals = self._process_batch_on_gpu(raw_signals, t1_values=t1_inputs, z_values=z_inputs)
                
                clean_signals_only = targets[:, :-2]
                pcasl_clean = clean_signals_only[:, :n_plds]
                vsasl_clean = clean_signals_only[:, n_plds:]
                
                pcasl_mu = torch.mean(pcasl_clean, dim=1, keepdim=True)
                pcasl_sigma = torch.std(pcasl_clean, dim=1, keepdim=True) + 1e-6
                pcasl_shape = (pcasl_clean - pcasl_mu) / pcasl_sigma
                vsasl_mu = torch.mean(vsasl_clean, dim=1, keepdim=True)
                vsasl_sigma = torch.std(vsasl_clean, dim=1, keepdim=True) + 1e-6
                vsasl_shape = (vsasl_clean - vsasl_mu) / vsasl_sigma
                target_for_loss = torch.cat([pcasl_shape, vsasl_shape], dim=1)
            
            for model_idx, model in enumerate(models):
                optimizers[model_idx].zero_grad(set_to_none=False)
                
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                    outputs = model(processed_signals)
                    outputs_f32 = tuple(o.float() if isinstance(o, torch.Tensor) else o for o in outputs)
                    loss, comps = self.custom_loss_fn(model_outputs=outputs_f32, targets=target_for_loss, global_epoch=epoch)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizers[model_idx])
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(optimizers[model_idx])
                self.scaler.update()
                schedulers[model_idx].step()
                
                if model_idx == 0: total_loss += loss.item()
                
                if wandb.run and model_idx == 0 and self.global_step % 20 == 0:
                    lrs = schedulers[model_idx].get_last_lr()
                    log_dict = {"train/total_loss": loss.item(), "train/lr_head": lrs[-1]}
                    if self.stage == 2:
                        log_dict["train_comps/nll_loss"] = comps.get('param_nll_loss', torch.tensor(0.0)).item()
                        log_dict["train_comps/mse_loss"] = comps.get('param_mse_loss', torch.tensor(0.0)).item()
                    wandb.log(log_dict, step=self.global_step)

            self.global_step += 1
        return total_loss / steps, {}

    def _validate(self, models, loader, epoch):
        if not loader: return [{'val_loss': float('inf')} for _ in models]
        for m in models: m.eval()
        val_losses = [0.0] * len(models)
        num_steps = 0

        with torch.no_grad():
            for inputs, targets in loader:
                for i, model in enumerate(models):
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                        outputs = model(inputs)
                        outputs_f32 = tuple(o.float() if isinstance(o, torch.Tensor) else o for o in outputs)
                        # Handle Stage 1 Validation where targets might include T1 (13 cols) vs Output (12 cols)
                        target_for_loss = targets
                        if self.stage == 1 and targets.shape[1] > outputs_f32[0].shape[1]:
                            target_for_loss = targets[:, :-1]
                        
                        loss, _ = self.custom_loss_fn(model_outputs=outputs_f32, targets=target_for_loss, global_epoch=epoch)
                    if not torch.isnan(loss): val_losses[i] += loss.item()
                num_steps += 1
        return [{'val_loss': v / num_steps if num_steps > 0 else float('nan')} for v in val_losses]
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

    def setup_gpu_noise_params(self, simulator, pld_list, norm_stats):
        if simulator is None or pld_list is None:
            return

        ref_signal = simulator._compute_reference_signal()
        self.ref_signal_gpu = torch.tensor(ref_signal, device=self.device, dtype=torch.float32)
        
        scalings = simulator.compute_tr_noise_scaling(np.array(pld_list))
        n_plds = len(pld_list)
        
        scale_vec = np.concatenate([
            np.full(n_plds, scalings['PCASL']),
            np.full(n_plds, scalings['VSASL'])
        ])
        self.noise_scale_vec_gpu = torch.tensor(scale_vec, device=self.device, dtype=torch.float32)
        
        self.norm_stats_gpu = {}
        for k, v in norm_stats.items():
            if isinstance(v, (list, np.ndarray)):
                self.norm_stats_gpu[k] = torch.tensor(v, device=self.device, dtype=torch.float32)
            else:
                self.norm_stats_gpu[k] = torch.tensor(v, device=self.device, dtype=torch.float32)
        logger.info("GPU Noise parameters and Norm Stats loaded successfully.")

    def _process_batch_on_gpu(self, raw_signals):
        if self.noise_scale_vec_gpu is None:
            return raw_signals 

        batch_size = raw_signals.shape[0]
        n_plds = raw_signals.shape[1] // 2
        
        # CRITICAL CHANGE: Harder Training Noise (1.5 - 10.0 SNR)
        # Stops model from "cheating" on easy data
        current_snr = torch.empty(batch_size, 1, device=self.device).uniform_(1.5, 10.0)
        
        noise_sigma = (self.ref_signal_gpu / current_snr) * self.noise_scale_vec_gpu
        
        gaussian_noise = torch.randn_like(raw_signals) * noise_sigma
        noisy_signals = raw_signals + gaussian_noise
        
        eng_features = engineer_signal_features_torch(noisy_signals, n_plds)
        
        pcasl_raw = noisy_signals[:, :n_plds]
        vsasl_raw = noisy_signals[:, n_plds:]
        
        pcasl_mu = torch.mean(pcasl_raw, dim=1, keepdim=True)
        pcasl_sigma = torch.std(pcasl_raw, dim=1, keepdim=True) + 1e-6
        pcasl_shape = (pcasl_raw - pcasl_mu) / pcasl_sigma
        
        vsasl_mu = torch.mean(vsasl_raw, dim=1, keepdim=True)
        vsasl_sigma = torch.std(vsasl_raw, dim=1, keepdim=True) + 1e-6
        vsasl_shape = (vsasl_raw - vsasl_mu) / vsasl_sigma
        
        shape_vector = torch.cat([pcasl_shape, vsasl_shape], dim=1)
        
        scalars = torch.cat([pcasl_mu, pcasl_sigma, vsasl_mu, vsasl_sigma, eng_features], dim=1)
        
        s_mean = self.norm_stats_gpu['scalar_features_mean']
        s_std = self.norm_stats_gpu['scalar_features_std'] + 1e-6
        
        scalars_norm = (scalars - s_mean) / s_std
        
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
            
            processed_signals = self._process_batch_on_gpu(raw_signals)
            target_for_loss = targets

            if self.stage == 1:
                n_plds = targets.shape[1] // 2
                pcasl_clean = targets[:, :n_plds]
                vsasl_clean = targets[:, n_plds:]
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
                        loss, _ = self.custom_loss_fn(model_outputs=outputs_f32, targets=targets, global_epoch=epoch)
                    if not torch.isnan(loss): val_losses[i] += loss.item()
                num_steps += 1
        return [{'val_loss': v / num_steps if num_steps > 0 else float('nan')} for v in val_losses]
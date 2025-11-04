# FILE: asl_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, IterableDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path
import wandb 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import time
from itertools import islice
import traceback
import optuna

from enhanced_asl_network import CustomLoss, DisentangledASLNet
from enhanced_simulation import RealisticASLSimulator
from utils import engineer_signal_features

import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ASLInMemoryDataset(Dataset):
    def __init__(self, data_dir: Optional[str], norm_stats: dict, stage: int, disentangled_mode: bool = False, num_samples_to_load: Optional[int] = None):
        self.data_dir = Path(data_dir) if data_dir else None; self.norm_stats = norm_stats
        self.stage = stage; self.disentangled_mode = disentangled_mode
        self.num_plds = len(norm_stats.get('pcasl_mean', []))
        if self.data_dir:
            logger.info(f"Loading dataset chunks from {self.data_dir} for stage {self.stage}..."); 
            files = sorted(list(self.data_dir.glob('dataset_chunk_*.npz')))
            if not files: raise FileNotFoundError(f"No dataset chunks in {data_dir}.")
            
            all_signals_noisy = [np.load(f)['signals_noisy'] for f in files]
            self.signals_noisy_unprocessed = np.concatenate(all_signals_noisy, axis=0)
            
            if self.stage == 1:
                all_signals_clean = [np.load(f)['signals_clean'] for f in files]
                self.signals_clean_unprocessed = np.concatenate(all_signals_clean, axis=0)
            
            all_params = [np.load(f)['params'] for f in files]
            self.params_unnormalized = np.concatenate(all_params, axis=0)

            if num_samples_to_load is not None and num_samples_to_load < len(self.signals_noisy_unprocessed):
                logger.info(f"Subsetting data to {num_samples_to_load} samples for accelerated training.")
                indices = np.random.permutation(len(self.signals_noisy_unprocessed))[:num_samples_to_load]
                self.signals_noisy_unprocessed = self.signals_noisy_unprocessed[indices]
                if self.stage == 1:
                    self.signals_clean_unprocessed = self.signals_clean_unprocessed[indices]
                self.params_unnormalized = self.params_unnormalized[indices]
            
            logger.info(f"Loaded {len(self.signals_noisy_unprocessed)} samples. Processing..."); 
            self.signals_processed = self._process_signals(self.signals_noisy_unprocessed)
            self.signals_tensor = torch.from_numpy(self.signals_processed.astype(np.float32))

            if self.stage == 1:
                self.targets_tensor = torch.from_numpy(self.signals_clean_unprocessed.astype(np.float32))
            else: # stage 2
                self.params_normalized = self._normalize_params(self.params_unnormalized)
                self.targets_tensor = torch.from_numpy(self.params_normalized.astype(np.float32))
        else:
            logger.info("ASLInMemoryDataset initialized in manual mode.")
            self.signals_tensor = torch.empty(0); self.targets_tensor = torch.empty(0)
            self.signals_noisy_unprocessed = np.empty(0)
            self.params_unnormalized = np.empty(0)

    def _process_signals(self, signals_unnorm: np.ndarray) -> np.ndarray:
        raw, eng = signals_unnorm[:, :self.num_plds*2], signals_unnorm[:, self.num_plds*2:]
        if self.disentangled_mode:
            amp = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-6; shape = raw / amp
            amp_norm = (amp - self.norm_stats['amplitude_mean']) / (self.norm_stats['amplitude_std'] + 1e-6)
            return np.concatenate([shape, eng, amp_norm], axis=1)
        else:
            p_raw, v_raw = raw[:, :self.num_plds], raw[:, self.num_plds:]; p_mean, p_std = np.array(self.norm_stats['pcasl_mean']), np.array(self.norm_stats['pcasl_std']) + 1e-6
            v_mean, v_std = np.array(self.norm_stats['vsasl_mean']), np.array(self.norm_stats['vsasl_std']) + 1e-6
            p_norm = (p_raw - p_mean) / p_std; v_norm = (v_raw - v_mean) / v_std
            return np.concatenate([p_norm, v_norm, eng], axis=1)

    def _normalize_params(self, params_unnorm: np.ndarray) -> np.ndarray:
        cbf, att = params_unnorm[:, 0], params_unnorm[:, 1]
        cbf_norm = (cbf - self.norm_stats['y_mean_cbf']) / self.norm_stats['y_std_cbf']
        att_norm = (att - self.norm_stats['y_mean_att']) / self.norm_stats['y_std_att']
        return np.stack([cbf_norm, att_norm], axis=1)

    def __len__(self): return self.signals_tensor.shape[0]
    def __getitem__(self, idx): return self.signals_tensor[idx], self.targets_tensor[idx]

class ASLIterableDataset(IterableDataset):
    def __init__(self, simulator: RealisticASLSimulator, plds: np.ndarray, 
                 noise_levels: List[float], norm_stats: Dict, stage: int, disentangled_mode: bool = False):
        super().__init__(); self.simulator, self.plds, self.num_plds = simulator, plds, len(plds)
        self.noise_levels, self.norm_stats, self.stage = noise_levels, norm_stats, stage
        self.base_params, self.physio_var = simulator.params, simulator.physio_var
        self.disentangled_mode = disentangled_mode
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.id + int(time.time() * 1000) if worker_info else int(time.time() * 1000)
        np.random.seed(seed % (2**32))
        while True:
            try:
                true_att = np.random.uniform(*self.physio_var.att_range); true_cbf = np.random.uniform(*self.physio_var.cbf_range)
                snr = np.random.choice(self.noise_levels)
                
                vsasl_clean = self.simulator._generate_vsasl_signal(self.plds, true_att, true_cbf, self.base_params.T1_artery, self.base_params.alpha_VSASL)
                pcasl_clean = self.simulator._generate_pcasl_signal(self.plds, true_att, true_cbf, self.base_params.T1_artery, self.base_params.T_tau, self.base_params.alpha_PCASL)
                clean_signal = np.concatenate([pcasl_clean, vsasl_clean])
                
                vsasl_noisy = self.simulator.add_realistic_noise(vsasl_clean, snr=snr)
                pcasl_noisy = self.simulator.add_realistic_noise(pcasl_clean, snr=snr)
                noisy_signal = np.concatenate([pcasl_noisy, vsasl_noisy])
                
                eng_feat = engineer_signal_features(noisy_signal, self.num_plds)
                
                if self.disentangled_mode:
                    amp = np.linalg.norm(noisy_signal) + 1e-6; shape = noisy_signal / amp
                    amp_norm = (amp - self.norm_stats['amplitude_mean']) / (self.norm_stats['amplitude_std'] + 1e-6)
                    final_input = np.concatenate([shape, eng_feat.flatten(), np.array([amp_norm])])
                else:
                    p_norm = (noisy_signal[:self.num_plds] - self.norm_stats['pcasl_mean']) / (np.array(self.norm_stats['pcasl_std']) + 1e-6)
                    v_norm = (noisy_signal[self.num_plds:] - self.norm_stats['vsasl_mean']) / (np.array(self.norm_stats['vsasl_std']) + 1e-6)
                    final_input = np.concatenate([p_norm, v_norm, eng_feat.flatten()])
                
                if self.stage == 1:
                    target = clean_signal
                else: # stage 2
                    target = np.array([(true_cbf - self.norm_stats['y_mean_cbf']) / self.norm_stats['y_std_cbf'], (true_att - self.norm_stats['y_mean_att']) / self.norm_stats['y_std_att']])

                yield torch.from_numpy(final_input.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
            except Exception as e: logger.error(f"DataLoader worker failed: {e}"); continue

class EnhancedASLTrainer:
    def __init__(self,
                 stage: int, model_config: Dict, model_class: callable,
                 weight_decay: float = 1e-5, batch_size: int = 256, n_ensembles: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        self.device = torch.device(device); self.model_config = model_config; self.stage = stage
        self.lr_base = float(model_config.get('learning_rate', 0.001)); self.weight_decay = float(weight_decay)
        self.n_ensembles = n_ensembles; self.validation_steps_per_epoch = model_config.get('validation_steps_per_epoch', 50)
        logger.info("Initializing models and casting to bfloat16..."); self.models = [model_class(**model_config).to(self.device, dtype=torch.bfloat16) for _ in range(n_ensembles)]
        logger.info("Compiling models with torch.compile()..."); self.models = [torch.compile(m, mode="max-autotune") for m in self.models]
        logger.info("Model compilation complete.")
        if wandb.run:
            for i, model in enumerate(self.models): wandb.watch(model, log='gradients', log_freq=200, idx=i)
        
        loss_params = {'training_stage': stage, 'w_cbf': model_config.get('loss_weight_cbf', 1.0), 'w_att': model_config.get('loss_weight_att', 1.0),
                       'log_var_reg_lambda': model_config.get('loss_log_var_reg_lambda', 0.0)}; self.custom_loss_fn = CustomLoss(**loss_params)
        self.global_step = 0; self.norm_stats = None

    def train_ensemble(self,
                       train_loader: DataLoader, val_loader: Optional[DataLoader],
                       n_epochs: int, steps_per_epoch: Optional[int], output_dir: Path,
                       early_stopping_patience: int = 10, early_stopping_min_delta: float = 0.0,
                       optuna_trial: Optional[Any] = None, fine_tuning_config: Optional[Dict] = None):
        if steps_per_epoch is None:
            if isinstance(train_loader.dataset, IterableDataset):
                raise ValueError("steps_per_epoch must be specified for an IterableDataset.")
            steps_per_epoch = len(train_loader)
        
        is_finetuning = fine_tuning_config is not None and fine_tuning_config.get('enabled', False)
        run_type = "Fine-Tuning" if is_finetuning else "Training"
        logger.info(f"--- Starting Stage {self.stage} {run_type} for {n_epochs} epochs ---")
        
        if is_finetuning:
            logger.info(f"--- Fine-tuning Mode: Freezing encoder and creating optimizer for head parameters. ---")
            for m in self.models:
                if hasattr(m, 'freeze_encoder'): m.freeze_encoder()
            self.optimizers = [torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=self.lr_base, weight_decay=self.weight_decay) for m in self.models]
        else:
            logger.info(f"--- Standard Training Mode: Creating optimizer for all parameters. ---")
            self.optimizers = [torch.optim.AdamW(m.parameters(), lr=self.lr_base, weight_decay=self.weight_decay) for m in self.models]
        
        total_steps = steps_per_epoch * n_epochs
        self.schedulers = [torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr_base, total_steps=total_steps) for opt in self.optimizers]
        
        patience_counters = [0] * self.n_ensembles; self.overall_best_val_losses = [float('inf')] * self.n_ensembles
        
        models_save_dir = output_dir / 'trained_models'
        models_save_dir.mkdir(exist_ok=True)

        for epoch in range(n_epochs):
            train_loss, _ = self._train_epoch(self.models, train_loader, self.optimizers, self.schedulers, epoch, steps_per_epoch)
            val_metrics = self._validate(self.models, val_loader, epoch)
            mean_val_loss = np.nanmean([m.get('val_loss', np.nan) for m in val_metrics])
            logger.info(f"Epoch {epoch + 1}/{n_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {mean_val_loss:.6f}")
            if wandb.run: wandb.log({'epoch': epoch, 'mean_train_loss': train_loss, 'mean_val_loss': mean_val_loss}, step=self.global_step)
            for i, vm in enumerate(val_metrics):
                val_loss = vm.get('val_loss', float('inf'))
                if val_loss < self.overall_best_val_losses[i] - early_stopping_min_delta:
                    self.overall_best_val_losses[i] = val_loss; patience_counters[i] = 0
                    unwrapped_model = getattr(self.models[i], '_orig_mod', self.models[i])
                    
                    if self.stage == 1:
                        model_path = output_dir / 'encoder_pretrained.pt'
                        torch.save(unwrapped_model.encoder.state_dict(), model_path)
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
            for model_idx, model in enumerate(models):
                try: signals, targets = next(loader_iter)
                except StopIteration: loader_iter = iter(loader); signals, targets = next(loader_iter)
                signals, targets = signals.to(self.device), targets.to(self.device)
                optimizers[model_idx].zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = model(signals)
                    loss, comps = self.custom_loss_fn(model_outputs=outputs, targets=targets, global_epoch=epoch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[model_idx].step()
                schedulers[model_idx].step()
                
                if model_idx == 0: total_loss += loss.item()
                
                if wandb.run and model_idx == 0 and self.global_step % 20 == 0:
                    log_dict = {"train/total_loss": loss.item(), "train/learning_rate": schedulers[model_idx].get_last_lr()[0]}
                    if self.stage == 1:
                        log_dict["train_comps/denoising_loss"] = comps.get('denoising_loss', torch.tensor(0.0)).item()
                    else: # stage 2
                        log_dict["train_comps/nll_loss"] = comps.get('param_nll_loss', torch.tensor(0.0)).item()
                        log_dict["train_comps/log_var_reg"] = comps.get('log_var_reg_loss', torch.tensor(0.0)).item()
                        cbf_lvar, att_lvar = outputs[2], outputs[3]
                        with torch.no_grad():
                            cbf_std_learned = torch.mean(torch.exp(cbf_lvar * 0.5))
                            att_std_learned = torch.mean(torch.exp(att_lvar * 0.5))
                        log_dict["uncertainty/cbf_std_learned_norm"] = cbf_std_learned.item()
                        log_dict["uncertainty/att_std_learned_norm"] = att_std_learned.item()
                    wandb.log(log_dict, step=self.global_step)

            self.global_step += 1
        return total_loss / steps, {}

    def _validate(self, models, loader, epoch):
        if not loader: return [{'val_loss': float('inf')} for _ in models]
        for m in models: m.eval()
        val_losses = [0.0] * len(models)
        
        is_iterable = isinstance(loader.dataset, IterableDataset)
        val_iterator = islice(loader, self.validation_steps_per_epoch) if is_iterable else loader
        num_steps = 0

        with torch.no_grad():
            for signals, targets in val_iterator:
                signals, targets = signals.to(self.device), targets.to(self.device)
                for i, model in enumerate(models):
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        outputs = model(signals)
                        loss, _ = self.custom_loss_fn(model_outputs=outputs, targets=targets, global_epoch=epoch)
                    val_losses[i] += loss.item()
                num_steps += 1
                if is_iterable and num_steps >= self.validation_steps_per_epoch:
                    break
        
        return [{'val_loss': v / num_steps if num_steps > 0 else float('inf')} for v in val_losses]
    
    def predict(self, signals: np.ndarray): pass
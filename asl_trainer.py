# FILE: asl_trainer.py
# FINAL CORRECTED VERSION FOR UNIFIED TRAINING V2

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

from enhanced_asl_network import EnhancedASLNet, CustomLoss, DisentangledASLNet
from enhanced_simulation import RealisticASLSimulator
from utils import engineer_signal_features

import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Helper classes (ASLNet, ASLDataset, etc.) are included for file completeness ---

class ASLNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32, 16]):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(0.1)])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 2))
        self.network = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ASLDataset(Dataset):
    def __init__(self, signals: np.ndarray, params: np.ndarray):
        self.signals = torch.FloatTensor(signals)
        self.params = torch.FloatTensor(params)
    def __len__(self) -> int:
        return len(self.signals)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.params[idx]

class ASLTrainer:
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32, 16], learning_rate: float = 1e-3, batch_size: int = 32, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device; self.batch_size = batch_size; self.model = ASLNet(input_size, hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate); self.criterion = nn.MSELoss()
        self.train_losses = []; self.val_losses = []
    def train(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int = 100, early_stopping_patience: int = 10): pass

class EnhancedASLDataset(Dataset):
    def __init__(self, signals: np.ndarray, params: np.ndarray, **kwargs):
        self.signals = torch.FloatTensor(signals); self.params = torch.FloatTensor(params)
    def __len__(self) -> int: return len(self.signals)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: return self.signals[idx], self.params[idx]

class ASLIterableDataset(IterableDataset):
    def __init__(self, simulator: RealisticASLSimulator, plds: np.ndarray, 
                 noise_levels: List[float], norm_stats: Dict, num_att_bins: int = 14,
                 disentangled_mode: bool = False):
        super().__init__()
        self.simulator, self.plds, self.num_plds = simulator, plds, len(plds)
        self.noise_levels, self.norm_stats = noise_levels, norm_stats
        self.base_params, self.physio_var = simulator.params, simulator.physio_var
        self.disentangled_mode = disentangled_mode
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.id + int(time.time() * 1000) if worker_info else int(time.time() * 1000)
        np.random.seed(seed % (2**32))
        while True:
            try:
                true_att = np.random.uniform(*self.physio_var.att_range)
                true_cbf = np.random.uniform(*self.physio_var.cbf_range)
                true_t1 = np.random.uniform(*self.physio_var.t1_artery_range)
                snr = np.random.choice(self.noise_levels)
                t_tau = self.base_params.T_tau * (1 + np.random.uniform(*self.physio_var.t_tau_perturb_range))
                alpha_p = np.clip(self.base_params.alpha_PCASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range)), 0.1, 1.1)
                alpha_v = np.clip(self.base_params.alpha_VSASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range)), 0.1, 1.0)
                data = self.simulator.generate_synthetic_data(self.plds, np.array([true_att]), 1, snr, true_cbf, true_t1, t_tau, alpha_p, alpha_v)
                raw_sig = np.concatenate([data['MULTIVERSE'][0,0,:,0], data['MULTIVERSE'][0,0,:,1]])
                eng_feat = engineer_signal_features(raw_sig, self.num_plds)
                if self.disentangled_mode:
                    amp = np.linalg.norm(raw_sig) + 1e-6
                    shape = raw_sig / amp
                    amp_norm = (amp - self.norm_stats['amplitude_mean']) / (self.norm_stats['amplitude_std'] + 1e-6)
                    final_input = np.concatenate([shape, eng_feat.flatten(), np.array([amp_norm])])
                else:
                    p_norm = (raw_sig[:self.num_plds] - self.norm_stats['pcasl_mean']) / (np.array(self.norm_stats['pcasl_std']) + 1e-6)
                    v_norm = (raw_sig[self.num_plds:] - self.norm_stats['vsasl_mean']) / (np.array(self.norm_stats['vsasl_std']) + 1e-6)
                    final_input = np.concatenate([p_norm, v_norm, eng_feat.flatten()])
                params = np.array([(true_cbf - self.norm_stats['y_mean_cbf']) / self.norm_stats['y_std_cbf'], (true_att - self.norm_stats['y_mean_att']) / self.norm_stats['y_std_att']])
                yield torch.from_numpy(final_input.astype(np.float32)), torch.from_numpy(params.astype(np.float32))
            except Exception as e:
                logger.error(f"DataLoader worker failed: {e}"); continue

class EnhancedASLTrainer:
    def __init__(self,
                 model_config: Dict, model_class: callable, input_size: int,
                 weight_decay: float = 1e-5, batch_size: int = 256, n_ensembles: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model_config = model_config
        self.lr_base = model_config.get('learning_rate', 0.001)
        self.weight_decay = weight_decay
        self.n_ensembles = n_ensembles
        self.validation_steps_per_epoch = model_config.get('validation_steps_per_epoch', 50)
        self.scalers = [torch.cuda.amp.GradScaler() for _ in range(self.n_ensembles)]

        logger.info("Initializing models and casting to bfloat16...")
        self.models = [model_class(**model_config).to(self.device, dtype=torch.bfloat16) for _ in range(n_ensembles)]
        
        logger.info("Compiling models with torch.compile()...")
        self.models = [torch.compile(m, mode="max-autotune") for m in self.models]
        logger.info("Model compilation complete.")

        if wandb.run:
            for i, model in enumerate(self.models):
                wandb.watch(model, log='all', log_freq=200, idx=i)

        self.best_states = [None] * self.n_ensembles
        
        # --- MODIFIED: Correctly instantiate the simplified CustomLoss ---
        loss_params = {
            'w_cbf': model_config.get('loss_weight_cbf', 1.0), 
            'w_att': model_config.get('loss_weight_att', 1.0),
            'log_var_reg_lambda': model_config.get('loss_log_var_reg_lambda', 0.0),
            'pinn_weight': model_config.get('loss_pinn_weight', 0.0), # Use the single unified key
            'model_params': model_config,
        }
        self.custom_loss_fn = CustomLoss(**loss_params)
        # --- END MODIFICATION ---

        self.global_step = 0
        self.norm_stats = None

    def train_ensemble(self,
                       train_loader: DataLoader, val_loader: Optional[DataLoader],
                       n_epochs: int, steps_per_epoch: Optional[int],
                       early_stopping_patience: int = 20, optuna_trial: Optional[Any] = None) -> Dict[str, Any]:
        
        histories = defaultdict(list)
        self.global_step = 0
        
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)
            logger.info(f"Using map-style dataset. One epoch = {steps_per_epoch} steps.")
        else:
            logger.info(f"Using iterable dataset. One epoch = {steps_per_epoch} steps.")
        
        logger.info(f"--- Starting Unified Training for {n_epochs} epochs ---")

        self.optimizers = [torch.optim.AdamW(m.parameters(), lr=self.lr_base, weight_decay=self.weight_decay) for m in self.models]
        total_steps = steps_per_epoch * n_epochs
        self.schedulers = [torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr_base, total_steps=total_steps) for opt in self.optimizers]
        
        patience_counters = [0] * self.n_ensembles
        self.overall_best_val_losses = [float('inf')] * self.n_ensembles

        for epoch in range(n_epochs):
            epoch_train_losses = defaultdict(list)
            epoch_val_metrics = []
            
            train_loader_iter = iter(train_loader)
            for model_idx in range(self.n_ensembles):
                train_loss, components = self._train_epoch(self.models[model_idx], train_loader_iter, self.optimizers[model_idx], self.scalers[model_idx], self.schedulers[model_idx], epoch, steps_per_epoch)
                epoch_train_losses['total_loss'].append(train_loss)
                for key, val in components.items(): epoch_train_losses[key].append(val)
                if model_idx < self.n_ensembles - 1: train_loader_iter = iter(train_loader)

            if val_loader:
                for model_idx in range(self.n_ensembles):
                    val_metrics = self._validate(self.models[model_idx], val_loader, epoch)
                    epoch_val_metrics.append(val_metrics)
                    val_loss = val_metrics.get('val_loss', float('inf'))
                    if val_loss < self.overall_best_val_losses[model_idx]:
                        self.overall_best_val_losses[model_idx] = val_loss
                        patience_counters[model_idx] = 0
                        unwrapped = getattr(self.models[model_idx], '_orig_mod', self.models[model_idx])
                        self.best_states[model_idx] = unwrapped.state_dict()
                    else:
                        patience_counters[model_idx] += 1
            
            mean_train_loss = np.nanmean(epoch_train_losses['total_loss'])
            mean_val_loss = np.nanmean([m.get('val_loss', np.nan) for m in epoch_val_metrics])
            logger.info(f"Epoch {epoch + 1}/{n_epochs}: Train Loss = {mean_train_loss:.6f}, Val Loss = {mean_val_loss:.6f}")

            if wandb.run:
                log_dict = {'epoch': epoch, 'mean_train_loss': mean_train_loss, 'mean_val_loss': mean_val_loss}
                wandb.log(log_dict, step=self.global_step)

            if all(p >= early_stopping_patience for p in patience_counters):
                logger.info("All models have early stopped. Ending training.")
                break
        
        for i, state in enumerate(self.best_states):
            if state:
                unwrapped = getattr(self.models[i], '_orig_mod', self.models[i])
                unwrapped.load_state_dict(state)
        
        final_mean_val_loss = np.nanmean([l for l in self.overall_best_val_losses if l != float('inf')] or [np.nan])
        return {'final_mean_val_loss': final_mean_val_loss, 'all_histories': histories}

    def _train_epoch(self, model, train_loader_iter, optimizer, scaler, scheduler, epoch, steps_per_epoch):
        model.train(); total_loss, total_components = 0.0, defaultdict(float)
        for i, (signals, params_norm) in enumerate(islice(train_loader_iter, steps_per_epoch)):
            signals, params_norm = signals.to(self.device), params_norm.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = model(signals)
                loss, components = self.custom_loss_fn(signals, *outputs, cbf_true_norm=params_norm[:, 0:1], att_true_norm=params_norm[:, 1:2], global_epoch=epoch)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += loss.item()
            for k, v in components.items(): total_components[k] += v.item()
            self.global_step += 1
        return total_loss / steps_per_epoch, {k: v / steps_per_epoch for k, v in total_components.items()}

    def _validate(self, model, val_loader, epoch):
        model.eval(); total_loss = 0.0
        val_iterator = islice(val_loader, self.validation_steps_per_epoch) if isinstance(val_loader.dataset, IterableDataset) else val_loader
        num_steps = self.validation_steps_per_epoch if isinstance(val_loader.dataset, IterableDataset) else len(val_loader)
        with torch.no_grad():
            for signals, params_norm in val_iterator:
                signals, params_norm = signals.to(self.device), params_norm.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = model(signals)
                    loss, _ = self.custom_loss_fn(signals, *outputs, cbf_true_norm=params_norm[:, 0:1], att_true_norm=params_norm[:, 1:2], global_epoch=epoch)
                total_loss += loss.item()
        return {'val_loss': total_loss / num_steps if num_steps > 0 else float('inf')}
        
    def predict(self, signals: np.ndarray): pass

class ASLInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Optional[str], norm_stats: dict, disentangled_mode: bool = False):
        self.data_dir = Path(data_dir) if data_dir else None; self.norm_stats = norm_stats
        self.num_plds = len(norm_stats.get('pcasl_mean', [])); self.disentangled_mode = disentangled_mode
        if self.data_dir:
            files = sorted(list(self.data_dir.glob('dataset_chunk_*.npz')))
            if not files: raise FileNotFoundError(f"No dataset chunks in {data_dir}.")
            all_signals = [np.load(f)['signals'] for f in files]
            all_params = [np.load(f)['params'] for f in files]
            self.signals_unnormalized = np.concatenate(all_signals, axis=0)
            self.params_unnormalized = np.concatenate(all_params, axis=0)
            self.signals_processed = self._process_signals(self.signals_unnormalized)
            self.params_normalized = self._normalize_params(self.params_unnormalized)
            self.signals_tensor = torch.from_numpy(self.signals_processed.astype(np.float32))
            self.params_tensor = torch.from_numpy(self.params_normalized.astype(np.float32))
        else:
            self.signals_tensor = torch.empty(0); self.params_tensor = torch.empty(0)
    def _process_signals(self, signals_unnorm: np.ndarray) -> np.ndarray:
        raw, eng = signals_unnorm[:, :self.num_plds*2], signals_unnorm[:, self.num_plds*2:]
        if self.disentangled_mode:
            amp = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-6
            shape = raw / amp
            amp_norm = (amp - self.norm_stats['amplitude_mean']) / (self.norm_stats['amplitude_std'] + 1e-6)
            return np.concatenate([shape, eng, amp_norm], axis=1)
        else:
            p_raw, v_raw = raw[:, :self.num_plds], raw[:, self.num_plds:]
            p_norm = (p_raw - self.norm_stats['pcasl_mean']) / (np.array(self.norm_stats['pcasl_std']) + 1e-6)
            v_norm = (v_raw - self.norm_stats['vsasl_mean']) / (np.array(self.norm_stats['vsasl_std']) + 1e-6)
            return np.concatenate([p_norm, v_norm, eng], axis=1)
    def _normalize_params(self, params_unnorm: np.ndarray) -> np.ndarray:
        cbf, att = params_unnorm[:, 0], params_unnorm[:, 1]
        cbf_norm = (cbf - self.norm_stats['y_mean_cbf']) / self.norm_stats['y_std_cbf']
        att_norm = (att - self.norm_stats['y_mean_att']) / self.norm_stats['y_std_att']
        return np.stack([cbf_norm, att_norm], axis=1)
    def __len__(self): return self.signals_tensor.shape[0]
    def __getitem__(self, idx): return self.signals_tensor[idx], self.params_tensor[idx]
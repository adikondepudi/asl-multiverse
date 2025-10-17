# FILE: asl_trainer.py
# FILE: asl_trainer.py
# FINAL, DEFINITIVE, CORRECTED VERSION

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

# --- Helper classes ---
class ASLNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32, 16]):
        super().__init__()
        layers = []
        prev_size = input_size
        for hs in hidden_sizes: layers.extend([nn.Linear(prev_size, hs), nn.ReLU(), nn.BatchNorm1d(hs), nn.Dropout(0.1)]); prev_size = hs
        layers.append(nn.Linear(prev_size, 2)); self.network = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.network(x)

class ASLDataset(Dataset):
    def __init__(self, signals: np.ndarray, params: np.ndarray): self.signals, self.params = torch.FloatTensor(signals), torch.FloatTensor(params)
    def __len__(self) -> int: return len(self.signals)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: return self.signals[idx], self.params[idx]

class ASLTrainer: # Old trainer
    def train(self): pass

class EnhancedASLDataset(Dataset):
    def __init__(self, signals: np.ndarray, params: np.ndarray, **kwargs): self.signals, self.params = torch.FloatTensor(signals), torch.FloatTensor(params)
    def __len__(self) -> int: return len(self.signals)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: return self.signals[idx], self.params[idx]

class ASLIterableDataset(IterableDataset):
    def __init__(self, simulator: RealisticASLSimulator, plds: np.ndarray, 
                 noise_levels: List[float], norm_stats: Dict, disentangled_mode: bool = False):
        super().__init__(); self.simulator, self.plds, self.num_plds = simulator, plds, len(plds)
        self.noise_levels, self.norm_stats = noise_levels, norm_stats
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
                data = self.simulator.generate_synthetic_data(self.plds, np.array([true_att]), n_noise=1, tsnr=snr, cbf_val=true_cbf)
                raw_sig = np.concatenate([data['MULTIVERSE'][0,0,:,0], data['MULTIVERSE'][0,0,:,1]])
                eng_feat = engineer_signal_features(raw_sig, self.num_plds)
                if self.disentangled_mode:
                    amp = np.linalg.norm(raw_sig) + 1e-6; shape = raw_sig / amp
                    amp_norm = (amp - self.norm_stats['amplitude_mean']) / (self.norm_stats['amplitude_std'] + 1e-6)
                    final_input = np.concatenate([shape, eng_feat.flatten(), np.array([amp_norm])])
                else:
                    p_norm = (raw_sig[:self.num_plds] - self.norm_stats['pcasl_mean']) / (np.array(self.norm_stats['pcasl_std']) + 1e-6)
                    v_norm = (raw_sig[self.num_plds:] - self.norm_stats['vsasl_mean']) / (np.array(self.norm_stats['vsasl_std']) + 1e-6)
                    final_input = np.concatenate([p_norm, v_norm, eng_feat.flatten()])
                params = np.array([(true_cbf - self.norm_stats['y_mean_cbf']) / self.norm_stats['y_std_cbf'], (true_att - self.norm_stats['y_mean_att']) / self.norm_stats['y_std_att']])
                yield torch.from_numpy(final_input.astype(np.float32)), torch.from_numpy(params.astype(np.float32))
            except Exception as e: logger.error(f"DataLoader worker failed: {e}"); continue

class EnhancedASLTrainer:
    def __init__(self,
                 model_config: Dict, model_class: callable, input_size: int,
                 weight_decay: float = 1e-5, batch_size: int = 256, n_ensembles: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        self.device = torch.device(device); self.model_config = model_config
        self.lr_base = float(model_config.get('learning_rate', 0.001)); self.weight_decay = float(weight_decay)
        self.n_ensembles = n_ensembles; self.validation_steps_per_epoch = model_config.get('validation_steps_per_epoch', 50)
        # --- FIX: Removed GradScaler initialization, as it's not compatible with BFloat16 ---
        logger.info("Initializing models and casting to bfloat16..."); self.models = [model_class(**model_config).to(self.device, dtype=torch.bfloat16) for _ in range(n_ensembles)]
        logger.info("Compiling models with torch.compile()..."); self.models = [torch.compile(m, mode="max-autotune") for m in self.models]
        logger.info("Model compilation complete.")
        if wandb.run:
            for i, model in enumerate(self.models): wandb.watch(model, log='all', log_freq=200, idx=i)
        self.best_states = [None] * self.n_ensembles
        loss_params = {'w_cbf': model_config.get('loss_weight_cbf', 1.0), 'w_att': model_config.get('loss_weight_att', 1.0),
                       'log_var_reg_lambda': model_config.get('loss_log_var_reg_lambda', 0.0), 'pinn_weight': model_config.get('loss_pinn_weight', 0.0),
                       'residual_pinn_weight': model_config.get('residual_pinn_weight', 0.0), # NEW for v3
                       'model_params': model_config}; self.custom_loss_fn = CustomLoss(**loss_params)
        self.global_step = 0; self.norm_stats = None

    def train_ensemble(self,
                       train_loader: DataLoader, val_loader: Optional[DataLoader],
                       n_epochs: int, steps_per_epoch: Optional[int],
                       early_stopping_patience: int = 20, optuna_trial: Optional[Any] = None):
        if steps_per_epoch is None: steps_per_epoch = len(train_loader)
        logger.info(f"--- Starting Unified Training for {n_epochs} epochs ---")
        self.optimizers = [torch.optim.AdamW(m.parameters(), lr=self.lr_base, weight_decay=self.weight_decay) for m in self.models]
        total_steps = steps_per_epoch * n_epochs
        self.schedulers = [torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr_base, total_steps=total_steps) for opt in self.optimizers]
        patience_counters = [0] * self.n_ensembles; self.overall_best_val_losses = [float('inf')] * self.n_ensembles
        for epoch in range(n_epochs):
            # --- FIX: Removed scalers from the arguments ---
            train_loss, _ = self._train_epoch(self.models, train_loader, self.optimizers, self.schedulers, epoch, steps_per_epoch)
            val_metrics = self._validate(self.models, val_loader, epoch)
            mean_val_loss = np.nanmean([m.get('val_loss', np.nan) for m in val_metrics])
            logger.info(f"Epoch {epoch + 1}/{n_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {mean_val_loss:.6f}")
            if wandb.run: wandb.log({'epoch': epoch, 'mean_train_loss': train_loss, 'mean_val_loss': mean_val_loss}, step=self.global_step)
            for i, vm in enumerate(val_metrics):
                val_loss = vm.get('val_loss', float('inf'))
                if val_loss < self.overall_best_val_losses[i]:
                    self.overall_best_val_losses[i] = val_loss; patience_counters[i] = 0
                    unwrapped = getattr(self.models[i], '_orig_mod', self.models[i]); self.best_states[i] = unwrapped.state_dict()
                else: patience_counters[i] += 1
            if all(p >= early_stopping_patience for p in patience_counters): logger.info("All models early stopped."); break
        for i, state in enumerate(self.best_states):
            if state: getattr(self.models[i], '_orig_mod', self.models[i]).load_state_dict(state)
        return {'final_mean_val_loss': np.nanmean(self.overall_best_val_losses)}

    # --- MODIFIED: Major changes for Online Hard Example Mining (OHEM) ---
    def _train_epoch(self, models, loader, optimizers, schedulers, epoch, steps):
        for m in models: m.train()
        total_loss = 0.0; loader_iter = iter(loader)
        # Get OHEM fraction from config, default to 1.0 (no OHEM)
        ohem_fraction = self.model_config.get('ohem_fraction', 1.0)

        for i in range(steps):
            for model_idx, model in enumerate(models):
                try: signals, params_norm = next(loader_iter)
                except StopIteration: loader_iter = iter(loader); signals, params_norm = next(loader_iter)
                signals, params_norm = signals.to(self.device), params_norm.to(self.device)
                optimizers[model_idx].zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    cbf_mean, att_mean, cbf_lvar, att_lvar, cbf_r, att_r = model(signals)
                    # Calculate the full batch loss AND get the unreduced components
                    loss_full_batch, comps = self.custom_loss_fn(
                        normalized_input_signal=signals, cbf_pred_norm=cbf_mean, att_pred_norm=att_mean,
                        cbf_true_norm=params_norm[:, 0:1], att_true_norm=params_norm[:, 1:2],
                        cbf_log_var=cbf_lvar, att_log_var=att_lvar,
                        cbf_rough_physical=cbf_r, att_rough_physical=att_r, global_epoch=epoch)
                
                # --- OHEM LOGIC ---
                if ohem_fraction < 1.0:
                    # Get the per-sample NLL loss (now attached to the graph)
                    unreduced_loss = comps['unreduced_loss']
                    num_hard_examples = int(signals.shape[0] * ohem_fraction)
                    
                    # Select the top 'k' losses
                    top_k_losses, _ = torch.topk(unreduced_loss.view(-1), k=num_hard_examples)
                    
                    # The final loss for backpropagation is the mean of only the hardest examples
                    final_loss = top_k_losses.mean()
                else:
                    # Default behavior: use the mean loss of the full batch
                    final_loss = loss_full_batch
                # --- END OHEM LOGIC ---
                
                # The GradScaler logic is removed. We call backward() directly on the final_loss.
                final_loss.backward() # Backpropagate only on the loss from the hardest examples
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[model_idx].step()
                schedulers[model_idx].step()
                
                if model_idx == 0: total_loss += final_loss.item()

            self.global_step += 1
        return total_loss / steps, {}

    def _validate(self, models, loader, epoch):
        if not loader: return [{'val_loss': float('inf')} for _ in models]
        for m in models: m.eval()
        val_losses = [0.0] * len(models)
        val_iterator = islice(loader, self.validation_steps_per_epoch) if isinstance(loader.dataset, IterableDataset) else loader
        num_steps = 0
        with torch.no_grad():
            for signals, params_norm in val_iterator:
                signals, params_norm = signals.to(self.device), params_norm.to(self.device)
                for i, model in enumerate(models):
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        cbf_mean, att_mean, cbf_lvar, att_lvar, cbf_r, att_r = model(signals)
                        loss, _ = self.custom_loss_fn(
                            normalized_input_signal=signals, cbf_pred_norm=cbf_mean, att_pred_norm=att_mean,
                            cbf_true_norm=params_norm[:, 0:1], att_true_norm=params_norm[:, 1:2],
                            cbf_log_var=cbf_lvar, att_log_var=att_lvar,
                            cbf_rough_physical=cbf_r, att_rough_physical=att_r, global_epoch=epoch)
                    val_losses[i] += loss.item()
                num_steps += 1
        return [{'val_loss': v / num_steps if num_steps > 0 else float('inf')} for v in val_losses]
    
    def predict(self, signals: np.ndarray): pass

class ASLInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Optional[str], norm_stats: dict, disentangled_mode: bool = False):
        self.data_dir = Path(data_dir) if data_dir else None; self.norm_stats = norm_stats
        self.num_plds = len(norm_stats.get('pcasl_mean', [])); self.disentangled_mode = disentangled_mode
        if self.data_dir:
            logger.info(f"Loading dataset chunks from {self.data_dir}..."); files = sorted(list(self.data_dir.glob('dataset_chunk_*.npz')))
            if not files: raise FileNotFoundError(f"No dataset chunks in {data_dir}.")
            all_signals = [np.load(f)['signals'] for f in files]; all_params = [np.load(f)['params'] for f in files]
            self.signals_unnormalized = np.concatenate(all_signals, axis=0); self.params_unnormalized = np.concatenate(all_params, axis=0)
            logger.info(f"Loaded {len(self.signals_unnormalized)} samples. Processing..."); self.signals_processed = self._process_signals(self.signals_unnormalized)
            self.params_normalized = self._normalize_params(self.params_unnormalized)
            self.signals_tensor = torch.from_numpy(self.signals_processed.astype(np.float32)); self.params_tensor = torch.from_numpy(self.params_normalized.astype(np.float32))
        else:
            logger.info("ASLInMemoryDataset initialized in manual mode."); self.signals_unnormalized=np.array([]); self.params_unnormalized=np.array([]); self.signals_tensor = torch.empty(0); self.params_tensor = torch.empty(0)
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
    def __getitem__(self, idx): return self.signals_tensor[idx], self.params_tensor[idx]
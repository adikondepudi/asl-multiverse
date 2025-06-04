import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from torch.optim.lr_scheduler import OneCycleLR 
import math 
import multiprocessing as mp
from pathlib import Path # Added Path

num_workers = mp.cpu_count()

# Assuming EnhancedASLNet and CustomLoss are correctly imported from enhanced_asl_network
from enhanced_asl_network import EnhancedASLNet, CustomLoss
# Assuming RealisticASLSimulator is imported if type hints are to be strictly checked
# from enhanced_simulation import RealisticASLSimulator # For type hinting

# For logging within this file if not passed from main.py
import logging
logger = logging.getLogger(__name__)
# Basic config for logger if it's not configured by main.py when this module is imported
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ASLNet(nn.Module):
    """Neural network for ASL parameter estimation"""

    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32, 16]):
        super().__init__()

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        # Output layer for CBF and ATT
        layers.append(nn.Linear(prev_size, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ASLDataset(Dataset):
    """Dataset for ASL signals and parameters"""

    def __init__(self, signals: np.ndarray, params: np.ndarray):
        self.signals = torch.FloatTensor(signals)
        self.params = torch.FloatTensor(params)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.params[idx]

class ASLTrainer:
    """Training manager for ASL parameter estimation network"""

    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int] = [64, 32, 16],
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.batch_size = batch_size

        # Initialize network
        self.model = ASLNet(input_size, hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Track training progress
        self.train_losses = []
        self.val_losses = []

    def prepare_data(self,
                    simulator, # ASLSimulator instance
                    n_samples: int = 10000,
                    val_split: float = 0.2,
                    plds_definition: Optional[np.ndarray] = None) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        # Generate synthetic data
        if plds_definition is None:
            plds = np.arange(500, 3001, 500)  # 6 PLDs
        else:
            plds = plds_definition

        att_values = np.arange(0, 4001, 100) # This generates many ATT points per noise realization
        # The original generate_synthetic_data creates (n_noise, n_att, n_plds)
        # n_samples here refers to n_noise in generate_synthetic_data
        # Pass current simulator CBF for data generation
        signals_dict = simulator.generate_synthetic_data(
            plds, att_values, n_noise=n_samples, tsnr=5.0, 
            cbf_val=simulator.params.CBF # Explicitly pass CBF from simulator config
        )


        # Correctly reshape the signals
        # Total samples will be n_samples (noise realizations) * len(att_values)
        num_total_instances = n_samples * len(att_values)
        X = np.zeros((num_total_instances, len(plds) * 2)) # PCASL + VSASL

        # PCASL signals: (n_noise, n_att, n_plds)
        # VSASL signals: (n_noise, n_att, n_plds)
        pcasl_all = signals_dict['PCASL'].reshape(num_total_instances, len(plds))
        vsasl_all = signals_dict['VSASL'].reshape(num_total_instances, len(plds))

        X[:, :len(plds)] = pcasl_all
        X[:, len(plds):] = vsasl_all

        # Generate corresponding parameters
        # CBF is fixed by simulator.params.CBF for generate_synthetic_data
        cbf_true_val = simulator.params.CBF
        cbf_params = np.full(num_total_instances, cbf_true_val)
        att_params = np.tile(np.repeat(att_values, 1), n_samples) # att_values repeated for each noise realization

        y = np.column_stack((cbf_params, att_params))

        # Split data
        if num_total_instances == 0:
            raise ValueError("No data generated by simulator.prepare_data.")
        n_val = int(num_total_instances * val_split)
        if n_val == 0 and num_total_instances > 1: n_val = 1
        if n_val >= num_total_instances : n_val = num_total_instances -1 if num_total_instances > 0 else 0


        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        if X_train.shape[0] == 0:
            raise ValueError("Training set is empty after split.")


        # Create data loaders
        train_dataset = ASLDataset(X_train, y_train)
        val_dataset = ASLDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size if X_val.shape[0] > 0 else 1)


        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for signals, params in train_loader:
            signals = signals.to(self.device)
            params = params.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, params)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        if val_loader is None or len(val_loader) == 0:
            logger.warning("Validation loader is empty or None, skipping validation.")
            return float('inf') # Or an appropriate high loss value

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for signals, params in val_loader:
                signals = signals.to(self.device)
                params = params.to(self.device)

                outputs = self.model(signals)
                loss = self.criterion(outputs, params)
                total_loss += loss.item()

        return total_loss / len(val_loader) if len(val_loader) > 0 else float('inf') # Added check for len


    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int = 100,
              early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """Train the model with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model_ASLTrainer.pt') # Differentiate from EnhancedASLTrainer
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                break

            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')

        # Load best model if it exists
        model_path = Path('best_model_ASLTrainer.pt')
        if model_path.exists():
            self.model.load_state_dict(torch.load(str(model_path))) # Use str(model_path) for PyTorch
            logger.info(f"Loaded best model from {model_path}")
        else:
            logger.warning(f"Best model file {model_path} not found. Current model state will be used.")


        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Predict CBF and ATT from ASL signals"""
        self.model.eval()
        with torch.no_grad():
            signals_tensor = torch.FloatTensor(signals).to(self.device)
            if signals_tensor.ndim == 1: # Handle single sample case
                signals_tensor = signals_tensor.unsqueeze(0)
            predictions = self.model(signals_tensor)
            return predictions.cpu().numpy()

    def evaluate_performance(self,
                           test_signals: np.ndarray,
                           true_params: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        predictions = self.predict(test_signals)

        # Calculate metrics
        mae_cbf = np.mean(np.abs(predictions[:,0] - true_params[:,0]))
        mae_att = np.mean(np.abs(predictions[:,1] - true_params[:,1]))

        rmse_cbf = np.sqrt(np.mean((predictions[:,0] - true_params[:,0])**2))
        rmse_att = np.sqrt(np.mean((predictions[:,1] - true_params[:,1])**2))

        # Relative errors (avoid division by zero)
        rel_error_cbf = np.mean(np.abs(predictions[:,0] - true_params[:,0]) / np.clip(true_params[:,0], 1e-6, None))
        rel_error_att = np.mean(np.abs(predictions[:,1] - true_params[:,1]) / np.clip(true_params[:,1], 1e-6, None))

        return {
            'MAE_CBF': mae_cbf,
            'MAE_ATT': mae_att,
            'RMSE_CBF': rmse_cbf,
            'RMSE_ATT': rmse_att,
            'RelError_CBF': rel_error_cbf,
            'RelError_ATT': rel_error_att
        }


class EnhancedASLDataset(Dataset):
    """Enhanced dataset with noise augmentation and weighted sampling"""

    def __init__(self,
                 signals: np.ndarray, # Expected shape (N_samples, N_features)
                 params: np.ndarray,  # Expected shape (N_samples, N_param_outputs e.g. 2 for CBF,ATT)
                 noise_levels: List[float] = [0.01, 0.02, 0.05], # Relative to signal std dev
                 dropout_range: Tuple[float, float] = (0.05, 0.15),
                 global_scale_range: Tuple[float, float] = (0.95, 1.05), # For global signal scaling
                 baseline_shift_std_factor: float = 0.01 # Factor of signal mean for baseline shift std
                ):
        self.signals = torch.FloatTensor(signals)
        self.params = torch.FloatTensor(params)
        self.noise_levels = noise_levels
        self.dropout_range = dropout_range
        self.global_scale_range = global_scale_range
        self.baseline_shift_std_factor = baseline_shift_std_factor


    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx].clone()
        param = self.params[idx].clone()

        # Apply noise augmentation (relative to signal's std dev)
        if self.noise_levels and np.random.rand() < 0.5: 
            noise_level_factor = np.random.choice(self.noise_levels)
            signal_std = torch.std(signal)
            if signal_std > 1e-6: # Avoid division by zero or amplification of noise for zero signals
                effective_noise_std = noise_level_factor * signal_std
                noise = torch.randn_like(signal) * effective_noise_std
                signal += noise

        # Apply random signal dropout (zeroing out some PLD points)
        if self.dropout_range and np.random.rand() < 0.5: 
            dropout_prob = np.random.uniform(*self.dropout_range)
            mask = torch.rand_like(signal) > dropout_prob
            signal *= mask
        
        # Apply global signal scaling
        if self.global_scale_range and np.random.rand() < 0.5:
            scale_factor = np.random.uniform(*self.global_scale_range)
            signal *= scale_factor

        # Apply baseline shift
        if self.baseline_shift_std_factor > 0 and np.random.rand() < 0.5:
            signal_mean_abs = torch.mean(torch.abs(signal))
            if signal_mean_abs > 1e-6: # Avoid large shifts for near-zero signals
                shift_std = self.baseline_shift_std_factor * signal_mean_abs
                shift = torch.randn(1) * shift_std 
                signal += shift.item() # Add scalar shift to all elements

        return signal, param

class EnhancedASLTrainer:
    """Enhanced training manager with curriculum learning and ensemble support"""

    def __init__(self,
                 model_class, # This is a callable that returns a model instance
                 input_size: int, # Will be determined by plds (+1 if M0 used)
                 hidden_sizes: List[int] = [256, 128, 64], # Passed to model_class
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 n_ensembles: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 n_plds_for_model: Optional[int] = None, # Num PLDs for model (e.g. transformer seq_len)
                 m0_input_feature_model: bool = False # For model_class if it needs this flag
                ): 

        self.device = device
        self.batch_size = batch_size
        self.n_ensembles = n_ensembles
        self.n_plds_for_model = n_plds_for_model # Used by EnhancedASLNet if it has transformer
        self.m0_input_feature_model = m0_input_feature_model # For model instantiation

        self.hidden_sizes = hidden_sizes 
        self.input_size_model = input_size # Store input size for model factory

        # Initialize ensemble models using the factory function
        self.models = [
            model_class().to(device) # model_class() creates EnhancedASLNet with current config
            for _ in range(n_ensembles)
        ]
        self.best_states = [None] * self.n_ensembles # To store state_dict of best model per ensemble member

        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=learning_rate)
            for model in self.models
        ]
        self.schedulers = []  # Will be initialized after dataloader creation

        self.train_losses = defaultdict(list)
        self.val_losses = defaultdict(list)
        self.metrics_history = defaultdict(lambda: defaultdict(list))

    def prepare_curriculum_data(self,
                                simulator, # Instance of RealisticASLSimulator
                                n_training_subjects: int = 10000, 
                                val_split: float = 0.2,
                                plds: Optional[np.ndarray] = None,
                                curriculum_att_ranges_config: Optional[List[Tuple[float, float, str]]] = None,
                                training_conditions_config: Optional[List[str]] = None,
                                training_noise_levels_config: Optional[List[float]] = None,
                                n_epochs_for_scheduler: int = 200,
                                include_m0_in_data: bool = False # Flag to generate M0 with data
                                ) -> Tuple[List[DataLoader], Optional[DataLoader]]:
        """Prepare curriculum learning datasets using diverse data from RealisticASLSimulator."""
        if plds is None:
            plds = np.arange(500, 3001, 500)  # Default PLDs

        conditions = training_conditions_config if training_conditions_config is not None else ['healthy', 'stroke', 'tumor', 'elderly']
        noise_levels = training_noise_levels_config if training_noise_levels_config is not None else [3.0, 5.0, 10.0, 15.0]

        logger.info(f"Generating diverse training data with {n_training_subjects} base subjects, conditions: {conditions}, SNRs: {noise_levels}")

        # NOTE: `generate_diverse_dataset` currently returns signals (N, n_plds*2) and parameters (N,2)
        # If include_m0_in_data is True, this function (or RealisticASLSimulator) needs to be updated
        # to also return M0 values. For now, assuming M0 is not included in `raw_dataset` output directly.
        raw_dataset = simulator.generate_diverse_dataset(
            plds=plds,
            n_subjects=n_training_subjects,
            conditions=conditions,
            noise_levels=noise_levels
        )

        X_all_asl = raw_dataset['signals']  # Shape: (total_generated_samples, num_plds * 2)
        y_all = raw_dataset['parameters'] # Shape: (total_generated_samples, 2) for [CBF_ml/100g/min, ATT_ms]

        if include_m0_in_data:
            # Placeholder: M0 generation logic would go here.
            # For example, M0 could be a fixed value, or drawn from a distribution,
            # or related to the tissue type/condition.
            # For now, if include_m0_in_data is True but not generated, we might add a dummy M0 or raise error.
            # Let's assume a dummy M0 (e.g., scaled by CBF) for demonstration if M0 is needed.
            # This part needs proper implementation if M0 is a true feature.
            m0_dummy_values = np.random.normal(1.0, 0.1, size=(X_all_asl.shape[0], 1)) # Example: M0 around 1.0
            X_all = np.concatenate((X_all_asl, m0_dummy_values), axis=1)
            logger.info("Included dummy M0 feature in X_all.")
        else:
            X_all = X_all_asl

        logger.info(f"Total generated diverse samples for training/validation: {X_all.shape[0]}")

        if X_all.shape[0] == 0:
            raise ValueError("No data generated by simulator.generate_diverse_dataset. Check parameters.")

        n_total_samples = X_all.shape[0]
        n_val = int(n_total_samples * val_split)
        if n_val == 0 and n_total_samples > 1: n_val = 1 
        if n_val >= n_total_samples : n_val = n_total_samples - 1 if n_total_samples > 0 else 0

        indices = np.random.permutation(n_total_samples)
        train_idx, val_idx = indices[:-n_val], indices[-n_val:]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        logger.info(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        if X_train.shape[0] == 0:
            raise ValueError("Training set is empty after split.")

        if curriculum_att_ranges_config is None:
            min_att_sim, max_att_sim = simulator.physio_var.att_range
            curriculum_stages_def = [
                (min_att_sim, 1500.0), (1500.0, 2500.0), (2500.0, max_att_sim)
            ]
        else: 
            curriculum_stages_def = [(r[0], r[1]) for r in curriculum_att_ranges_config]

        train_loaders = []
        for i, (att_min_stage, att_max_stage) in enumerate(curriculum_stages_def):
            if i == len(curriculum_stages_def) - 1: 
                mask = (y_train[:, 1] >= att_min_stage) & (y_train[:, 1] <= att_max_stage)
            else:
                mask = (y_train[:, 1] >= att_min_stage) & (y_train[:, 1] < att_max_stage)

            stage_X, stage_y = X_train[mask], y_train[mask]

            if len(stage_X) == 0:
                logger.warning(f"Curriculum stage {i+1} (ATT {att_min_stage}-{att_max_stage}ms) has no samples. Skipping.")
                continue
            logger.info(f"Curriculum stage {i+1} (ATT {att_min_stage}-{att_max_stage}ms): {len(stage_X)} samples.")

            att_for_weights = np.clip(stage_y[:, 1], a_min=100.0, a_max=None)
            weights = np.exp(-att_for_weights / 2000.0) # Higher weight for shorter ATT
            
            sampler = None
            if np.sum(weights) > 1e-9 and np.all(np.isfinite(weights)) and len(weights) > 0:
                try:
                    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                except RuntimeError as e: # Can happen if weights sum to zero or are invalid
                    logger.warning(f"Failed to create WeightedRandomSampler for stage {i+1}: {e}. Using uniform sampling.")
            else:
                 logger.warning(f"Invalid weights (sum too small, NaNs, or empty) in curriculum stage {i+1}. Using uniform sampling.")

            # Use EnhancedASLDataset which applies augmentations
            dataset = EnhancedASLDataset(stage_X, stage_y) 
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, 
                                num_workers=num_workers if num_workers > 0 else 0, # Ensure num_workers >=0
                                pin_memory=True, drop_last=(len(stage_X) > self.batch_size))
            train_loaders.append(loader)

        if not train_loaders:
             logger.error("No training data loaders created. All curriculum stages might be empty.")

        val_loader = None
        if X_val.shape[0] > 0:
            val_dataset = EnhancedASLDataset(X_val, y_val) 
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                    num_workers=num_workers if num_workers > 0 else 0, 
                                    pin_memory=True, drop_last=False)
        else:
            logger.warning("Validation set is empty. Validation loader will not be created.")

        self.schedulers = [] 
        if train_loaders:
            total_steps_per_epoch = sum(len(loader) for loader in train_loaders)
            if total_steps_per_epoch > 0 :
                total_steps = total_steps_per_epoch * n_epochs_for_scheduler
                for opt_idx, opt in enumerate(self.optimizers):
                    current_lr = opt.param_groups[0]['lr'] if opt.param_groups else self.learning_rate # Fallback
                    scheduler = OneCycleLR(opt, max_lr=current_lr, total_steps=total_steps)
                    self.schedulers.append(scheduler)
            else:
                logger.warning("Total steps per epoch is 0. Schedulers not configured with OneCycleLR.")
        else:
            logger.warning("No training loaders available. Schedulers not initialized.")
            
        return train_loaders, val_loader


    def train_ensemble(self,
                   train_loaders: List[DataLoader],
                   val_loader: Optional[DataLoader], 
                   n_epochs: int = 200,
                   early_stopping_patience: int = 20) -> Dict[str, any]: # Return type changed
        """Train ensemble models with curriculum learning"""

        best_val_losses = [float('inf')] * self.n_ensembles
        patience_counters = [0] * self.n_ensembles
        # self.best_states already initialized in __init__

        histories = defaultdict(lambda: defaultdict(list)) 

        if not train_loaders:
            logger.error("train_loaders is empty. Aborting training.")
            return {'final_mean_train_loss': float('nan'), 'final_mean_val_loss': float('nan'), 'all_histories': histories}


        for stage, train_loader in enumerate(train_loaders):
            logger.info(f"\nStarting curriculum stage {stage + 1}/{len(train_loaders)} with {len(train_loader)} batches.")
            if len(train_loader) == 0:
                logger.warning(f"Skipping empty curriculum stage {stage + 1}.")
                continue

            for epoch in range(n_epochs):
                active_models_in_stage = 0
                for model_idx in range(self.n_ensembles):
                    # If a model has early stopped (its best_state is saved and patience ran out),
                    # it might skip further training in this stage or subsequent stages.
                    # This logic depends on whether patience resets per stage or is global.
                    # Current: Global patience. If it met patience in a prev stage, it might not train more.
                    if patience_counters[model_idx] >= early_stopping_patience and self.best_states[model_idx] is not None:
                        continue 
                    active_models_in_stage +=1

                    model = self.models[model_idx]
                    optimizer = self.optimizers[model_idx]
                    
                    train_loss = self._train_epoch(model, train_loader, optimizer,
                                                 self.schedulers[model_idx] if self.schedulers and len(self.schedulers) > model_idx else None,
                                                 epoch, stage, n_epochs) 
                    histories[model_idx]['train_losses'].append(train_loss)

                    if val_loader:
                        val_loss = self._validate(model, val_loader, epoch, stage, n_epochs)
                        histories[model_idx]['val_losses'].append(val_loss)

                        if val_loss < best_val_losses[model_idx]:
                            best_val_losses[model_idx] = val_loss
                            patience_counters[model_idx] = 0
                            self.best_states[model_idx] = model.state_dict() # Save best state
                        else:
                            patience_counters[model_idx] += 1
                    else: 
                        histories[model_idx]['val_losses'].append(float('inf'))


                if active_models_in_stage == 0 and epoch > 0: 
                    logger.info(f"All active models early stopped at stage {stage+1}, epoch {epoch+1}. Moving to next stage or finishing.")
                    break 

                if (epoch + 1) % 10 == 0:
                    current_train_losses = [h['train_losses'][-1] for h_idx, h in histories.items() if h['train_losses'] and (patience_counters[h_idx] < early_stopping_patience or self.best_states[h_idx] is None)]
                    current_val_losses = [h['val_losses'][-1] for h_idx, h in histories.items() if h['val_losses'] and h['val_losses'][-1] != float('inf') and (patience_counters[h_idx] < early_stopping_patience or self.best_states[h_idx] is None)]
                    
                    mean_train_loss_epoch = np.nanmean(current_train_losses) if current_train_losses else float('nan')
                    mean_val_loss_epoch = np.nanmean(current_val_losses) if current_val_losses else float('nan')
                    logger.info(f"Stage {stage+1}, Epoch {epoch + 1}: Mean Active Train Loss = {mean_train_loss_epoch:.6f}, Mean Active Val Loss = {mean_val_loss_epoch:.6f}")
            
            # Optional: reset patience for each new stage if desired
            # patience_counters = [0] * self.n_ensembles # If reset, models train fully on each stage.

        # Restore best states found across all epochs and stages for each model
        for model_idx, state in enumerate(self.best_states):
            if state is not None:
                self.models[model_idx].load_state_dict(state)
                logger.info(f"Loaded best state for model {model_idx} (Val Loss: {best_val_losses[model_idx]:.4f})")
            else: # This can happen if no validation was done, or model never improved.
                logger.warning(f"No best state found for model {model_idx} (Val Loss: {best_val_losses[model_idx]:.4f}). Using final state.")


        final_train_losses = [np.nanmean(histories[i]['train_losses']) for i in range(self.n_ensembles) if histories[i]['train_losses']]
        final_val_losses = [best_val_losses[i] for i in range(self.n_ensembles) if best_val_losses[i] != float('inf')]

        return {
            'final_mean_train_loss': np.nanmean(final_train_losses) if final_train_losses else float('nan'),
            'final_mean_val_loss': np.nanmean(final_val_losses) if final_val_losses else float('nan'),
            'all_histories': histories 
        }


    def _train_epoch(self,
                    model: torch.nn.Module,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], 
                    epoch: int, 
                    stage: int, 
                    n_epochs_per_stage: int) -> float: 
        model.train()
        total_loss = 0.0
        global_epoch_for_loss = stage * n_epochs_per_stage + epoch

        for signals, params in train_loader:
            signals = signals.to(self.device)
            params = params.to(self.device) 

            optimizer.zero_grad()
            # Model expects (batch, features); if M0 is used, it should be part of `signals`
            cbf_mean, att_mean, cbf_log_var, att_log_var = model(signals)

            loss = CustomLoss()( 
                cbf_mean, att_mean,
                params[:, 0:1], params[:, 1:2], 
                cbf_log_var, att_log_var,
                global_epoch_for_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            if scheduler: 
                scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

    def _validate(self,
                 model: torch.nn.Module,
                 val_loader: DataLoader,
                 epoch: int, 
                 stage: int, 
                 n_epochs_per_stage: int) -> float: 
        model.eval()
        total_loss = 0.0
        global_epoch_for_loss = stage * n_epochs_per_stage + epoch

        with torch.no_grad():
            for signals, params in val_loader:
                signals = signals.to(self.device)
                params = params.to(self.device)

                cbf_mean, att_mean, cbf_log_var, att_log_var = model(signals)

                loss = CustomLoss()(
                    cbf_mean, att_mean,
                    params[:, 0:1], params[:, 1:2],
                    cbf_log_var, att_log_var,
                    global_epoch_for_loss 
                )
                total_loss += loss.item()

        return total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')


    def predict(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using the ensemble.
        `signals` is a numpy array of shape (N_samples, N_features).
        If M0 is used by the model, N_features should include M0.
        """
        signals_tensor = torch.FloatTensor(signals).to(self.device)
        if signals_tensor.ndim == 1: 
            signals_tensor = signals_tensor.unsqueeze(0)

        all_cbf_means, all_att_means = [], []
        all_cbf_aleatoric_vars, all_att_aleatoric_vars = [], []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                cbf_mean, att_mean, cbf_log_var, att_log_var = model(signals_tensor)
                all_cbf_means.append(cbf_mean.cpu().numpy()) 
                all_att_means.append(att_mean.cpu().numpy()) 
                all_cbf_aleatoric_vars.append(torch.exp(cbf_log_var).cpu().numpy()) 
                all_att_aleatoric_vars.append(torch.exp(att_log_var).cpu().numpy())

        all_cbf_means_np = np.concatenate(all_cbf_means, axis=1) # (N_samples, N_ensemble)
        all_att_means_np = np.concatenate(all_att_means, axis=1) # (N_samples, N_ensemble)
        all_cbf_aleatoric_vars_np = np.concatenate(all_cbf_aleatoric_vars, axis=1)
        all_att_aleatoric_vars_np = np.concatenate(all_att_aleatoric_vars, axis=1)

        ensemble_cbf_mean = np.mean(all_cbf_means_np, axis=1) 
        ensemble_att_mean = np.mean(all_att_means_np, axis=1) 

        mean_aleatoric_cbf_var = np.mean(all_cbf_aleatoric_vars_np, axis=1)
        mean_aleatoric_att_var = np.mean(all_att_aleatoric_vars_np, axis=1)

        epistemic_cbf_var = np.var(all_cbf_means_np, axis=1)
        epistemic_att_var = np.var(all_att_means_np, axis=1)

        total_cbf_var = mean_aleatoric_cbf_var + epistemic_cbf_var
        total_att_var = mean_aleatoric_att_var + epistemic_att_var

        total_cbf_std = np.sqrt(np.maximum(total_cbf_var, 0)) # Ensure non-negative before sqrt
        total_att_std = np.sqrt(np.maximum(total_att_var, 0))

        return ensemble_cbf_mean, ensemble_att_mean, total_cbf_std, total_att_std

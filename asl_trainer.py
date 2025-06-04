import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from torch.optim.lr_scheduler import OneCycleLR # Import was missing for EnhancedASLTrainer
import math # Import was missing
import multiprocessing as mp
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
        signals_dict = simulator.generate_synthetic_data(plds, att_values, n_noise=n_samples, tsnr=5.0)


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

        return total_loss / len(val_loader)

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

        # Load best model
        if Path('best_model_ASLTrainer.pt').exists():
            self.model.load_state_dict(torch.load('best_model_ASLTrainer.pt'))

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Predict CBF and ATT from ASL signals"""
        self.model.eval()
        with torch.no_grad():
            signals_tensor = torch.FloatTensor(signals).to(self.device)
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
                 signals: np.ndarray,
                 params: np.ndarray,
                 noise_levels: List[float] = [0.01, 0.02, 0.05], # Relative to signal max, or absolute? Needs clarification. Assuming absolute for now.
                 dropout_range: Tuple[float, float] = (0.05, 0.15)):
        self.signals = torch.FloatTensor(signals)
        self.params = torch.FloatTensor(params)
        self.noise_levels = noise_levels
        self.dropout_range = dropout_range

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx].clone()
        param = self.params[idx].clone()

        # Apply noise augmentation
        if self.noise_levels and np.random.rand() < 0.5: # Apply with some probability
            noise_level_val = np.random.choice(self.noise_levels)
            # Scale noise_level_val by signal magnitude if it's meant to be relative
            # For now, assuming absolute noise values if they are small (e.g. 0.01)
            # If signals are e.g. ~0.005, noise of 0.01 is very high.
            # Let's assume noise_levels are relative to current signal's std for more robustness
            effective_noise_level = noise_level_val * torch.std(signal) if torch.std(signal) > 1e-6 else noise_level_val

            noise = torch.randn_like(signal) * effective_noise_level
            signal += noise

        # Apply random signal dropout
        if self.dropout_range and np.random.rand() < 0.5: # Apply with some probability
            dropout_prob = np.random.uniform(*self.dropout_range)
            mask = torch.rand_like(signal) > dropout_prob
            signal *= mask

        return signal, param

class EnhancedASLTrainer:
    """Enhanced training manager with curriculum learning and ensemble support"""

    def __init__(self,
                 model_class, # This is a callable that returns a model instance
                 input_size: int, # Will be determined by plds
                 hidden_sizes: List[int] = [256, 128, 64], # Passed to model_class
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 n_ensembles: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 n_plds_for_model: Optional[int] = None): # Add n_plds for model init

        self.device = device
        self.batch_size = batch_size
        self.n_ensembles = n_ensembles
        self.n_plds_for_model = n_plds_for_model
        self.hidden_sizes = hidden_sizes # Store for model creation

        # Initialize ensemble models using the factory function
        self.models = [
            model_class().to(device) # model_class should be a factory, e.g., lambda: EnhancedASLNet(...)
            for _ in range(n_ensembles)
        ]

        # Initialize optimizers with OneCycleLR scheduler
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=learning_rate)
            for model in self.models
        ]

        self.schedulers = []  # Will be initialized after dataloader creation

        # Track training progress
        self.train_losses = defaultdict(list)
        self.val_losses = defaultdict(list)
        self.metrics_history = defaultdict(lambda: defaultdict(list))

    def prepare_curriculum_data(self,
                                simulator, # Instance of RealisticASLSimulator
                                n_training_subjects: int = 10000, # Renamed from n_samples for clarity
                                val_split: float = 0.2,
                                plds: Optional[np.ndarray] = None,
                                curriculum_att_ranges_config: Optional[List[Tuple[float, float, float]]] = None,
                                training_conditions_config: Optional[List[str]] = None,
                                training_noise_levels_config: Optional[List[float]] = None,
                                n_epochs_for_scheduler: int = 200
                                ) -> Tuple[List[DataLoader], Optional[DataLoader]]:
        """Prepare curriculum learning datasets using diverse data from RealisticASLSimulator."""
        if plds is None:
            plds = np.arange(500, 3001, 500)  # Default PLDs

        # Use config or defaults for diverse data generation
        conditions = training_conditions_config if training_conditions_config is not None else ['healthy', 'stroke', 'tumor', 'elderly']
        noise_levels = training_noise_levels_config if training_noise_levels_config is not None else [3.0, 5.0, 10.0, 15.0]

        logger.info(f"Generating diverse training data with {n_training_subjects} base subjects, conditions: {conditions}, SNRs: {noise_levels}")

        raw_dataset = simulator.generate_diverse_dataset(
            plds=plds,
            n_subjects=n_training_subjects,
            conditions=conditions,
            noise_levels=noise_levels
        )

        X_all = raw_dataset['signals']  # Shape: (total_generated_samples, num_plds * 2)
        y_all = raw_dataset['parameters'] # Shape: (total_generated_samples, 2) for [CBF_ml/100g/min, ATT_ms]

        logger.info(f"Total generated diverse samples for training/validation: {X_all.shape[0]}")

        if X_all.shape[0] == 0:
            raise ValueError("No data generated by simulator.generate_diverse_dataset. Check parameters.")

        # Split into train/val
        n_total_samples = X_all.shape[0]
        n_val = int(n_total_samples * val_split)
        
        if n_val == 0 and n_total_samples > 1: n_val = 1 # Ensure at least one val sample if possible
        if n_val >= n_total_samples : n_val = n_total_samples - 1 if n_total_samples > 0 else 0


        indices = np.random.permutation(n_total_samples)
        train_idx, val_idx = indices[:-n_val], indices[-n_val:]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        logger.info(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        if X_train.shape[0] == 0:
            raise ValueError("Training set is empty after split.")


        # Create curriculum stages based on ATT
        # Default ATT ranges for curriculum stages if not provided
        if curriculum_att_ranges_config is None:
             # Use ATT range from simulator if available, or default
            min_att_sim, max_att_sim = simulator.physio_var.att_range
            curriculum_stages_def = [
                (min_att_sim, 1500.0),
                (1500.0, 2500.0),
                (2500.0, max_att_sim)
            ]
        else: # expect format [(min_att1, max_att1), (min_att2, max_att2), ...]
            curriculum_stages_def = [(r[0], r[1]) for r in curriculum_att_ranges_config]


        train_loaders = []
        for i, (att_min_stage, att_max_stage) in enumerate(curriculum_stages_def):
            if i == len(curriculum_stages_def) - 1: # Last stage, include upper bound
                mask = (y_train[:, 1] >= att_min_stage) & (y_train[:, 1] <= att_max_stage)
            else:
                mask = (y_train[:, 1] >= att_min_stage) & (y_train[:, 1] < att_max_stage)

            stage_X = X_train[mask]
            stage_y = y_train[mask]

            if len(stage_X) == 0:
                logger.warning(f"Curriculum stage {i+1} (ATT {att_min_stage}-{att_max_stage}ms) has no samples. Skipping.")
                continue
            logger.info(f"Curriculum stage {i+1} (ATT {att_min_stage}-{att_max_stage}ms): {len(stage_X)} samples.")

            att_for_weights = np.clip(stage_y[:, 1], a_min=100.0, a_max=None)
            weights = np.exp(-att_for_weights / 2000.0)
            
            sampler = None
            if np.sum(weights) > 1e-9 and np.all(np.isfinite(weights)):
                try:
                    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                except Exception as e:
                    logger.warning(f"Failed to create WeightedRandomSampler for stage {i+1}: {e}. Using uniform sampling.")
            else:
                 logger.warning(f"Invalid weights in curriculum stage {i+1}. Using uniform sampling.")


            dataset = EnhancedASLDataset(stage_X, stage_y)
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last= (len(stage_X) > self.batch_size) ) # drop_last if more than one batch
            train_loaders.append(loader)

        if not train_loaders:
             logger.error("No training data loaders created. All curriculum stages might be empty.")
             # Depending on strictness, could raise ValueError or return empty list
             # return [], None # Or handle as error in main.py

        # Create validation loader
        val_loader = None
        if X_val.shape[0] > 0:
            val_dataset = EnhancedASLDataset(X_val, y_val) # Use EnhancedASLDataset for val too for consistency in input
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False) # No need to drop last for validation
        else:
            logger.warning("Validation set is empty. Validation loader will not be created.")


        # Initialize schedulers
        self.schedulers = [] # Reset schedulers
        if train_loaders:
            total_steps_per_epoch = sum(len(loader) for loader in train_loaders)
            if total_steps_per_epoch > 0 :
                total_steps = total_steps_per_epoch * n_epochs_for_scheduler
                for opt_idx, opt in enumerate(self.optimizers):
                    # Ensure model learning rate is used if optimizer's default LR is not set by Adam.
                    # Adam's lr parameter sets opt.defaults['lr'].
                    current_lr = opt.param_groups[0]['lr'] if opt.param_groups else self.models[opt_idx].learning_rate_placeholder # Assuming model has such a placeholder
                    scheduler = OneCycleLR(opt, max_lr=current_lr, total_steps=total_steps)
                    self.schedulers.append(scheduler)
            else:
                logger.warning("Total steps per epoch is 0. Schedulers not configured with OneCycleLR.")
        else:
            logger.warning("No training loaders available. Schedulers not initialized.")
            
        return train_loaders, val_loader


    def train_ensemble(self,
                   train_loaders: List[DataLoader],
                   val_loader: Optional[DataLoader], # Can be None
                   n_epochs: int = 200,
                   early_stopping_patience: int = 20) -> Dict[str, List[float]]: # Return type changed
        """Train ensemble models with curriculum learning"""

        best_val_losses = [float('inf')] * self.n_ensembles
        patience_counters = [0] * self.n_ensembles
        best_states = [None] * self.n_ensembles # To store state_dict of best model per ensemble member

        histories = defaultdict(lambda: defaultdict(list)) # Store train/val loss per model

        if not train_loaders:
            logger.error("train_loaders is empty. Aborting training.")
            return histories # Return empty histories

        for stage, train_loader in enumerate(train_loaders):
            logger.info(f"\nStarting curriculum stage {stage + 1}/{len(train_loaders)} with {len(train_loader)} batches.")
            if len(train_loader) == 0:
                logger.warning(f"Skipping empty curriculum stage {stage + 1}.")
                continue

            for epoch in range(n_epochs):
                active_models_in_stage = 0
                for model_idx in range(self.n_ensembles):
                    if patience_counters[model_idx] >= early_stopping_patience and best_states[model_idx] is not None:
                        # This model has already early-stopped in a previous stage or epoch
                        # Or, if we want to reset patience per stage, remove `and best_states[model_idx] is not None`
                        continue
                    active_models_in_stage +=1

                    model = self.models[model_idx]
                    optimizer = self.optimizers[model_idx]
                    # Scheduler step is per batch, not per epoch for OneCycleLR
                    # scheduler = self.schedulers[model_idx] # Access scheduler if it exists

                    train_loss = self._train_epoch(model, train_loader, optimizer,
                                                 self.schedulers[model_idx] if self.schedulers and len(self.schedulers) > model_idx else None,
                                                 epoch, stage, n_epochs) # Pass n_epochs for CustomLoss epoch calculation
                    histories[model_idx]['train_losses'].append(train_loss)

                    if val_loader:
                        val_loss = self._validate(model, val_loader, epoch, stage, n_epochs) # Pass n_epochs for CustomLoss
                        histories[model_idx]['val_losses'].append(val_loss)

                        if val_loss < best_val_losses[model_idx]:
                            best_val_losses[model_idx] = val_loss
                            patience_counters[model_idx] = 0
                            best_states[model_idx] = model.state_dict() # Save best state
                            # torch.save(model.state_dict(), f'best_model_ensemble_{model_idx}_stage_{stage}.pt') # Optional: save per stage
                        else:
                            patience_counters[model_idx] += 1
                    else: # No validation loader
                        histories[model_idx]['val_losses'].append(float('inf')) # Or some other indicator


                if active_models_in_stage == 0 and epoch > 0: # All models early stopped
                    logger.info(f"All models early stopped at stage {stage+1}, epoch {epoch+1}. Moving to next stage or finishing.")
                    break # Break epoch loop for this stage

                if (epoch + 1) % 10 == 0:
                    mean_train_loss_epoch = np.nanmean([h['train_losses'][-1] for h_idx, h in histories.items() if h['train_losses']])
                    mean_val_loss_epoch = np.nanmean([h['val_losses'][-1] for h_idx, h in histories.items() if h['val_losses'] and h['val_losses'][-1] != float('inf')])
                    logger.info(f"Stage {stage+1}, Epoch {epoch + 1}: Mean Train Loss = {mean_train_loss_epoch:.6f}, Mean Val Loss = {mean_val_loss_epoch:.6f}")
            
            # Optional: reset patience for each new stage if desired
            # patience_counters = [0] * self.n_ensembles

        # Restore best states found across all epochs and stages for each model
        for model_idx, state in enumerate(best_states):
            if state is not None:
                self.models[model_idx].load_state_dict(state)
                logger.info(f"Loaded best state for model {model_idx} (Val Loss: {best_val_losses[model_idx]:.4f})")
            else:
                logger.warning(f"No best state found for model {model_idx} (possibly no validation or training).")


        # For returning, maybe average losses or return all histories
        final_train_losses = [np.nanmean(histories[i]['train_losses']) for i in range(self.n_ensembles) if histories[i]['train_losses']]
        final_val_losses = [np.nanmean(histories[i]['val_losses']) for i in range(self.n_ensembles) if histories[i]['val_losses'] and histories[i]['val_losses'][-1] != float('inf')]

        return {
            'final_mean_train_loss': np.nanmean(final_train_losses) if final_train_losses else float('nan'),
            'final_mean_val_loss': np.nanmean(final_val_losses) if final_val_losses else float('nan'),
            'all_histories': histories # For detailed analysis
        }


    def _train_epoch(self,
                    model: torch.nn.Module,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # Scheduler can be None
                    epoch: int, # Current epoch within a stage
                    stage: int, # Current curriculum stage
                    n_epochs_per_stage: int) -> float: # Total epochs for this stage
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        
        # Calculate global epoch for CustomLoss or other epoch-dependent logic
        # This assumes each stage runs for n_epochs_per_stage
        # If curriculum stages have different epoch counts, this needs adjustment.
        # For OneCycleLR, scheduler.step() is per batch.
        global_epoch_for_loss = stage * n_epochs_per_stage + epoch


        for signals, params in train_loader:
            signals = signals.to(self.device)
            params = params.to(self.device) # params[:,0] is CBF, params[:,1] is ATT

            optimizer.zero_grad()

            cbf_mean, att_mean, cbf_log_var, att_log_var = model(signals)

            loss = CustomLoss()( # Assuming CustomLoss handles epoch-dependent weighting
                cbf_mean, att_mean,
                params[:, 0:1], params[:, 1:2], # Ensure targets are (batch, 1)
                cbf_log_var, att_log_var,
                global_epoch_for_loss
            )

            loss.backward()
            optimizer.step()
            if scheduler: # Scheduler steps per batch for OneCycleLR
                scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

    def _validate(self,
                 model: torch.nn.Module,
                 val_loader: DataLoader,
                 epoch: int, # Current epoch
                 stage: int, # Current stage
                 n_epochs_per_stage: int) -> float: # Total epochs for this stage
        """Validate the model"""
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
                    global_epoch_for_loss # Use consistent epoch for loss calculation
                )
                total_loss += loss.item()

        return total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')


    def predict(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using the ensemble

        Returns
        -------
        Tuple containing:
            - Mean CBF predictions (N,)
            - Mean ATT predictions (N,)
            - CBF uncertainties (std dev) (N,)
            - ATT uncertainties (std dev) (N,)
        """
        signals_tensor = torch.FloatTensor(signals).to(self.device)
        if signals_tensor.ndim == 1: # Single sample
            signals_tensor = signals_tensor.unsqueeze(0)


        all_cbf_means = []
        all_att_means = []
        all_cbf_aleatoric_vars = [] # Aleatoric variance from each model
        all_att_aleatoric_vars = []


        for model in self.models:
            model.eval()
            with torch.no_grad():
                cbf_mean, att_mean, cbf_log_var, att_log_var = model(signals_tensor)
                all_cbf_means.append(cbf_mean.cpu().numpy()) # Shape (N, 1)
                all_att_means.append(att_mean.cpu().numpy()) # Shape (N, 1)
                all_cbf_aleatoric_vars.append(torch.exp(cbf_log_var).cpu().numpy()) # Shape (N, 1)
                all_att_aleatoric_vars.append(torch.exp(att_log_var).cpu().numpy()) # Shape (N, 1)

        # Stack along a new dimension (models) -> (num_models, N, 1)
        all_cbf_means_np = np.stack(all_cbf_means, axis=0)
        all_att_means_np = np.stack(all_att_means, axis=0)
        all_cbf_aleatoric_vars_np = np.stack(all_cbf_aleatoric_vars, axis=0)
        all_att_aleatoric_vars_np = np.stack(all_att_aleatoric_vars, axis=0)

        # Ensemble mean prediction (average over models) -> (N, 1)
        ensemble_cbf_mean = np.mean(all_cbf_means_np, axis=0).squeeze(-1) # -> (N,)
        ensemble_att_mean = np.mean(all_att_means_np, axis=0).squeeze(-1) # -> (N,)

        # Aleatoric uncertainty (average of individual model variances) -> (N, 1)
        mean_aleatoric_cbf_var = np.mean(all_cbf_aleatoric_vars_np, axis=0).squeeze(-1)
        mean_aleatoric_att_var = np.mean(all_att_aleatoric_vars_np, axis=0).squeeze(-1)

        # Epistemic uncertainty (variance of ensemble model means) -> (N, 1)
        epistemic_cbf_var = np.var(all_cbf_means_np, axis=0).squeeze(-1)
        epistemic_att_var = np.var(all_att_means_np, axis=0).squeeze(-1)

        # Total variance = mean aleatoric + epistemic -> (N, 1)
        total_cbf_var = mean_aleatoric_cbf_var + epistemic_cbf_var
        total_att_var = mean_aleatoric_att_var + epistemic_att_var

        # Return standard deviations
        total_cbf_std = np.sqrt(total_cbf_var)
        total_att_std = np.sqrt(total_att_var)

        return ensemble_cbf_mean, ensemble_att_mean, total_cbf_std, total_att_std
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

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
                    simulator,
                    n_samples: int = 10000,
                    val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        # Generate synthetic data
        plds = np.arange(500, 3001, 500)  # 6 PLDs
        att_values = np.arange(0, 4001, 100)
        signals = simulator.generate_synthetic_data(plds, att_values, n_samples)
        
        # Correctly reshape the signals
        X = np.zeros((n_samples * len(att_values), len(plds) * 2))
        for i in range(n_samples):
            pcasl = signals['PCASL'][i].flatten()  # Shape: (num_att * num_plds,)
            vsasl = signals['VSASL'][i].flatten()  # Shape: (num_att * num_plds,)
            start_idx = i * len(att_values)
            end_idx = (i + 1) * len(att_values)
            X[start_idx:end_idx, :len(plds)] = pcasl.reshape(-1, len(plds))
            X[start_idx:end_idx, len(plds):] = vsasl.reshape(-1, len(plds))
        
        # Generate corresponding parameters
        cbf = np.full(n_samples * len(att_values), simulator.params.CBF)
        att = np.tile(att_values, n_samples)
        y = np.column_stack((cbf, att))
        
        # Split data
        n_val = int(len(X) * val_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        # Create data loaders
        train_dataset = ASLDataset(X_train, y_train)
        val_dataset = ASLDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
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
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
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
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pt'))
        
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
        
        # Relative errors
        rel_error_cbf = np.mean(np.abs(predictions[:,0] - true_params[:,0]) / true_params[:,0])
        rel_error_att = np.mean(np.abs(predictions[:,1] - true_params[:,1]) / true_params[:,1])
        
        return {
            'MAE_CBF': mae_cbf,
            'MAE_ATT': mae_att,
            'RMSE_CBF': rmse_cbf,
            'RMSE_ATT': rmse_att,
            'RelError_CBF': rel_error_cbf,
            'RelError_ATT': rel_error_att
        }
    
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
from sklearn.model_selection import KFold
from collections import defaultdict

class EnhancedASLDataset(Dataset):
    """Enhanced dataset with noise augmentation and weighted sampling"""
    
    def __init__(self, 
                 signals: np.ndarray,
                 params: np.ndarray,
                 noise_levels: List[float] = [0.01, 0.02, 0.05],
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
        if self.noise_levels:
            noise_level = np.random.choice(self.noise_levels)
            noise = torch.randn_like(signal) * noise_level
            signal += noise
        
        # Apply random signal dropout
        if self.dropout_range:
            dropout_prob = np.random.uniform(*self.dropout_range)
            mask = torch.rand_like(signal) > dropout_prob
            signal *= mask
        
        return signal, param

class EnhancedASLTrainer:
    """Enhanced training manager with curriculum learning and ensemble support"""
    
    def __init__(self,
                 model_class,
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 n_ensembles: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.batch_size = batch_size
        self.n_ensembles = n_ensembles
        
        # Initialize ensemble models
        self.models = [
            model_class(input_size, hidden_sizes).to(device)
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
                              simulator,
                              n_samples: int = 20000,
                              val_split: float = 0.2) -> Tuple[List[DataLoader], DataLoader]:
        """Prepare curriculum learning datasets"""
        
        # Generate more samples for problematic ATT ranges
        att_ranges = [
            (500, 1500, 0.4),   # 40% samples
            (1500, 2500, 0.3),  # 30% samples
            (2500, 4000, 0.3)   # 30% samples
        ]
        
        datasets = []
        for att_min, att_max, proportion in att_ranges:
            n_range_samples = int(n_samples * proportion)
            att_values = np.random.uniform(att_min, att_max, n_range_samples)
            data = simulator.generate_synthetic_data(att_values, n_noise=50)
            datasets.append((data, att_values))
        
        # Combine datasets with proper weighting
        X = np.concatenate([d[0] for d in datasets])
        y = np.concatenate([np.column_stack((np.full_like(att, simulator.params.CBF), att))
                          for d, att in datasets])
        
        # Split into train/val
        n_val = int(len(X) * val_split)
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:-n_val], indices[-n_val:]
        
        # Create curriculum stages
        n_stages = 3
        stage_ranges = np.linspace(500, 4000, n_stages + 1)
        train_loaders = []
        
        for i in range(n_stages):
            mask = (y[train_idx, 1] >= stage_ranges[i]) & (y[train_idx, 1] < stage_ranges[i+1])
            stage_X = X[train_idx][mask]
            stage_y = y[train_idx][mask]
            
            # Create weighted sampler (higher weights for shorter ATT)
            weights = np.exp(-stage_y[:, 1] / 2000)
            sampler = WeightedRandomSampler(weights, len(weights))
            
            dataset = EnhancedASLDataset(stage_X, stage_y)
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
            train_loaders.append(loader)
        
        # Create validation loader
        val_dataset = EnhancedASLDataset(X[val_idx], y[val_idx])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Initialize schedulers
        total_steps = sum(len(loader) for loader in train_loaders) * 200  # 200 epochs
        for opt in self.optimizers:
            scheduler = OneCycleLR(opt, max_lr=0.001, total_steps=total_steps)
            self.schedulers.append(scheduler)
        
        return train_loaders, val_loader
    
    def train_ensemble(self,
                   train_loaders: List[DataLoader],
                   val_loader: DataLoader,
                   n_epochs: int = 200,
                   early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """Train ensemble models with curriculum learning"""
        
        best_val_losses = [float('inf')] * self.n_ensembles
        patience_counters = [0] * self.n_ensembles
        best_states = [None] * self.n_ensembles
        
        # Training history for each model
        histories = defaultdict(lambda: defaultdict(list))
        
        # Curriculum stages
        for stage, train_loader in enumerate(train_loaders):
            print(f"\nStarting curriculum stage {stage + 1}/{len(train_loaders)}")
            
            for epoch in range(n_epochs):
                # Train each model in ensemble
                for model_idx in range(self.n_ensembles):
                    if patience_counters[model_idx] >= early_stopping_patience:
                        continue
                        
                    model = self.models[model_idx]
                    optimizer = self.optimizers[model_idx]
                    scheduler = self.schedulers[model_idx]
                    
                    # Training
                    train_loss = self._train_epoch(model, train_loader, optimizer, 
                                                 scheduler, epoch, stage)
                    histories[model_idx]['train_losses'].append(train_loss)
                    
                    # Validation
                    val_loss = self._validate(model, val_loader, epoch)
                    histories[model_idx]['val_losses'].append(val_loss)
                    
                    # Early stopping check
                    if val_loss < best_val_losses[model_idx]:
                        best_val_losses[model_idx] = val_loss
                        patience_counters[model_idx] = 0
                        best_states[model_idx] = model.state_dict()
                    else:
                        patience_counters[model_idx] += 1
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    mean_train = np.mean([h['train_losses'][-1] for h in histories.values()])
                    mean_val = np.mean([h['val_losses'][-1] for h in histories.values()])
                    print(f"Epoch {epoch + 1}: Mean Train Loss = {mean_train:.6f}, "
                          f"Mean Val Loss = {mean_val:.6f}")
            
        # Restore best states
        for model_idx, state in enumerate(best_states):
            if state is not None:
                self.models[model_idx].load_state_dict(state)
        
        return histories
    
    def _train_epoch(self, 
                    model: torch.nn.Module,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    epoch: int,
                    stage: int) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        
        for signals, params in train_loader:
            signals = signals.to(self.device)
            params = params.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with uncertainty
            cbf_mean, att_mean, cbf_log_var, att_log_var = model(signals)
            
            # Compute loss with uncertainty
            loss = CustomLoss()(
                cbf_mean, att_mean,
                params[:, 0:1], params[:, 1:2],
                cbf_log_var, att_log_var,
                epoch + stage * 200
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def _validate(self, 
                 model: torch.nn.Module,
                 val_loader: DataLoader,
                 epoch: int) -> float:
        """Validate the model"""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for signals, params in val_loader:
                signals = signals.to(self.device)
                params = params.to(self.device)
                
                # Forward pass with uncertainty
                cbf_mean, att_mean, cbf_log_var, att_log_var = model(signals)
                
                # Compute loss
                loss = CustomLoss()(
                    cbf_mean, att_mean,
                    params[:, 0:1], params[:, 1:2],
                    cbf_log_var, att_log_var,
                    epoch
                )
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def predict(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using the ensemble
        
        Returns
        -------
        Tuple containing:
            - Mean CBF predictions
            - Mean ATT predictions
            - CBF uncertainties
            - ATT uncertainties
        """
        signals_tensor = torch.FloatTensor(signals).to(self.device)
        
        # Initialize storage for predictions
        cbf_preds = []
        att_preds = []
        cbf_vars = []
        att_vars = []
        
        # Get predictions from each model
        for model in self.models:
            model.eval()
            with torch.no_grad():
                cbf_mean, att_mean, cbf_log_var, att_log_var = model(signals_tensor)
                cbf_preds.append(cbf_mean.cpu().numpy())
                att_preds.append(att_mean.cpu().numpy())
                cbf_vars.append(np.exp(cbf_log_var.cpu().numpy()))
                att_vars.append(np.exp(att_log_var.cpu().numpy()))
        
        # Combine predictions
        cbf_mean = np.mean(cbf_preds, axis=0)
        att_mean = np.mean(att_preds, axis=0)
        
        # Total uncertainty is aleatoric (mean of individual variances) 
        # plus epistemic (variance of predictions)
        cbf_uncertainty = (np.mean(cbf_vars, axis=0) + 
                         np.var(cbf_preds, axis=0))
        att_uncertainty = (np.mean(att_vars, axis=0) + 
                         np.var(att_preds, axis=0))
        
        return cbf_mean, att_mean, cbf_uncertainty, att_uncertainty
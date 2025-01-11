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
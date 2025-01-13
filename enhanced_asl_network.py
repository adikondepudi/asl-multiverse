import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.layers(x))

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for PLD dependencies"""
    def __init__(self, n_plds: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Project features first
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output single attention weight per timestep
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, channels_per_pld, n_plds)
        batch_size, channels, n_plds = x.size()
        
        # Transpose for attention calculation: (batch_size, n_plds, channels_per_pld)
        x_trans = x.transpose(1, 2)
        
        # Calculate attention scores: (batch_size, n_plds, 1)
        attention_scores = self.attention(x_trans)
        
        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)
        
        # Apply attention: (batch_size, 1, n_plds) @ (batch_size, n_plds, channels_per_pld)
        attended = torch.bmm(attention_weights.unsqueeze(1), x_trans)
        
        return attended.squeeze(1), attention_weights
    
class UncertaintyHead(nn.Module):
    """Uncertainty estimation head"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var

class EnhancedASLNet(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 n_plds: int = 6,
                 dropout_rate: float = 0.1,
                 norm_type: str = 'batch'):
        super().__init__()
        
        # Store parameters
        self.n_plds = n_plds
        self.input_size = input_size
        
        # Calculate dimensions
        self.channels_per_pld = hidden_sizes[-1]
        
        # Input processing layers
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_norm = self._get_norm_layer(hidden_sizes[0], norm_type)
        
        # Shared feature extraction
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                self._get_norm_layer(hidden_sizes[i+1], norm_type),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for i in range(len(hidden_sizes)-1)
        ])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_sizes[-1], dropout_rate)
            for _ in range(3)
        ])
        
        # Add a dimension adjustment layer
        self.dim_adjust = nn.Linear(hidden_sizes[-1], self.channels_per_pld * n_plds)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(n_plds, hidden_sizes[-1])
        
        # Separate branches for CBF and ATT
        combined_size = self.channels_per_pld + hidden_sizes[-1]
        branch_size = combined_size
        
        self.cbf_branch = nn.Sequential(
            nn.Linear(branch_size, branch_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(branch_size // 2, branch_size // 4)
        )
        
        self.att_branch = nn.Sequential(
            nn.Linear(branch_size, branch_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(branch_size // 2, branch_size // 4)
        )
        
        # Uncertainty estimation heads
        self.cbf_uncertainty = UncertaintyHead(branch_size // 4, 1)
        self.att_uncertainty = UncertaintyHead(branch_size // 4, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Initial processing
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # Shared feature extraction
        for layer in self.shared_layers:
            x = layer(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Adjust dimensions and reshape for temporal attention
        x_adjusted = self.dim_adjust(x)
        x_reshaped = x_adjusted.view(batch_size, self.channels_per_pld, self.n_plds)
        
        # Apply temporal attention
        x_attended, attention_weights = self.temporal_attention(x_reshaped)
        
        # Concatenate attended features with original
        x_combined = torch.cat([x_attended, x], dim=1)
        
        # Separate branches
        cbf_features = self.cbf_branch(x_combined)
        att_features = self.att_branch(x_combined)
        
        # Uncertainty estimation
        cbf_mean, cbf_log_var = self.cbf_uncertainty(cbf_features)
        att_mean, att_log_var = self.att_uncertainty(att_features)
        
        return cbf_mean, att_mean, cbf_log_var, att_log_var

    def _get_norm_layer(self, size: int, norm_type: str) -> nn.Module:
        """Get normalization layer based on specified type"""
        if norm_type == 'batch':
            return nn.BatchNorm1d(size)
        elif norm_type == 'layer':
            return nn.LayerNorm(size)
        elif norm_type == 'instance':
            return nn.InstanceNorm1d(size)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
            
class CustomLoss(nn.Module):
    """Custom loss function with uncertainty estimation and temporal consistency"""
    
    def __init__(self, att_weight_schedule: Optional[callable] = None):
        super().__init__()
        self.att_weight_schedule = att_weight_schedule or (lambda _: 1.0)
        
    def forward(self, 
                cbf_pred: torch.Tensor,
                att_pred: torch.Tensor,
                cbf_true: torch.Tensor,
                att_true: torch.Tensor,
                cbf_log_var: torch.Tensor,
                att_log_var: torch.Tensor,
                epoch: int) -> torch.Tensor:
        
        # Get ATT weight for current epoch
        att_weight = self.att_weight_schedule(epoch)
        
        # Uncertainty weighted MSE loss
        cbf_loss = 0.5 * (torch.exp(-cbf_log_var) * (cbf_pred - cbf_true)**2 + cbf_log_var)
        att_loss = 0.5 * (torch.exp(-att_log_var) * (att_pred - att_true)**2 + att_log_var)
        
        # Weight ATT loss more heavily for short ATT values
        att_weights = torch.exp(-att_true / 2000)  # Higher weights for shorter ATT
        weighted_att_loss = att_loss * att_weights * att_weight
        
        return torch.mean(cbf_loss + weighted_att_loss)
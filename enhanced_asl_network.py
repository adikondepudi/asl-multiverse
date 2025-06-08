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

class UncertaintyHead(nn.Module):
    """Uncertainty estimation head with bounded log_var output."""
    def __init__(self, input_dim: int, output_dim: int, log_var_min: float = -7.0, log_var_max: float = 7.0):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_var_raw = nn.Linear(input_dim, output_dim) # Predicts raw value for log_var
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(x)
        raw_log_var = self.log_var_raw(x)
        # Scale tanh output to [log_var_min, log_var_max]
        log_var_range = self.log_var_max - self.log_var_min
        log_var = self.log_var_min + (torch.tanh(raw_log_var) + 1.0) * 0.5 * log_var_range
        return mean, log_var

class EnhancedASLNet(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 n_plds: int = 6,
                 dropout_rate: float = 0.1,
                 norm_type: str = 'batch',
                 
                 use_transformer_temporal: bool = True,
                 use_focused_transformer: bool = False,
                 transformer_d_model: int = 64,
                 transformer_d_model_focused: int = 32,
                 transformer_nhead: int = 4,
                 transformer_nlayers: int = 2,
                 
                 m0_input_feature: bool = False,
                 
                 log_var_cbf_min: float = -6.0,
                 log_var_cbf_max: float = 7.0,
                 log_var_att_min: float = -2.0,
                 log_var_att_max: float = 14.0
                ):
        super().__init__()
        
        self.n_plds = n_plds
        self.use_transformer_temporal = use_transformer_temporal
        self.use_focused_transformer = use_focused_transformer
        self.m0_input_feature = m0_input_feature
        
        # --- FIX: Define feature processors based on the nature of the input ---
        self.num_raw_signal_features = n_plds * 2
        # All other features (engineered, M0, etc.) are processed by a simple linear layer.
        self.num_other_features = input_size - self.num_raw_signal_features
        
        fused_feature_size = 0

        if self.num_other_features > 0:
            # Allocate a portion of the first hidden layer's capacity to other features
            other_features_out_dim = hidden_sizes[0] // 4
            self.other_features_processor = nn.Linear(self.num_other_features, other_features_out_dim)
            fused_feature_size += other_features_out_dim
        else:
            self.other_features_processor = None
            
        # --- Temporal processing modules (applied FIRST to raw signals) ---
        if self.use_transformer_temporal:
            if self.use_focused_transformer:
                self.d_model_eff = transformer_d_model_focused
                # PCASL branch: projects from 1 feature per PLD to d_model
                self.pcasl_input_proj = nn.Linear(1, self.d_model_eff)
                encoder_pcasl = nn.TransformerEncoderLayer(self.d_model_eff, transformer_nhead, self.d_model_eff * 2, dropout_rate, batch_first=True)
                self.pcasl_transformer = nn.TransformerEncoder(encoder_pcasl, transformer_nlayers)
                
                # VSASL branch
                self.vsasl_input_proj = nn.Linear(1, self.d_model_eff)
                encoder_vsasl = nn.TransformerEncoderLayer(self.d_model_eff, transformer_nhead, self.d_model_eff * 2, dropout_rate, batch_first=True)
                self.vsasl_transformer = nn.TransformerEncoder(encoder_vsasl, transformer_nlayers)
                
                # The fused feature size will be the combination of both transformer outputs
                fused_feature_size += self.d_model_eff * 2
            else: # Shared Transformer
                self.d_model_eff = transformer_d_model
                # Project from 2 features per PLD (PCASL, VSASL) to d_model
                self.shared_input_proj = nn.Linear(2, self.d_model_eff)
                encoder_shared = nn.TransformerEncoderLayer(self.d_model_eff, transformer_nhead, self.d_model_eff * 2, dropout_rate, batch_first=True)
                self.shared_transformer = nn.TransformerEncoder(encoder_shared, transformer_nlayers)
                fused_feature_size += self.d_model_eff
        else:
            # If not using transformers, the raw signal features will be passed directly to the main MLP
            fused_feature_size += self.num_raw_signal_features

        # --- Main MLP backbone (operates on FUSED features) ---
        self.input_layer = nn.Linear(fused_feature_size, hidden_sizes[0])
        self.input_norm = self._get_norm_layer(hidden_sizes[0], norm_type)
        
        self.shared_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self._get_norm_layer(hidden_sizes[i+1], norm_type), nn.ReLU(), nn.Dropout(dropout_rate))
            for i in range(len(hidden_sizes)-1)
        ])
        
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_sizes[-1], dropout_rate) for _ in range(3)])

        # --- Output Branches ---
        branch_input_dim = hidden_sizes[-1]
        self.cbf_branch = nn.Sequential(nn.Linear(branch_input_dim, branch_input_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(branch_input_dim // 2, branch_input_dim // 4))
        self.att_branch = nn.Sequential(nn.Linear(branch_input_dim, branch_input_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(branch_input_dim // 2, branch_input_dim // 4))
        self.cbf_uncertainty = UncertaintyHead(branch_input_dim // 4, 1, log_var_min=log_var_cbf_min, log_var_max=log_var_cbf_max)
        self.att_uncertainty = UncertaintyHead(branch_input_dim // 4, 1, log_var_min=log_var_att_min, log_var_max=log_var_att_max)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # FIX: Split input into signal sequences and other features at the very beginning
        pcasl_seq = x[:, :self.n_plds]
        vsasl_seq = x[:, self.n_plds:self.num_raw_signal_features]
        
        # FIX: Process signal sequences with transformers FIRST
        if self.use_transformer_temporal:
            if self.use_focused_transformer:
                # Process PCASL: (B, N_plds) -> (B, N_plds, 1) -> (B, N_plds, D_model)
                pcasl_in = self.pcasl_input_proj(pcasl_seq.unsqueeze(-1))
                pcasl_out = self.pcasl_transformer(pcasl_in)
                pcasl_features = torch.mean(pcasl_out, dim=1) # Global average pooling over time

                # Process VSASL
                vsasl_in = self.vsasl_input_proj(vsasl_seq.unsqueeze(-1))
                vsasl_out = self.vsasl_transformer(vsasl_in)
                vsasl_features = torch.mean(vsasl_out, dim=1)
                
                temporal_features = torch.cat([pcasl_features, vsasl_features], dim=1)
            else: # Shared Transformer
                # Stack signals: (B, N_plds, 2)
                shared_seq = torch.stack([pcasl_seq, vsasl_seq], dim=-1)
                shared_in = self.shared_input_proj(shared_seq)
                shared_out = self.shared_transformer(shared_in)
                temporal_features = torch.mean(shared_out, dim=1)
            
            # Fuse temporal features with other features
            if self.num_other_features > 0:
                other_features = x[:, self.num_raw_signal_features:]
                processed_other_features = self.other_features_processor(other_features)
                fused_features = torch.cat([temporal_features, processed_other_features], dim=1)
            else:
                fused_features = temporal_features

        else: # No transformer, just fuse raw signals with other features
            if self.num_other_features > 0:
                other_features = x[:, self.num_raw_signal_features:]
                processed_other_features = self.other_features_processor(other_features)
                fused_features = torch.cat([pcasl_seq, vsasl_seq, processed_other_features], dim=1)
            else:
                fused_features = torch.cat([pcasl_seq, vsasl_seq], dim=1)

        # FIX: The main MLP backbone now operates on the properly fused feature vector
        h = self.input_layer(fused_features)
        h = self.input_norm(h)
        h = F.relu(h)
        
        for layer in self.shared_layers:
            h = layer(h)
        
        for block in self.residual_blocks:
            h = block(h)

        # Output branches remain the same
        cbf_features = self.cbf_branch(h)
        att_features = self.att_branch(h)
        cbf_mean, cbf_log_var = self.cbf_uncertainty(cbf_features)
        att_mean, att_log_var = self.att_uncertainty(att_features)
        
        return cbf_mean, att_mean, cbf_log_var, att_log_var

    def _get_norm_layer(self, size: int, norm_type: str) -> nn.Module:
        if norm_type == 'batch': return nn.BatchNorm1d(size)
        elif norm_type == 'layer': return nn.LayerNorm(size)
        else:
            print(f"Warning: Unknown normalization type '{norm_type}'. Using BatchNorm1d.")
            return nn.BatchNorm1d(size)

class CustomLoss(nn.Module):
    """
    Custom loss: NLL with regression-focal loss on ATT and optional log_var regularization.
    The focal loss dynamically weights samples based on prediction error, forcing the
    model to focus on harder examples.
    """
    
    def __init__(self, 
                 w_cbf: float = 1.0, 
                 w_att: float = 1.0, 
                 log_var_reg_lambda: float = 0.0,
                 focal_gamma: float = 1.5, # Focal loss focusing parameter
                 att_epoch_weight_schedule: Optional[callable] = None
                ):
        super().__init__()
        self.w_cbf = w_cbf
        self.w_att = w_att
        self.log_var_reg_lambda = log_var_reg_lambda
        self.focal_gamma = focal_gamma
        # Fallback for epoch weighting if provided, though focal loss is more powerful
        self.att_epoch_weight_schedule = att_epoch_weight_schedule or (lambda _: 1.0)
        
    def forward(self, 
                cbf_pred_norm: torch.Tensor, att_pred_norm: torch.Tensor, 
                cbf_true_norm: torch.Tensor, att_true_norm: torch.Tensor, 
                cbf_log_var: torch.Tensor, att_log_var: torch.Tensor, 
                epoch: int) -> torch.Tensor:
        
        # --- Standard NLL calculation (Aleatoric Uncertainty Loss) ---
        cbf_nll_loss = 0.5 * (torch.exp(-cbf_log_var) * (cbf_pred_norm - cbf_true_norm)**2 + cbf_log_var)
        att_nll_loss = 0.5 * (torch.exp(-att_log_var) * (att_pred_norm - att_true_norm)**2 + att_log_var)

        # --- Focal Weighting for ATT based on prediction error ---
        focal_weight = torch.ones_like(att_nll_loss) # Default weight is 1
        if self.focal_gamma > 0:
            with torch.no_grad(): # Don't backprop through the weight calculation
                # Get the absolute error (residual) for the normalized ATT
                att_residual = torch.abs(att_pred_norm - att_true_norm)
                
                # Normalize the residual to a [0, 1] range to act like a probability of error
                # A normalized target has std=1, so an error of 4 is ~4 std devs, which is very high.
                # Clamping ensures the weight doesn't become excessively large.
                att_error_norm = torch.clamp(att_residual / 4.0, 0.0, 1.0)

            # The focal modulating factor. We want to *increase* weight for large errors.
            # Weight is proportional to the normalized error. Add epsilon for stability.
            focal_weight = (att_error_norm + 0.1).pow(self.focal_gamma)

        # --- Apply weights and combine losses ---
        weighted_cbf_loss = self.w_cbf * cbf_nll_loss
        
        # Get epoch-based global weight for ATT
        att_epoch_weight_factor = self.att_epoch_weight_schedule(epoch) 

        # Apply both the dynamic focal weight and the global epoch weight to the ATT loss
        weighted_att_loss = self.w_att * att_nll_loss * focal_weight * att_epoch_weight_factor
        
        total_param_loss = torch.mean(weighted_cbf_loss + weighted_att_loss)
        
        # --- Optional regularization on the magnitude of predicted uncertainty ---
        log_var_regularization = 0.0
        if self.log_var_reg_lambda > 0:
            log_var_regularization = self.log_var_reg_lambda * \
                                     (torch.mean(cbf_log_var**2) + torch.mean(att_log_var**2))
            
        total_loss = total_param_loss + log_var_regularization
        
        return total_loss
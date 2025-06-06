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
                 input_size: int, # Base input size (e.g., n_plds * 2 for PCASL+VSASL)
                 hidden_sizes: List[int] = [256, 128, 64],
                 n_plds: int = 6, # Number of PLDs per modality (PCASL or VSASL)
                 dropout_rate: float = 0.1,
                 norm_type: str = 'batch',
                 
                 # Transformer settings
                 use_transformer_temporal: bool = True,
                 use_focused_transformer: bool = False, # New: For split PCASL/VSASL transformers
                 transformer_d_model: int = 64, # d_model for shared transformer OR total if focused implies split
                 transformer_d_model_focused: int = 32, # d_model per branch if focused transformer
                 transformer_nhead: int = 4,
                 transformer_nlayers: int = 2,
                 
                 m0_input_feature: bool = False,
                 
                 # Uncertainty head settings
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
        
        actual_input_size = input_size # This is n_plds_pcasl + n_plds_vsasl if they are distinct, or n_plds*2
        if self.m0_input_feature:
            actual_input_size += 1

        self.input_layer = nn.Linear(actual_input_size, hidden_sizes[0])
        self.input_norm = self._get_norm_layer(hidden_sizes[0], norm_type)
        
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                self._get_norm_layer(hidden_sizes[i+1], norm_type),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for i in range(len(hidden_sizes)-1)
        ])
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_sizes[-1], dropout_rate)
            for _ in range(3) 
        ])
        
        # Temporal processing module
        if self.use_transformer_temporal:
            if self.use_focused_transformer:
                self.d_model_transformer_eff = transformer_d_model_focused
                self.temporal_projection_pcasl = nn.Linear(hidden_sizes[-1], n_plds * self.d_model_transformer_eff)
                self.temporal_projection_vsasl = nn.Linear(hidden_sizes[-1], n_plds * self.d_model_transformer_eff)

                encoder_layer_pcasl = nn.TransformerEncoderLayer(
                    d_model=self.d_model_transformer_eff, nhead=transformer_nhead,
                    dim_feedforward=self.d_model_transformer_eff * 2, dropout=dropout_rate, batch_first=True
                )
                self.temporal_transformer_pcasl = nn.TransformerEncoder(encoder_layer_pcasl, num_layers=transformer_nlayers)
                
                encoder_layer_vsasl = nn.TransformerEncoderLayer(
                    d_model=self.d_model_transformer_eff, nhead=transformer_nhead,
                    dim_feedforward=self.d_model_transformer_eff * 2, dropout=dropout_rate, batch_first=True
                )
                self.temporal_transformer_vsasl = nn.TransformerEncoder(encoder_layer_vsasl, num_layers=transformer_nlayers)
                self.temporal_feature_size = self.d_model_transformer_eff * 2
            else: # Original shared transformer path
                self.d_model_transformer_eff = transformer_d_model
                self.temporal_projection = nn.Linear(hidden_sizes[-1], n_plds * self.d_model_transformer_eff)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model_transformer_eff, nhead=transformer_nhead,
                    dim_feedforward=self.d_model_transformer_eff * 2, dropout=dropout_rate, batch_first=True
                )
                self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)
                self.temporal_feature_size = self.d_model_transformer_eff
        else: 
            self.temporal_feature_size = hidden_sizes[-1]

        branch_input_dim = self.temporal_feature_size 
        
        self.cbf_branch = nn.Sequential(
            nn.Linear(branch_input_dim, branch_input_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(branch_input_dim // 2, branch_input_dim // 4)
        )
        self.att_branch = nn.Sequential(
            nn.Linear(branch_input_dim, branch_input_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(branch_input_dim // 2, branch_input_dim // 4)
        )
        
        self.cbf_uncertainty = UncertaintyHead(branch_input_dim // 4, 1, log_var_min=log_var_cbf_min, log_var_max=log_var_cbf_max)
        self.att_uncertainty = UncertaintyHead(branch_input_dim // 4, 1, log_var_min=log_var_att_min, log_var_max=log_var_att_max)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        for layer in self.shared_layers:
            x = layer(x)
        
        for block in self.residual_blocks:
            x = block(x) # x shape: (batch_size, hidden_sizes[-1])
        
        if self.use_transformer_temporal:
            if self.use_focused_transformer:
                # PCASL Stream
                x_pcasl_projected = self.temporal_projection_pcasl(x) 
                x_seq_pcasl = x_pcasl_projected.view(x.size(0), self.n_plds, self.d_model_transformer_eff)
                transformer_out_pcasl = self.temporal_transformer_pcasl(x_seq_pcasl)
                pooled_pcasl = torch.mean(transformer_out_pcasl, dim=1)
                
                # VSASL Stream
                x_vsasl_projected = self.temporal_projection_vsasl(x)
                x_seq_vsasl = x_vsasl_projected.view(x.size(0), self.n_plds, self.d_model_transformer_eff)
                transformer_out_vsasl = self.temporal_transformer_vsasl(x_seq_vsasl)
                pooled_vsasl = torch.mean(transformer_out_vsasl, dim=1)
                
                branch_features = torch.cat((pooled_pcasl, pooled_vsasl), dim=1)
            else: # Original shared transformer path
                x_projected = self.temporal_projection(x) 
                x_seq = x_projected.view(x.size(0), self.n_plds, self.d_model_transformer_eff)
                transformer_out = self.temporal_transformer(x_seq)
                branch_features = torch.mean(transformer_out, dim=1)
        else:
            branch_features = x

        cbf_features = self.cbf_branch(branch_features)
        att_features = self.att_branch(branch_features)
        
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
    """Custom loss: NLL with optional task weighting and log_var regularization."""
    
    def __init__(self, 
                 att_weight_schedule: Optional[callable] = None, # Retained for now, but effect nullified if not used
                 log_var_clamp_min: float = -10.0, # Retained, but bounded head output might make it less critical
                 log_var_clamp_max: float = 10.0,
                 w_cbf: float = 1.0, # New: weight for CBF loss component
                 w_att: float = 1.0, # New: weight for ATT loss component
                 log_var_reg_lambda: float = 0.0 # New: regularization strength for log_variances
                ):
        super().__init__()
        self.att_weight_schedule = att_weight_schedule or (lambda _: 1.0)
        self.log_var_clamp_min = log_var_clamp_min 
        self.log_var_clamp_max = log_var_clamp_max
        self.w_cbf = w_cbf
        self.w_att = w_att
        self.log_var_reg_lambda = log_var_reg_lambda
        
    def forward(self, 
                cbf_pred: torch.Tensor, att_pred: torch.Tensor, 
                cbf_true: torch.Tensor, att_true: torch.Tensor, 
                cbf_log_var: torch.Tensor, att_log_var: torch.Tensor, 
                epoch: int) -> torch.Tensor:
        
        # Log variances are now bounded by UncertaintyHead. Clamping here is a secondary check.
        cbf_log_var_clamped = torch.clamp(cbf_log_var, self.log_var_clamp_min, self.log_var_clamp_max)
        att_log_var_clamped = torch.clamp(att_log_var, self.log_var_clamp_min, self.log_var_clamp_max)
        
        cbf_nll_loss = 0.5 * (torch.exp(-cbf_log_var_clamped) * (cbf_pred - cbf_true)**2 + cbf_log_var_clamped)
        att_nll_loss = 0.5 * (torch.exp(-att_log_var_clamped) * (att_pred - att_true)**2 + att_log_var_clamped)
        
        # Phase 1, Item 1.1: Removed att_instance_weights. 
        # att_epoch_weight_factor defaults to 1.0 if schedule is None.
        att_epoch_weight_factor = self.att_weight_schedule(epoch) 
        
        weighted_cbf_loss = self.w_cbf * cbf_nll_loss
        weighted_att_loss = self.w_att * att_nll_loss * att_epoch_weight_factor # att_epoch_weight_factor kept for now
        
        total_param_loss = torch.mean(weighted_cbf_loss + weighted_att_loss)
        
        # Phase 3, Item 3.3: Regularization of log_var
        log_var_regularization = 0.0
        if self.log_var_reg_lambda > 0:
            # Penalize squared log_var (encourages log_var around 0, i.e., variance around 1)
            # This might need adjustment if parameters are not normalized.
            # For now, penalizing large magnitude log_vars.
            log_var_regularization = self.log_var_reg_lambda * \
                                     (torch.mean(cbf_log_var_clamped**2) + torch.mean(att_log_var_clamped**2))
            
        total_loss = total_param_loss + log_var_regularization
        
        return total_loss

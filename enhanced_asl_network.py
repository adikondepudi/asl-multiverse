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
    """Uncertainty estimation head"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(x)
        log_var = self.log_var(x) # Predict log variance for stability
        return mean, log_var

class EnhancedASLNet(nn.Module):
    def __init__(self, 
                 input_size: int, # Base input size (e.g., n_plds * 2 for PCASL+VSASL)
                 hidden_sizes: List[int] = [256, 128, 64],
                 n_plds: int = 6, # Number of PLDs per modality (PCASL or VSASL)
                 dropout_rate: float = 0.1,
                 norm_type: str = 'batch',
                 use_transformer_temporal: bool = True, # Flag to use Transformer
                 transformer_nhead: int = 4, # Num heads for Transformer
                 transformer_nlayers: int = 2, # Num layers for Transformer
                 m0_input_feature: bool = False # Flag if M0 is an additional input feature
                ):
        super().__init__()
        
        self.n_plds = n_plds
        self.use_transformer_temporal = use_transformer_temporal
        self.m0_input_feature = m0_input_feature
        
        actual_input_size = input_size
        if self.m0_input_feature:
            actual_input_size += 1 # Add one dimension for M0 scalar

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
            for _ in range(3) # Number of residual blocks
        ])
        
        # Temporal processing module
        if self.use_transformer_temporal:
            # Transformer expects (seq_len, batch_size, embed_dim)
            # Our features per PLD will be the "embedding"
            # Assuming hidden_sizes[-1] is the feature dim per combined PLD step after shared layers.
            # If we want to process PCASL and VSASL PLD sequences somewhat independently before combining,
            # the architecture would need adjustment here. For now, assume input is (PCASL_pld1, VSASL_pld1, PCASL_pld2, VSASL_pld2, ...)
            # And we want to treat each (PCASL_pld_i, VSASL_pld_i) pair as a "timestep" or we reshape.
            # Let's assume the input `x` to the transformer part is features extracted from the full (n_plds*2) vector.
            # The original TemporalAttention was applied after `dim_adjust` which created `channels_per_pld`.
            # Let's use hidden_sizes[-1] as the embedding dimension for the transformer.
            # The sequence length for the transformer would be n_plds.
            # This requires reshaping the output of shared_layers if it's a flat vector.

            self.channels_per_pld_transformer = hidden_sizes[-1] // 2 # Arbitrary split for PCASL/VSASL features if processing separately
                                                                      # Or use hidden_sizes[-1] if processing combined features per PLD
            self.transformer_embed_dim = hidden_sizes[-1] # Each PLD's features become an embedding
            
            # We need a way to get per-PLD features if input `x` is flat.
            # Let's assume `x` after shared_layers and residual_blocks is (batch_size, hidden_sizes[-1])
            # We need to project this into something suitable for the transformer's sequence.
            # This layer will project the flat features into (batch_size, n_plds, transformer_embed_dim)
            self.temporal_projection = nn.Linear(hidden_sizes[-1], n_plds * self.transformer_embed_dim)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.transformer_embed_dim, 
                nhead=transformer_nhead,
                dim_feedforward=self.transformer_embed_dim * 2, # Typical feedforward size
                dropout=dropout_rate,
                batch_first=True # Expects (batch, seq, feature)
            )
            self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)
            # Output of transformer will be (batch, n_plds, transformer_embed_dim)
            # We'll likely mean-pool this over the n_plds dimension.
            self.temporal_feature_size = self.transformer_embed_dim
        else: # Fallback or alternative (e.g., if simple MLP is better)
            # If not using transformer, the final feature vector from residual_blocks is used directly.
            self.temporal_feature_size = hidden_sizes[-1]


        # Branches for CBF and ATT
        # The input to branches will be the pooled transformer output or the direct output of residual blocks
        branch_input_dim = self.temporal_feature_size 
        
        self.cbf_branch = nn.Sequential(
            nn.Linear(branch_input_dim, branch_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(branch_input_dim // 2, branch_input_dim // 4)
        )
        
        self.att_branch = nn.Sequential(
            nn.Linear(branch_input_dim, branch_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(branch_input_dim // 2, branch_input_dim // 4)
        )
        
        self.cbf_uncertainty = UncertaintyHead(branch_input_dim // 4, 1)
        self.att_uncertainty = UncertaintyHead(branch_input_dim // 4, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x is assumed to be (batch_size, input_features)
        # If m0_input_feature is True, x should be (batch_size, n_plds*2 + 1)
        # where the last feature is M0.

        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        for layer in self.shared_layers:
            x = layer(x) # x shape: (batch_size, hidden_sizes[-1])
        
        for block in self.residual_blocks:
            x = block(x) # x shape: (batch_size, hidden_sizes[-1])
        
        # Temporal processing
        if self.use_transformer_temporal:
            # Project x into (batch, n_plds, transformer_embed_dim)
            x_projected = self.temporal_projection(x) # Output: (batch, n_plds * transformer_embed_dim)
            x_seq = x_projected.view(x.size(0), self.n_plds, self.transformer_embed_dim) # Reshape to (batch, n_plds, embed_dim)
            
            # Transformer expects (seq_len, batch, embed_dim) if batch_first=False
            # or (batch, seq_len, embed_dim) if batch_first=True. We use batch_first=True.
            transformer_out = self.temporal_transformer(x_seq) # Output: (batch, n_plds, transformer_embed_dim)
            
            # Pool the transformer output over the sequence dimension (n_plds)
            # This gives a fixed-size representation for the branches.
            branch_features = torch.mean(transformer_out, dim=1) # (batch, transformer_embed_dim)
        else:
            branch_features = x # Use features directly from residual blocks

        cbf_features = self.cbf_branch(branch_features)
        att_features = self.att_branch(branch_features)
        
        cbf_mean, cbf_log_var = self.cbf_uncertainty(cbf_features)
        att_mean, att_log_var = self.att_uncertainty(att_features)
        
        return cbf_mean, att_mean, cbf_log_var, att_log_var

    def _get_norm_layer(self, size: int, norm_type: str) -> nn.Module:
        if norm_type == 'batch':
            return nn.BatchNorm1d(size)
        elif norm_type == 'layer':
            return nn.LayerNorm(size)
        # InstanceNorm1d is typically for conv nets, not ideal for Linear features here.
        # elif norm_type == 'instance': 
        #     return nn.InstanceNorm1d(size) 
        else:
            print(f"Warning: Unknown normalization type '{norm_type}'. Using BatchNorm1d.")
            return nn.BatchNorm1d(size)
            
class CustomLoss(nn.Module):
    """Custom loss function with uncertainty estimation and ATT-based weighting"""
    
    def __init__(self, att_weight_schedule: Optional[callable] = None,
                 log_var_clamp_min: float = -10.0,
                 log_var_clamp_max: float = 10.0):
        super().__init__()
        # att_weight_schedule: lambda function epoch -> weight_factor, defaults to 1.0
        self.att_weight_schedule = att_weight_schedule or (lambda _: 1.0)
        self.log_var_clamp_min = log_var_clamp_min
        self.log_var_clamp_max = log_var_clamp_max
        
    def forward(self, 
                cbf_pred: torch.Tensor, # (batch, 1)
                att_pred: torch.Tensor, # (batch, 1)
                cbf_true: torch.Tensor, # (batch, 1)
                att_true: torch.Tensor, # (batch, 1)
                cbf_log_var: torch.Tensor, # (batch, 1)
                att_log_var: torch.Tensor, # (batch, 1)
                epoch: int) -> torch.Tensor:
        
        # Clamp log variances to prevent numerical instability
        cbf_log_var_clamped = torch.clamp(cbf_log_var, self.log_var_clamp_min, self.log_var_clamp_max)
        att_log_var_clamped = torch.clamp(att_log_var, self.log_var_clamp_min, self.log_var_clamp_max)
        
        # Aleatoric Uncertainty weighted MSE loss (Gaussian NLL)
        # Loss_i = 0.5 * ( (y_pred_i - y_true_i)^2 / exp(log_var_i) + log_var_i )
        # Loss_i = 0.5 * ( exp(-log_var_i) * (y_pred_i - y_true_i)^2 + log_var_i )
        cbf_nll_loss = 0.5 * (torch.exp(-cbf_log_var_clamped) * (cbf_pred - cbf_true)**2 + cbf_log_var_clamped)
        att_nll_loss = 0.5 * (torch.exp(-att_log_var_clamped) * (att_pred - att_true)**2 + att_log_var_clamped)
        
        # Get ATT weight for current epoch from schedule
        att_epoch_weight_factor = self.att_weight_schedule(epoch)
        
        # Heuristic weighting for ATT: higher weights for shorter true ATT values.
        # This encourages the network to be more accurate for shorter ATTs, which can be challenging.
        # Ensure att_true is positive before division, and scale appropriately.
        # Max ATT could be around 4000ms. exp(-500/2000) ~ 0.77, exp(-3000/2000) ~ 0.22
        # This gives higher weights to shorter ATTs.
        att_instance_weights = torch.exp(-torch.clamp(att_true, min=100.0) / 2000.0) 
        
        weighted_att_loss = att_nll_loss * att_instance_weights * att_epoch_weight_factor
        
        # Combine losses
        # Taking mean over the batch
        total_loss = torch.mean(cbf_nll_loss + weighted_att_loss)
        
        return total_loss

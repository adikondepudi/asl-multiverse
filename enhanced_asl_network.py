import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
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

class CrossAttentionBlock(nn.Module):
    """
    A cross-attention block that allows a query sequence to attend to a key-value sequence.
    Includes a residual connection and layer normalization.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: The sequence to be updated (e.g., PCASL features).
            key_value: The sequence to attend to (e.g., VSASL features).
        Returns:
            The updated query sequence after cross-attention.
        """
        attn_output, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        # Residual connection and normalization
        out = query + self.dropout(attn_output)
        out = self.norm(out)
        return out

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
    """
    Disentangled two-stream architecture for ASL parameter estimation.
    - ATT Stream: Processes signal *shape* using transformers to capture timing.
    - CBF Stream: Processes signal *amplitude* and engineered features using a simple MLP.
    - Cross-Attention: Fuses information between the PCASL and VSASL streams.
    """
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 n_plds: int = 6,
                 dropout_rate: float = 0.1,
                 norm_type: str = 'batch',
                 
                 use_transformer_temporal: bool = True, # Kept for signature compatibility, logic now fixed to transformer for ATT
                 use_focused_transformer: bool = True,  # Kept for signature compatibility
                 transformer_d_model_focused: int = 32,
                 transformer_nhead: int = 4,
                 transformer_nlayers: int = 2,
                 
                 m0_input_feature: bool = False, # Handled by engineered features
                 
                 log_var_cbf_min: float = -6.0,
                 log_var_cbf_max: float = 7.0,
                 log_var_att_min: float = -2.0,
                 log_var_att_max: float = 14.0
                ):
        super().__init__()
        
        self.n_plds = n_plds
        self.num_raw_signal_features = n_plds * 2
        # All other features (engineered, M0, etc.) are processed by the CBF stream
        self.num_engineered_features = input_size - self.num_raw_signal_features

        # --- Buffers for normalization statistics ---
        self.register_buffer('pcasl_mean', torch.zeros(n_plds))
        self.register_buffer('pcasl_std', torch.ones(n_plds))
        self.register_buffer('vsasl_mean', torch.zeros(n_plds))
        self.register_buffer('vsasl_std', torch.ones(n_plds))
        self.register_buffer('amplitude_mean', torch.tensor(0.0))
        self.register_buffer('amplitude_std', torch.tensor(1.0))

        # --- ATT Stream (Shape-based) ---
        # This stream uses focused transformers on the shape-normalized signal.
        self.att_d_model = transformer_d_model_focused
        
        # PCASL branch for ATT stream
        self.pcasl_input_proj_att = nn.Linear(1, self.att_d_model)
        encoder_pcasl_att = nn.TransformerEncoderLayer(self.att_d_model, transformer_nhead, self.att_d_model * 2, dropout_rate, batch_first=True)
        self.pcasl_transformer_att = nn.TransformerEncoder(encoder_pcasl_att, transformer_nlayers)
        
        # VSASL branch for ATT stream
        self.vsasl_input_proj_att = nn.Linear(1, self.att_d_model)
        encoder_vsasl_att = nn.TransformerEncoderLayer(self.att_d_model, transformer_nhead, self.att_d_model * 2, dropout_rate, batch_first=True)
        self.vsasl_transformer_att = nn.TransformerEncoder(encoder_vsasl_att, transformer_nlayers)

        # NEW: Cross-Attention layers to fuse information between PCASL and VSASL streams
        self.pcasl_to_vsasl_cross_attn = CrossAttentionBlock(self.att_d_model, transformer_nhead, dropout_rate)
        self.vsasl_to_pcasl_cross_attn = CrossAttentionBlock(self.att_d_model, transformer_nhead, dropout_rate)

        # MLP for ATT stream after transformer feature fusion
        att_mlp_input_size = self.att_d_model * 2
        self.att_mlp = nn.Sequential(
            nn.Linear(att_mlp_input_size, hidden_sizes[0]),
            self._get_norm_layer(hidden_sizes[0], norm_type),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            self._get_norm_layer(hidden_sizes[1], norm_type),
            nn.ReLU()
        )
        self.att_uncertainty = UncertaintyHead(hidden_sizes[1], 1, log_var_min=log_var_att_min, log_var_max=log_var_att_max)
        
        # --- CBF Stream (Amplitude-based) ---
        # This stream uses a simple MLP on the signal amplitude and engineered features.
        cbf_mlp_input_size = 1 + self.num_engineered_features # 1 for normalized amplitude
        self.cbf_mlp = nn.Sequential(
            nn.Linear(cbf_mlp_input_size, 64),
            nn.ReLU(),
            self._get_norm_layer(64, norm_type),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32)
        )
        self.cbf_uncertainty = UncertaintyHead(32, 1, log_var_min=log_var_cbf_min, log_var_max=log_var_cbf_max)

    def set_norm_stats(self, norm_stats: Dict):
        """A helper method to set the normalization statistics from a dictionary."""
        if not norm_stats:
            return
        
        def to_tensor(val):
            return torch.tensor(val, device=self.pcasl_mean.device, dtype=torch.float32)

        self.pcasl_mean.data = to_tensor(norm_stats.get('pcasl_mean', 0.0))
        self.pcasl_std.data = to_tensor(norm_stats.get('pcasl_std', 1.0))
        self.vsasl_mean.data = to_tensor(norm_stats.get('vsasl_mean', 0.0))
        self.vsasl_std.data = to_tensor(norm_stats.get('vsasl_std', 1.0))
        self.amplitude_mean.data = to_tensor(norm_stats.get('amplitude_mean', 0.0))
        self.amplitude_std.data = to_tensor(norm_stats.get('amplitude_std', 1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- 1. Disentangle Input ---
        # The input `x` contains the Z-scored signal and un-normalized engineered features.
        normalized_signal = x[:, :self.num_raw_signal_features]
        engineered_features = x[:, self.num_raw_signal_features:]

        # --- 2. Un-normalize to get physical signal ---
        pcasl_norm = normalized_signal[:, :self.n_plds]
        vsasl_norm = normalized_signal[:, self.n_plds:self.num_raw_signal_features]
        pcasl_raw = pcasl_norm * self.pcasl_std + self.pcasl_mean
        vsasl_raw = vsasl_norm * self.vsasl_std + self.vsasl_mean
        raw_signal = torch.cat([pcasl_raw, vsasl_raw], dim=1)

        # --- 3. Extract physical features & re-normalize for network input ---
        # Extract physical amplitude for the CBF stream
        amplitude_physical = torch.linalg.norm(raw_signal, dim=1, keepdim=True)
        # Normalize the physical amplitude before feeding to the network
        amplitude_norm = (amplitude_physical - self.amplitude_mean) / (self.amplitude_std + 1e-6)
        
        # Use the original Z-scored signal as the shape input for the ATT stream
        pcasl_shape_seq = normalized_signal[:, :self.n_plds]
        vsasl_shape_seq = normalized_signal[:, self.n_plds:self.num_raw_signal_features]

        # --- 4. CBF Stream Processing ---
        cbf_stream_input = torch.cat([amplitude_norm, engineered_features], dim=1)
        cbf_final_features = self.cbf_mlp(cbf_stream_input)
        cbf_mean, cbf_log_var = self.cbf_uncertainty(cbf_final_features)

        # --- 5. ATT Stream Processing ---
        # PCASL branch (self-attention)
        pcasl_in_att = self.pcasl_input_proj_att(pcasl_shape_seq.unsqueeze(-1))
        pcasl_self_attn_out = self.pcasl_transformer_att(pcasl_in_att)

        # VSASL branch (self-attention)
        vsasl_in_att = self.vsasl_input_proj_att(vsasl_shape_seq.unsqueeze(-1))
        vsasl_self_attn_out = self.vsasl_transformer_att(vsasl_in_att)

        # Cross-attention to allow streams to inform each other
        pcasl_out_att = self.pcasl_to_vsasl_cross_attn(query=pcasl_self_attn_out, key_value=vsasl_self_attn_out)
        vsasl_out_att = self.vsasl_to_pcasl_cross_attn(query=vsasl_self_attn_out, key_value=pcasl_self_attn_out)

        # Global average pooling on the cross-attended features
        pcasl_features_att = torch.mean(pcasl_out_att, dim=1)
        vsasl_features_att = torch.mean(vsasl_out_att, dim=1)

        # Fuse ATT transformer features and process with MLP
        att_stream_features = torch.cat([pcasl_features_att, vsasl_features_att], dim=1)
        att_final_features = self.att_mlp(att_stream_features)
        att_mean, att_log_var = self.att_uncertainty(att_final_features)
        
        return cbf_mean, att_mean, cbf_log_var, att_log_var

    def _get_norm_layer(self, size: int, norm_type: str) -> nn.Module:
        if norm_type == 'batch': return nn.BatchNorm1d(size)
        elif norm_type == 'layer': return nn.LayerNorm(size)
        else:
            print(f"Warning: Unknown normalization type '{norm_type}'. Using BatchNorm1d.")
            return nn.BatchNorm1d(size)

def torch_kinetic_model(pred_cbf_norm: torch.Tensor, pred_att_norm: torch.Tensor,
                        norm_stats: Dict, model_params: Dict) -> torch.Tensor:
    """
    Differentiable PyTorch implementation of the ASL kinetic models.
    This function acts as a "physics decoder" for the PINN loss.
    """
    # 1. Denormalize predictions back to physical units
    pred_cbf = pred_cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    pred_att = pred_att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    
    # Convert CBF to ml/g/s
    pred_cbf_cgs = pred_cbf / 6000.0

    # 2. Get constants and PLDs as torch tensors on the correct device
    device = pred_cbf.device
    plds = torch.tensor(model_params['pld_values'], device=device, dtype=torch.float32)
    T1_artery = torch.tensor(model_params['T1_artery'], device=device, dtype=torch.float32)
    T_tau = torch.tensor(model_params['T_tau'], device=device, dtype=torch.float32)
    alpha_PCASL = torch.tensor(model_params['alpha_PCASL'], device=device, dtype=torch.float32)
    alpha_VSASL = torch.tensor(model_params['alpha_VSASL'], device=device, dtype=torch.float32)
    alpha_BS1 = torch.tensor(1.0, device=device, dtype=torch.float32) # Assuming 1.0 from config
    T2_factor = torch.tensor(1.0, device=device, dtype=torch.float32) # Assuming 1.0 from config
    lambda_blood = 0.90; M0_b = 1.0

    # Ensure inputs are correctly broadcastable for batched operations
    # pred_cbf_cgs and pred_att are (B, 1), plds is (N_plds)
    # We want results of shape (B, N_plds)
    B = pred_cbf_cgs.shape[0]
    N = plds.shape[0]
    plds_b = plds.unsqueeze(0).expand(B, -1) # (B, N_plds)
    
    # --- 3. PCASL Kinetic Model (PyTorch) ---
    alpha1 = alpha_PCASL * (alpha_BS1**4)
    pcasl_prefactor = (2 * M0_b * pred_cbf_cgs * alpha1 / lambda_blood * T1_artery / 1000) * T2_factor
    
    cond_full_bolus = plds_b >= pred_att
    cond_trailing_edge = (plds_b < pred_att) & (plds_b >= (pred_att - T_tau))
    
    pcasl_sig = torch.zeros_like(plds_b)
    
    # Full bolus decay part
    pcasl_sig = torch.where(
        cond_full_bolus,
        pcasl_prefactor * torch.exp(-plds_b / T1_artery) * (1 - torch.exp(-T_tau / T1_artery)),
        pcasl_sig
    )
    # Trailing edge part
    pcasl_sig = torch.where(
        cond_trailing_edge,
        pcasl_prefactor * (torch.exp(-pred_att / T1_artery) - torch.exp(-(T_tau + plds_b) / T1_artery)),
        pcasl_sig
    )

    # --- 4. VSASL Kinetic Model (PyTorch) ---
    alpha2 = alpha_VSASL * (alpha_BS1**3)
    vsasl_prefactor = (2 * M0_b * pred_cbf_cgs * alpha2 / lambda_blood) * T2_factor

    cond_ti_le_att = plds_b <= pred_att # TI is assumed equal to PLD here
    
    vsasl_sig = torch.where(
        cond_ti_le_att,
        vsasl_prefactor * (plds_b / 1000) * torch.exp(-plds_b / T1_artery),
        vsasl_prefactor * (pred_att / 1000) * torch.exp(-plds_b / T1_artery)
    )

    # --- 5. Concatenate and re-normalize the signal ---
    reconstructed_signal = torch.cat([pcasl_sig, vsasl_sig], dim=1) # (B, N_plds * 2)

    pcasl_mean = torch.tensor(norm_stats['pcasl_mean'], device=device, dtype=torch.float32)
    pcasl_std = torch.tensor(norm_stats['pcasl_std'], device=device, dtype=torch.float32)
    vsasl_mean = torch.tensor(norm_stats['vsasl_mean'], device=device, dtype=torch.float32)
    vsasl_std = torch.tensor(norm_stats['vsasl_std'], device=device, dtype=torch.float32)
    
    pcasl_recon_norm = (reconstructed_signal[:, :N] - pcasl_mean) / (pcasl_std + 1e-6)
    vsasl_recon_norm = (reconstructed_signal[:, N:] - vsasl_mean) / (vsasl_std + 1e-6)
    
    reconstructed_signal_norm = torch.cat([pcasl_recon_norm, vsasl_recon_norm], dim=1)
    
    return reconstructed_signal_norm

class CustomLoss(nn.Module):
    """
    Custom loss: NLL with regression-focal loss on ATT and optional log_var regularization
    and a physics-informed (PINN) reconstruction loss.
    """
    
    def __init__(self, 
                 w_cbf: float = 1.0, 
                 w_att: float = 1.0, 
                 log_var_reg_lambda: float = 0.0,
                 focal_gamma: float = 1.5,
                 pinn_weight: float = 0.0,
                 model_params: Optional[Dict[str, Any]] = None,
                 att_epoch_weight_schedule: Optional[callable] = None,
                 pinn_att_weighting_sigma: float = 500.0
                ):
        super().__init__()
        self.w_cbf = w_cbf
        self.w_att = w_att
        self.log_var_reg_lambda = log_var_reg_lambda
        self.focal_gamma = focal_gamma
        self.pinn_weight = pinn_weight
        self.model_params = model_params if model_params is not None else {}
        self.norm_stats = None
        self.att_epoch_weight_schedule = att_epoch_weight_schedule or (lambda _: 1.0)
        self.pinn_att_weighting_sigma = pinn_att_weighting_sigma

    def forward(self,
                normalized_input_signal: torch.Tensor,
                cbf_pred_norm: torch.Tensor, att_pred_norm: torch.Tensor, 
                cbf_true_norm: torch.Tensor, att_true_norm: torch.Tensor, 
                cbf_log_var: torch.Tensor, att_log_var: torch.Tensor, 
                global_epoch: int) -> torch.Tensor:
        
        # --- Standard NLL calculation (Aleatoric Uncertainty Loss) ---
        cbf_nll_loss = 0.5 * (torch.exp(-cbf_log_var) * (cbf_pred_norm - cbf_true_norm)**2 + cbf_log_var)
        att_nll_loss = 0.5 * (torch.exp(-att_log_var) * (att_pred_norm - att_true_norm)**2 + att_log_var)

        # --- Focal Weighting for ATT based on prediction error ---
        focal_weight = torch.ones_like(att_nll_loss)
        if self.focal_gamma > 0:
            with torch.no_grad():
                att_residual = torch.abs(att_pred_norm - att_true_norm)
                att_error_norm = torch.clamp(att_residual / 4.0, 0.0, 1.0)
            focal_weight = (att_error_norm + 0.1).pow(self.focal_gamma)

        # --- Apply weights and combine losses ---
        weighted_cbf_loss = self.w_cbf * cbf_nll_loss
        att_epoch_weight_factor = self.att_epoch_weight_schedule(global_epoch) 
        weighted_att_loss = self.w_att * att_nll_loss * focal_weight * att_epoch_weight_factor
        total_param_loss = torch.mean(weighted_cbf_loss + weighted_att_loss)
        
        # --- Optional regularization on the magnitude of predicted uncertainty ---
        log_var_regularization = 0.0
        if self.log_var_reg_lambda > 0:
            log_var_regularization = self.log_var_reg_lambda * (torch.mean(cbf_log_var**2) + torch.mean(att_log_var**2))
            
        # --- Physics-Informed (PINN) Regularization with Physics-Guided Attention ---
        pinn_loss = 0.0
        if self.pinn_weight > 0 and self.norm_stats and self.model_params and global_epoch > 10:
            reconstructed_signal_norm = torch_kinetic_model(
                cbf_pred_norm, att_pred_norm, self.norm_stats, self.model_params
            )
            
            # Create physics-guided weights for the PINN loss
            with torch.no_grad():
                device = att_true_norm.device
                plds = torch.tensor(self.model_params['pld_values'], device=device, dtype=torch.float32)
                y_mean_att = self.norm_stats['y_mean_att']
                y_std_att = self.norm_stats['y_std_att']
                att_true_ms = att_true_norm * y_std_att + y_mean_att # De-normalize ATT to ms
                
                # Create Gaussian weights centered at the true ATT for each sample in the batch
                plds_b = plds.unsqueeze(0).expand(att_true_ms.shape[0], -1) # (B, N_plds)
                pinn_weights_pcasl = torch.exp(-((plds_b - att_true_ms)**2) / (2 * self.pinn_att_weighting_sigma**2))
                
                # Apply same weighting to both PCASL and VSASL parts of the signal vector
                pinn_loss_weights = torch.cat([pinn_weights_pcasl, pinn_weights_pcasl], dim=1)
                
                # Normalize weights to keep loss magnitude consistent
                pinn_loss_weights = pinn_loss_weights * pinn_loss_weights.numel() / (pinn_loss_weights.sum() + 1e-9)

            # Calculate weighted MSE for PINN loss
            num_raw_signal_feats = len(self.model_params.get('pld_values', [])) * 2
            input_signal_norm = normalized_input_signal[:, :num_raw_signal_feats]
            
            pinn_loss = torch.mean(pinn_loss_weights * (reconstructed_signal_norm - input_signal_norm)**2)

        total_loss = total_param_loss + log_var_regularization + self.pinn_weight * pinn_loss
        
        return total_loss
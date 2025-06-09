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
        cbf_mlp_input_size = 1 + self.num_engineered_features # 1 for amplitude
        self.cbf_mlp = nn.Sequential(
            nn.Linear(cbf_mlp_input_size, 64),
            nn.ReLU(),
            self._get_norm_layer(64, norm_type),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32)
        )
        self.cbf_uncertainty = UncertaintyHead(32, 1, log_var_min=log_var_cbf_min, log_var_max=log_var_cbf_max)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- 1. Disentangle Input ---
        raw_signal = x[:, :self.num_raw_signal_features]
        engineered_features = x[:, self.num_raw_signal_features:]
        
        # Calculate amplitude (L2 norm) and shape-normalized signal
        amplitude = torch.linalg.norm(raw_signal, dim=1, keepdim=True) + 1e-6
        shape_input = raw_signal / amplitude
        
        pcasl_shape_seq = shape_input[:, :self.n_plds]
        vsasl_shape_seq = shape_input[:, self.n_plds:self.num_raw_signal_features]
        
        # Prepare input for the CBF stream
        cbf_stream_input = torch.cat([amplitude - 1e-6, engineered_features], dim=1)

        # --- 2. ATT Stream Processing ---
        # PCASL branch
        pcasl_in_att = self.pcasl_input_proj_att(pcasl_shape_seq.unsqueeze(-1))
        pcasl_out_att = self.pcasl_transformer_att(pcasl_in_att)
        pcasl_features_att = torch.mean(pcasl_out_att, dim=1) # Global average pooling

        # VSASL branch
        vsasl_in_att = self.vsasl_input_proj_att(vsasl_shape_seq.unsqueeze(-1))
        vsasl_out_att = self.vsasl_transformer_att(vsasl_in_att)
        vsasl_features_att = torch.mean(vsasl_out_att, dim=1)

        # Fuse ATT transformer features and process with MLP
        att_stream_features = torch.cat([pcasl_features_att, vsasl_features_att], dim=1)
        att_final_features = self.att_mlp(att_stream_features)
        att_mean, att_log_var = self.att_uncertainty(att_final_features)

        # --- 3. CBF Stream Processing ---
        cbf_final_features = self.cbf_mlp(cbf_stream_input)
        cbf_mean, cbf_log_var = self.cbf_uncertainty(cbf_final_features)
        
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
                 pinn_weight: float = 0.0, # NEW: Weight for the physics-informed loss
                 model_params: Optional[Dict[str, Any]] = None, # NEW: For physics constants
                 att_epoch_weight_schedule: Optional[callable] = None
                ):
        super().__init__()
        self.w_cbf = w_cbf
        self.w_att = w_att
        self.log_var_reg_lambda = log_var_reg_lambda
        self.focal_gamma = focal_gamma
        self.pinn_weight = pinn_weight
        self.model_params = model_params if model_params is not None else {}
        self.norm_stats = None # To be populated by the trainer
        self.att_epoch_weight_schedule = att_epoch_weight_schedule or (lambda _: 1.0)
        
    def forward(self,
                normalized_input_signal: torch.Tensor, # NEW: The original input to the model
                cbf_pred_norm: torch.Tensor, att_pred_norm: torch.Tensor, 
                cbf_true_norm: torch.Tensor, att_true_norm: torch.Tensor, 
                cbf_log_var: torch.Tensor, att_log_var: torch.Tensor, 
                epoch: int) -> torch.Tensor:
        
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
        att_epoch_weight_factor = self.att_epoch_weight_schedule(epoch) 
        weighted_att_loss = self.w_att * att_nll_loss * focal_weight * att_epoch_weight_factor
        total_param_loss = torch.mean(weighted_cbf_loss + weighted_att_loss)
        
        # --- Optional regularization on the magnitude of predicted uncertainty ---
        log_var_regularization = 0.0
        if self.log_var_reg_lambda > 0:
            log_var_regularization = self.log_var_reg_lambda * (torch.mean(cbf_log_var**2) + torch.mean(att_log_var**2))
            
        # --- NEW: Physics-Informed (PINN) Regularization ---
        pinn_loss = 0.0
        if self.pinn_weight > 0 and self.norm_stats and self.model_params and epoch > 10: # Activate after some epochs
            # Reconstruct signal from model predictions
            reconstructed_signal_norm = torch_kinetic_model(
                cbf_pred_norm, att_pred_norm, self.norm_stats, self.model_params
            )
            # Compare reconstructed signal with the actual input signal (raw signal part only)
            num_raw_signal_feats = len(self.model_params.get('pld_values', [])) * 2
            pinn_loss = F.mse_loss(
                reconstructed_signal_norm,
                normalized_input_signal[:, :num_raw_signal_feats]
            )

        total_loss = total_param_loss + log_var_regularization + self.pinn_weight * pinn_loss
        
        return total_loss

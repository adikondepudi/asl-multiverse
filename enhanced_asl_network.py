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

def _torch_physical_kinetic_model(
    pred_cbf: torch.Tensor, pred_att: torch.Tensor,
    plds: torch.Tensor, model_params: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable PyTorch implementation of the ASL kinetic models returning PHYSICAL signals.
    """
    pred_cbf_cgs = pred_cbf / 6000.0
    device = pred_cbf.device
    
    T1_artery = torch.tensor(model_params['T1_artery'], device=device, dtype=torch.float32)
    T_tau = torch.tensor(model_params['T_tau'], device=device, dtype=torch.float32)
    alpha_PCASL = torch.tensor(model_params['alpha_PCASL'], device=device, dtype=torch.float32)
    alpha_VSASL = torch.tensor(model_params['alpha_VSASL'], device=device, dtype=torch.float32)
    alpha_BS1 = torch.tensor(model_params.get('alpha_BS1', 1.0), device=device, dtype=torch.float32)
    T2_factor = torch.tensor(model_params.get('T2_factor', 1.0), device=device, dtype=torch.float32)
    lambda_blood = 0.90; M0_b = 1.0

    B = pred_cbf_cgs.shape[0]
    plds_b = plds.unsqueeze(0).expand(B, -1)

    alpha1 = alpha_PCASL * (alpha_BS1**4)
    pcasl_prefactor = (2 * M0_b * pred_cbf_cgs * alpha1 / lambda_blood * T1_artery / 1000) * T2_factor
    
    cond_full_bolus = plds_b >= pred_att
    cond_trailing_edge = (plds_b < pred_att) & (plds_b >= (pred_att - T_tau))
    
    pcasl_sig = torch.zeros_like(plds_b)
    pcasl_sig = torch.where(
        cond_full_bolus,
        pcasl_prefactor * torch.exp(-plds_b / T1_artery) * (1 - torch.exp(-T_tau / T1_artery)),
        pcasl_sig)
    pcasl_sig = torch.where(
        cond_trailing_edge,
        pcasl_prefactor * (torch.exp(-pred_att / T1_artery) - torch.exp(-(T_tau + plds_b) / T1_artery)),
        pcasl_sig)

    alpha2 = alpha_VSASL * (alpha_BS1**3)
    vsasl_prefactor = (2 * M0_b * pred_cbf_cgs * alpha2 / lambda_blood) * T2_factor
    cond_ti_le_att = plds_b <= pred_att
    
    vsasl_sig = torch.where(
        cond_ti_le_att,
        vsasl_prefactor * (plds_b / 1000) * torch.exp(-plds_b / T1_artery),
        vsasl_prefactor * (pred_att / 1000) * torch.exp(-plds_b / T1_artery))

    return pcasl_sig, vsasl_sig

def _torch_analytic_gradients(
    pred_cbf: torch.Tensor, pred_att: torch.Tensor,
    plds: torch.Tensor, model_params: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes analytical gradients of the physical ASL signals with respect to CBF and ATT.
    This is the core of the Differentiable Physics-Encoder.
    """
    pcasl_sig, vsasl_sig = _torch_physical_kinetic_model(pred_cbf, pred_att, plds, model_params)
    
    # 1. dS/dCBF is simple due to linear relationship
    # Add a small epsilon to avoid division by zero if cbf is zero
    dSdC_pcasl = pcasl_sig / (pred_cbf + 1e-6)
    dSdC_vsasl = vsasl_sig / (pred_cbf + 1e-6)
    
    # 2. dS/dATT requires evaluating the derivatives of the kinetic equations
    device = pred_cbf.device
    T1_artery = torch.tensor(model_params['T1_artery'], device=device, dtype=torch.float32)
    T_tau = torch.tensor(model_params['T_tau'], device=device, dtype=torch.float32)
    B = pred_cbf.shape[0]
    plds_b = plds.unsqueeze(0).expand(B, -1)

    # PCASL dS/dATT
    dSdA_pcasl = torch.zeros_like(pcasl_sig)
    cond_trailing_edge = (plds_b < pred_att) & (plds_b >= (pred_att - T_tau))
    pcasl_prefactor_for_grad = (pcasl_sig / (pred_cbf + 1e-6)) * pred_cbf
    dSdA_pcasl = torch.where(
        cond_trailing_edge,
        pcasl_prefactor_for_grad * (-1.0 / T1_artery) * torch.exp(-pred_att / T1_artery),
        dSdA_pcasl
    )

    # VSASL dS/dATT
    vsasl_prefactor_for_grad = (vsasl_sig / (pred_cbf + 1e-6)) * pred_cbf
    cond_ti_gt_att = plds_b > pred_att
    dSdA_vsasl = torch.where(
        cond_ti_gt_att,
        vsasl_prefactor_for_grad / (pred_att + 1e-6), # derivative of (K * ATT * exp(...)) wrt ATT is K*exp(...) = S/ATT
        torch.zeros_like(vsasl_sig)
    )

    return dSdC_pcasl, dSdA_pcasl, dSdC_vsasl, dSdA_vsasl

class EnhancedASLNet(nn.Module):
    """
    Disentangled two-stream architecture for ASL parameter estimation.
    - Differentiable Physics-Encoder: Enriches input features with signal sensitivities (dS/dCBF, dS/dATT).
    - Multi-Scale Transformer: Processes temporal information at short and long scales for robust ATT estimation.
    - Cross-Attention: Fuses information between the PCASL and VSASL streams.
    """
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 n_plds: int = 6,
                 dropout_rate: float = 0.1,
                 norm_type: str = 'batch',
                 
                 use_transformer_temporal: bool = True,
                 use_focused_transformer: bool = True,
                 transformer_d_model_focused: int = 32,
                 transformer_nhead: int = 4,
                 transformer_nlayers: int = 2,
                 
                 m0_input_feature: bool = False,
                 
                 log_var_cbf_min: float = -6.0,
                 log_var_cbf_max: float = 7.0,
                 log_var_att_min: float = -2.0,
                 log_var_att_max: float = 14.0,
                 **kwargs
                ):
        super().__init__()
        
        self.n_plds = n_plds
        self.num_raw_signal_features = n_plds * 2
        self.num_engineered_features = input_size - self.num_raw_signal_features
        
        physics_keys = ['T1_artery', 'T_tau', 'alpha_PCASL', 'alpha_VSASL', 'alpha_BS1', 'T2_factor', 'pld_values']
        self.model_params_for_physics = {key: kwargs[key] for key in physics_keys if key in kwargs}

        # --- Buffers for normalization statistics ---
        self.register_buffer('pcasl_mean', torch.zeros(n_plds))
        self.register_buffer('pcasl_std', torch.ones(n_plds))
        self.register_buffer('vsasl_mean', torch.zeros(n_plds))
        self.register_buffer('vsasl_std', torch.ones(n_plds))
        self.register_buffer('amplitude_mean', torch.tensor(0.0))
        self.register_buffer('amplitude_std', torch.tensor(1.0))

        if 'pld_values' in self.model_params_for_physics:
            self.register_buffer('plds_tensor', torch.tensor(self.model_params_for_physics['pld_values'], dtype=torch.float32))
        else:
            self.register_buffer('plds_tensor', torch.zeros(n_plds))

        self.pre_estimator = nn.Sequential(
            nn.Linear(1 + self.num_engineered_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.grad_norm_pcasl_cbf = nn.LayerNorm(n_plds)
        self.grad_norm_pcasl_att = nn.LayerNorm(n_plds)
        self.grad_norm_vsasl_cbf = nn.LayerNorm(n_plds)
        self.grad_norm_vsasl_att = nn.LayerNorm(n_plds)

        self.att_d_model = transformer_d_model_focused
        self.pcasl_input_proj_att = nn.Linear(3, self.att_d_model)
        self.vsasl_input_proj_att = nn.Linear(3, self.att_d_model)
        
        encoder_long = nn.TransformerEncoderLayer(self.att_d_model, transformer_nhead, self.att_d_model * 2, dropout_rate, batch_first=True)
        self.pcasl_transformer_att_long = nn.TransformerEncoder(encoder_long, transformer_nlayers)
        self.vsasl_transformer_att_long = nn.TransformerEncoder(encoder_long, transformer_nlayers)
        
        encoder_short = nn.TransformerEncoderLayer(self.att_d_model, transformer_nhead, self.att_d_model * 2, dropout_rate, batch_first=True)
        self.pcasl_transformer_att_short = nn.TransformerEncoder(encoder_short, max(1, transformer_nlayers // 2))
        self.vsasl_transformer_att_short = nn.TransformerEncoder(encoder_short, max(1, transformer_nlayers // 2))

        self.pcasl_to_vsasl_cross_attn = CrossAttentionBlock(self.att_d_model, transformer_nhead, dropout_rate)
        self.vsasl_to_pcasl_cross_attn = CrossAttentionBlock(self.att_d_model, transformer_nhead, dropout_rate)

        att_mlp_input_size = self.att_d_model * 4
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
        
        cbf_mlp_input_size = 1 + self.num_engineered_features
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # --- 1. Disentangle Input ---
        normalized_signal = x[:, :self.num_raw_signal_features]
        engineered_features = x[:, self.num_raw_signal_features:]

        # --- 2. Un-normalize to get physical signal ---
        pcasl_norm = normalized_signal[:, :self.n_plds]
        vsasl_norm = normalized_signal[:, self.n_plds:self.num_raw_signal_features]
        pcasl_raw = pcasl_norm * self.pcasl_std + self.pcasl_mean
        vsasl_raw = vsasl_norm * self.vsasl_std + self.vsasl_mean
        raw_signal = torch.cat([pcasl_raw, vsasl_raw], dim=1)

        # --- 3. Extract physical features & re-normalize for network input ---
        amplitude_physical = torch.linalg.norm(raw_signal, dim=1, keepdim=True)
        amplitude_norm = (amplitude_physical - self.amplitude_mean) / (self.amplitude_std + 1e-6)
        
        pcasl_shape_seq = normalized_signal[:, :self.n_plds]
        vsasl_shape_seq = normalized_signal[:, self.n_plds:self.num_raw_signal_features]

        # --- 4. CBF Stream Processing ---
        cbf_stream_input = torch.cat([amplitude_norm, engineered_features], dim=1)
        cbf_final_features = self.cbf_mlp(cbf_stream_input)
        cbf_mean, cbf_log_var = self.cbf_uncertainty(cbf_final_features)

        # --- 5. ATT Stream Processing (with Differentiable Physics-Encoder) ---
        pre_estimator_input = torch.cat([amplitude_physical.detach(), engineered_features], dim=1)
        rough_params = F.softplus(self.pre_estimator(pre_estimator_input))
        cbf_rough, att_rough = rough_params[:, 0:1], rough_params[:, 1:2]
        cbf_rough = torch.clamp(cbf_rough, min=1.0)
        att_rough = torch.clamp(att_rough, min=100.0)

        dSdC_p, dSdA_p, dSdC_v, dSdA_v = _torch_analytic_gradients(
            cbf_rough, att_rough, self.plds_tensor, self.model_params_for_physics
        )
        
        pcasl_feature_seq = torch.stack([pcasl_shape_seq, self.grad_norm_pcasl_cbf(dSdC_p), self.grad_norm_pcasl_att(dSdA_p)], dim=-1)
        vsasl_feature_seq = torch.stack([vsasl_shape_seq, self.grad_norm_vsasl_cbf(dSdC_v), self.grad_norm_vsasl_att(dSdA_v)], dim=-1)

        pcasl_in_att = self.pcasl_input_proj_att(pcasl_feature_seq)
        vsasl_in_att = self.vsasl_input_proj_att(vsasl_feature_seq)
        
        n_plds_short = self.n_plds // 2
        pcasl_in_att_short, vsasl_in_att_short = pcasl_in_att[:, :n_plds_short, :], vsasl_in_att[:, :n_plds_short, :]
        
        pcasl_long_out = self.pcasl_transformer_att_long(pcasl_in_att)
        vsasl_long_out = self.vsasl_transformer_att_long(vsasl_in_att)
        pcasl_short_out = self.pcasl_transformer_att_short(pcasl_in_att_short)
        vsasl_short_out = self.vsasl_transformer_att_short(vsasl_in_att_short)
        
        pcasl_fused_long = self.pcasl_to_vsasl_cross_attn(query=pcasl_long_out, key_value=vsasl_long_out)
        vsasl_fused_long = self.vsasl_to_pcasl_cross_attn(query=vsasl_long_out, key_value=pcasl_long_out)

        pcasl_feat_long, vsasl_feat_long = torch.mean(pcasl_fused_long, dim=1), torch.mean(vsasl_fused_long, dim=1)
        pcasl_feat_short, vsasl_feat_short = torch.mean(pcasl_short_out, dim=1), torch.mean(vsasl_short_out, dim=1)

        att_stream_features = torch.cat([pcasl_feat_long, vsasl_feat_long, pcasl_feat_short, vsasl_feat_short], dim=1)
        att_final_features = self.att_mlp(att_stream_features)
        att_mean, att_log_var = self.att_uncertainty(att_final_features)
        
        # Return final predictions AND the rough estimates for the auxiliary loss
        return cbf_mean, att_mean, cbf_log_var, att_log_var, cbf_rough, att_rough

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
    device = pred_cbf_norm.device
    pred_cbf = pred_cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    pred_att = pred_att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    plds = torch.tensor(model_params['pld_values'], device=device, dtype=torch.float32)

    pcasl_sig, vsasl_sig = _torch_physical_kinetic_model(pred_cbf, pred_att, plds, model_params)

    reconstructed_signal = torch.cat([pcasl_sig, vsasl_sig], dim=1)

    pcasl_mean = torch.tensor(norm_stats['pcasl_mean'], device=device, dtype=torch.float32)
    pcasl_std = torch.tensor(norm_stats['pcasl_std'], device=device, dtype=torch.float32)
    vsasl_mean = torch.tensor(norm_stats['vsasl_mean'], device=device, dtype=torch.float32)
    vsasl_std = torch.tensor(norm_stats['vsasl_std'], device=device, dtype=torch.float32)
    
    N = len(plds)
    pcasl_recon_norm = (reconstructed_signal[:, :N] - pcasl_mean) / (pcasl_std + 1e-6)
    vsasl_recon_norm = (reconstructed_signal[:, N:] - vsasl_mean) / (vsasl_std + 1e-6)
    
    reconstructed_signal_norm = torch.cat([pcasl_recon_norm, vsasl_recon_norm], dim=1)
    
    return reconstructed_signal_norm

class CustomLoss(nn.Module):
    """
    Custom loss: NLL with regression-focal loss on ATT and optional log_var regularization
    and a physics-informed (PINN) reconstruction loss.
    NEW: Includes an auxiliary loss to train the pre-estimator network.
    """
    
    def __init__(self, 
                 w_cbf: float = 1.0, 
                 w_att: float = 1.0, 
                 log_var_reg_lambda: float = 0.0,
                 focal_gamma: float = 1.5,
                 pinn_weight: float = 0.0,
                 model_params: Optional[Dict[str, Any]] = None,
                 att_epoch_weight_schedule: Optional[callable] = None,
                 pinn_att_weighting_sigma: float = 500.0,
                 pre_estimator_loss_weight: float = 0.5 # New weight for auxiliary loss
                ):
        super().__init__()
        self.w_cbf = w_cbf
        self.w_att = w_att
        self.log_var_reg_lambda = log_var_reg_lambda
        self.focal_gamma = focal_gamma
        self.pinn_weight = pinn_weight
        self.pre_estimator_loss_weight = pre_estimator_loss_weight
        self.model_params = model_params if model_params is not None else {}
        self.norm_stats = None
        self.att_epoch_weight_schedule = att_epoch_weight_schedule or (lambda _: 1.0)
        self.pinn_att_weighting_sigma = pinn_att_weighting_sigma
        self.mse_loss = nn.MSELoss()

    def forward(self,
                normalized_input_signal: torch.Tensor,
                cbf_pred_norm: torch.Tensor, att_pred_norm: torch.Tensor, 
                cbf_true_norm: torch.Tensor, att_true_norm: torch.Tensor, 
                cbf_log_var: torch.Tensor, att_log_var: torch.Tensor, 
                cbf_rough_physical: torch.Tensor, att_rough_physical: torch.Tensor,
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
            
        # --- Physics-Informed (PINN) Regularization ---
        pinn_loss = 0.0
        if self.pinn_weight > 0 and self.norm_stats and self.model_params and global_epoch > 10:
            reconstructed_signal_norm = torch_kinetic_model(cbf_pred_norm, att_pred_norm, self.norm_stats, self.model_params)
            num_raw_signal_feats = len(self.model_params.get('pld_values', [])) * 2
            input_signal_norm = normalized_input_signal[:, :num_raw_signal_feats]
            pinn_loss = self.mse_loss(reconstructed_signal_norm, input_signal_norm)

        # --- Auxiliary Loss for Pre-Estimator (in NORMALIZED space) ---
        pre_estimator_loss = 0.0
        if self.pre_estimator_loss_weight > 0 and self.norm_stats:
            # Normalize the pre-estimator's physical outputs to match the ground truth's scale
            cbf_rough_norm = (cbf_rough_physical - self.norm_stats['y_mean_cbf']) / (self.norm_stats['y_std_cbf'] + 1e-6)
            att_rough_norm = (att_rough_physical - self.norm_stats['y_mean_att']) / (self.norm_stats['y_std_att'] + 1e-6)
            
            loss_cbf_pre = self.mse_loss(cbf_rough_norm, cbf_true_norm)
            loss_att_pre = self.mse_loss(att_rough_norm, att_true_norm)
            pre_estimator_loss = loss_cbf_pre + loss_att_pre

        total_loss = total_param_loss + log_var_regularization + self.pinn_weight * pinn_loss + self.pre_estimator_loss_weight * pre_estimator_loss
        
        return total_loss
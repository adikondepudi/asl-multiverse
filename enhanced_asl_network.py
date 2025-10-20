# FILE: enhanced_asl_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
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
    
class AttentionPooling(nn.Module):
    """
    Attention-based pooling to create a learned weighted average of sequence features.
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        attn_weights = self.attention_net(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1) # Normalize weights across sequence
        # Weighted sum: (batch, 1, seq_len) @ (batch, seq_len, d_model) -> (batch, 1, d_model)
        pooled = torch.bmm(attn_weights.transpose(1, 2), x)
        return pooled.squeeze(1) # (batch, d_model)

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
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 log_var_min: Union[float, List[float]] = -7.0, 
                 log_var_max: Union[float, List[float]] = 7.0):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_var_raw = nn.Linear(input_dim, output_dim) # Predicts raw value for log_var

        # Register min/max as buffers so they move to the correct device and are saved with the state_dict
        if isinstance(log_var_min, list):
            assert len(log_var_min) == output_dim, "log_var_min list must match output_dim"
            self.register_buffer('log_var_min_val', torch.tensor(log_var_min, dtype=torch.float32))
        else:
            self.register_buffer('log_var_min_val', torch.tensor([log_var_min] * output_dim, dtype=torch.float32))

        if isinstance(log_var_max, list):
            assert len(log_var_max) == output_dim, "log_var_max list must match output_dim"
            self.register_buffer('log_var_max_val', torch.tensor(log_var_max, dtype=torch.float32))
        else:
            self.register_buffer('log_var_max_val', torch.tensor([log_var_max] * output_dim, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(x)
        raw_log_var = self.log_var_raw(x)
        
        # Scale tanh output to [log_var_min, log_var_max] for each dimension
        log_var_range = self.log_var_max_val - self.log_var_min_val
        log_var = self.log_var_min_val + (torch.tanh(raw_log_var) + 1.0) * 0.5 * log_var_range
        return mean, log_var

# --- NEW: MoE Gating Network ---
class GatingNetwork(nn.Module):
    """A simple MLP to produce expert weights from input features."""
    def __init__(self, input_dim: int, num_experts: int, dropout_rate: float):
        super().__init__()
        hidden_dim = (input_dim + num_experts) // 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a probability distribution over the experts."""
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

# --- NEW: Encapsulated Expert Module ---
class Expert(nn.Module):
    """An individual expert in the MoE head, containing a full prediction pipeline."""
    def __init__(self, input_dim: int, hidden_sizes: List[int], norm_type: str, dropout_rate: float,
                 log_var_att_min: float, log_var_att_max: float, log_var_cbf_min: float, log_var_cbf_max: float):
        super().__init__()
        # Helper function to get normalization layer
        def _get_norm_layer(size: int, norm_type: str) -> nn.Module:
            if norm_type == 'batch': return nn.BatchNorm1d(size)
            elif norm_type == 'layer': return nn.LayerNorm(size)
            else: return nn.BatchNorm1d(size)

        self.joint_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            _get_norm_layer(hidden_sizes[0], norm_type),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1])
        )
        self.att_head = UncertaintyHead(hidden_sizes[1], 1, log_var_min=log_var_att_min, log_var_max=log_var_att_max)
        self.cbf_head = UncertaintyHead(hidden_sizes[1], 1, log_var_min=log_var_cbf_min, log_var_max=log_var_cbf_max)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        joint_features = self.joint_mlp(x)
        cbf_mean, cbf_log_var = self.cbf_head(joint_features)
        att_mean, att_log_var = self.att_head(joint_features)
        return cbf_mean, att_mean, cbf_log_var, att_log_var

# --- NEW: Mixture of Experts Head ---
class MixtureOfExpertsHead(nn.Module):
    """Combines a gating network and multiple expert predictors."""
    def __init__(self, input_dim: int, hidden_sizes: List[int], norm_type: str, dropout_rate: float,
                 log_var_cbf_min: float, log_var_cbf_max: float, log_var_att_min: float, log_var_att_max: float,
                 num_experts: int, gating_dropout_rate: float):
        super().__init__()
        self.gating_network = GatingNetwork(input_dim, num_experts, gating_dropout_rate)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_sizes, norm_type, dropout_rate, 
                   log_var_att_min, log_var_att_max, log_var_cbf_min, log_var_cbf_max)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Get weights for each expert: [batch_size, num_experts]
        expert_weights = self.gating_network(x)

        # Get predictions from all experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Stack the outputs from each expert
        # e.g., cbf_means_stacked shape: [batch_size, num_experts, 1]
        cbf_means_stacked = torch.stack([out[0] for out in expert_outputs], dim=1)
        att_means_stacked = torch.stack([out[1] for out in expert_outputs], dim=1)
        cbf_log_vars_stacked = torch.stack([out[2] for out in expert_outputs], dim=1)
        att_log_vars_stacked = torch.stack([out[3] for out in expert_outputs], dim=1)

        # Calculate the weighted average of the predictions
        # Unsqueeze weights to [batch_size, 1, num_experts] for bmm
        weights = expert_weights.unsqueeze(1)
        
        final_cbf_mean = torch.bmm(weights, cbf_means_stacked).squeeze(1)
        final_att_mean = torch.bmm(weights, att_means_stacked).squeeze(1)
        final_cbf_log_var = torch.bmm(weights, cbf_log_vars_stacked).squeeze(1)
        final_att_log_var = torch.bmm(weights, att_log_vars_stacked).squeeze(1)
        
        return final_cbf_mean, final_att_mean, final_cbf_log_var, final_att_log_var

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

    B = pred_cbf.shape[0]
    plds_b = plds.unsqueeze(0).expand(B, -1) if B > 1 or plds.dim() == 1 else plds

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

# NEW: Physics Residual PINN helper function
def _calculate_pcasl_gkm_residual(pcasl_signal_curve, t_points, cbf_phys, att_phys, t1_tissue, t1_blood, labeling_duration):
    """
    Calculates the residual of the simplified 1-compartment GKM differential equation for PCASL.
    The equation is: dM(t)/dt = 2 * f_cbf * AIF(t) - (f_cbf/lambda + 1/T1_tissue) * M(t)
    All time units are in seconds.
    """
    t_points.requires_grad_(True)
    
    # Use autograd to get the time derivative of the signal curve
    dM_dt = torch.autograd.grad(
        outputs=pcasl_signal_curve,
        inputs=t_points,
        grad_outputs=torch.ones_like(pcasl_signal_curve),
        create_graph=True
    )[0]
    
    # Define the Arterial Input Function (AIF) for PCASL (boxcar)
    aif = torch.zeros_like(t_points)
    att_s = att_phys / 1000.0
    tau_s = labeling_duration / 1000.0
    
    # Bolus arrives at ATT, lasts for labeling_duration
    aif = torch.where((t_points > att_s) & (t_points <= att_s + tau_s), 1.0, 0.0)
    
    # The right-hand side of the differential equation
    lambda_blood = 0.9
    f_cbf_s = cbf_phys / 60.0 # to ml/g/s
    k_clearance = f_cbf_s / lambda_blood + 1.0 / (t1_tissue / 1000.0)
    
    rhs = 2 * f_cbf_s * aif - k_clearance * pcasl_signal_curve
    
    residual = dM_dt - rhs
    return residual

class DisentangledEncoder(nn.Module):
    """The feature extraction backbone for DisentangledASLNet."""
    def __init__(self, n_plds, dropout_rate, transformer_d_model_focused, transformer_nhead_model, transformer_nlayers_model, **kwargs):
        super().__init__()
        self.n_plds = n_plds
        self.num_shape_features = n_plds * 2
        self.num_engineered_features = 4  # TTP_p, TTP_v, CoM_p, CoM_v
        self.num_amplitude_features = 1
        
        # --- 1. SHAPE STREAM (Transformers for temporal features) ---
        self.att_d_model = transformer_d_model_focused
        self.pcasl_input_proj = nn.Linear(1, self.att_d_model)
        self.vsasl_input_proj = nn.Linear(1, self.att_d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(self.att_d_model, transformer_nhead_model, self.att_d_model * 2, dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, transformer_nlayers_model)
        
        # --- 2. AMPLITUDE STREAM (Simple MLP for energy mapping) ---
        self.amplitude_mlp = nn.Sequential(
            nn.Linear(self.num_amplitude_features + self.num_engineered_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # --- 3. DYNAMIC FUSION ENGINE (Cross-Attention) ---
        amplitude_feature_size = 64
        self.query_proj = nn.Linear(amplitude_feature_size, self.att_d_model)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.att_d_model, num_heads=transformer_nhead_model, 
            dropout=dropout_rate, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(self.att_d_model)
        self.fusion_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape_vector = x[:, :self.num_shape_features]
        engineered_features = x[:, self.num_shape_features : self.num_shape_features + self.num_engineered_features]
        amplitude_scalar = x[:, -self.num_amplitude_features:]

        # --- SHAPE STREAM ---
        pcasl_shape = shape_vector[:, :self.n_plds].unsqueeze(-1)
        vsasl_shape = shape_vector[:, self.n_plds:].unsqueeze(-1)
        pcasl_proj = self.pcasl_input_proj(pcasl_shape)
        vsasl_proj = self.vsasl_input_proj(vsasl_shape)
        pcasl_encoded = self.transformer_encoder(pcasl_proj)
        vsasl_encoded = self.transformer_encoder(vsasl_proj)
        
        # --- AMPLITUDE STREAM ---
        amplitude_input = torch.cat([amplitude_scalar, engineered_features], dim=1)
        amplitude_features = self.amplitude_mlp(amplitude_input)

        # --- DYNAMIC FUSION ENGINE ---
        shape_sequence = torch.cat([pcasl_encoded, vsasl_encoded], dim=1)
        query_proj = self.query_proj(amplitude_features)
        query = query_proj.unsqueeze(1)
        attn_output, _ = self.cross_attention(query=query, key=shape_sequence, value=shape_sequence)
        contextualized_query_unnorm = query + self.fusion_dropout(attn_output)
        contextualized_query = self.fusion_norm(contextualized_query_unnorm).squeeze(1)

        # Final feature vector for prediction, combining context-aware and original features
        fused_features = torch.cat([contextualized_query, amplitude_features], dim=1)
        return fused_features

class DisentangledASLNet(nn.Module):
    """
    DisentangledASLNet v5: Refactored architecture with a Mixture-of-Experts (MoE) head
    to replace the single regression head for improved specialization.
    """
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 n_plds: int = 6,
                 norm_type: str = 'batch',
                 log_var_cbf_min: float = 0.0,
                 log_var_cbf_max: float = 7.0,
                 log_var_att_min: float = 0.0,
                 log_var_att_max: float = 14.0,
                 moe: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__()
        
        self.encoder = DisentangledEncoder(n_plds=n_plds, **kwargs)
        
        # Fused features will be concatenation of attention output and original amplitude features
        fused_feature_size = self.encoder.att_d_model + 64 # From query and amplitude_features
        dropout_rate = kwargs.get('dropout_rate', 0.1)

        # --- MODIFIED: Replace single head with Mixture of Experts Head ---
        if moe and moe.get('num_experts', 0) > 0:
            self.moe_head = MixtureOfExpertsHead(
                input_dim=fused_feature_size,
                hidden_sizes=hidden_sizes,
                norm_type=norm_type,
                dropout_rate=dropout_rate,
                log_var_cbf_min=log_var_cbf_min,
                log_var_cbf_max=log_var_cbf_max,
                log_var_att_min=log_var_att_min,
                log_var_att_max=log_var_att_max,
                num_experts=moe['num_experts'],
                gating_dropout_rate=moe['gating_dropout_rate']
            )
        else: # Fallback to original single head if MoE is not configured
            self.joint_mlp = nn.Sequential(
                nn.Linear(fused_feature_size, hidden_sizes[0]),
                self._get_norm_layer(hidden_sizes[0], norm_type),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_sizes[0], hidden_sizes[1])
            )
            self.att_head = UncertaintyHead(hidden_sizes[1], 1, log_var_min=log_var_att_min, log_var_max=log_var_att_max)
            self.cbf_head = UncertaintyHead(hidden_sizes[1], 1, log_var_min=log_var_cbf_min, log_var_max=log_var_cbf_max)
            self.moe_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        fused_features = self.encoder(x)
        
        if self.moe_head:
            cbf_mean, att_mean, cbf_log_var, att_log_var = self.moe_head(fused_features)
        else: # Fallback for non-MoE configuration
            joint_features = self.joint_mlp(fused_features)
            cbf_mean, cbf_log_var = self.cbf_head(joint_features)
            att_mean, att_log_var = self.att_head(joint_features)

        return cbf_mean, att_mean, cbf_log_var, att_log_var, None, None

    def _get_norm_layer(self, size: int, norm_type: str) -> nn.Module:
        if norm_type == 'batch': return nn.BatchNorm1d(size)
        elif norm_type == 'layer': return nn.LayerNorm(size)
        else: return nn.BatchNorm1d(size)

    def get_head_parameters(self):
        """Returns all parameters related to the regression heads."""
        if self.moe_head:
            return list(self.moe_head.parameters())
        else:
            return list(self.joint_mlp.parameters()) + \
                   list(self.att_head.parameters()) + \
                   list(self.cbf_head.parameters())
    
    def freeze_encoder(self):
        """Freezes all parameters in the encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Sets requires_grad=True for all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

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
    Custom loss for v3: Unified loss with UW, Reconstruction PINN, and a true
    Physics Residual PINN for guaranteed physical plausibility.
    """
    
    def __init__(self, 
                 w_cbf: float = 1.0, 
                 w_att: float = 1.0, 
                 log_var_reg_lambda: float = 0.0,
                 pinn_weight: float = 0.0,
                 residual_pinn_weight: float = 0.0, # NEW: For GKM residual loss
                 model_params: Optional[Dict[str, Any]] = None,
                ):
        super().__init__()
        self.w_cbf = w_cbf
        self.w_att = w_att
        self.log_var_reg_lambda = log_var_reg_lambda
        self.pinn_weight = pinn_weight
        self.residual_pinn_weight = residual_pinn_weight # NEW
        self.model_params = model_params if model_params is not None else {}
        self.norm_stats = None
        self.mse_loss = nn.MSELoss()

    def forward(self,
                normalized_input_signal: torch.Tensor,
                cbf_pred_norm: torch.Tensor, att_pred_norm: torch.Tensor, 
                cbf_true_norm: torch.Tensor, att_true_norm: torch.Tensor, 
                cbf_log_var: torch.Tensor, att_log_var: torch.Tensor, 
                cbf_rough_physical: Optional[torch.Tensor], att_rough_physical: Optional[torch.Tensor],
                global_epoch: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # cbf_log_var and att_log_var tensors are now used directly
        # Prevents both exploding precision and pathologically negative loss.
        # cbf_log_var = torch.clamp(cbf_log_var, min=-6.0, max=10.0)
        # att_log_var = torch.clamp(att_log_var, min=-6.0, max=10.0)

        # --- Standard NLL calculation (Aleatoric Uncertainty Loss) ---
        cbf_precision = torch.exp(-cbf_log_var)
        att_precision = torch.exp(-att_log_var)

        cbf_nll_loss = 0.5 * (cbf_precision * (cbf_pred_norm - cbf_true_norm)**2 + cbf_log_var)
        att_nll_loss = 0.5 * (att_precision * (att_pred_norm - att_true_norm)**2 + att_log_var)

        # --- Apply weights and combine losses ---
        weighted_cbf_loss = self.w_cbf * cbf_nll_loss
        weighted_att_loss = self.w_att * att_nll_loss
        
        # MODIFIED: Get handle on unreduced loss for OHEM
        combined_nll_loss = weighted_cbf_loss + weighted_att_loss
        total_param_loss = torch.mean(combined_nll_loss)
        
        # --- Optional regularization on the magnitude of predicted uncertainty ---
        log_var_regularization = torch.tensor(0.0, device=total_param_loss.device)
        if self.log_var_reg_lambda > 0:
            log_var_regularization = self.log_var_reg_lambda * (torch.mean(cbf_log_var**2) + torch.mean(att_log_var**2))
            
        # --- Physics-Informed (PINN) Regularization ---
        recon_loss = torch.tensor(0.0, device=total_param_loss.device)
        residual_loss = torch.tensor(0.0, device=total_param_loss.device)
        
        if (self.pinn_weight > 0 or self.residual_pinn_weight > 0) and self.norm_stats and self.model_params:
            num_raw_signal_feats = len(self.model_params.get('pld_values', [])) * 2
            
            # PINN 1: Reconstruction Consistency
            if self.pinn_weight > 0:
                reconstructed_signal_norm = torch_kinetic_model(cbf_pred_norm, att_pred_norm, self.norm_stats, self.model_params)
                input_signal_norm = normalized_input_signal[:, :num_raw_signal_feats]
                recon_loss = self.mse_loss(reconstructed_signal_norm, input_signal_norm)

            # PINN 2: GKM Physics Residual (NEW)
            if self.residual_pinn_weight > 0 and 'T1_tissue' in self.model_params:
                # 1. Get network's predicted parameters in physical units
                pred_cbf_phys = cbf_pred_norm * self.norm_stats['y_std_cbf'] + self.norm_stats['y_mean_cbf']
                pred_att_phys = att_pred_norm * self.norm_stats['y_std_att'] + self.norm_stats['y_mean_att']

                # 2. Create dense time points ("collocation points") in seconds
                max_pld = self.model_params['pld_values'][-1]
                t_collocation_ms = torch.linspace(0, max_pld * 1.1, 100, device=cbf_pred_norm.device)
                t_collocation_s = t_collocation_ms.unsqueeze(0).expand(cbf_pred_norm.shape[0], -1) / 1000.0

                # 3. Generate the continuous signal curve over the collocation points
                pcasl_curve_continuous, _ = _torch_physical_kinetic_model(pred_cbf_phys, pred_att_phys, t_collocation_ms, self.model_params)
                
                # 4. Calculate the physics residual
                gkm_residual = _calculate_pcasl_gkm_residual(
                    pcasl_signal_curve=pcasl_curve_continuous, t_points=t_collocation_s,
                    cbf_phys=pred_cbf_phys, att_phys=pred_att_phys,
                    t1_tissue=self.model_params['T1_tissue'],
                    t1_blood=self.model_params['T1_artery'],
                    labeling_duration=self.model_params['T_tau']
                )
                residual_loss = torch.mean(gkm_residual**2)
                
        # --- Final Unified Loss ---
        total_loss = total_param_loss + log_var_regularization + self.pinn_weight * recon_loss + self.residual_pinn_weight * residual_loss
        
        loss_components = {
            'param_nll_loss': total_param_loss,
            'log_var_reg_loss': log_var_regularization,
            'pinn_recon_loss': recon_loss,
            'pinn_residual_loss': residual_loss, # NEW
            'pre_estimator_loss': torch.tensor(0.0)
        }
        # MODIFIED: Pass unreduced loss (NOT detached) to trainer for OHEM
        loss_components['unreduced_loss'] = combined_nll_loss
        
        return total_loss, loss_components
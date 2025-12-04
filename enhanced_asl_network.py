# FILE: enhanced_asl_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
import math

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
        attn_weights = self.attention_net(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = torch.bmm(attn_weights.transpose(1, 2), x)
        return pooled.squeeze(1)

class UncertaintyHead(nn.Module):
    """Uncertainty estimation head with bounded log_var output."""
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 log_var_min: Union[float, List[float]] = -7.0, 
                 log_var_max: Union[float, List[float]] = 7.0):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_var_raw = nn.Linear(input_dim, output_dim)

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
        log_var_range = self.log_var_max_val - self.log_var_min_val
        log_var = self.log_var_min_val + (torch.tanh(raw_log_var) + 1.0) * 0.5 * log_var_range
        return mean, log_var

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
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

class Expert(nn.Module):
    """An individual expert in the MoE head, containing a full prediction pipeline."""
    def __init__(self, input_dim: int, hidden_sizes: List[int], norm_type: str, dropout_rate: float,
                 log_var_att_min: float, log_var_att_max: float, log_var_cbf_min: float, log_var_cbf_max: float):
        super().__init__()
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
        expert_weights = self.gating_network(x)
        expert_outputs = [expert(x) for expert in self.experts]
        cbf_means_stacked = torch.stack([out[0] for out in expert_outputs], dim=1)
        att_means_stacked = torch.stack([out[1] for out in expert_outputs], dim=1)
        cbf_log_vars_stacked = torch.stack([out[2] for out in expert_outputs], dim=1)
        att_log_vars_stacked = torch.stack([out[3] for out in expert_outputs], dim=1)
        weights = expert_weights.unsqueeze(1)
        final_cbf_mean = torch.bmm(weights, cbf_means_stacked).squeeze(1)
        final_att_mean = torch.bmm(weights, att_means_stacked).squeeze(1)
        final_cbf_log_var = torch.bmm(weights, cbf_log_vars_stacked).squeeze(1)
        final_att_log_var = torch.bmm(weights, att_log_vars_stacked).squeeze(1)
        return final_cbf_mean, final_att_mean, final_cbf_log_var, final_att_log_var

class SignalDecoder(nn.Module):
    """
    A simple MLP decoder to reconstruct a clean ASL signal from latent features.
    Used in Stage 1 for self-supervised denoising.
    """
    def __init__(self, latent_dim: int, output_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        input_d = latent_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(input_d, h_dim))
            layers.append(nn.ReLU())
            input_d = h_dim
        layers.append(nn.Linear(input_d, output_dim))
        self.decoder_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_mlp(x)

class Conv1DFeatureExtractor(nn.Module):
    """
    A dedicated 1D-ConvNet to extract local, shape-based features from the ASL signal.
    Crucially includes BatchNorm1d for per-batch stability.
    """
    def __init__(self, in_channels: int, feature_dim: int, dropout_rate: float):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def __init__(self, in_channels: int, conditioning_dim: int):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(conditioning_dim, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels * 2)
        )
        self.in_channels = in_channels

    def forward(self, features: torch.Tensor, conditioning_vector: torch.Tensor) -> torch.Tensor:
        params = self.generator(conditioning_vector)
        gamma = params[:, :self.in_channels].unsqueeze(-1)
        beta = params[:, self.in_channels:].unsqueeze(-1)
        return gamma * features + beta

class PhysicsInformedASLProcessor(nn.Module):
    """
    V6 Physics-Conditioned encoder. It uses a dual-stream Conv1D architecture to process
    per-curve normalized shape vectors for PCASL and VSASL independently. It then uses
    separate FiLM layers to inject rich physical context (amplitudes, timing features)
    into each stream before fusing them for the prediction head.
    """
    def __init__(self, n_plds: int, feature_dim: int, nhead: int, dropout_rate: float, num_scalar_features: int = 11, **kwargs):
        super().__init__()
        self.n_plds = n_plds
        self.num_scalar_features = num_scalar_features

        # Two separate towers for the disentangled shape vectors
        self.pcasl_tower = Conv1DFeatureExtractor(in_channels=1, feature_dim=feature_dim, dropout_rate=dropout_rate)
        self.vsasl_tower = Conv1DFeatureExtractor(in_channels=1, feature_dim=feature_dim, dropout_rate=dropout_rate)

        # Two separate FiLM layers to modulate each stream independently
        self.pcasl_film = FiLMLayer(in_channels=feature_dim, conditioning_dim=self.num_scalar_features)
        self.vsasl_film = FiLMLayer(in_channels=feature_dim, conditioning_dim=self.num_scalar_features)
        
        self.attention_pooling = AttentionPooling(d_model=feature_dim)
        
        # Fusion MLP combines information from both processed streams and the original scalar features
        fusion_input_dim = (feature_dim * 2) + self.num_scalar_features
        self.final_fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x structure:
        # [pcasl_shape (6), vsasl_shape(6), standardized_scalars (11)]
        
        pcasl_shape_input = x[:, :self.n_plds].unsqueeze(1)
        vsasl_shape_input = x[:, self.n_plds:self.n_plds * 2].unsqueeze(1)
        
        # Extract the scalar features (including the appended T1)
        scalar_features = x[:, self.n_plds * 2:]

        # 1. Process shapes independently
        pcasl_features = self.pcasl_tower(pcasl_shape_input) # -> (batch, feature_dim, n_plds)
        vsasl_features = self.vsasl_tower(vsasl_shape_input) # -> (batch, feature_dim, n_plds)

        # 2. Condition each stream's features with the full scalar context vector
        conditioned_pcasl = self.pcasl_film(pcasl_features, scalar_features)
        conditioned_vsasl = self.vsasl_film(vsasl_features, scalar_features)

        # 3. Pool each conditioned sequence into a single context vector
        pcasl_seq = conditioned_pcasl.transpose(1, 2) # -> (batch, n_plds, feature_dim)
        vsasl_seq = conditioned_vsasl.transpose(1, 2) # -> (batch, n_plds, feature_dim)
        pcasl_context = self.attention_pooling(pcasl_seq) # -> (batch, feature_dim)
        vsasl_context = self.attention_pooling(vsasl_seq) # -> (batch, feature_dim)

        # 4. Fuse pooled contexts with original scalar features for the head
        final_input_to_head = torch.cat([pcasl_context, vsasl_context, scalar_features], dim=1)
        final_output_vector = self.final_fusion_mlp(final_input_to_head)
        
        return final_output_vector

class DisentangledASLNet(nn.Module):
    """
    Main network class. It uses a specified encoder (like the V5 PhysicsInformedASLProcessor)
    and then applies either a denoising decoder (Stage 1) or a regression head (Stage 2).
    """
    def __init__(self, 
                 mode: str,
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 n_plds: int = 6,
                 norm_type: str = 'batch',
                 log_var_cbf_min: float = 0.0,
                 log_var_cbf_max: float = 7.0,
                 log_var_att_min: float = 0.0,
                 log_var_att_max: float = 14.0,
                 moe: Optional[Dict[str, Any]] = None,
                 encoder_type: str = 'physics_processor',
                 num_scalar_features: int = 11,
                 active_features_list: Optional[List[str]] = None,
                 **kwargs):
        super().__init__()
        
        self.mode = mode
        self.encoder_frozen = False
        
        # DYNAMIC CALCULATION of scalar dimension from active_features_list
        # This prevents shape mismatch errors when ablating different feature combinations
        if active_features_list is not None:
            scalar_dim = 0
            for feat in active_features_list:
                if feat in ['mean', 'std', 'ttp', 'com', 'peak']: 
                    scalar_dim += 2
                elif feat in ['t1_artery', 'z_coord']: 
                    scalar_dim += 1
            num_scalar_features = scalar_dim
        
        if encoder_type.lower() == 'physics_processor':
            self.encoder = PhysicsInformedASLProcessor(
                n_plds=n_plds, 
                feature_dim=kwargs.get('transformer_d_model_focused'),
                nhead=kwargs.get('transformer_nhead_model'),
                dropout_rate=kwargs.get('dropout_rate'),
                num_scalar_features=num_scalar_features
            )
            fused_feature_size = 256 # Output of the final_fusion_mlp
        else:
            raise ValueError(f"Unknown encoder_type: '{encoder_type}'. Only 'physics_processor' is supported in this version.")
        
        dropout_rate = kwargs.get('dropout_rate', 0.1)

        if self.mode == 'denoising':
            self.decoder = SignalDecoder(
                latent_dim=fused_feature_size,
                output_dim=n_plds * 2,
                hidden_sizes=hidden_sizes
            )
        elif self.mode == 'regression':
            if moe and moe.get('num_experts', 0) > 0:
                self.head = MixtureOfExpertsHead(
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
            else:
                joint_mlp = nn.Sequential(
                    nn.Linear(fused_feature_size, hidden_sizes[0]),
                    self._get_norm_layer(hidden_sizes[0], norm_type),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_sizes[0], hidden_sizes[1])
                )
                att_head = UncertaintyHead(hidden_sizes[1], 1, log_var_min=log_var_att_min, log_var_max=log_var_att_max)
                cbf_head = UncertaintyHead(hidden_sizes[1], 1, log_var_min=log_var_cbf_min, log_var_max=log_var_cbf_max)
                self.head = nn.ModuleDict({'joint_mlp': joint_mlp, 'att_head': att_head, 'cbf_head': cbf_head})
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'denoising' or 'regression'.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        fused_features = self.encoder(x)
        
        if self.mode == 'denoising':
            reconstructed_signal = self.decoder(fused_features)
            return (reconstructed_signal,)
        
        elif self.mode == 'regression':
            if isinstance(self.head, MixtureOfExpertsHead):
                cbf_mean, att_mean, cbf_log_var, att_log_var = self.head(fused_features)
            else:
                joint_features = self.head['joint_mlp'](fused_features)
                cbf_mean, cbf_log_var = self.head['cbf_head'](joint_features)
                att_mean, att_log_var = self.head['att_head'](joint_features)
            return cbf_mean, att_mean, cbf_log_var, att_log_var, None, None

    def _get_norm_layer(self, size: int, norm_type: str) -> nn.Module:
        if norm_type == 'batch': return nn.BatchNorm1d(size)
        elif norm_type == 'layer': return nn.LayerNorm(size)
        else: return nn.BatchNorm1d(size)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

class CustomLoss(nn.Module):
    """
    Custom loss for the two-stage training strategy.
    - Stage 1: Self-supervised denoising (MSE loss).
    - Stage 2: Supervised regression (NLL loss).
    """
    def __init__(self, 
                 training_stage: int,
                 w_cbf: float = 1.0, 
                 w_att: float = 1.0,
                 log_var_reg_lambda: float = 0.0,
                 mse_weight: float = 0.0):
        super().__init__()
        if training_stage not in [1, 2]:
            raise ValueError("training_stage must be 1 or 2.")
        self.training_stage = training_stage
        self.w_cbf = w_cbf
        self.w_att = w_att
        self.log_var_reg_lambda = log_var_reg_lambda
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()

    def forward(self,
                model_outputs: Tuple,
                targets: torch.Tensor,
                global_epoch: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if self.training_stage == 1:
            reconstructed_signal = model_outputs[0]
            clean_signal_target = targets
            denoising_loss = self.mse_loss(reconstructed_signal, clean_signal_target)
            return denoising_loss, {'denoising_loss': denoising_loss}

        elif self.training_stage == 2:
            cbf_pred_norm, att_pred_norm, cbf_log_var, att_log_var, _, _ = model_outputs
            cbf_true_norm, att_true_norm = targets[:, 0:1], targets[:, 1:2]
            
            cbf_precision = torch.exp(-cbf_log_var)
            att_precision = torch.exp(-att_log_var)

            cbf_nll_loss = 0.5 * (cbf_precision * (cbf_pred_norm - cbf_true_norm)**2 + cbf_log_var)
            att_nll_loss = 0.5 * (att_precision * (att_pred_norm - att_true_norm)**2 + att_log_var)

            weighted_cbf_loss = self.w_cbf * cbf_nll_loss
            weighted_att_loss = self.w_att * att_nll_loss
            
            combined_nll_loss = weighted_cbf_loss + weighted_att_loss
            total_param_loss = torch.mean(combined_nll_loss)
            
            log_var_regularization = torch.tensor(0.0, device=total_param_loss.device)
            if self.log_var_reg_lambda > 0:
                log_var_regularization = self.log_var_reg_lambda * (torch.mean(cbf_log_var**2) + torch.mean(att_log_var**2))
            
            # --- NEW LOGIC: Add MSE Component if mse_weight > 0 ---
            mse_component = torch.tensor(0.0, device=total_param_loss.device)
            if self.mse_weight > 0:
                # Calculate simple MSE for CBF and ATT
                cbf_mse = F.mse_loss(cbf_pred_norm, cbf_true_norm)
                att_mse = F.mse_loss(att_pred_norm, att_true_norm)
                mse_component = self.mse_weight * (cbf_mse + att_mse)

            total_loss = total_param_loss + log_var_regularization + mse_component
            
            loss_components = {
                'param_nll_loss': total_param_loss,
                'log_var_reg_loss': log_var_regularization,
                'param_mse_loss': mse_component,
                'unreduced_loss': combined_nll_loss
            }
            return total_loss, loss_components
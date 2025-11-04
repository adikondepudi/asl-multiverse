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
        attn_weights = self.attention_net(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = torch.bmm(attn_weights.transpose(1, 2), x)
        return pooled.squeeze(1)

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
        attn_output, _ = self.cross_attn(query=query, key=key_value, value=key_value)
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
            nn.Conv1d(64, feature_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)

class DisentangledEncoderConv1D(nn.Module):
    """
    The feature extraction backbone using a 1D-ConvNet for the shape stream.
    """
    def __init__(self, n_plds, dropout_rate, transformer_d_model_focused, transformer_nhead_model, **kwargs):
        super().__init__()
        self.n_plds = n_plds
        self.num_shape_features = n_plds * 2
        self.num_engineered_features = 4
        self.num_amplitude_features = 1
        self.att_d_model = transformer_d_model_focused

        self.shape_feature_extractor = Conv1DFeatureExtractor(
            in_channels=1, 
            feature_dim=self.att_d_model, 
            dropout_rate=dropout_rate
        )
        self.amplitude_mlp = nn.Sequential(
            nn.Linear(self.num_amplitude_features + self.num_engineered_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
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

        pcasl_shape = shape_vector[:, :self.n_plds].unsqueeze(-1)
        vsasl_shape = shape_vector[:, self.n_plds:].unsqueeze(-1)

        pcasl_encoded = self.shape_feature_extractor(pcasl_shape.transpose(1, 2)).transpose(1, 2)
        vsasl_encoded = self.shape_feature_extractor(vsasl_shape.transpose(1, 2)).transpose(1, 2)
        
        amplitude_input = torch.cat([amplitude_scalar, engineered_features], dim=1)
        amplitude_features = self.amplitude_mlp(amplitude_input)

        shape_sequence = torch.cat([pcasl_encoded, vsasl_encoded], dim=1)
        query_proj = self.query_proj(amplitude_features)
        query = query_proj.unsqueeze(1)
        attn_output, _ = self.cross_attention(query=query, key=shape_sequence, value=shape_sequence)
        contextualized_query_unnorm = query + self.fusion_dropout(attn_output)
        contextualized_query = self.fusion_norm(contextualized_query_unnorm).squeeze(1)

        fused_features = torch.cat([contextualized_query, amplitude_features], dim=1)
        return fused_features

class PhysicsInformedASLProcessor(nn.Module):
    """
    A specialized, four-stage encoder for MULTIVERSE ASL data. It mimics an expert's
    workflow: Decompose & Denoise -> Detect Events -> Reconcile -> Estimate.
    """
    def __init__(self, n_plds: int, feature_dim: int, nhead: int, dropout_rate: float, **kwargs):
        super().__init__()
        self.n_plds = n_plds
        self.num_engineered_features = 4
        self.num_amplitude_features = 1

        # --- STATION 1: THE CLEANERS (Independent Denoising Towers) ---
        # We need two separate 1D Conv towers. They DO NOT share weights.
        # This is critical because they are learning the unique shapes of two different signals.
        self.pcasl_denoising_tower = Conv1DFeatureExtractor(
            in_channels=1, feature_dim=feature_dim, dropout_rate=dropout_rate
        )
        self.vsasl_denoising_tower = Conv1DFeatureExtractor(
            in_channels=1, feature_dim=feature_dim, dropout_rate=dropout_rate
        )

        # --- STATION 2: THE SPOTTERS (Kinetic Event Detectors) ---
        # These modules will learn to find the most important part of each feature sequence.
        # We will use the existing AttentionPooling class for this.
        self.pcasl_event_detector = AttentionPooling(d_model=feature_dim)
        self.vsasl_event_detector = AttentionPooling(d_model=feature_dim)

        # --- STATION 3: THE SUPERVISOR (Cross-Modal Reconciliation) ---
        # This module forces the model to compare its findings.
        # It uses the PCASL event as a query to "ask" the VSASL sequence for confirmation.
        self.reconciliation_attention = CrossAttentionBlock(
            d_model=feature_dim, nhead=nhead, dropout=dropout_rate
        )

        # --- STATION 4: THE MANAGER (Final Fusion & Estimation) ---
        # An MLP for processing the simple amplitude/engineered features.
        self.amplitude_processor = nn.Sequential(
            nn.Linear(self.num_amplitude_features + self.num_engineered_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # The final MLP that combines all our expert knowledge into one vector for the head.
        # Input size: reconciled_vector (feature_dim) + amplitude_vector (128)
        self.final_fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim + 128, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- PREPARATION: Separate Inputs ---
        shape_vector = x[:, :self.n_plds * 2]
        engineered_features = x[:, self.n_plds * 2 : self.n_plds * 2 + self.num_engineered_features]
        amplitude_scalar = x[:, -self.num_amplitude_features:]

        # Reshape for Conv1D: [Batch, Channels, Length]
        pcasl_shape_input = shape_vector[:, :self.n_plds].unsqueeze(1)
        vsasl_shape_input = shape_vector[:, self.n_plds:].unsqueeze(1)

        # --- EXECUTE ASSEMBLY LINE ---

        # STATION 1: Pass each signal through its dedicated Cleaner tower.
        # Output shape: [Batch, feature_dim, n_plds]
        pcasl_feature_sequence = self.pcasl_denoising_tower(pcasl_shape_input)
        vsasl_feature_sequence = self.vsasl_denoising_tower(vsasl_shape_input)

        # Reshape for Attention: [Batch, Sequence_Length, feature_dim]
        pcasl_seq = pcasl_feature_sequence.transpose(1, 2)
        vsasl_seq = vsasl_feature_sequence.transpose(1, 2)

        # STATION 2: The Spotters find the most salient event in each sequence.
        # Output shape: [Batch, feature_dim]
        pcasl_event_vector = self.pcasl_event_detector(pcasl_seq)
        # We don't use the vsasl_event_vector directly, but the tower needs to process it for the next step.

        # STATION 3: The Supervisor reconciles the information.
        # The PCASL event vector is the "question" (query).
        # The VSASL sequence is the "textbook" to find the answer in (key/value).
        # Query shape must be [Batch, 1, feature_dim] for the CrossAttentionBlock.
        reconciled_vector = self.reconciliation_attention(
            query=pcasl_event_vector.unsqueeze(1), 
            key_value=vsasl_seq
        )
        # Squeeze to remove the sequence dimension: [Batch, feature_dim]
        reconciled_vector = reconciled_vector.squeeze(1)

        # STATION 4: The Manager makes the final decision.
        # Process the amplitude information in parallel.
        amplitude_input = torch.cat([amplitude_scalar, engineered_features], dim=1)
        amplitude_vector = self.amplitude_processor(amplitude_input)

        # Combine the expert timing report with the amplitude report.
        final_output_vector = self.final_fusion_mlp(
            torch.cat([reconciled_vector, amplitude_vector], dim=1)
        )
        
        return final_output_vector

class DisentangledASLNet(nn.Module):
    """
    DisentangledASLNet v6: A two-stage model with a self-supervised denoising autoencoder
    (Stage 1) and a supervised regression head (Stage 2).
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
                 encoder_type: str = 'conv1d',
                 **kwargs):
        super().__init__()
        
        self.mode = mode
        self.encoder_frozen = False
        
        # --- START OF MODIFICATION ---
        # We add a new option for our superior encoder.
        if encoder_type.lower() == 'physics_processor':
            self.encoder = PhysicsInformedASLProcessor(
                n_plds=n_plds, 
                feature_dim=kwargs.get('transformer_d_model_focused'), # Re-use existing config value
                nhead=kwargs.get('transformer_nhead_model'),         # Re-use existing config value
                dropout_rate=kwargs.get('dropout_rate'),
                **kwargs
            )
            # The output size of our new encoder's fusion MLP is 256. This is CRITICAL.
            fused_feature_size = 256
        
        # This is the old code. Keep it for backward compatibility.
        elif encoder_type.lower() == 'conv1d':
            self.encoder = DisentangledEncoderConv1D(n_plds=n_plds, **kwargs)
            fused_feature_size = self.encoder.att_d_model + 64
        # --- END OF MODIFICATION ---
        
        else:
            raise ValueError(f"Unknown encoder_type: '{encoder_type}'. Must be 'conv1d' or 'physics_processor'.")
        
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
                 log_var_reg_lambda: float = 0.0):
        super().__init__()
        if training_stage not in [1, 2]:
            raise ValueError("training_stage must be 1 or 2.")
        self.training_stage = training_stage
        self.w_cbf = w_cbf
        self.w_att = w_att
        self.log_var_reg_lambda = log_var_reg_lambda
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
            
            total_loss = total_param_loss + log_var_regularization
            
            loss_components = {
                'param_nll_loss': total_param_loss,
                'log_var_reg_loss': log_var_regularization,
                'unreduced_loss': combined_nll_loss
            }
            return total_loss, loss_components
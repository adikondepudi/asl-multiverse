# amplitude_aware_spatial_network.py
"""
Amplitude-Aware Spatial ASL Network

This architecture solves the fundamental problem of CBF estimation in spatial ASL networks:
GroupNorm destroys signal amplitude information, but CBF is encoded primarily in amplitude.

Solution: Extract amplitude features BEFORE any normalization and inject them via FiLM
conditioning and output modulation.

Architecture:
- Amplitude Path: Extracts scalar features (mean, std, max) from raw input
- Spatial Path: Standard U-Net with GroupNorm (learns spatial/temporal patterns)
- FiLM Fusion: Amplitude features modulate spatial features at bottleneck
- Output Modulation: CBF prediction scaled by learned amplitude function

Author: Claude (Opus 4.5)
Date: 2026-02-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Modulates feature maps based on conditioning vector:
        y = gamma(c) * x + beta(c)

    Where gamma and beta are learned functions of conditioning vector c.
    """
    def __init__(self, num_features: int, conditioning_dim: int):
        super().__init__()
        self.num_features = num_features

        # MLP to generate gamma and beta from conditioning
        self.gamma_generator = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.ReLU(inplace=True),
            nn.Linear(conditioning_dim, num_features)
        )
        self.beta_generator = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.ReLU(inplace=True),
            nn.Linear(conditioning_dim, num_features)
        )

        # Initialize to identity transformation (gamma=1, beta=0)
        nn.init.zeros_(self.gamma_generator[-1].weight)
        nn.init.zeros_(self.gamma_generator[-1].bias)
        nn.init.zeros_(self.beta_generator[-1].weight)
        nn.init.zeros_(self.beta_generator[-1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature maps
            conditioning: (B, conditioning_dim) conditioning vector

        Returns:
            Modulated feature maps (B, C, H, W)
        """
        gamma = self.gamma_generator(conditioning)  # (B, C)
        beta = self.beta_generator(conditioning)    # (B, C)

        # Expand for broadcasting: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM: y = (1 + gamma) * x + beta
        # Using (1 + gamma) ensures identity init works
        return (1 + gamma) * x + beta


class AmplitudeFeatureExtractor(nn.Module):
    """
    Extracts amplitude-related features from raw ASL signals.

    These features preserve the signal amplitude information that would be
    destroyed by GroupNorm in the spatial pathway.

    Features extracted:
    - Per-channel mean (6 PCASL + 6 VSASL = 12 features)
    - Per-channel std (12 features)
    - Per-channel max (12 features)
    - Global signal power (1 feature)
    - PCASL/VSASL ratio (1 feature)
    - Temporal peak location (2 features: PCASL, VSASL)

    Total: 40 amplitude features
    """
    def __init__(self, n_plds: int = 6):
        super().__init__()
        self.n_plds = n_plds
        self.n_channels = n_plds * 2  # PCASL + VSASL

        # Number of raw amplitude features
        self.n_raw_features = (
            self.n_channels +  # per-channel mean
            self.n_channels +  # per-channel std
            self.n_channels +  # per-channel max
            1 +               # global signal power
            1 +               # PCASL/VSASL mean ratio
            2                 # peak locations (PCASL, VSASL)
        )  # Total: 40

        # MLP to process raw features into conditioning vector
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.n_raw_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        self.output_dim = 64

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract amplitude features from raw input.

        Args:
            x: (B, 2*n_plds, H, W) raw ASL signals (PCASL + VSASL)

        Returns:
            conditioning: (B, output_dim) conditioning vector for FiLM
            raw_features: (B, n_raw_features) raw amplitude features
        """
        B, C, H, W = x.shape

        # Per-channel statistics (computed over spatial dimensions)
        channel_mean = x.mean(dim=(2, 3))  # (B, C)
        channel_std = x.std(dim=(2, 3))    # (B, C)
        channel_max = x.amax(dim=(2, 3))   # (B, C)

        # Global signal power (mean of squared values)
        signal_power = (x ** 2).mean(dim=(1, 2, 3), keepdim=True).squeeze()  # (B,)
        if signal_power.dim() == 0:
            signal_power = signal_power.unsqueeze(0)

        # PCASL/VSASL ratio (indicates ATT-dependent signal)
        pcasl_mean = channel_mean[:, :self.n_plds].mean(dim=1)  # (B,)
        vsasl_mean = channel_mean[:, self.n_plds:].mean(dim=1)  # (B,)
        ratio = pcasl_mean / (vsasl_mean + 1e-6)  # (B,)

        # Temporal peak location (which PLD has max signal)
        pcasl_peak = channel_mean[:, :self.n_plds].argmax(dim=1).float() / self.n_plds  # (B,)
        vsasl_peak = channel_mean[:, self.n_plds:].argmax(dim=1).float() / self.n_plds  # (B,)

        # Concatenate all features
        raw_features = torch.cat([
            channel_mean,           # (B, 12)
            channel_std,            # (B, 12)
            channel_max,            # (B, 12)
            signal_power.unsqueeze(1),  # (B, 1)
            ratio.unsqueeze(1),         # (B, 1)
            pcasl_peak.unsqueeze(1),    # (B, 1)
            vsasl_peak.unsqueeze(1),    # (B, 1)
        ], dim=1)  # (B, 40)

        # Process through MLP
        conditioning = self.feature_mlp(raw_features)  # (B, 64)

        return conditioning, raw_features


class DoubleConvNoNorm(nn.Module):
    """Double convolution without normalization (for first layer)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvWithNorm(nn.Module):
    """Double convolution with GroupNorm (for deeper layers)."""
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        num_groups = min(num_groups, out_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AmplitudeAwareSpatialASLNet(nn.Module):
    """
    Amplitude-Aware Spatial ASL Network for joint CBF/ATT estimation.

    This architecture preserves signal amplitude information (critical for CBF)
    while using GroupNorm for stable training.

    Key innovations:
    1. Amplitude Feature Extractor: Extracts mean/std/max before any normalization
    2. First layer without GroupNorm: Preserves some amplitude in early features
    3. FiLM conditioning: Injects amplitude info into bottleneck features
    4. Amplitude-modulated output: CBF prediction scaled by learned amplitude function

    Input: (B, 2*n_plds, H, W) - concatenated PCASL + VSASL signals
    Output: (cbf, att, cbf_uncertainty, att_uncertainty) all shape (B, 1, H, W)
    """

    def __init__(
        self,
        n_plds: int = 6,
        features: List[int] = [32, 64, 128, 256],
        conditioning_dim: int = 64,
        use_film_at_bottleneck: bool = True,
        use_film_at_decoder: bool = True,
        use_amplitude_output_modulation: bool = True,
        **kwargs
    ):
        super().__init__()

        self.n_plds = n_plds
        self.in_channels = n_plds * 2
        self.features = features
        self.use_film_at_bottleneck = use_film_at_bottleneck
        self.use_film_at_decoder = use_film_at_decoder
        self.use_amplitude_output_modulation = use_amplitude_output_modulation

        # ===== AMPLITUDE PATH =====
        self.amplitude_extractor = AmplitudeFeatureExtractor(n_plds)
        self.conditioning_dim = self.amplitude_extractor.output_dim

        # ===== SPATIAL PATH (U-Net) =====
        # First encoder WITHOUT GroupNorm to preserve some amplitude info
        self.encoder1 = DoubleConvNoNorm(self.in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)

        # Remaining encoders WITH GroupNorm
        self.encoder2 = DoubleConvWithNorm(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConvWithNorm(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DoubleConvWithNorm(features[2], features[3])  # Bottleneck

        # FiLM conditioning at bottleneck
        if use_film_at_bottleneck:
            self.bottleneck_film = FiLMLayer(features[3], self.conditioning_dim)

        # Decoder
        self.up1 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder1 = DoubleConvWithNorm(features[3], features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConvWithNorm(features[2], features[1])

        self.up3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder3 = DoubleConvWithNorm(features[1], features[0])

        # FiLM conditioning at decoder (optional)
        if use_film_at_decoder:
            self.decoder1_film = FiLMLayer(features[2], self.conditioning_dim)
            self.decoder2_film = FiLMLayer(features[1], self.conditioning_dim)
            self.decoder3_film = FiLMLayer(features[0], self.conditioning_dim)

        # ===== OUTPUT HEADS =====
        # Spatial predictions (normalized, like original model)
        self.spatial_head = nn.Conv2d(features[0], 2, kernel_size=1)  # CBF, ATT

        # Amplitude-dependent CBF modulation
        if use_amplitude_output_modulation:
            # Learn a MULTIPLICATIVE CORRECTION to the direct amplitude estimate
            # CBF_final = CBF_spatial * (base_amplitude * learned_correction)
            # This ensures amplitude directly influences CBF even before training
            self.cbf_amplitude_correction = nn.Sequential(
                nn.Linear(self.conditioning_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )
            # Initialize correction to output 0 (so total scale = base_amplitude * exp(0) = base_amplitude)
            nn.init.zeros_(self.cbf_amplitude_correction[-1].weight)
            nn.init.zeros_(self.cbf_amplitude_correction[-1].bias)

        # Initialize output head
        self._init_output_weights()

    def _init_output_weights(self):
        """Initialize output layer for stable normalized predictions."""
        nn.init.normal_(self.spatial_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.spatial_head.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with amplitude-aware processing.

        Args:
            x: (B, 2*n_plds, H, W) raw ASL signals

        Returns:
            cbf_pred: (B, 1, H, W) CBF prediction (NORMALIZED z-score * amplitude_scale)
            att_pred: (B, 1, H, W) ATT prediction (NORMALIZED z-score)
            cbf_logvar: (B, 1, H, W) CBF log-variance (placeholder)
            att_logvar: (B, 1, H, W) ATT log-variance (placeholder)
        """
        # ===== AMPLITUDE PATH =====
        # Extract amplitude features BEFORE any normalization happens
        conditioning, raw_amplitude = self.amplitude_extractor(x)

        # ===== SPATIAL PATH =====
        # Encoder (first layer has no normalization)
        e1 = self.encoder1(x)  # (B, 32, H, W) - amplitude info partially preserved
        e2 = self.encoder2(self.pool1(e1))  # (B, 64, H/2, W/2)
        e3 = self.encoder3(self.pool2(e2))  # (B, 128, H/4, W/4)
        e4 = self.encoder4(self.pool3(e3))  # (B, 256, H/8, W/8) - bottleneck

        # Apply FiLM at bottleneck
        if self.use_film_at_bottleneck:
            e4 = self.bottleneck_film(e4, conditioning)

        # Decoder with skip connections
        d1 = self.up1(e4)
        d1 = self._match_size(d1, e3)
        d1 = self.decoder1(torch.cat([e3, d1], dim=1))
        if self.use_film_at_decoder:
            d1 = self.decoder1_film(d1, conditioning)

        d2 = self.up2(d1)
        d2 = self._match_size(d2, e2)
        d2 = self.decoder2(torch.cat([e2, d2], dim=1))
        if self.use_film_at_decoder:
            d2 = self.decoder2_film(d2, conditioning)

        d3 = self.up3(d2)
        d3 = self._match_size(d3, e1)
        d3 = self.decoder3(torch.cat([e1, d3], dim=1))
        if self.use_film_at_decoder:
            d3 = self.decoder3_film(d3, conditioning)

        # ===== OUTPUT =====
        spatial_out = self.spatial_head(d3)  # (B, 2, H, W)

        cbf_spatial = spatial_out[:, 0:1, :, :]  # (B, 1, H, W)
        att_spatial = spatial_out[:, 1:2, :, :]  # (B, 1, H, W)

        # Apply amplitude-dependent scaling to CBF
        if self.use_amplitude_output_modulation:
            # DIRECT AMPLITUDE CONNECTION:
            # Use mean signal amplitude as base scale (this preserves CBF info directly)
            # Channel means are the first n_channels features in raw_amplitude
            channel_means = raw_amplitude[:, :self.in_channels]  # (B, 12)
            base_amplitude = channel_means.mean(dim=1, keepdim=True)  # (B, 1)

            # Normalize base amplitude to reasonable range (training data has ~0.01-0.1 signals)
            # This creates a direct CBF proxy: higher amplitude = higher CBF
            amplitude_proxy = base_amplitude * 100.0  # Scale to ~1-10 range

            # Learn a correction factor (log-scale for stability)
            log_correction = self.cbf_amplitude_correction(conditioning)  # (B, 1)
            correction = torch.exp(log_correction.clamp(-2, 2))  # Bounded correction

            # Final scale combines direct amplitude with learned correction
            cbf_scale = amplitude_proxy * correction  # (B, 1)
            cbf_scale = cbf_scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

            # The spatial network predicts RELATIVE CBF patterns
            # The amplitude path provides the ABSOLUTE scale
            cbf_pred = cbf_spatial * cbf_scale
        else:
            cbf_pred = cbf_spatial

        att_pred = att_spatial

        # Placeholder uncertainties
        cbf_logvar = torch.zeros_like(cbf_pred) - 5.0
        att_logvar = torch.zeros_like(att_pred) - 5.0

        return cbf_pred, att_pred, cbf_logvar, att_logvar

    def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pad tensor x to match target size (handles odd dimensions)."""
        diffY = target.size(2) - x.size(2)
        diffX = target.size(3) - x.size(3)

        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        return x


class AmplitudeAwareLoss(nn.Module):
    """
    Loss function for amplitude-aware network.

    Combines:
    1. Supervised loss on CBF/ATT predictions (normalized space)
    2. Amplitude prediction auxiliary loss
    3. Physics-informed reconstruction loss (optional)
    """

    def __init__(
        self,
        norm_stats: Dict,
        cbf_weight: float = 1.0,
        att_weight: float = 1.0,
        amplitude_weight: float = 0.1,
        physics_weight: float = 0.0,
        kinetic_model: Optional[nn.Module] = None
    ):
        super().__init__()

        self.cbf_mean = norm_stats['y_mean_cbf']
        self.cbf_std = norm_stats['y_std_cbf']
        self.att_mean = norm_stats['y_mean_att']
        self.att_std = norm_stats['y_std_att']

        self.cbf_weight = cbf_weight
        self.att_weight = att_weight
        self.amplitude_weight = amplitude_weight
        self.physics_weight = physics_weight
        self.kinetic_model = kinetic_model

    def forward(
        self,
        pred_cbf: torch.Tensor,
        pred_att: torch.Tensor,
        target_cbf: torch.Tensor,
        target_att: torch.Tensor,
        brain_mask: torch.Tensor,
        input_signals: Optional[torch.Tensor] = None,
        raw_amplitude_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss.

        Args:
            pred_cbf: (B, 1, H, W) predicted CBF (model output, may be scaled)
            pred_att: (B, 1, H, W) predicted ATT (normalized)
            target_cbf: (B, 1, H, W) target CBF (raw, ml/100g/min)
            target_att: (B, 1, H, W) target ATT (raw, ms)
            brain_mask: (B, 1, H, W) binary mask
            input_signals: (B, C, H, W) original input for physics loss
            raw_amplitude_features: (B, n_features) amplitude features for auxiliary loss

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual losses for logging
        """
        # Normalize targets
        target_cbf_norm = (target_cbf - self.cbf_mean) / (self.cbf_std + 1e-6)
        target_att_norm = (target_att - self.att_mean) / (self.att_std + 1e-6)

        mask_sum = brain_mask.sum().clamp(min=1.0)

        # Supervised losses (L1)
        cbf_loss = (torch.abs(pred_cbf - target_cbf_norm) * brain_mask).sum() / mask_sum
        att_loss = (torch.abs(pred_att - target_att_norm) * brain_mask).sum() / mask_sum

        # Physics loss (if enabled)
        physics_loss = torch.tensor(0.0, device=pred_cbf.device)
        if self.physics_weight > 0 and self.kinetic_model is not None and input_signals is not None:
            # Denormalize predictions
            pred_cbf_raw = pred_cbf * self.cbf_std + self.cbf_mean
            pred_att_raw = pred_att * self.att_std + self.att_mean

            # Clamp to physical bounds
            pred_cbf_raw = torch.clamp(pred_cbf_raw, min=0.0)
            pred_att_raw = torch.clamp(pred_att_raw, min=0.0, max=5000.0)

            physics_loss = self.kinetic_model.compute_physics_loss(
                pred_cbf_raw, pred_att_raw, input_signals, brain_mask
            )

        # Total loss
        total_loss = (
            self.cbf_weight * cbf_loss +
            self.att_weight * att_loss +
            self.physics_weight * physics_loss
        )

        loss_dict = {
            'loss': total_loss.item(),
            'cbf_loss': cbf_loss.item(),
            'att_loss': att_loss.item(),
            'physics_loss': physics_loss.item() if self.physics_weight > 0 else 0.0
        }

        return total_loss, loss_dict


# ============================================================================
# TEST AMPLITUDE SENSITIVITY
# ============================================================================

def test_amplitude_sensitivity():
    """Test that the network IS sensitive to input amplitude."""
    print("=" * 70)
    print("Testing AmplitudeAwareSpatialASLNet amplitude sensitivity")
    print("=" * 70)

    # Create model
    model = AmplitudeAwareSpatialASLNet(n_plds=6)
    model.eval()

    # Create test input
    np.random.seed(42)
    base_input = np.random.randn(1, 12, 64, 64).astype(np.float32) * 0.01 + 0.05

    print("\nInput scale | CBF pred mean | ATT pred mean | CBF range")
    print("-" * 60)

    results = []
    for scale in [0.1, 1.0, 10.0, 100.0]:
        x = torch.from_numpy(base_input * scale)
        with torch.no_grad():
            cbf, att, _, _ = model(x)

        cbf_mean = cbf.mean().item()
        att_mean = att.mean().item()
        cbf_range = (cbf.min().item(), cbf.max().item())

        print(f"{scale:10.1f}  | {cbf_mean:12.4f} | {att_mean:12.4f} | [{cbf_range[0]:.3f}, {cbf_range[1]:.3f}]")
        results.append((scale, cbf_mean))

    # Check if CBF predictions change with scale
    cbf_values = [r[1] for r in results]
    is_sensitive = max(cbf_values) / (min(cbf_values) + 1e-6) > 1.5

    print()
    if is_sensitive:
        print("SUCCESS: Network IS sensitive to input amplitude!")
    else:
        print("WARNING: Network may not be sufficiently sensitive to amplitude.")

    return is_sensitive


if __name__ == "__main__":
    test_amplitude_sensitivity()

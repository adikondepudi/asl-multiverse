# spatial_asl_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class KineticModel(nn.Module):
    """
    Differentiable Forward Model for ASL Kinetics.
    Converts CBF and ATT maps back into raw ASL signals for Data Consistency Loss.

    Implements the General Kinetic Model for both PCASL and VSASL.

    Physics-Informed Loss Usage:
    ---------------------------
    This module enables physics-informed training by:
    1. Taking predicted CBF/ATT parameters
    2. Reconstructing the expected ASL signal
    3. Computing loss between reconstructed and actual input signal

    This ensures predictions are physically consistent with the kinetic model,
    acting as a strong regularizer that anchors the network to physical reality.

    Domain Randomization:
    --------------------
    Supports per-batch variation of physics parameters (T1, alpha) to prevent
    the network from overfitting to fixed parameter values. This is critical
    for generalization to real patient data where parameters vary.
    """
    def __init__(self, pld_values, t1_blood=1850.0, t_tau=1800.0,
                 alpha_pcasl=0.85, alpha_vsasl=0.56, alpha_bs=1.0,
                 t2_factor=1.0, t_sat_vs=2000.0,
                 domain_randomization: dict = None):
        super().__init__()
        # Register all physics parameters as buffers for device compatibility
        self.register_buffer('plds', torch.tensor(pld_values, dtype=torch.float32))
        self.register_buffer('t1_blood_default', torch.tensor(t1_blood, dtype=torch.float32))
        self.register_buffer('t_tau_default', torch.tensor(t_tau, dtype=torch.float32))
        self.register_buffer('t2_factor', torch.tensor(t2_factor, dtype=torch.float32))
        self.register_buffer('t_sat_vs', torch.tensor(t_sat_vs, dtype=torch.float32))

        # Default efficiencies
        self.register_buffer('alpha_pcasl_default', torch.tensor(alpha_pcasl * (alpha_bs**4), dtype=torch.float32))
        self.register_buffer('alpha_vsasl_default', torch.tensor(alpha_vsasl * (alpha_bs**3), dtype=torch.float32))
        self.register_buffer('lambda_blood', torch.tensor(0.90, dtype=torch.float32))
        self.register_buffer('unit_conv', torch.tensor(6000.0, dtype=torch.float32))

        # Domain randomization configuration
        self.domain_randomization = domain_randomization or {}
        self.use_domain_rand = self.domain_randomization.get('enabled', False)

        # Randomization ranges (default values if not specified)
        if self.use_domain_rand:
            self.t1_range = self.domain_randomization.get('T1_artery_range', [1550.0, 2150.0])
            self.alpha_pcasl_range = self.domain_randomization.get('alpha_PCASL_range', [0.75, 0.95])
            self.alpha_vsasl_range = self.domain_randomization.get('alpha_VSASL_range', [0.40, 0.70])
            self.t_tau_perturb = self.domain_randomization.get('T_tau_perturb', 0.10)
            # Background suppression: 1.0 = no BS, 0.85-0.95 = typical in-vivo BS
            self.alpha_bs1_range = self.domain_randomization.get('alpha_BS1_range', [0.85, 1.0])

    def forward(self, cbf, att, randomize_params: bool = False):
        """
        Generate ASL signals from parameter maps.

        Args:
            cbf: (Batch, 1, H, W) - CBF in ml/100g/min
            att: (Batch, 1, H, W) - ATT in ms
            randomize_params: If True and domain_randomization enabled,
                             sample physics parameters per-batch

        Returns:
            signals: (Batch, 2*N_plds, H, W) - [PCASL_t1...tn, VSASL_t1...tn]
        """
        batch_size = cbf.shape[0]
        device = cbf.device

        # Get physics parameters (with optional randomization)
        if randomize_params and self.use_domain_rand:
            # Sample per-batch physics parameters
            t1_blood = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(*self.t1_range)
            # Sample background suppression efficiency (1.0 = no BS, <1 = with BS)
            alpha_bs1 = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(*self.alpha_bs1_range)
            # Effective labeling efficiency = raw efficiency * BS attenuation
            # PCASL uses 4 BS pulses (alpha_BS1^4), VSASL uses 3 BS pulses (alpha_BS1^3)
            alpha_pcasl = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(*self.alpha_pcasl_range) * (alpha_bs1 ** 4)
            alpha_vsasl = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(*self.alpha_vsasl_range) * (alpha_bs1 ** 3)
            t_tau = self.t_tau_default * (1 + torch.empty(1, device=device).uniform_(-self.t_tau_perturb, self.t_tau_perturb))
        else:
            # Use default parameters (alpha_BS1 = 1.0, no attenuation)
            t1_blood = self.t1_blood_default
            alpha_pcasl = self.alpha_pcasl_default
            alpha_vsasl = self.alpha_vsasl_default
            t_tau = self.t_tau_default

        # Expand dims for broadcasting: (Batch, N_plds, H, W)
        pld_exp = self.plds.view(1, -1, 1, 1)

        # Convert CBF to physiological units (ml/g/s)
        f = cbf / self.unit_conv

        # --- PCASL GENERATION ---
        # Based on Buxton General Kinetic Model
        # Condition 2: ATT - tau <= PLD < ATT (bolus in transit)
        term2_p = (2 * alpha_pcasl * f * t1_blood / 1000.0 *
                   (torch.exp(-att / t1_blood) - torch.exp(-(t_tau + pld_exp) / t1_blood)) *
                   self.t2_factor) / self.lambda_blood

        # Condition 3: PLD >= ATT (bolus arrived)
        term3_p = (2 * alpha_pcasl * f * t1_blood / 1000.0 *
                   torch.exp(-pld_exp / t1_blood) *
                   (1 - torch.exp(-t_tau / t1_blood)) *
                   self.t2_factor) / self.lambda_blood

        # Soft masks for differentiability
        steep = 10.0
        mask_arrived = torch.sigmoid((pld_exp - att) * steep)
        mask_transit = torch.sigmoid((pld_exp - (att - t_tau)) * steep) * (1 - mask_arrived)

        pcasl_sig = (term3_p * mask_arrived) + (term2_p * mask_transit)

        # --- VSASL GENERATION ---
        # Condition 1: PLD <= ATT (vascular signal)
        term1_v = (2 * alpha_vsasl * f * (pld_exp / 1000.0) *
                   torch.exp(-pld_exp / t1_blood) * self.t2_factor) / self.lambda_blood

        # Condition 2: PLD > ATT (tissue signal)
        term2_v = (2 * alpha_vsasl * f * (att / 1000.0) *
                   torch.exp(-pld_exp / t1_blood) * self.t2_factor) / self.lambda_blood

        mask_vs_arrived = torch.sigmoid((pld_exp - att) * steep)
        vsasl_sig = (term2_v * mask_vs_arrived) + (term1_v * (1 - mask_vs_arrived))

        # Concatenate Channel-wise: [PCASL_t1...tn, VSASL_t1...tn]
        # Scale output to match the SpatialDataset's *100 normalization
        return torch.cat([pcasl_sig, vsasl_sig], dim=1) * 100.0

    def compute_physics_loss(self, pred_cbf, pred_att, input_signals, brain_mask,
                            randomize_params: bool = True):
        """
        Compute physics-informed loss (data consistency).

        This loss ensures that the predicted CBF/ATT values, when plugged back
        into the kinetic equations, reproduce the observed MRI signal intensities.

        Args:
            pred_cbf: (B, 1, H, W) Predicted CBF (raw units, ml/100g/min)
            pred_att: (B, 1, H, W) Predicted ATT (raw units, ms)
            input_signals: (B, 2*N_plds, H, W) Actual input signals
            brain_mask: (B, 1, H, W) Binary mask
            randomize_params: Whether to use domain randomization

        Returns:
            physics_loss: Scalar tensor - L1 loss between predicted and actual signals
        """
        # Reconstruct signals from predicted parameters
        pred_signals = self.forward(pred_cbf, pred_att, randomize_params=randomize_params)

        # Expand mask to match signal channels
        n_channels = pred_signals.shape[1]
        expanded_mask = brain_mask.expand(-1, n_channels, -1, -1)

        # L1 loss on signal reconstruction (masked)
        signal_diff = torch.abs(pred_signals - input_signals) * expanded_mask
        mask_sum = expanded_mask.sum() + 1e-6

        physics_loss = signal_diff.sum() / mask_sum

        return physics_loss


class DoubleConv(nn.Module):
    """
    Standard U-Net double convolution block.

    Uses GroupNorm instead of BatchNorm for two reasons:
    1. GroupNorm is consistent between train and eval modes (no running stats)
    2. BatchNorm with small batches is unstable and can cause train/eval mismatch
       where the model appears to learn during training but fails at validation.

    The train/eval mismatch with BatchNorm was causing the model to appear to
    converge (loss going down) but actually predict garbage at inference time.
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        # Use GroupNorm with num_groups. If out_channels < num_groups, use out_channels.
        num_groups_1 = min(num_groups, out_channels)
        num_groups_2 = min(num_groups, out_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups_1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups_2, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SpatialASLNet(nn.Module):
    """
    U-Net architecture for Spatio-Temporal ASL Processing.

    Input: (Batch, N_plds * 2, Height, Width) - PCASL + VSASL channels
    Output: CBF_map, ATT_map, log_var_cbf, log_var_att

    IMPORTANT: Model outputs NORMALIZED predictions (z-scores), not raw values.
    Targets must also be normalized during training. Denormalization happens at inference.

    This avoids initialization bias where softplus(0)*100=69.3 and sigmoid(0)*3000=1500
    would cause the model to predict near the dataset mean.

    Architecture notes:
    - Uses GroupNorm instead of BatchNorm for train/eval consistency
    - Kaiming initialization for stable gradient flow
    - Output is unbounded (no activation) for normalized prediction
    """
    def __init__(self, n_plds=6, features=[32, 64, 128, 256], **kwargs):
        super().__init__()
        in_channels = n_plds * 2  # PCASL + VSASL input channels

        # Encoder (Contracting Path)
        self.encoder1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DoubleConv(features[2], features[3])

        # Decoder (Expanding Path)
        self.up1 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features[3], features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features[2], features[1])
        self.up3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features[1], features[0])

        # Output Head: 2 channels (CBF, ATT)
        self.out_conv = nn.Conv2d(features[0], 2, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize output conv to small values for stable start
        # This ensures initial predictions are near 0 (the normalized mean)
        nn.init.normal_(self.out_conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.out_conv.bias, 0)

    def forward(self, x):
        """
        Forward pass of U-Net.

        Args:
            x: (Batch, Time, H, W) - Multi-PLD ASL signal

        Returns:
            cbf_map: (Batch, 1, H, W) - NORMALIZED CBF prediction (z-score)
            att_map: (Batch, 1, H, W) - NORMALIZED ATT prediction (z-score)
            log_var_cbf: (Batch, 1, H, W) - Placeholder uncertainty
            log_var_att: (Batch, 1, H, W) - Placeholder uncertainty

        NOTE: Output is UNBOUNDED normalized predictions. Denormalize at inference:
            cbf_raw = cbf_norm * std_cbf + mean_cbf
            att_raw = att_norm * std_att + mean_att
        """
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Decoder with skip connections
        d1 = self.up1(e4)
        # Pad if dimensions don't match exactly due to pooling odd shapes
        diffY = e3.size()[2] - d1.size()[2]
        diffX = e3.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = self.decoder1(torch.cat([e3, d1], dim=1))

        d2 = self.up2(d1)
        diffY = e2.size()[2] - d2.size()[2]
        diffX = e2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = self.decoder2(torch.cat([e2, d2], dim=1))

        d3 = self.up3(d2)
        diffY = e1.size()[2] - d3.size()[2]
        diffX = e1.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = self.decoder3(torch.cat([e1, d3], dim=1))

        out = self.out_conv(d3)

        # UNBOUNDED output - no activation constraints!
        # Model predicts normalized values (z-scores), not raw CBF/ATT.
        # This avoids initialization bias: softplus(0)*100=69.3, sigmoid(0)*3000=1500
        # would cause mean prediction. With unbounded output, init ≈ 0 = normalized mean.
        cbf_norm = out[:, 0:1, :, :]
        att_norm = out[:, 1:2, :, :]

        # Return 4 values to match existing trainer signature
        # (cbf, att, log_var_cbf, log_var_att)
        # Placeholder log_var for compatibility
        zero_log = torch.zeros_like(cbf_norm) - 5.0

        return cbf_norm, att_norm, zero_log, zero_log


class DualEncoderSpatialASLNet(nn.Module):
    """
    Dual-Encoder Y-Net Architecture for Joint PCASL-VSASL Processing.

    This architecture processes PCASL and VSASL signals through separate encoder
    streams before fusion, allowing the network to learn modality-specific features:

    - Stream A (PCASL): Extracts transit-dependent features
      PCASL is sensitive to ATT but provides high SNR measurements

    - Stream B (VSASL): Extracts transit-independent perfusion features
      VSASL is insensitive to ATT, maintaining signal in delayed flow regions

    Fusion happens at the bottleneck layer, allowing the network to learn
    distinct hierarchical representations before combining them.

    Architecture:
        Input: (Batch, 2*N_plds, H, W) - [PCASL_t1...tn, VSASL_t1...tn]

        PCASL Stream:  Input → Enc1 → Enc2 → Enc3 → Bottleneck_A
        VSASL Stream:  Input → Enc1 → Enc2 → Enc3 → Bottleneck_B

        Fusion: Concat(Bottleneck_A, Bottleneck_B) → Fused Bottleneck

        Decoder: Fused → Dec1 → Dec2 → Dec3 → Output

        Output: (cbf_norm, att_norm, log_var_cbf, log_var_att)

    Output Activations (as per technical recommendations):
    - CBF: Softplus for strict positivity (Softplus(0) ≈ 0.69, smooth ReLU)
    - ATT: Sigmoid × max_att for bounded positive range [0, 3000] ms

    References:
    - IVIM-NET architecture with dual streams
    - Y-Net for multi-modal medical imaging
    """

    def __init__(self, n_plds=6, features=[32, 64, 128, 256],
                 use_constrained_output: bool = False,
                 max_cbf: float = 200.0,
                 max_att: float = 5000.0,
                 **kwargs):
        """
        Args:
            n_plds: Number of Post-Labeling Delays per modality
            features: Feature dimensions at each encoder level [32, 64, 128, 256]
            use_constrained_output: If True, use Softplus/Sigmoid for physics constraints
                                   If False, output unbounded normalized predictions
            max_cbf: Maximum CBF for constraint (ml/100g/min)
            max_att: Maximum ATT for constraint (ms)
        """
        super().__init__()
        self.n_plds = n_plds
        self.use_constrained_output = use_constrained_output
        self.max_cbf = max_cbf
        self.max_att = max_att

        # ============= PCASL Stream (Transit-Dependent Features) =============
        self.pcasl_encoder1 = DoubleConv(n_plds, features[0])
        self.pcasl_pool1 = nn.MaxPool2d(2)
        self.pcasl_encoder2 = DoubleConv(features[0], features[1])
        self.pcasl_pool2 = nn.MaxPool2d(2)
        self.pcasl_encoder3 = DoubleConv(features[1], features[2])
        self.pcasl_pool3 = nn.MaxPool2d(2)
        self.pcasl_bottleneck = DoubleConv(features[2], features[3])

        # ============= VSASL Stream (Transit-Independent Features) =============
        self.vsasl_encoder1 = DoubleConv(n_plds, features[0])
        self.vsasl_pool1 = nn.MaxPool2d(2)
        self.vsasl_encoder2 = DoubleConv(features[0], features[1])
        self.vsasl_pool2 = nn.MaxPool2d(2)
        self.vsasl_encoder3 = DoubleConv(features[1], features[2])
        self.vsasl_pool3 = nn.MaxPool2d(2)
        self.vsasl_bottleneck = DoubleConv(features[2], features[3])

        # ============= Fusion Layer =============
        # Concatenate PCASL and VSASL bottleneck features
        # Then process through a fusion block
        self.fusion_conv = DoubleConv(features[3] * 2, features[3])

        # ============= Shared Decoder =============
        # Uses skip connections from PCASL stream (could be fused skips)
        self.up1 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features[2] * 3, features[2])  # 3x: fused + pcasl_skip + vsasl_skip

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features[1] * 3, features[1])

        self.up3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features[0] * 3, features[0])

        # ============= Output Head =============
        # Separate heads for CBF and ATT (allows different activations)
        self.cbf_head = nn.Conv2d(features[0], 1, kernel_size=1)
        self.att_head = nn.Conv2d(features[0], 1, kernel_size=1)

        # Optional: Uncertainty estimation heads
        self.cbf_logvar_head = nn.Conv2d(features[0], 1, kernel_size=1)
        self.att_logvar_head = nn.Conv2d(features[0], 1, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize output heads to small values for stable start
        for head in [self.cbf_head, self.att_head]:
            nn.init.normal_(head.weight, mean=0, std=0.01)
            nn.init.constant_(head.bias, 0)

        # Initialize log_var heads to predict low variance initially
        for head in [self.cbf_logvar_head, self.att_logvar_head]:
            nn.init.normal_(head.weight, mean=0, std=0.01)
            nn.init.constant_(head.bias, -5.0)  # exp(-5) ≈ 0.007, low initial uncertainty

    def forward(self, x):
        """
        Forward pass of Dual-Encoder Y-Net.

        Args:
            x: (Batch, 2*N_plds, H, W) - Concatenated [PCASL, VSASL] signals

        Returns:
            cbf_out: (Batch, 1, H, W) - CBF prediction
            att_out: (Batch, 1, H, W) - ATT prediction
            log_var_cbf: (Batch, 1, H, W) - CBF uncertainty (log variance)
            log_var_att: (Batch, 1, H, W) - ATT uncertainty (log variance)
        """
        # Split input into PCASL and VSASL channels
        pcasl_input = x[:, :self.n_plds, :, :]  # First n_plds channels
        vsasl_input = x[:, self.n_plds:, :, :]  # Last n_plds channels

        # ============= PCASL Stream =============
        pe1 = self.pcasl_encoder1(pcasl_input)
        pe2 = self.pcasl_encoder2(self.pcasl_pool1(pe1))
        pe3 = self.pcasl_encoder3(self.pcasl_pool2(pe2))
        pb = self.pcasl_bottleneck(self.pcasl_pool3(pe3))

        # ============= VSASL Stream =============
        ve1 = self.vsasl_encoder1(vsasl_input)
        ve2 = self.vsasl_encoder2(self.vsasl_pool1(ve1))
        ve3 = self.vsasl_encoder3(self.vsasl_pool2(ve2))
        vb = self.vsasl_bottleneck(self.vsasl_pool3(ve3))

        # ============= Fusion =============
        # Concatenate bottleneck features and process
        fused = torch.cat([pb, vb], dim=1)
        fused = self.fusion_conv(fused)

        # ============= Decoder with Dual Skip Connections =============
        # Level 1
        d1 = self.up1(fused)
        # Handle size mismatch
        d1 = self._match_size(d1, pe3)
        d1 = self.decoder1(torch.cat([d1, pe3, self._match_size(ve3, pe3)], dim=1))

        # Level 2
        d2 = self.up2(d1)
        d2 = self._match_size(d2, pe2)
        d2 = self.decoder2(torch.cat([d2, pe2, self._match_size(ve2, pe2)], dim=1))

        # Level 3
        d3 = self.up3(d2)
        d3 = self._match_size(d3, pe1)
        d3 = self.decoder3(torch.cat([d3, pe1, self._match_size(ve1, pe1)], dim=1))

        # ============= Output Heads =============
        cbf_raw = self.cbf_head(d3)
        att_raw = self.att_head(d3)
        log_var_cbf = self.cbf_logvar_head(d3)
        log_var_att = self.att_logvar_head(d3)

        if self.use_constrained_output:
            # Apply physics-motivated constraints:
            # CBF: Softplus for strict positivity (smooth ReLU)
            # Note: Softplus(0) ≈ 0.69, so we subtract to center at 0
            cbf_out = F.softplus(cbf_raw) * (self.max_cbf / 10.0)  # Scale to reasonable range

            # ATT: Sigmoid × max_att for bounded [0, max_att]
            att_out = torch.sigmoid(att_raw) * self.max_att
        else:
            # Unbounded normalized output (recommended for training stability)
            # Denormalization happens at inference time
            cbf_out = cbf_raw
            att_out = att_raw

        return cbf_out, att_out, log_var_cbf, log_var_att

    def _match_size(self, x, target):
        """Pad tensor x to match target's spatial dimensions."""
        diffY = target.size(2) - x.size(2)
        diffX = target.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x


class MaskedSpatialLoss(nn.Module):
    """
    Masked loss function that only counts brain pixels, not background air.

    This prevents the network from learning the trivial solution of predicting
    zero everywhere (which would score well on 80% air background).

    IMPORTANT: This loss expects NORMALIZED predictions and will NORMALIZE targets
    using the provided norm_stats. This is critical for preventing mean prediction!

    Loss Modes:
    - 'l1' or 'mae': L1 loss (Mean Absolute Error) - RECOMMENDED
    - 'l2' or 'mse': L2 loss (Mean Squared Error)
    - 'huber': Huber loss (robust to outliers)

    Distribution Matching (variance_weight > 0):
    Adds a penalty if prediction variance is much lower than target variance.
    This prevents the model from collapsing to mean prediction - if the model
    just predicts the average, this term will be high even if per-sample L1 is low.

    Note: With normalized targets, ATT scaling is no longer needed since both
    CBF and ATT are in z-score units with similar magnitude (~0-3 std).
    """
    def __init__(self, loss_type: str = 'l1', dc_weight: float = 0.0,
                 kinetic_model: nn.Module = None,
                 att_scale: float = 1.0,  # Default 1.0 since we normalize targets
                 cbf_weight: float = 1.0,
                 att_weight: float = 1.0,
                 norm_stats: dict = None,
                 variance_weight: float = 0.1):  # NEW: Penalize low prediction variance
        super().__init__()
        self.loss_type = loss_type.lower()
        self.dc_weight = dc_weight
        self.kinetic_model = kinetic_model
        self.att_scale = att_scale
        self.cbf_weight = cbf_weight
        self.att_weight = att_weight
        self.variance_weight = variance_weight

        # Store normalization statistics for target normalization
        # These should contain: y_mean_cbf, y_std_cbf, y_mean_att, y_std_att
        self.norm_stats = norm_stats
        self._norm_tensors_initialized = False

        print(f"[MaskedSpatialLoss] loss_type={loss_type}, att_scale={att_scale}, "
              f"cbf_weight={cbf_weight}, att_weight={att_weight}, dc_weight={dc_weight}, "
              f"variance_weight={variance_weight}, "
              f"norm_stats={'provided' if norm_stats else 'MISSING - targets will NOT be normalized!'}")

    def _init_norm_tensors(self, device):
        """Lazily initialize normalization tensors on the correct device."""
        if self._norm_tensors_initialized:
            return

        if self.norm_stats is not None:
            self.register_buffer('cbf_mean', torch.tensor(self.norm_stats['y_mean_cbf'], dtype=torch.float32, device=device))
            self.register_buffer('cbf_std', torch.tensor(self.norm_stats['y_std_cbf'], dtype=torch.float32, device=device))
            self.register_buffer('att_mean', torch.tensor(self.norm_stats['y_mean_att'], dtype=torch.float32, device=device))
            self.register_buffer('att_std', torch.tensor(self.norm_stats['y_std_att'], dtype=torch.float32, device=device))
            self._norm_tensors_initialized = True

    def forward(self, pred_cbf: torch.Tensor, pred_att: torch.Tensor,
                target_cbf: torch.Tensor, target_att: torch.Tensor,
                brain_mask: torch.Tensor,
                input_signals: torch.Tensor = None) -> dict:
        """
        Compute masked loss.

        IMPORTANT: pred_cbf and pred_att are NORMALIZED predictions (z-scores).
        target_cbf and target_att are RAW values that will be normalized here.

        Args:
            pred_cbf: (B, 1, H, W) predicted CBF (NORMALIZED z-score)
            pred_att: (B, 1, H, W) predicted ATT (NORMALIZED z-score)
            target_cbf: (B, 1, H, W) ground truth CBF (RAW, will be normalized)
            target_att: (B, 1, H, W) ground truth ATT (RAW, will be normalized)
            brain_mask: (B, 1, H, W) binary mask (1=brain, 0=air)
            input_signals: (B, 2*PLDs, H, W) for data consistency loss

        Returns:
            dict with 'total_loss' and component losses
        """
        # Expand mask if needed
        if brain_mask.dim() == 3:
            brain_mask = brain_mask.unsqueeze(1)

        mask_sum = brain_mask.sum() + 1e-6

        # --- NORMALIZE TARGETS ---
        # This is CRITICAL! Model outputs normalized predictions, so targets must match.
        # Without this, the model learns to predict the mean because:
        # 1. Constrained activations (softplus/sigmoid) have initialization bias
        # 2. Raw targets have different scales (CBF: 20-100, ATT: 500-3000)
        if self.norm_stats is not None:
            self._init_norm_tensors(target_cbf.device)
            target_cbf_norm = (target_cbf - self.cbf_mean) / (self.cbf_std + 1e-6)
            target_att_norm = (target_att - self.att_mean) / (self.att_std + 1e-6)
        else:
            # Fallback: no normalization (will likely cause mean prediction!)
            target_cbf_norm = target_cbf
            target_att_norm = target_att

        # --- Compute errors in NORMALIZED space ---
        cbf_err = pred_cbf - target_cbf_norm
        att_err = pred_att - target_att_norm

        # --- Apply loss function ---
        if self.loss_type in ['l1', 'mae']:
            cbf_loss_map = torch.abs(cbf_err) * brain_mask
            att_loss_map = torch.abs(att_err) * brain_mask
        elif self.loss_type in ['l2', 'mse']:
            cbf_loss_map = (cbf_err ** 2) * brain_mask
            att_loss_map = (att_err ** 2) * brain_mask
        elif self.loss_type == 'huber':
            # With normalized targets, both CBF and ATT errors are in z-score units
            # so we use the same delta for both
            delta = 1.0
            cbf_loss_map = torch.where(
                torch.abs(cbf_err) < delta,
                0.5 * cbf_err ** 2,
                delta * (torch.abs(cbf_err) - 0.5 * delta)
            ) * brain_mask
            att_loss_map = torch.where(
                torch.abs(att_err) < delta,
                0.5 * att_err ** 2,
                delta * (torch.abs(att_err) - 0.5 * delta)
            ) * brain_mask
        else:
            # Default to L1
            cbf_loss_map = torch.abs(cbf_err) * brain_mask
            att_loss_map = torch.abs(att_err) * brain_mask

        # --- Compute mean losses over brain voxels ---
        cbf_loss = cbf_loss_map.sum() / mask_sum
        att_loss = att_loss_map.sum() / mask_sum

        # Scale ATT loss to balance with CBF
        att_loss_scaled = att_loss * self.att_scale

        # Weighted combination
        supervised_loss = self.cbf_weight * cbf_loss + self.att_weight * att_loss_scaled

        # --- Variance Penalty (Anti-Mean-Collapse) ---
        # If the model just predicts the mean, variance will be near zero.
        # This term penalizes when pred variance << target variance.
        variance_loss = torch.tensor(0.0, device=pred_cbf.device)

        if self.variance_weight > 0:
            # Flatten spatial dimensions but keep batch: (B, 1, H, W) -> (B, H*W)
            B = pred_cbf.shape[0]
            pred_cbf_flat = pred_cbf.view(B, -1)
            pred_att_flat = pred_att.view(B, -1)
            target_cbf_flat = target_cbf_norm.view(B, -1)
            target_att_flat = target_att_norm.view(B, -1)
            mask_flat = brain_mask.view(B, -1)

            # Compute variance only over masked (brain) pixels within each sample
            # Then average across batch
            cbf_var_penalties = []
            att_var_penalties = []

            for b in range(B):
                mask_b = mask_flat[b] > 0.5
                if mask_b.sum() < 10:  # Skip if too few brain pixels
                    continue

                pred_cbf_b = pred_cbf_flat[b][mask_b]
                pred_att_b = pred_att_flat[b][mask_b]
                target_cbf_b = target_cbf_flat[b][mask_b]
                target_att_b = target_att_flat[b][mask_b]

                # Compute per-sample variance
                pred_cbf_var = pred_cbf_b.var()
                target_cbf_var = target_cbf_b.var()
                pred_att_var = pred_att_b.var()
                target_att_var = target_att_b.var()

                # Penalize if prediction variance is lower than target variance
                cbf_var_penalties.append(F.relu(target_cbf_var - pred_cbf_var))
                att_var_penalties.append(F.relu(target_att_var - pred_att_var))

            if cbf_var_penalties:
                avg_cbf_var_penalty = torch.stack(cbf_var_penalties).mean()
                avg_att_var_penalty = torch.stack(att_var_penalties).mean()
                variance_loss = self.variance_weight * (avg_cbf_var_penalty + avg_att_var_penalty)

        # --- Data Consistency Loss (Physics-Informed) ---
        # This loss ensures that predicted CBF/ATT values, when plugged back into
        # the kinetic equations, reproduce the observed MRI signal intensities.
        # This acts as a powerful regularizer anchoring the network to physical reality.
        dc_loss = torch.tensor(0.0, device=pred_cbf.device)

        if self.dc_weight > 0 and self.kinetic_model is not None and input_signals is not None:
            # DENORMALIZE predictions for kinetic model (which expects raw CBF/ATT)
            if self.norm_stats is not None:
                pred_cbf_raw = pred_cbf * self.cbf_std + self.cbf_mean
                pred_att_raw = pred_att * self.att_std + self.att_mean
                # Apply physical constraints
                pred_cbf_raw = torch.clamp(pred_cbf_raw, min=0.0)  # CBF must be positive
                pred_att_raw = torch.clamp(pred_att_raw, min=0.0, max=5000.0)  # ATT range
            else:
                pred_cbf_raw = pred_cbf
                pred_att_raw = pred_att

            # Use KineticModel's physics loss computation with domain randomization
            # This varies T1, alpha parameters to prevent overfitting to fixed values
            if hasattr(self.kinetic_model, 'compute_physics_loss'):
                dc_loss = self.dc_weight * self.kinetic_model.compute_physics_loss(
                    pred_cbf_raw, pred_att_raw, input_signals, brain_mask,
                    randomize_params=self.training if hasattr(self, 'training') else True
                )
            else:
                # Fallback to direct computation
                pred_signals = self.kinetic_model(pred_cbf_raw, pred_att_raw)

                # Expand mask for signal channels
                n_channels = pred_signals.shape[1]
                expanded_mask = brain_mask.expand(-1, n_channels, -1, -1)

                # L1 difference between predicted and actual signals (masked)
                signal_diff = torch.abs(pred_signals - input_signals) * expanded_mask
                dc_loss = self.dc_weight * signal_diff.sum() / (expanded_mask.sum() + 1e-6)

        total_loss = supervised_loss + dc_loss + variance_loss

        # Return losses for logging
        # Note: cbf_loss and att_loss are in NORMALIZED units (z-scores)
        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'cbf_loss': cbf_loss,  # In normalized units
            'att_loss': att_loss,  # In normalized units
            'att_loss_scaled': att_loss_scaled,
            'dc_loss': dc_loss,
            'variance_loss': variance_loss  # NEW: Tracks anti-collapse penalty
        }


def denormalize_spatial_predictions(pred_cbf_norm: torch.Tensor,
                                     pred_att_norm: torch.Tensor,
                                     norm_stats: dict,
                                     apply_constraints: bool = True) -> tuple:
    """
    Convert normalized predictions back to physical units.

    Args:
        pred_cbf_norm: (B, 1, H, W) normalized CBF prediction (z-score)
        pred_att_norm: (B, 1, H, W) normalized ATT prediction (z-score)
        norm_stats: dict with y_mean_cbf, y_std_cbf, y_mean_att, y_std_att
        apply_constraints: if True, clamp to physical ranges

    Returns:
        cbf_raw: (B, 1, H, W) CBF in ml/100g/min
        att_raw: (B, 1, H, W) ATT in ms
    """
    cbf_raw = pred_cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att_raw = pred_att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']

    if apply_constraints:
        # Apply physical constraints
        cbf_raw = torch.clamp(cbf_raw, min=0.0, max=200.0)  # CBF: 0-200 ml/100g/min
        att_raw = torch.clamp(att_raw, min=0.0, max=5000.0)  # ATT: 0-5000 ms

    return cbf_raw, att_raw


class BiasReducedLoss(nn.Module):
    """
    Bias-Reduced Loss Function for Quantitative MRI Parameter Estimation.

    Standard L1/L2 losses encourage the network to predict the conditional mean,
    leading to variance collapse (smoothed outputs) at low SNR. This loss addresses
    this by combining:

    1. Point estimate loss (MAE/MSE): Ensures accuracy on average
    2. Negative Log-Likelihood loss: Models uncertainty explicitly
    3. Variance matching: Prevents prediction variance from being too low

    Based on Mao et al. (2023/2024): "Bias-Reduced Neural Networks for Parameter
    Estimation in Quantitative MRI"

    Loss Modes:
    -----------
    - 'mae_only': Pure L1 loss (forces accuracy, may cause variance collapse)
    - 'nll': Negative Log-Likelihood (models uncertainty, but can cheat with high variance)
    - 'mae_nll': MAE + NLL balanced (RECOMMENDED - best of both worlds)
    - 'probabilistic': Full probabilistic loss with KL divergence regularization

    The probabilistic loss explicitly models the posterior distribution p(y|x) and
    forces the network to produce well-calibrated uncertainty estimates.
    """

    def __init__(self, loss_mode: str = 'mae_nll',
                 mae_weight: float = 1.0,
                 nll_weight: float = 0.1,
                 variance_match_weight: float = 0.05,
                 kl_weight: float = 0.01,
                 norm_stats: dict = None):
        """
        Args:
            loss_mode: 'mae_only', 'nll', 'mae_nll', or 'probabilistic'
            mae_weight: Weight for MAE component
            nll_weight: Weight for NLL component
            variance_match_weight: Weight for variance matching penalty
            kl_weight: Weight for KL divergence regularization (probabilistic mode)
            norm_stats: Normalization statistics for target denormalization
        """
        super().__init__()
        self.loss_mode = loss_mode.lower()
        self.mae_weight = mae_weight
        self.nll_weight = nll_weight
        self.variance_match_weight = variance_match_weight
        self.kl_weight = kl_weight
        self.norm_stats = norm_stats

        print(f"[BiasReducedLoss] mode={loss_mode}, mae_weight={mae_weight}, "
              f"nll_weight={nll_weight}, variance_match={variance_match_weight}")

    def forward(self, pred_mean: torch.Tensor, pred_log_var: torch.Tensor,
                target: torch.Tensor, brain_mask: torch.Tensor = None) -> dict:
        """
        Compute bias-reduced loss.

        Args:
            pred_mean: (B, 1, H, W) Predicted parameter mean
            pred_log_var: (B, 1, H, W) Predicted log variance (uncertainty)
            target: (B, 1, H, W) Ground truth parameter
            brain_mask: (B, 1, H, W) Binary mask (optional)

        Returns:
            dict with 'total_loss' and component losses
        """
        if brain_mask is None:
            brain_mask = torch.ones_like(pred_mean)
        if brain_mask.dim() == 3:
            brain_mask = brain_mask.unsqueeze(1)

        mask_sum = brain_mask.sum() + 1e-6

        # Error between prediction and target
        error = pred_mean - target

        # ========== Component 1: MAE Loss ==========
        mae_loss = (torch.abs(error) * brain_mask).sum() / mask_sum

        # ========== Component 2: NLL Loss ==========
        # NLL = 0.5 * (precision * error² + log_var)
        # where precision = exp(-log_var)
        precision = torch.exp(-pred_log_var)
        nll_loss = 0.5 * ((precision * error**2 + pred_log_var) * brain_mask).sum() / mask_sum

        # Clamp log_var to prevent numerical issues
        pred_log_var_clamped = torch.clamp(pred_log_var, min=-10, max=5)

        # ========== Component 3: Variance Matching ==========
        # Penalize if predicted variance is much lower than observed variance
        variance_loss = torch.tensor(0.0, device=pred_mean.device)

        if self.variance_match_weight > 0:
            B = pred_mean.shape[0]
            pred_flat = pred_mean.view(B, -1)
            target_flat = target.view(B, -1)
            mask_flat = brain_mask.view(B, -1)

            var_penalties = []
            for b in range(B):
                mask_b = mask_flat[b] > 0.5
                if mask_b.sum() < 10:
                    continue
                pred_var = pred_flat[b][mask_b].var()
                target_var = target_flat[b][mask_b].var()
                # Penalize if pred_var < target_var
                var_penalties.append(F.relu(target_var - pred_var))

            if var_penalties:
                variance_loss = self.variance_match_weight * torch.stack(var_penalties).mean()

        # ========== Component 4: KL Divergence (Probabilistic Mode) ==========
        # Regularizes predicted uncertainty toward prior (standard normal)
        kl_loss = torch.tensor(0.0, device=pred_mean.device)

        if self.loss_mode == 'probabilistic' and self.kl_weight > 0:
            # KL(q(y|x) || p(y)) where q is predicted Gaussian, p is N(0,1)
            # KL = 0.5 * (log_var + exp(-log_var) * mean² - 1)
            # Simplified: regularize log_var toward 0 (unit variance)
            kl_loss = self.kl_weight * (
                (0.5 * (pred_log_var_clamped + torch.exp(-pred_log_var_clamped) - 1)) * brain_mask
            ).sum() / mask_sum

        # ========== Combine Components ==========
        if self.loss_mode == 'mae_only':
            total_loss = self.mae_weight * mae_loss + variance_loss
        elif self.loss_mode == 'nll':
            total_loss = self.nll_weight * nll_loss + variance_loss
        elif self.loss_mode == 'mae_nll':
            total_loss = self.mae_weight * mae_loss + self.nll_weight * nll_loss + variance_loss
        elif self.loss_mode == 'probabilistic':
            total_loss = self.mae_weight * mae_loss + self.nll_weight * nll_loss + variance_loss + kl_loss
        else:
            # Default to mae_nll
            total_loss = self.mae_weight * mae_loss + self.nll_weight * nll_loss + variance_loss

        return {
            'total_loss': total_loss,
            'mae_loss': mae_loss,
            'nll_loss': nll_loss,
            'variance_loss': variance_loss,
            'kl_loss': kl_loss,
        }


class CramerRaoBoundEstimator:
    """
    Estimator for the Cramér-Rao Lower Bound (CRLB) for ASL parameter estimation.

    The CRLB provides the theoretical minimum variance achievable by any unbiased
    estimator. It's computed from the Fisher Information Matrix:

        CRLB = F^(-1) where F_ij = E[∂log p(x|θ) / ∂θ_i * ∂log p(x|θ) / ∂θ_j]

    For ASL, this depends on:
    - The kinetic model (∂S/∂CBF, ∂S/∂ATT)
    - The noise variance σ²
    - The experimental design (PLDs)

    Usage:
    ------
    Can be used during training to weight loss based on theoretical difficulty:
    - Easy voxels (low CRLB): Weight loss higher - model should be accurate
    - Hard voxels (high CRLB): Weight loss lower - some error is expected

    References:
    -----------
    - Mao et al. (2023): Bias-Reduced Neural Networks for Parameter Estimation
    - Woods et al. (2006): Fisher Information in ASL MRI
    """

    def __init__(self, kinetic_model: 'KineticModel', noise_sigma: float = 0.01):
        self.kinetic_model = kinetic_model
        self.noise_sigma = noise_sigma

    def compute_fisher_information(self, cbf: torch.Tensor, att: torch.Tensor,
                                   eps: float = 1e-4) -> torch.Tensor:
        """
        Compute Fisher Information Matrix via numerical differentiation.

        Args:
            cbf: (B, 1, H, W) CBF values
            att: (B, 1, H, W) ATT values
            eps: Step size for numerical differentiation

        Returns:
            fisher: (B, 2, 2, H, W) Fisher information matrix at each voxel
        """
        # Enable gradients for kinetic model
        cbf = cbf.requires_grad_(True)
        att = att.requires_grad_(True)

        # Forward model
        signals = self.kinetic_model(cbf, att)  # (B, 2*n_plds, H, W)

        # Compute gradients ∂S/∂CBF and ∂S/∂ATT
        # Using numerical differentiation for stability
        with torch.no_grad():
            signals_cbf_plus = self.kinetic_model(cbf + eps, att)
            signals_cbf_minus = self.kinetic_model(cbf - eps, att)
            grad_cbf = (signals_cbf_plus - signals_cbf_minus) / (2 * eps)

            signals_att_plus = self.kinetic_model(cbf, att + eps)
            signals_att_minus = self.kinetic_model(cbf, att - eps)
            grad_att = (signals_att_plus - signals_att_minus) / (2 * eps)

        # Fisher Information: F = (1/σ²) * Σ_t (∂S_t/∂θ_i) * (∂S_t/∂θ_j)
        # Sum over time (PLDs)
        inv_var = 1.0 / (self.noise_sigma ** 2)

        # F_11 = (1/σ²) * Σ (∂S/∂CBF)²
        f_11 = inv_var * (grad_cbf ** 2).sum(dim=1, keepdim=True)
        # F_22 = (1/σ²) * Σ (∂S/∂ATT)²
        f_22 = inv_var * (grad_att ** 2).sum(dim=1, keepdim=True)
        # F_12 = F_21 = (1/σ²) * Σ (∂S/∂CBF) * (∂S/∂ATT)
        f_12 = inv_var * (grad_cbf * grad_att).sum(dim=1, keepdim=True)

        # Assemble 2x2 matrix
        B, _, H, W = cbf.shape
        fisher = torch.zeros(B, 2, 2, H, W, device=cbf.device)
        fisher[:, 0, 0] = f_11.squeeze(1)
        fisher[:, 0, 1] = f_12.squeeze(1)
        fisher[:, 1, 0] = f_12.squeeze(1)
        fisher[:, 1, 1] = f_22.squeeze(1)

        return fisher

    def compute_crlb(self, cbf: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        """
        Compute CRLB (minimum variance bound) for each parameter.

        Args:
            cbf: (B, 1, H, W) CBF values
            att: (B, 1, H, W) ATT values

        Returns:
            crlb: (B, 2, H, W) CRLB for [CBF, ATT] at each voxel
        """
        fisher = self.compute_fisher_information(cbf, att)

        # CRLB = diag(F^(-1))
        # For 2x2: F^(-1) = (1/det) * [[F_22, -F_12], [-F_21, F_11]]
        det = fisher[:, 0, 0] * fisher[:, 1, 1] - fisher[:, 0, 1] * fisher[:, 1, 0]
        det = det.clamp(min=1e-10)  # Avoid division by zero

        crlb_cbf = fisher[:, 1, 1] / det
        crlb_att = fisher[:, 0, 0] / det

        crlb = torch.stack([crlb_cbf, crlb_att], dim=1)

        return crlb


class SpatialDataset(torch.utils.data.Dataset):
    """
    Dataset for spatial ASL training with proper normalization and augmentation.

    Implements:
    - M0 normalization: Input = (ΔS / M0) * 100
    - Configurable normalization mode:
      - 'global_scale': Preserves CBF-proportional amplitude (RECOMMENDED for CBF estimation)
      - 'per_curve': Per-pixel temporal z-score (DESTROYS CBF information!)
    - Random horizontal/vertical flips for spatial invariance
    - Random 90-degree rotations
    - Brain masking

    CRITICAL: Per-pixel z-score normalization DESTROYS CBF information!
    The signal model is: x ≈ CBF · M0 · k(ATT, T1)
    Z-score: z = (x - mean) / std cancels out CBF · M0 completely!

    Use normalization_mode='global_scale' to PRESERVE amplitude (CBF) information.
    The trainer applies normalization in _process_batch_on_gpu based on config.
    """

    M0_SCALE_FACTOR = 100.0  # Scales ~0.01-0.05 signals to ~1.0-5.0 range

    def __init__(self, data_dir: str, transform: bool = True,
                 flip_prob: float = 0.5, per_pixel_norm: bool = False,
                 rotation_prob: float = 0.25, normalization_mode: str = 'global_scale'):
        """
        Args:
            data_dir: Path to directory with spatial_chunk_*.npz files
            transform: Whether to apply augmentation
            flip_prob: Probability of random flips
            per_pixel_norm: DEPRECATED - use normalization_mode instead.
                           If True, applies z-score (DESTROYS CBF info!)
            rotation_prob: Probability of 90-degree rotation
            normalization_mode: 'global_scale' (RECOMMENDED) or 'per_curve' (z-score)
                              This should match the config's normalization_mode.
                              NOTE: Normalization is now handled in trainer's _process_batch_on_gpu,
                              so this is mainly for backward compatibility.
        """
        import glob
        self.data_files = sorted(glob.glob(f"{data_dir}/spatial_chunk_*.npz"))
        self.transform = transform
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob

        # Normalization mode: 'global_scale' preserves CBF, 'per_curve' destroys it
        self.normalization_mode = normalization_mode

        # Handle deprecated per_pixel_norm parameter
        if per_pixel_norm:
            import warnings
            warnings.warn(
                "per_pixel_norm=True is DEPRECATED and DESTROYS CBF information! "
                "Use normalization_mode='global_scale' instead. "
                "Setting normalization_mode='per_curve' for backward compatibility.",
                DeprecationWarning
            )
            self.normalization_mode = 'per_curve'

        # NOTE: We no longer apply normalization here - it's done in trainer
        # to ensure consistency between training and validation.
        # This flag is kept for backward compatibility but should be False.
        self.per_pixel_norm = False  # Always False now - trainer handles normalization

        # Preload data into RAM
        if self.data_files:
            print(f"[SpatialDataset] Pre-loading {len(self.data_files)} chunks to RAM...")
            all_sig, all_tgt = [], []
            for f in self.data_files:
                d = np.load(f)
                all_sig.append(d['signals'])
                all_tgt.append(d['targets'])

            # Concatenate and apply M0 scaling
            self.signals = np.concatenate(all_sig, axis=0) * self.M0_SCALE_FACTOR

            # Keep targets in original units (CBF: ml/100g/min, ATT: ms)
            self.targets = np.concatenate(all_tgt, axis=0)

            self.total_samples = len(self.signals)
            print(f"[SpatialDataset] Loaded {self.total_samples} samples. "
                  f"RAM: {self.signals.nbytes/1e9:.1f} GB. "
                  f"per_pixel_norm={per_pixel_norm}")
        else:
            self.total_samples = 0

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # RAM Access (Instant)
        signals = self.signals[idx].copy()  # (C, H, W) where C = 2*n_plds
        targets = self.targets[idx]

        # --- Create brain mask (non-zero regions) ---
        # Use mean signal across time as proxy for tissue
        mean_signal = np.mean(np.abs(signals), axis=0)
        brain_mask = (mean_signal > np.percentile(mean_signal, 5)).astype(np.float32)

        # --- Normalization ---
        # NOTE: Normalization is now handled in the trainer's _process_batch_on_gpu
        # to ensure consistency between training and validation modes.
        #
        # CRITICAL: Per-pixel z-score normalization DESTROYS CBF information!
        # The signal model is: x ≈ CBF · M0 · k(ATT, T1)
        # Z-score: z = (x - mean) / std cancels out CBF · M0 completely!
        #
        # The trainer will apply the correct normalization based on config:
        # - 'global_scale': Preserves CBF-proportional amplitude (RECOMMENDED)
        # - 'per_curve': Per-pixel z-score (destroys CBF, only ATT shape remains)
        #
        # We return raw M0-normalized signals here (already scaled by M0_SCALE_FACTOR=100).

        # --- Augmentation ---
        if self.transform:
            # Random horizontal flip
            if np.random.rand() < self.flip_prob:
                signals = np.flip(signals, axis=-1).copy()
                targets = np.flip(targets, axis=-1).copy()
                brain_mask = np.flip(brain_mask, axis=-1).copy()

            # Random vertical flip
            if np.random.rand() < self.flip_prob:
                signals = np.flip(signals, axis=-2).copy()
                targets = np.flip(targets, axis=-2).copy()
                brain_mask = np.flip(brain_mask, axis=-2).copy()

            # Random 90-degree rotation
            if np.random.rand() < self.rotation_prob:
                k = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
                signals = np.rot90(signals, k=k, axes=(-2, -1)).copy()
                targets = np.rot90(targets, k=k, axes=(-2, -1)).copy()
                brain_mask = np.rot90(brain_mask, k=k, axes=(-2, -1)).copy()

        return {
            'signals': torch.from_numpy(signals).float(),
            'cbf': torch.from_numpy(targets[0:1]).float(),
            'att': torch.from_numpy(targets[1:2]).float(),
            'mask': torch.from_numpy(brain_mask[np.newaxis, ...]).float()
        }


def normalize_by_m0(signals: np.ndarray, m0: np.ndarray, 
                    scale_factor: float = 100.0) -> np.ndarray:
    """
    Normalize ASL signals by M0 calibration scan.
    
    Input = (ΔS_raw / M0) * scale_factor
    
    This scales perfusion signals (~0.01-0.05) to a neural network-friendly
    range (~1.0-5.0).
    
    Args:
        signals: (..., H, W) raw difference signals
        m0: (H, W) M0 calibration image
        scale_factor: Multiplier (default 100)
        
    Returns:
        Normalized signals with same shape
    """
    # Avoid division by zero - use 5th percentile as floor
    m0_safe = np.maximum(m0, np.percentile(m0[m0 > 0], 5) if np.any(m0 > 0) else 1.0)
    
    # Handle NaN/Inf
    normalized = (signals / m0_safe) * scale_factor
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return normalized.astype(np.float32)


def create_brain_mask(m0: np.ndarray, threshold_percentile: float = 50) -> np.ndarray:
    """
    Create binary brain mask from M0 image.
    
    Args:
        m0: (H, W) or (H, W, Z) M0 image
        threshold_percentile: Percentile for threshold (default 50)
        
    Returns:
        Binary mask (1=brain, 0=air)
    """
    if np.any(m0 > 0):
        threshold = np.percentile(m0[m0 > 0], threshold_percentile) * 0.3
    else:
        threshold = 0
    
    return (m0 > threshold).astype(np.float32)

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
    """
    def __init__(self, pld_values, t1_blood=1850.0, t_tau=1800.0, 
                 alpha_pcasl=0.85, alpha_vsasl=0.56, alpha_bs=1.0, 
                 t2_factor=1.0, t_sat_vs=2000.0):
        super().__init__()
        # Register all physics parameters as buffers for device compatibility
        self.register_buffer('plds', torch.tensor(pld_values, dtype=torch.float32))
        self.register_buffer('t1_blood', torch.tensor(t1_blood, dtype=torch.float32))
        self.register_buffer('t_tau', torch.tensor(t_tau, dtype=torch.float32))
        self.register_buffer('t2_factor', torch.tensor(t2_factor, dtype=torch.float32))
        self.register_buffer('t_sat_vs', torch.tensor(t_sat_vs, dtype=torch.float32))
        
        # Combined efficiencies
        self.register_buffer('alpha_pcasl_eff', torch.tensor(alpha_pcasl * (alpha_bs**4), dtype=torch.float32))
        self.register_buffer('alpha_vsasl_eff', torch.tensor(alpha_vsasl * (alpha_bs**3), dtype=torch.float32))
        self.register_buffer('lambda_blood', torch.tensor(0.90, dtype=torch.float32))
        self.register_buffer('unit_conv', torch.tensor(6000.0, dtype=torch.float32)) # Conversion ml/100g/min -> ml/g/s

    def forward(self, cbf, att):
        """
        Generate ASL signals from parameter maps.
        
        Args:
            cbf: (Batch, 1, H, W) - CBF in ml/100g/min
            att: (Batch, 1, H, W) - ATT in ms
        
        Returns:
            signals: (Batch, 2*N_plds, H, W) - [PCASL_t1...tn, VSASL_t1...tn]
        """
        # Expand dims for broadcasting: (Batch, N_plds, H, W)
        pld_exp = self.plds.view(1, -1, 1, 1)
        
        # Convert CBF to physiological units (ml/g/s)
        f = cbf / self.unit_conv
        
        # --- PCASL GENERATION ---
        # Condition 1: PLD < ATT - tau (Zero signal) - handled by masks
        # Condition 2: ATT - tau <= PLD < ATT
        term2_p = (2 * self.alpha_pcasl_eff * f * self.t1_blood / 1000.0 * 
                   (torch.exp(-att / self.t1_blood) - torch.exp(-(self.t_tau + pld_exp) / self.t1_blood)) * 
                   self.t2_factor) / self.lambda_blood
        
        # Condition 3: PLD >= ATT
        term3_p = (2 * self.alpha_pcasl_eff * f * self.t1_blood / 1000.0 * 
                   torch.exp(-pld_exp / self.t1_blood) * 
                   (1 - torch.exp(-self.t_tau / self.t1_blood)) * 
                   self.t2_factor) / self.lambda_blood
        
        # Soft masks for differentiability
        # steepness controls how sharp the transition is at ATT
        steep = 10.0 
        mask_arrived = torch.sigmoid((pld_exp - att) * steep)  # PLD >= ATT
        mask_transit = torch.sigmoid((pld_exp - (att - self.t_tau)) * steep) * (1 - mask_arrived)  # Window
        
        pcasl_sig = (term3_p * mask_arrived) + (term2_p * mask_transit)

        # --- VSASL GENERATION ---
        # Condition 1: PLD <= ATT
        term1_v = (2 * self.alpha_vsasl_eff * f * (pld_exp / 1000.0) * 
                   torch.exp(-pld_exp / self.t1_blood) * self.t2_factor) / self.lambda_blood
        
        # Condition 2: PLD > ATT
        term2_v = (2 * self.alpha_vsasl_eff * f * (att / 1000.0) * 
                   torch.exp(-pld_exp / self.t1_blood) * self.t2_factor) / self.lambda_blood

        mask_vs_arrived = torch.sigmoid((pld_exp - att) * steep)
        vsasl_sig = (term2_v * mask_vs_arrived) + (term1_v * (1 - mask_vs_arrived))

        # Concatenate Channel-wise: [PCASL_t1...tn, VSASL_t1...tn]
        # CRITICAL: Scale output to match the SpatialDataset's *100 normalization
        return torch.cat([pcasl_sig, vsasl_sig], dim=1) * 100.0


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

        # --- Data Consistency Loss (Self-Supervised) ---
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

            # Reconstruct signals from predicted parameters
            pred_signals = self.kinetic_model(pred_cbf_raw, pred_att_raw)

            # L1 difference between predicted and actual signals (masked)
            signal_diff = torch.abs(pred_signals - input_signals) * brain_mask
            dc_loss = self.dc_weight * signal_diff.sum() / mask_sum

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


class SpatialDataset(torch.utils.data.Dataset):
    """
    Dataset for spatial ASL training with proper normalization and augmentation.

    Implements:
    - M0 normalization: Input = (ΔS / M0) * 100
    - Per-pixel temporal normalization (optional): Z-score each pixel's temporal signal
      This creates "shape vectors" that are SNR-invariant, like the 1D model.
    - Random horizontal/vertical flips for spatial invariance
    - Random 90-degree rotations
    - Brain masking

    The per-pixel normalization is CRITICAL for learning because:
    1. It removes absolute signal magnitude, focusing on temporal shape
    2. Makes the model invariant to SNR variations
    3. Matches what the 1D model does (and the 1D model works!)
    """

    M0_SCALE_FACTOR = 100.0  # Scales ~0.01-0.05 signals to ~1.0-5.0 range

    def __init__(self, data_dir: str, transform: bool = True,
                 flip_prob: float = 0.5, per_pixel_norm: bool = True,
                 rotation_prob: float = 0.25):
        """
        Args:
            data_dir: Path to directory with spatial_chunk_*.npz files
            transform: Whether to apply augmentation
            flip_prob: Probability of random flips
            per_pixel_norm: If True, z-score normalize each pixel's temporal signal
            rotation_prob: Probability of 90-degree rotation
        """
        import glob
        self.data_files = sorted(glob.glob(f"{data_dir}/spatial_chunk_*.npz"))
        self.transform = transform
        self.flip_prob = flip_prob
        self.per_pixel_norm = per_pixel_norm
        self.rotation_prob = rotation_prob

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

        # --- Per-pixel temporal normalization ---
        # This is CRITICAL: z-score each pixel's temporal signal to create "shape vectors"
        # This makes the model SNR-invariant (same approach as 1D model)
        if self.per_pixel_norm:
            # signals shape: (C, H, W) - C is temporal dimension
            # Compute mean and std across temporal dimension for each pixel
            temporal_mean = np.mean(signals, axis=0, keepdims=True)  # (1, H, W)
            temporal_std = np.std(signals, axis=0, keepdims=True) + 1e-6  # (1, H, W)
            signals = (signals - temporal_mean) / temporal_std

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

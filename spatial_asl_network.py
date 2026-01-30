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
    """Standard U-Net double convolution block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SpatialASLNet(nn.Module):
    """
    U-Net architecture for Spatio-Temporal ASL Processing.
    
    Input: (Batch, N_plds * 2, Height, Width) - PCASL + VSASL channels
    Output: CBF_map, ATT_map, log_var_cbf, log_var_att
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

    def forward(self, x):
        """
        Forward pass of U-Net.
        
        Args:
            x: (Batch, Time, H, W) - Multi-PLD ASL signal
            
        Returns:
            cbf_map: (Batch, 1, H, W) - Positive via Softplus
            att_map: (Batch, 1, H, W) - 0-3000ms via Sigmoid scaling
            log_var_cbf: (Batch, 1, H, W) - Placeholder uncertainty
            log_var_att: (Batch, 1, H, W) - Placeholder uncertainty
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
        
        # Activation Constraints
        # Output is now normalized: 0.0 - 1.0 range implies 0 - 100 CBF, 0 - 3000 ATT
        cbf_map = F.softplus(out[:, 0:1, :, :]) * 100.0 
        att_map = 3000.0 * torch.sigmoid(out[:, 1:2, :, :])
        
        # Return 4 values to match existing trainer signature
        # (cbf, att, log_var_cbf, log_var_att)
        # We assume homoscedastic uncertainty for now (constant/learned scalar could be added)
        # For simplicity, returning constant log_var (exp(-5) ≈ 0.007 variance)
        zero_log = torch.zeros_like(cbf_map) - 5.0
        
        return cbf_map, att_map, zero_log, zero_log


class MaskedSpatialLoss(nn.Module):
    """
    Masked loss function that only counts brain pixels, not background air.

    This prevents the network from learning the trivial solution of predicting
    zero everywhere (which would score well on 80% air background).

    Loss Modes:
    - 'l1' or 'mae': L1 loss (Mean Absolute Error) - RECOMMENDED
    - 'l2' or 'mse': L2 loss (Mean Squared Error)
    - 'huber': Huber loss (robust to outliers)

    ATT Scaling:
    ATT values (500-3000ms) are ~30x larger than CBF (20-100), which can cause
    the loss to be dominated by ATT errors. We scale ATT loss by 1/30 by default.
    """
    def __init__(self, loss_type: str = 'l1', dc_weight: float = 0.0,
                 kinetic_model: nn.Module = None,
                 att_scale: float = 0.033,  # 1/30 to balance CBF vs ATT
                 cbf_weight: float = 1.0,
                 att_weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.dc_weight = dc_weight
        self.kinetic_model = kinetic_model
        self.att_scale = att_scale
        self.cbf_weight = cbf_weight
        self.att_weight = att_weight

        print(f"[MaskedSpatialLoss] loss_type={loss_type}, att_scale={att_scale}, "
              f"cbf_weight={cbf_weight}, att_weight={att_weight}, dc_weight={dc_weight}")

    def forward(self, pred_cbf: torch.Tensor, pred_att: torch.Tensor,
                target_cbf: torch.Tensor, target_att: torch.Tensor,
                brain_mask: torch.Tensor,
                input_signals: torch.Tensor = None) -> dict:
        """
        Compute masked loss.

        Args:
            pred_cbf: (B, 1, H, W) predicted CBF
            pred_att: (B, 1, H, W) predicted ATT
            target_cbf: (B, 1, H, W) ground truth CBF
            target_att: (B, 1, H, W) ground truth ATT
            brain_mask: (B, 1, H, W) binary mask (1=brain, 0=air)
            input_signals: (B, 2*PLDs, H, W) for data consistency loss

        Returns:
            dict with 'total_loss' and component losses
        """
        # Expand mask if needed
        if brain_mask.dim() == 3:
            brain_mask = brain_mask.unsqueeze(1)

        mask_sum = brain_mask.sum() + 1e-6

        # --- Compute errors ---
        cbf_err = pred_cbf - target_cbf
        att_err = pred_att - target_att

        # --- Apply loss function ---
        if self.loss_type in ['l1', 'mae']:
            cbf_loss_map = torch.abs(cbf_err) * brain_mask
            att_loss_map = torch.abs(att_err) * brain_mask
        elif self.loss_type in ['l2', 'mse']:
            cbf_loss_map = (cbf_err ** 2) * brain_mask
            att_loss_map = (att_err ** 2) * brain_mask
        elif self.loss_type == 'huber':
            delta = 1.0
            cbf_loss_map = torch.where(
                torch.abs(cbf_err) < delta,
                0.5 * cbf_err ** 2,
                delta * (torch.abs(cbf_err) - 0.5 * delta)
            ) * brain_mask
            att_loss_map = torch.where(
                torch.abs(att_err) < delta * 30,  # Scale delta for ATT range
                0.5 * att_err ** 2,
                delta * 30 * (torch.abs(att_err) - 0.5 * delta * 30)
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

        # --- Data Consistency Loss (Self-Supervised) ---
        dc_loss = torch.tensor(0.0, device=pred_cbf.device)

        if self.dc_weight > 0 and self.kinetic_model is not None and input_signals is not None:
            # Reconstruct signals from predicted parameters
            pred_signals = self.kinetic_model(pred_cbf, pred_att)

            # L1 difference between predicted and actual signals (masked)
            signal_diff = torch.abs(pred_signals - input_signals) * brain_mask
            dc_loss = self.dc_weight * signal_diff.sum() / mask_sum

        total_loss = supervised_loss + dc_loss

        # Return unscaled losses for logging (easier to interpret)
        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'cbf_loss': cbf_loss,  # Unscaled for interpretability
            'att_loss': att_loss,  # Unscaled
            'att_loss_scaled': att_loss_scaled,
            'dc_loss': dc_loss
        }


class SpatialDataset(torch.utils.data.Dataset):
    """
    Dataset for spatial ASL training with proper M0 normalization and augmentation.
    
    Implements:
    - M0 normalization: Input = (ΔS / M0) * 100
    - Random horizontal/vertical flips for spatial invariance
    - Brain masking
    """
    
    M0_SCALE_FACTOR = 100.0  # Scales ~0.01-0.05 signals to ~1.0-5.0 range
    
    def __init__(self, data_dir: str, transform: bool = True, 
                 flip_prob: float = 0.5):
        """
        Args:
            data_dir: Path to directory with spatial_chunk_*.npz files
            transform: Whether to apply augmentation
            flip_prob: Probability of random flips
        """
        import glob
        self.data_files = sorted(glob.glob(f"{data_dir}/spatial_chunk_*.npz"))
        self.transform = transform
        self.flip_prob = flip_prob
        
        # Preload data into RAM
        if self.data_files:
            # PRELOAD STRATEGY: Load all 20GB into RAM to stop I/O thrashing
            print(f"[SpatialDataset] Pre-loading {len(self.data_files)} chunks to RAM...")
            all_sig, all_tgt = [], []
            for f in self.data_files:
                d = np.load(f)
                all_sig.append(d['signals'])
                all_tgt.append(d['targets'])
            
            # Concatenate and normalize immediately
            self.signals = np.concatenate(all_sig, axis=0) * self.M0_SCALE_FACTOR

            # Keep targets in original units (CBF: ml/100g/min, ATT: ms)
            # Model outputs are scaled to match: CBF via softplus*100, ATT via sigmoid*3000
            self.targets = np.concatenate(all_tgt, axis=0)
            
            self.total_samples = len(self.signals)
            print(f"[SpatialDataset] Loaded {self.total_samples} samples. RAM: {self.signals.nbytes/1e9:.1f} GB")
        else:
            self.total_samples = 0
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # RAM Access (Instant)
        signals = self.signals[idx].copy()
        targets = self.targets[idx]
        
        # --- Create brain mask (non-zero regions) ---
        # Use mean signal across time as proxy for tissue
        mean_signal = np.mean(np.abs(signals), axis=0)
        brain_mask = (mean_signal > np.percentile(mean_signal, 5)).astype(np.float32)
        
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

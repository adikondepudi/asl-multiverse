import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Note on Future Spatial Architectures (Aligning with Shou et al. insights):
# While the CNNs below are good starting points, for SOTA performance on spatial ASL data,
# consider exploring Transformer-based architectures like:
# - SwinIR or Swin UNETR
# - UNETR
# - TransUNet
# These have shown strong performance in medical image analysis, including ASL,
# particularly when dealing with complex spatial dependencies and noise patterns.
# Key strategies from Shou et al. to incorporate would be:
# 1. Using M0 images as an additional input channel.
# 2. Employing pseudo-3D inputs (e.g., stacking 3-5 adjacent 2D slices as channels).
# 3. For multi-delay ASL image sequences, combining spatial and temporal processing,
#    potentially by treating PLDs as additional channels or using spatio-temporal Transformers.


class ASLSpatialDataset(Dataset):
    """Dataset for spatial ASL data"""
    
    def __init__(self, 
                 noisy_data: np.ndarray, # (n_samples, n_channels/PLDs, depth, height, width) for 3D or (n_samples, n_channels/PLDs, height, width) for 2D
                 clean_data: np.ndarray,
                 transform=None):
        self.noisy_data = torch.FloatTensor(noisy_data)
        self.clean_data = torch.FloatTensor(clean_data)
        self.transform = transform
        
    def __len__(self):
        return len(self.noisy_data)
    
    def __getitem__(self, idx):
        noisy = self.noisy_data[idx]
        clean = self.clean_data[idx]
        
        if self.transform: # Augmentations could be applied here
            # Example: random flips, rotations if applicable
            pass # Placeholder for actual transform logic
            
        return noisy, clean


class ResidualBlock3D(nn.Module):
    """3D Residual block for spatial-temporal processing"""
    
    def __init__(self, channels: int):
        super().__init__()
        # Using kernel size 3, padding 1 to keep spatial dimensions same
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False) # Bias often false with BN
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # Skip connection
        return F.relu(out)


class SpatialEnhancementCNN(nn.Module): # U-Net like architecture for 3D data
    """CNN for spatial enhancement of ASL images (e.g., denoising 3D ASL volumes over PLDs)"""
    
    def __init__(self,
                 in_channels: int = 6,  # E.g., Number of PLDs treated as channels if input is (Batch, PLDs, Depth, H, W)
                 out_channels: int = 6, # Usually same as in_channels for denoising
                 base_features: int = 32): # Number of features in the first conv layer
        super().__init__()
        
        # Encoder path (contracting path)
        self.enc1 = self._conv_block(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) # Downsample
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 4, base_features * 8)
        
        # Decoder path (expansive path)
        self.upconv3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4) # After concat: (base*4 from upconv + base*4 from enc3 skip)
        self.upconv2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2) # After concat: (base*2 + base*2)
        self.upconv1 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_features * 2, base_features) # After concat: (base + base)
        
        # Output layer
        self.output_conv = nn.Conv3d(base_features, out_channels, kernel_size=1) # 1x1x1 conv to map to out_channels
        
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        # Standard double convolution block
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1_out = self.enc1(x)    # -> base_features
        enc2_in = self.pool1(enc1_out)
        enc2_out = self.enc2(enc2_in)  # -> base_features * 2
        enc3_in = self.pool2(enc2_out)
        enc3_out = self.enc3(enc3_in)  # -> base_features * 4
        
        bottleneck_in = self.pool3(enc3_out)
        bottleneck_out = self.bottleneck(bottleneck_in) # -> base_features * 8
        
        # Decoder
        dec3_in_up = self.upconv3(bottleneck_out) 
        # Concatenate skip connection from encoder
        dec3_in_cat = torch.cat([dec3_in_up, enc3_out], dim=1) # (BF*4 + BF*4 = BF*8)
        dec3_out = self.dec3(dec3_in_cat) # -> BF*4
        
        dec2_in_up = self.upconv2(dec3_out)
        dec2_in_cat = torch.cat([dec2_in_up, enc2_out], dim=1) # (BF*2 + BF*2 = BF*4)
        dec2_out = self.dec2(dec2_in_cat) # -> BF*2
        
        dec1_in_up = self.upconv1(dec2_out)
        dec1_in_cat = torch.cat([dec1_in_up, enc1_out], dim=1) # (BF + BF = BF*2)
        dec1_out = self.dec1(dec1_in_cat) # -> BF
        
        return self.output_conv(dec1_out)


class SpatioTemporalASLNet(nn.Module):
    """Combined spatial (2D slice) and temporal (1D PLD vector) processing for ASL parameter estimation."""
    
    def __init__(self,
                 # Spatial branch (2D CNN for a single slice at multiple PLDs)
                 spatial_in_channels: int = 12, # e.g., 6 PLDs for PCASL + 6 PLDs for VSASL, stacked as channels
                 spatial_base_features: int = 32,
                 spatial_output_features: int = 64, # Features from spatial branch
                 
                 # Temporal branch (LSTM/Transformer for 1D signal vector)
                 temporal_input_size: int, # e.g., n_plds * 2 (if using flattened signal vector)
                 temporal_hidden_dim: int = 128,
                 temporal_n_layers: int = 2, # For LSTM or Transformer layers
                 use_transformer_temporal: bool = True, # Flag for temporal branch
                 transformer_nhead: int = 4,
                 
                 # Fusion and output
                 fusion_dropout_rate: float = 0.1,
                 num_output_params: int = 2 # CBF, ATT
                ):
        super().__init__()
        
        # Spatial processing branch (Simplified 2D CNN)
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(spatial_in_channels, spatial_base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_base_features), nn.ReLU(),
            nn.Conv2d(spatial_base_features, spatial_base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_base_features * 2), nn.ReLU(),
            nn.Conv2d(spatial_base_features * 2, spatial_output_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_output_features), nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # To get a feature vector from spatial maps
        
        # Temporal processing branch
        self.use_transformer_temporal = use_transformer_temporal
        if use_transformer_temporal:
            # Assuming temporal_input_size is the raw feature dim before projection
            # Let's project it to temporal_hidden_dim to serve as d_model for transformer
            self.temporal_input_projection = nn.Linear(temporal_input_size, temporal_hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=temporal_hidden_dim, nhead=transformer_nhead,
                dim_feedforward=temporal_hidden_dim * 2, batch_first=True)
            self.temporal_processor = nn.TransformerEncoder(encoder_layer, num_layers=temporal_n_layers)
            self.temporal_output_features = temporal_hidden_dim
        else: # LSTM
            self.temporal_processor = nn.LSTM(temporal_input_size, temporal_hidden_dim, 
                                            num_layers=temporal_n_layers, batch_first=True,
                                            bidirectional=True)
            self.temporal_output_features = temporal_hidden_dim * 2 # Bidirectional
            
        # Fusion and output layers
        fusion_input_dim = spatial_output_features + self.temporal_output_features
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2), nn.ReLU(),
            nn.Dropout(fusion_dropout_rate),
            nn.Linear(fusion_input_dim // 2, fusion_input_dim // 4), nn.ReLU(),
            nn.Dropout(fusion_dropout_rate)
        )
        
        # Parameter estimation heads (mean and log_var for uncertainty)
        self.param_heads = nn.ModuleList()
        for _ in range(num_output_params): # CBF, ATT
            self.param_heads.append(nn.Linear(fusion_input_dim // 4, 2)) # 2 for mean and log_var

    def forward(self, spatial_input: torch.Tensor, temporal_input: torch.Tensor) \
            -> Tuple[torch.Tensor, ...]:
        """
        spatial_input: (batch, spatial_in_channels, height, width) - e.g., multi-PLD image data for one slice
        temporal_input: (batch, temporal_input_size) - e.g., spatially averaged signal vector
        """
        # Spatial features
        spatial_f = self.spatial_cnn(spatial_input)
        spatial_f = self.global_avg_pool(spatial_f)
        spatial_f = spatial_f.view(spatial_f.size(0), -1) # Flatten: (batch, spatial_output_features)
        
        # Temporal features
        if self.use_transformer_temporal:
            # Temporal input might need reshaping or projection if not already (batch, seq_len, features)
            # Assuming temporal_input is (batch, flat_features)
            # Project to (batch, embed_dim) then unsqueeze to (batch, 1, embed_dim) for transformer
            # This implies a transformer processing a single "global" temporal feature vector.
            # If temporal_input is already (batch, seq_len, features_per_step), adjust accordingly.
            
            # If temporal_input is (batch, temporal_input_size_flat)
            projected_temp_input = self.temporal_input_projection(temporal_input) # (batch, temporal_hidden_dim)
            # Transformer expects a sequence. If we treat the whole vector as one step:
            temp_input_seq = projected_temp_input.unsqueeze(1) # (batch, 1, temporal_hidden_dim)
            temporal_out_seq = self.temporal_processor(temp_input_seq) # (batch, 1, temporal_hidden_dim)
            temporal_f = temporal_out_seq.squeeze(1) # (batch, temporal_hidden_dim)
        else: # LSTM
            # LSTM expects (batch, seq_len, input_size_per_step)
            # If temporal_input is (batch, flat_features), treat as seq_len=1
            temp_input_seq = temporal_input.unsqueeze(1) # (batch, 1, temporal_input_size)
            lstm_out, (hn, cn) = self.temporal_processor(temp_input_seq)
            # lstm_out is (batch, 1, hidden_dim*2). Squeeze seq_len dim.
            temporal_f = lstm_out.squeeze(1) # (batch, hidden_dim*2)
            
        # Fusion
        combined_features = torch.cat([spatial_f, temporal_f], dim=1)
        fused_out = self.fusion_layers(combined_features)
        
        # Parameter estimation
        outputs = []
        for head in self.param_heads:
            param_mean_logvar = head(fused_out) # (batch, 2)
            outputs.append(param_mean_logvar[:, 0:1]) # Mean
            outputs.append(param_mean_logvar[:, 1:2]) # Log_var
            
        return tuple(outputs) # (cbf_mean, cbf_log_var, att_mean, att_log_var, ...)


class PerceptualLoss(nn.Module):
    """Perceptual loss for preserving ASL signal characteristics (e.g., for image denoising)"""
    # This is more relevant for image-to-image tasks like denoising.
    # For parameter estimation, direct MSE/NLL on parameters is more common.
    
    def __init__(self, feature_extractor: nn.Module, feature_layer_names: List[str]):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_layer_names = feature_layer_names
        # Make sure feature_extractor doesn't update its weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
    def forward(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        pred_features = self._get_features(pred_img)
        target_features = self._get_features(target_img)
        
        for layer_name in self.feature_layer_names:
            loss += F.l1_loss(pred_features[layer_name], target_features[layer_name])
            
        return loss

    def _get_features(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Helper to extract features from specified layers of a pre-trained network
        # This depends heavily on the architecture of `feature_extractor`
        # Example for a VGG-like network:
        features = {}
        x = img
        # for name, module in self.feature_extractor.features.named_children():
        #     x = module(x)
        #     if name in self.feature_layer_names:
        #         features[name] = x
        # This part is highly dependent on the chosen feature extractor.
        # For ASL, a custom feature extractor might be more relevant than VGG.
        # Or, use simpler image gradient / structural similarity losses.
        return features # Placeholder


class SpatialEnhancementTrainer:
    """Training manager for spatial enhancement networks (e.g., U-Net for denoising)"""
    
    def __init__(self,
                 model: nn.Module, # e.g., SpatialEnhancementCNN
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        self.reconstruction_loss = nn.MSELoss() # Or L1Loss
        # self.perceptual_loss = PerceptualLoss(...) # If using perceptual loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True) # verbose for LR changes
        
    def _total_variation_loss(self, img_batch: torch.Tensor) -> torch.Tensor:
        """Total variation loss for spatial smoothness (for 3D data B,C,D,H,W)"""
        # Assumes img_batch is (B, C, D, H, W)
        tv_d = torch.sum(torch.abs(img_batch[:, :, :-1, :, :] - img_batch[:, :, 1:, :, :]))
        tv_h = torch.sum(torch.abs(img_batch[:, :, :, :-1, :] - img_batch[:, :, :, 1:, :]))
        tv_w = torch.sum(torch.abs(img_batch[:, :, :, :, :-1] - img_batch[:, :, :, :, 1:]))
        
        # Normalize by number of elements (pixels) where differences are computed
        # This normalization makes the loss less dependent on image size.
        num_elements = img_batch.size(0) * img_batch.size(1) * \
                       ((img_batch.size(2)-1)*img_batch.size(3)*img_batch.size(4) + \
                        img_batch.size(2)*(img_batch.size(3)-1)*img_batch.size(4) + \
                        img_batch.size(2)*img_batch.size(3)*(img_batch.size(4)-1))
        if num_elements == 0: return torch.tensor(0.0).to(img_batch.device) # Avoid div by zero for single pixel/slice dimensions
        
        return (tv_d + tv_h + tv_w) / num_elements if num_elements > 0 else torch.tensor(0.0).to(img_batch.device)

    def train_epoch(self, train_loader: DataLoader, tv_weight: float = 0.001) -> float:
        self.model.train()
        total_loss_epoch = 0.0
        
        for noisy_imgs, clean_imgs in train_loader: # Assuming loader yields (noisy_3D_volume, clean_3D_volume)
            noisy_imgs = noisy_imgs.to(self.device) # (B, C=PLDs, D, H, W)
            clean_imgs = clean_imgs.to(self.device)
            
            self.optimizer.zero_grad()
            enhanced_imgs = self.model(noisy_imgs)
            
            recon_loss = self.reconstruction_loss(enhanced_imgs, clean_imgs)
            # perceptual_loss_val = self.perceptual_loss(enhanced_imgs, clean_imgs) # If used
            tv_loss_val = self._total_variation_loss(enhanced_imgs)
            
            # Combined loss (adjust weights as needed)
            # loss = recon_loss + 0.1 * perceptual_loss_val + tv_weight * tv_loss_val
            loss = recon_loss + tv_weight * tv_loss_val
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clip gradients
            self.optimizer.step()
            
            total_loss_epoch += loss.item()
            
        avg_loss = total_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_mse, total_psnr, total_ssim = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(self.device), clean_imgs.to(self.device)
                enhanced_imgs = self.model(noisy_imgs)
                
                mse = F.mse_loss(enhanced_imgs, clean_imgs, reduction='mean') # Mean over all elements
                total_mse += mse.item()
                
                if mse.item() > 0: # Avoid log(0)
                    # PSNR is typically defined for images scaled [0, MAX_VAL]
                    # Assuming images are normalized, MAX_VAL might be 1.0 or 255.
                    # If signal range is small, PSNR can be tricky.
                    # For ASL diff signals, range might be e.g. [-0.01, 0.01] after M0 norm.
                    # Let's assume signals are roughly in [-1, 1] or [0,1] for PSNR.
                    # If data_max is not 1, it should be provided. For now, assume 1.
                    data_max = 1.0 
                    psnr = 20 * torch.log10(data_max / torch.sqrt(mse))
                    total_psnr += psnr.item()
                
                # For SSIM, need to ensure inputs are appropriate (e.g. single channel images)
                # This SSIM is very basic. A library like kornia or skimage.metrics.structural_similarity is better.
                # Assuming enhanced_imgs and clean_imgs are (B,C,D,H,W)
                # We might calculate SSIM per channel/slice and average.
                # For simplicity, applying basic SSIM logic, but this needs careful thought for volumetric multi-channel data.
                # ssim_val = self._compute_ssim_basic(enhanced_imgs, clean_imgs) # Custom simplified SSIM
                # total_ssim += ssim_val

        n_batches = len(val_loader) if len(val_loader) > 0 else 1
        return {
            'mse': total_mse / n_batches,
            'psnr': total_psnr / n_batches if total_psnr > 0 else np.nan, # Handle cases where PSNR wasn't computed
            # 'ssim': total_ssim / n_batches # Add back if _compute_ssim_basic is robust
        }
    
    def _compute_ssim_basic(self, img1: torch.Tensor, img2: torch.Tensor, C1=0.01**2, C2=0.03**2, window_size=7, sigma=1.5) -> float:
        # Simplified SSIM, not robust for all cases, especially 3D.
        # For 3D, consider processing slice by slice or using a 3D SSIM implementation.
        # This is a placeholder. For real use, use a library function.
        
        # Reduce to 2D if 3D for this basic version: take middle slice, first channel
        if img1.ndim == 5: # B, C, D, H, W
            img1_2d = img1[:, 0, img1.size(2)//2, :, :].unsqueeze(1) # B, 1, H, W
            img2_2d = img2[:, 0, img2.size(2)//2, :, :].unsqueeze(1)
        elif img1.ndim == 4: # B, C, H, W (e.g. multi-channel 2D)
             img1_2d = img1[:, 0, :, :].unsqueeze(1) # Take first channel
             img2_2d = img2[:, 0, :, :].unsqueeze(1)
        else: # Assume B, H, W or H, W, needs unsqueeze to B,1,H,W
            # This part needs to be robust based on expected input dims
            return 0.0 # Cannot compute for unexpected dims

        # Use 2D average pooling as a simple local mean/variance estimator
        mu1 = F.avg_pool2d(img1_2d, window_size, 1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2_2d, window_size, 1, padding=window_size//2)
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1_2d * img1_2d, window_size, 1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2_2d * img2_2d, window_size, 1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1_2d * img2_2d, window_size, 1, padding=window_size//2) - mu1_mu2

        ssim_num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ssim_den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = ssim_num / ssim_den
        return ssim_map.mean().item()


def create_synthetic_spatial_data(n_samples: int = 100,
                                matrix_size: Tuple[int, int] = (32, 32), # Smaller for faster test
                                n_slices_depth: int = 8, # Depth for 3D volume
                                n_plds_channels: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    from enhanced_simulation import RealisticASLSimulator # Local import
    simulator = RealisticASLSimulator()
    plds_arr = np.arange(500, 3001, 500)[:n_plds_channels]
    
    # Output shape: (n_samples, n_plds_channels, n_slices_depth, matrix_size_h, matrix_size_w)
    all_noisy_data = np.zeros((n_samples, n_plds_channels, n_slices_depth, matrix_size[0], matrix_size[1]))
    all_clean_data = np.zeros_like(all_noisy_data)
    
    print("Generating synthetic spatial data for CNN testing...")
    for i in tqdm(range(n_samples), desc="Spatial Samples"):
        # Generate 3D spatial data (H, W, D, PLDs) from simulator
        # For this example, we'll use the PCASL generation for simplicity.
        # A full MULTIVERSE spatial sim would generate both and stack them as channels.
        raw_spatial_pcasl, _, _ = simulator.generate_spatial_data(
            matrix_size=matrix_size, n_slices=n_slices_depth, plds=plds_arr
        ) # Returns (H, W, D, PLDs)
        
        # Transpose to (PLDs, D, H, W) - channels first for PyTorch Conv3D
        clean_sample_volume = raw_spatial_pcasl.transpose(3, 2, 0, 1) # PLDs, D, H, W
        
        # Add noise to each PLD volume - this is simplified.
        # In reality, noise is added per PLD acquisition instance.
        # Here, adding to the already generated "clean" volume for simplicity of example.
        noisy_sample_volume = np.zeros_like(clean_sample_volume)
        for p_idx in range(n_plds_channels):
            # Add noise to each (D,H,W) volume for this PLD
            # Using simulator.add_realistic_noise which expects 1D or 2D.
            # For 3D volume, need to adapt or apply noise differently.
            # Simple Gaussian noise for this example:
            pld_volume_clean = clean_sample_volume[p_idx]
            noise_level = np.mean(np.abs(pld_volume_clean)) / np.random.uniform(3,10) # Random SNR
            noise = np.random.normal(0, noise_level, pld_volume_clean.shape)
            noisy_sample_volume[p_idx] = pld_volume_clean + noise

        all_clean_data[i] = clean_sample_volume
        all_noisy_data[i] = noisy_sample_volume
            
    return all_noisy_data, all_clean_data


if __name__ == "__main__":
    print("Testing Spatial Enhancement CNN with 3D U-Net...")
    
    # Create small synthetic 3D dataset for testing
    # Shape: (n_samples, n_channels=PLDs, depth, height, width)
    test_noisy_data, test_clean_data = create_synthetic_spatial_data(
        n_samples=10, matrix_size=(32, 32), n_slices_depth=8, n_plds_channels=3 # Small for quick test
    )
    print(f"Generated data shape: Noisy {test_noisy_data.shape}, Clean {test_clean_data.shape}")
    
    dataset = ASLSpatialDataset(test_noisy_data, test_clean_data)
    # Batch size needs to be small if data is large (e.g. 1 or 2 for full 3D volumes)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True) 
    
    # Model: in_channels = n_plds_channels, out_channels = n_plds_channels
    model = SpatialEnhancementCNN(in_channels=test_noisy_data.shape[1], 
                                  out_channels=test_clean_data.shape[1],
                                  base_features=16) # Reduced base features for faster test
    trainer = SpatialEnhancementTrainer(model)
    
    print("Training SpatialEnhancementCNN for a few epochs...")
    for epoch in range(3): # Few epochs for demonstration
        avg_loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch+1}/3, Avg Training Loss: {avg_loss:.6f}")
        if epoch % 2 == 0 or epoch == 2: # Evaluate periodically
             eval_metrics = trainer.evaluate(train_loader) # Evaluate on same data for simplicity here
             print(f"  Evaluation on (pseudo) Val Set - MSE: {eval_metrics.get('mse',0):.6f}, PSNR: {eval_metrics.get('psnr',0):.2f} dB")
    
    print("\nTesting SpatioTemporalASLNet (Combined Model)...")
    # Dummy inputs for SpatioTemporalASLNet
    batch_s = 4
    n_plds_st = 6 # Number of PLDs for the temporal part
    spatial_channels_st = n_plds_st * 2 # PCASL+VSASL PLDs as channels for spatial input
    
    # Dummy spatial input: (batch, spatial_channels, H, W)
    dummy_spatial_input = torch.randn(batch_s, spatial_channels_st, 32, 32)
    # Dummy temporal input: (batch, n_plds*2 features) -> for one subject, the 1D signal vector
    dummy_temporal_input_flat = torch.randn(batch_s, n_plds_st * 2)
    
    st_model = SpatioTemporalASLNet(
        spatial_in_channels=spatial_channels_st,
        spatial_output_features=32, # Reduced for test
        temporal_input_size=n_plds_st * 2, # Flat input for temporal
        temporal_hidden_dim=64, # Reduced for test
        use_transformer_temporal=True, # Test with transformer
        transformer_nhead=2
    )
    
    print("Forward pass through SpatioTemporalASLNet...")
    cbf_m, cbf_lv, att_m, att_lv = st_model(dummy_spatial_input, dummy_temporal_input_flat)
    print(f"  Output shapes: CBF mean {cbf_m.shape}, CBF log_var {cbf_lv.shape}, ATT mean {att_m.shape}, ATT log_var {att_lv.shape}")
    print("SpatioTemporalASLNet test completed.")

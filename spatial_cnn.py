import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ASLSpatialDataset(Dataset):
    """Dataset for spatial ASL data"""
    
    def __init__(self, 
                 noisy_data: np.ndarray,
                 clean_data: np.ndarray,
                 transform=None):
        """
        Parameters
        ----------
        noisy_data : np.ndarray
            4D array (n_samples, n_channels, height, width)
        clean_data : np.ndarray
            4D array (n_samples, n_channels, height, width)
        """
        self.noisy_data = torch.FloatTensor(noisy_data)
        self.clean_data = torch.FloatTensor(clean_data)
        self.transform = transform
        
    def __len__(self):
        return len(self.noisy_data)
    
    def __getitem__(self, idx):
        noisy = self.noisy_data[idx]
        clean = self.clean_data[idx]
        
        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
            
        return noisy, clean


class ResidualBlock3D(nn.Module):
    """3D Residual block for spatial-temporal processing"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)


class SpatialEnhancementCNN(nn.Module):
    """CNN for spatial enhancement of ASL images"""
    
    def __init__(self,
                 in_channels: int = 6,  # Number of PLDs
                 out_channels: int = 6,
                 base_features: int = 32):
        super().__init__()
        
        # Encoder path
        self.enc1 = self._encoder_block(in_channels, base_features)
        self.enc2 = self._encoder_block(base_features, base_features * 2)
        self.enc3 = self._encoder_block(base_features * 2, base_features * 4)
        self.enc4 = self._encoder_block(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = ResidualBlock3D(base_features * 8)
        
        # Decoder path
        self.dec4 = self._decoder_block(base_features * 8, base_features * 4)
        self.dec3 = self._decoder_block(base_features * 8, base_features * 2)  # Skip connection
        self.dec2 = self._decoder_block(base_features * 4, base_features)
        self.dec1 = self._decoder_block(base_features * 2, base_features)
        
        # Output layer
        self.output = nn.Conv3d(base_features, out_channels, 1)
        
    def _encoder_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _decoder_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, 2))
        enc3 = self.enc3(F.max_pool3d(enc2, 2))
        enc4 = self.enc4(F.max_pool3d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, 2))
        
        # Decoder with skip connections
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        
        # Output
        return self.output(dec1)


class SpatioTemporalASLNet(nn.Module):
    """Combined spatial and temporal processing for ASL"""
    
    def __init__(self,
                 spatial_size: Tuple[int, int] = (64, 64),
                 n_plds: int = 6,
                 hidden_dim: int = 128):
        super().__init__()
        
        # Spatial processing branch
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(n_plds * 2, 64, 3, padding=1),  # PCASL + VSASL
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Global average pooling to combine spatial information
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal processing branch (similar to existing network)
        self.temporal_lstm = nn.LSTM(n_plds * 2, hidden_dim, 
                                    num_layers=2, 
                                    batch_first=True,
                                    bidirectional=True)
        
        # Fusion and output layers
        fusion_dim = 32 + hidden_dim * 2  # CNN features + bidirectional LSTM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Parameter estimation heads
        self.cbf_head = nn.Linear(64, 1)
        self.att_head = nn.Linear(64, 1)
        
        # Uncertainty estimation
        self.cbf_uncertainty = nn.Linear(64, 1)
        self.att_uncertainty = nn.Linear(64, 1)
        
    def forward(self, spatial_data: torch.Tensor, 
                temporal_data: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        spatial_data : torch.Tensor
            Shape (batch, channels, height, width)
        temporal_data : torch.Tensor
            Shape (batch, sequence_length)
        """
        # Spatial processing
        spatial_features = self.spatial_cnn(spatial_data)
        spatial_features = self.global_pool(spatial_features)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        
        # Temporal processing
        temporal_data = temporal_data.unsqueeze(1)  # Add sequence dimension
        lstm_out, _ = self.temporal_lstm(temporal_data)
        temporal_features = lstm_out.squeeze(1)
        
        # Fusion
        combined = torch.cat([spatial_features, temporal_features], dim=1)
        fused = self.fusion(combined)
        
        # Parameter estimation
        cbf = self.cbf_head(fused)
        att = self.att_head(fused)
        cbf_unc = self.cbf_uncertainty(fused)
        att_unc = self.att_uncertainty(fused)
        
        return cbf, att, cbf_unc, att_unc


class PerceptualLoss(nn.Module):
    """Perceptual loss for preserving ASL signal characteristics"""
    
    def __init__(self, feature_weights: List[float] = [1.0, 0.5, 0.25]):
        super().__init__()
        self.feature_weights = feature_weights
        
        # Feature extraction layers
        self.features = nn.ModuleList([
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1)
        ])
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0
        
        x_pred = pred
        x_target = target
        
        for i, (layer, weight) in enumerate(zip(self.features, self.feature_weights)):
            x_pred = F.relu(layer(x_pred))
            x_target = F.relu(layer(x_target))
            
            loss += weight * F.mse_loss(x_pred, x_target)
            
            if i < len(self.features) - 1:
                x_pred = F.max_pool2d(x_pred, 2)
                x_target = F.max_pool2d(x_target, 2)
                
        return loss


class SpatialEnhancementTrainer:
    """Training manager for spatial enhancement networks"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.smoothness_loss = self._total_variation_loss
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5)
        
    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation loss for spatial smoothness"""
        batch_size = x.size(0)
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return (h_tv + w_tv) / batch_size
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            enhanced = self.model(noisy)
            
            # Compute losses
            recon_loss = self.reconstruction_loss(enhanced, clean)
            percept_loss = self.perceptual_loss(enhanced, clean)
            smooth_loss = self.smoothness_loss(enhanced)
            
            # Combined loss
            loss = recon_loss + 0.1 * percept_loss + 0.01 * smooth_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        
        total_mse = 0
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                enhanced = self.model(noisy)
                
                # MSE
                mse = F.mse_loss(enhanced, clean)
                total_mse += mse.item()
                
                # PSNR
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr.item()
                
                # SSIM (simplified version)
                ssim = self._compute_ssim(enhanced, clean)
                total_ssim += ssim
                
        n_batches = len(val_loader)
        return {
            'mse': total_mse / n_batches,
            'psnr': total_psnr / n_batches,
            'ssim': total_ssim / n_batches
        }
    
    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Simplified SSIM computation"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        
        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
               
        return ssim.mean().item()


def create_synthetic_spatial_data(n_samples: int = 100,
                                matrix_size: Tuple[int, int] = (64, 64),
                                n_plds: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic spatial data for testing"""
    
    from enhanced_simulation import RealisticASLSimulator
    
    simulator = RealisticASLSimulator()
    plds = np.arange(500, 3001, 500)[:n_plds]
    
    noisy_data = []
    clean_data = []
    
    print("Generating synthetic spatial data...")
    for _ in range(n_samples):
        # Generate spatial data
        data_4d, cbf_map, att_map = simulator.generate_spatial_data(
            matrix_size=matrix_size,
            n_slices=1,
            plds=plds
        )
        
        # Extract single slice
        clean_slice = data_4d[:, :, 0, :]
        
        # Add noise
        noisy_slice = clean_slice + np.random.normal(0, 0.001, clean_slice.shape)
        
        # Reshape for CNN (channels first)
        clean_data.append(clean_slice.transpose(2, 0, 1))
        noisy_data.append(noisy_slice.transpose(2, 0, 1))
    
    return np.array(noisy_data), np.array(clean_data)


if __name__ == "__main__":
    # Test the spatial enhancement CNN
    print("Testing Spatial Enhancement CNN...")
    
    # Create synthetic data
    noisy_data, clean_data = create_synthetic_spatial_data(
        n_samples=50,
        matrix_size=(64, 64),
        n_plds=6
    )
    
    print(f"Data shape: {noisy_data.shape}")
    
    # Create dataset and dataloader
    dataset = ASLSpatialDataset(noisy_data, clean_data)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create and train model
    model = SpatialEnhancementCNN(in_channels=6, out_channels=6)
    trainer = SpatialEnhancementTrainer(model)
    
    # Train for a few epochs
    for epoch in range(5):
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # Evaluate
    metrics = trainer.evaluate(train_loader)
    print(f"Evaluation metrics: {metrics}")
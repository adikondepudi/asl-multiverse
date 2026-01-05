# FILE: noise_engine.py
"""
Single Source of Truth for Noise Generation.
Handles both PyTorch Tensors (Trainer) and NumPy Arrays (Validation/Sim).
Supports both 1D voxel-wise and 2D spatial processing.
"""
import torch
import numpy as np
from typing import Dict, Union, List, Tuple
from scipy.ndimage import gaussian_filter, affine_transform


class NoiseInjector:
    """
    Single Source of Truth for Noise Generation.
    Handles both PyTorch Tensors (Trainer) and NumPy Arrays (Validation/Sim).
    """
    def __init__(self, config: Dict):
        self.config = config.get('noise_config', {})
        self.components = config.get('data_noise_components', ['thermal'])
        
        # Ranges
        self.snr_range = self.config.get('snr_range', [1.0, 15.0])
        self.physio_amp_range = self.config.get('physio_amp_range', [0.05, 0.15])
        self.drift_range = self.config.get('drift_range', [-0.02, 0.02])
        self.spike_prob = self.config.get('spike_probability', 0.05)
        self.spike_mag = self.config.get('spike_magnitude_range', [2.0, 5.0])
        
        # Spatial noise parameters
        self.spatial_noise_sigma = self.config.get('spatial_noise_sigma', 0.8)
        self.motion_probability = self.config.get('motion_probability', 0.2)
        self.motion_shift_range = self.config.get('motion_shift_range', [1, 4])  # pixels
        self.motion_rotate_range = self.config.get('motion_rotate_range', [-3, 3])  # degrees

    def apply_noise(self, signals: Union[torch.Tensor, np.ndarray], ref_signal: float, pld_scaling: dict) -> Union[torch.Tensor, np.ndarray]:
        """Apply noise to 1D voxel signals (Batch, Features)."""
        is_torch = isinstance(signals, torch.Tensor)
        xp = torch if is_torch else np
        
        # Ensure 2D (Batch, Features)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1) if is_torch else signals[np.newaxis, :]
            
        batch_size, seq_len = signals.shape
        n_plds = seq_len // 2
        device = signals.device if is_torch else None
        
        # 1. Base Thermal Noise (SNR scaling)
        if is_torch:
            snr = torch.empty(batch_size, 1, device=device).uniform_(*self.snr_range)
        else:
            snr = np.random.uniform(*self.snr_range, size=(batch_size, 1))
            
        noise_sigma = ref_signal / snr
        
        # Prepare scaling vector
        scale_p = pld_scaling['PCASL']
        scale_v = pld_scaling['VSASL']
        
        if is_torch:
            s_vec = torch.cat([
                torch.full((n_plds,), scale_p, device=device), 
                torch.full((n_plds,), scale_v, device=device)
            ])
            s_vec = s_vec.unsqueeze(0)
        else:
            s_vec = np.concatenate([
                np.full(n_plds, scale_p),
                np.full(n_plds, scale_v)
            ])[np.newaxis, :]

        # Generate Noise
        total_noise = xp.zeros_like(signals)

        if 'thermal' in self.components:
            if is_torch:
                total_noise += torch.randn_like(signals) * noise_sigma * s_vec
            else:
                total_noise += np.random.randn(*signals.shape) * noise_sigma * s_vec

        if 'physio' in self.components:
            t = xp.arange(seq_len, device=device) if is_torch else np.arange(seq_len)
            if is_torch:
                amp = torch.empty(batch_size, 1, device=device).uniform_(*self.physio_amp_range)
                freq = torch.empty(batch_size, 1, device=device).uniform_(0.5, 2.0)
                phase = torch.rand(batch_size, 1, device=device) * 6.28
                phys = amp * noise_sigma * torch.sin(2 * 3.14159 * freq * t.unsqueeze(0) / seq_len + phase)
            else:
                amp = np.random.uniform(*self.physio_amp_range, size=(batch_size, 1))
                freq = np.random.uniform(0.5, 2.0, size=(batch_size, 1))
                phase = np.random.rand(batch_size, 1) * 6.28
                phys = amp * noise_sigma * np.sin(2 * np.pi * freq * t[np.newaxis, :] / seq_len + phase)
            total_noise += phys

        if 'drift' in self.components:
            t = xp.linspace(-1, 1, seq_len, device=device) if is_torch else np.linspace(-1, 1, seq_len)
            if is_torch:
                slope = torch.empty(batch_size, 1, device=device).uniform_(*self.drift_range)
                drift = slope * noise_sigma * t.unsqueeze(0)
            else:
                slope = np.random.uniform(*self.drift_range, size=(batch_size, 1))
                drift = slope * noise_sigma * t[np.newaxis, :]
            total_noise += drift

        if 'spikes' in self.components:
            if is_torch:
                spike_mask = (torch.rand(batch_size, seq_len, device=device) < self.spike_prob).float()
                spike_magnitude = torch.empty(batch_size, 1, device=device).uniform_(*self.spike_mag)
                spike_noise = spike_mask * spike_magnitude * noise_sigma
            else:
                spike_mask = (np.random.rand(batch_size, seq_len) < self.spike_prob).astype(np.float32)
                spike_magnitude = np.random.uniform(*self.spike_mag, size=(batch_size, 1))
                spike_noise = spike_mask * spike_magnitude * noise_sigma
            total_noise += spike_noise

        return signals + total_noise


class SpatialNoiseEngine:
    """
    Spatial noise engine for 2D/3D ASL image processing.
    Implements physics-correct Rician noise, spatially correlated noise, and motion artifacts.
    """
    def __init__(self, config: Dict = None):
        config = config or {}
        self.snr_range = config.get('snr_range', [3.0, 15.0])
        self.spatial_noise_sigma = config.get('spatial_noise_sigma', 0.8)
        self.motion_probability = config.get('motion_probability', 0.2)
        self.motion_shift_range = config.get('motion_shift_range', [1, 4])
        self.motion_rotate_range = config.get('motion_rotate_range', [-3, 3])
        self.static_tissue_fraction = config.get('static_tissue_fraction', 0.05)  # 5% of M0
        
    def add_rician_noise(self, signal: np.ndarray, snr: float = None) -> np.ndarray:
        """
        Apply Rician noise (correct MRI physics for magnitude images).
        
        Rician noise: S_noisy = sqrt((S + N_real)^2 + N_imag^2)
        
        At low SNR (ASL), this creates upward bias that U-Net must learn to correct.
        
        Args:
            signal: Clean signal (any shape)
            snr: Signal-to-noise ratio. If None, samples from snr_range.
            
        Returns:
            Noisy signal with same shape
        """
        if snr is None:
            snr = np.random.uniform(*self.snr_range)
        
        mean_signal = np.mean(np.abs(signal)) + 1e-10
        sigma = mean_signal / snr
        
        # Rician: magnitude of complex signal with Gaussian real/imag noise
        noise_real = np.random.normal(0, sigma, signal.shape)
        noise_imag = np.random.normal(0, sigma, signal.shape)
        
        noisy_signal = np.sqrt((signal + noise_real)**2 + noise_imag**2)
        
        return noisy_signal.astype(np.float32)
    
    def add_spatially_correlated_noise(self, signal: np.ndarray, snr: float = None) -> np.ndarray:
        """
        Apply spatially correlated (colored) Rician noise.
        
        This breaks the voxel-independence assumption that LS relies on.
        The U-Net can exploit spatial correlations in noise patterns.
        
        Args:
            signal: Clean signal (Time, H, W) or (H, W)
            snr: Signal-to-noise ratio
            
        Returns:
            Noisy signal with spatial noise blobs
        """
        if snr is None:
            snr = np.random.uniform(*self.snr_range)
            
        mean_signal = np.mean(np.abs(signal)) + 1e-10
        sigma = mean_signal / snr
        
        # Generate spatially correlated Gaussian noise
        noise_real = np.random.normal(0, sigma, signal.shape)
        noise_imag = np.random.normal(0, sigma, signal.shape)
        
        # Apply Gaussian blur to create spatial correlations ("noise blobs")
        # This creates correlations that LS cannot handle but U-Net can learn
        if signal.ndim == 3:  # (Time, H, W)
            for t in range(signal.shape[0]):
                noise_real[t] = gaussian_filter(noise_real[t], sigma=self.spatial_noise_sigma)
                noise_imag[t] = gaussian_filter(noise_imag[t], sigma=self.spatial_noise_sigma)
        else:  # (H, W)
            noise_real = gaussian_filter(noise_real, sigma=self.spatial_noise_sigma)
            noise_imag = gaussian_filter(noise_imag, sigma=self.spatial_noise_sigma)
        
        # Re-scale noise to maintain target SNR after blur
        noise_real = noise_real * (sigma / (np.std(noise_real) + 1e-10))
        noise_imag = noise_imag * (sigma / (np.std(noise_imag) + 1e-10))
        
        # Rician combination
        noisy_signal = np.sqrt((signal + noise_real)**2 + noise_imag**2)
        
        return noisy_signal.astype(np.float32)
    
    def add_motion_artifacts(self, signal_stack: np.ndarray) -> np.ndarray:
        """
        Add motion artifacts by randomly shifting/rotating specific time frames.
        
        This simulates patient motion - the #1 killer of clinical ASL.
        LS is very sensitive to outliers (squared error), while U-Net can
        learn to perform outlier rejection by looking at spatial neighbors.
        
        Args:
            signal_stack: (Batch, Time, H, W) or (Time, H, W)
            
        Returns:
            Corrupted signal stack (same shape)
        """
        has_batch = signal_stack.ndim == 4
        if not has_batch:
            signal_stack = signal_stack[np.newaxis, ...]
        
        batch_size, n_time, h, w = signal_stack.shape
        output = signal_stack.copy()
        
        for b in range(batch_size):
            if np.random.rand() < self.motion_probability:
                # Pick random time frame to corrupt
                corrupted_frame = np.random.randint(0, n_time)
                
                # Random shift
                shift_x = np.random.randint(*self.motion_shift_range) * np.random.choice([-1, 1])
                shift_y = np.random.randint(*self.motion_shift_range) * np.random.choice([-1, 1])
                
                # Random rotation (degrees)
                angle = np.random.uniform(*self.motion_rotate_range)
                theta = np.radians(angle)
                
                # Build affine transformation matrix
                # Center the rotation
                c_y, c_x = h / 2, w / 2
                
                # Rotation matrix
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                
                # Affine: first translate to center, rotate, translate back, then shift
                # For scipy's affine_transform, the matrix maps output to input
                rotation_matrix = np.array([
                    [cos_t, -sin_t],
                    [sin_t, cos_t]
                ])
                
                # Apply transformation
                frame = output[b, corrupted_frame]
                
                # Simple shift using numpy roll as approximation (faster)
                shifted_frame = np.roll(np.roll(frame, shift_x, axis=1), shift_y, axis=0)
                
                # For rotation: use scipy affine_transform for proper interpolation
                if abs(angle) > 0.5:  # Only rotate if significant
                    offset = np.array([c_y, c_x]) - rotation_matrix @ np.array([c_y, c_x])
                    shifted_frame = affine_transform(
                        shifted_frame, 
                        rotation_matrix, 
                        offset=offset + np.array([shift_y, shift_x]),
                        order=1,  # Bilinear interpolation
                        mode='constant',
                        cval=0.0
                    )
                
                output[b, corrupted_frame] = shifted_frame
        
        if not has_batch:
            output = output[0]
            
        return output.astype(np.float32)
    
    def simulate_realistic_acquisition(self, clean_difference: np.ndarray, 
                                       m0_map: np.ndarray = None,
                                       snr: float = None) -> np.ndarray:
        """
        Simulate realistic ASL acquisition with Control/Label pair noise.
        
        This properly models the ASL acquisition process:
        1. Control image ≈ Static tissue + Difference signal
        2. Label image ≈ Static tissue
        3. Both have Rician noise applied separately
        4. Final difference = Control - Label
        
        This creates the subtle biases at low SNR that break simple LS.
        
        Args:
            clean_difference: Clean perfusion difference signal (Time, H, W)
            m0_map: M0 calibration map (H, W). If None, uses uniform M0=1
            snr: SNR for noise level
            
        Returns:
            Realistic noisy difference signal
        """
        if snr is None:
            snr = np.random.uniform(*self.snr_range)
            
        if m0_map is None:
            m0_map = np.ones(clean_difference.shape[1:], dtype=np.float32)
        
        # Static tissue signal (imperfect background suppression)
        static_tissue = self.static_tissue_fraction * m0_map
        
        # Ensure proper broadcasting
        if clean_difference.ndim == 3 and static_tissue.ndim == 2:
            static_tissue = static_tissue[np.newaxis, :, :]
        
        # Control image: Static + Difference
        control = static_tissue + clean_difference
        
        # Label image: Static only  
        label = np.broadcast_to(static_tissue, control.shape).copy()
        
        # Add Rician noise to each separately
        control_noisy = self.add_spatially_correlated_noise(control, snr)
        label_noisy = self.add_spatially_correlated_noise(label, snr)
        
        # Difference of noisy images
        noisy_difference = control_noisy - label_noisy
        
        return noisy_difference.astype(np.float32)
    
    def apply_full_noise_model(self, clean_signal: np.ndarray, 
                                m0_map: np.ndarray = None,
                                snr: float = None,
                                use_motion: bool = True,
                                use_realistic_acquisition: bool = True) -> np.ndarray:
        """
        Apply the complete hostile noise model.
        
        Combines:
        1. Realistic acquisition simulation (Control/Label pairs)
        2. Spatially correlated Rician noise
        3. Motion artifacts
        
        This creates data that breaks LS assumptions and justifies spatial U-Net.
        
        Args:
            clean_signal: Clean signal (Time, H, W) or (Batch, Time, H, W)
            m0_map: M0 calibration map
            snr: Target SNR
            use_motion: Whether to add motion artifacts
            use_realistic_acquisition: Whether to use Control/Label simulation
            
        Returns:
            Noisy signal with all artifacts
        """
        has_batch = clean_signal.ndim == 4
        
        if not has_batch:
            clean_signal = clean_signal[np.newaxis, ...]
        
        batch_size = clean_signal.shape[0]
        output = np.zeros_like(clean_signal)
        
        for b in range(batch_size):
            sample = clean_signal[b]  # (Time, H, W)
            
            # Sample SNR for this sample
            sample_snr = snr if snr is not None else np.random.uniform(*self.snr_range)
            
            if use_realistic_acquisition:
                noisy = self.simulate_realistic_acquisition(sample, m0_map, sample_snr)
            else:
                noisy = self.add_spatially_correlated_noise(sample, sample_snr)
            
            output[b] = noisy
        
        # Motion artifacts apply across batch
        if use_motion:
            output = self.add_motion_artifacts(output)
        
        if not has_batch:
            output = output[0]
            
        return output.astype(np.float32)

# FILE: noise_engine.py
"""
Single Source of Truth for Noise Generation.
Handles both PyTorch Tensors (Trainer) and NumPy Arrays (Validation/Sim).
"""
import torch
import numpy as np
from typing import Dict, Union, List

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

    def apply_noise(self, signals: Union[torch.Tensor, np.ndarray], ref_signal: float, pld_scaling: dict) -> Union[torch.Tensor, np.ndarray]:
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
            # Check if scale vector is already tensor or needs creation
            # Assuming pld_scaling values are floats, we construct the vector
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
            # Simple modulation
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

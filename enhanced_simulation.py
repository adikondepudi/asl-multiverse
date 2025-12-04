# enhanced_simulation.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from asl_simulation import ASLSimulator, ASLParameters
import multiprocessing as mp
from tqdm import tqdm
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class PhysiologicalVariation:
    cbf_range: Tuple[float, float] = (20.0, 100.0)
    att_range: Tuple[float, float] = (500.0, 4000.0)
    t1_artery_range: Tuple[float, float] = (1650.0, 2050.0)
    stroke_cbf_range: Tuple[float, float] = (5.0, 30.0)
    stroke_att_range: Tuple[float, float] = (1500.0, 4500.0)
    tumor_cbf_range: Tuple[float, float] = (10.0, 150.0)
    tumor_att_range: Tuple[float, float] = (700.0, 3500.0)
    young_cbf_range: Tuple[float, float] = (60.0, 120.0)
    young_att_range: Tuple[float, float] = (500.0, 1500.0)
    elderly_cbf_range: Tuple[float, float] = (30.0, 70.0)
    elderly_att_range: Tuple[float, float] = (1500.0, 3500.0)
    t_tau_perturb_range: Tuple[float, float] = (-0.05, 0.05)
    alpha_perturb_range: Tuple[float, float] = (-0.10, 0.10)
    arterial_blood_volume_range: Tuple[float, float] = (0.00, 0.015) # 0 to 1.5%

class RealisticASLSimulator(ASLSimulator):
    def __init__(self, params: ASLParameters = ASLParameters()):
        super().__init__(params)
        self.physio_var = PhysiologicalVariation()

    def add_modular_noise(self, signal, snr, noise_components=['thermal']):
        """
        Applies noise components based on a list string.
        options: 'thermal', 'physio', 'drift', 'spikes'
        """
        noisy_signal = signal.copy()
        sig_len = signal.shape[-1]
        t = np.arange(sig_len)
        
        # 1. Base Thermal Level
        mean_sig = np.mean(np.abs(signal))
        noise_sd = (mean_sig / snr) if snr > 0 else 0
        
        # 2. Additive Layers
        if 'physio' in noise_components:
            # Cardiac (Fast) + Respiratory (Slow)
            cardiac = (noise_sd * 0.5) * np.sin(2 * np.pi * 1.0 * t / sig_len * 5 + np.random.rand())
            resp = (noise_sd * 0.3) * np.sin(2 * np.pi * 0.3 * t / sig_len * 5 + np.random.rand())
            noisy_signal += (cardiac + resp)

        if 'drift' in noise_components:
            # Low freq baseline shift
            drift = (noise_sd * 0.4) * np.linspace(-1, 1, sig_len)
            noisy_signal += drift

        if 'spikes' in noise_components:
            # Random outliers
            if np.random.rand() < 0.2: # 20% of samples get a spike
                idx = np.random.randint(0, sig_len)
                noisy_signal[idx] += 5 * noise_sd * np.random.choice([-1, 1])

        # 3. Final Rician/Thermal Noise (Correct MRI Physics)
        if 'thermal' in noise_components:
            n_real = np.random.normal(0, noise_sd, signal.shape)
            n_imag = np.random.normal(0, noise_sd, signal.shape)
            noisy_signal = np.sqrt((noisy_signal + n_real)**2 + n_imag**2)
            
        return noisy_signal

    def add_realistic_noise(self, signal: np.ndarray, snr: float = 5.0,
                            temporal_correlation: float = 0.3, include_spike_artifacts: bool = True,
                            spike_probability: float = 0.01, spike_magnitude_factor: float = 5.0,
                            include_baseline_drift: bool = True, drift_magnitude_factor: float = 0.1,
                            include_physiological: bool = True) -> np.ndarray:
        """
        Applies a comprehensive, physically layered noise model to a clean ASL signal.
        This models the sequence of real-world signal corruption:
        1. Physiological fluctuations and drift are added to the clean signal.
        2. Rician noise, the correct model for MR magnitude data, is applied.
        3. Sporadic spike artifacts corrupt the final noisy signal.
        """
        signal_with_phys = signal.copy()
        
        mean_abs_signal = np.mean(np.abs(signal))
        base_noise_level = mean_abs_signal / snr if snr > 0 and mean_abs_signal > 0 else 1e-5

        if include_physiological and signal.ndim > 0 and signal.shape[-1] > 1:
            t = np.arange(signal.shape[-1])
            cardiac_freq = np.random.uniform(0.8, 1.2)
            cardiac = (base_noise_level * 0.5) * np.sin(2 * np.pi * cardiac_freq * t / signal.shape[-1] * 5 + np.random.rand() * np.pi)
            respiratory_freq = np.random.uniform(0.2, 0.4)
            respiratory = (base_noise_level * 0.3) * np.sin(2 * np.pi * respiratory_freq * t / signal.shape[-1] * 5 + np.random.rand() * np.pi)
            signal_with_phys += cardiac + respiratory

        if include_baseline_drift and signal.ndim > 0 and signal.shape[-1] > 1:
            drift_freq = np.random.uniform(0.05, 0.2)
            drift_amp = drift_magnitude_factor * (mean_abs_signal if mean_abs_signal > 0 else base_noise_level)
            drift = drift_amp * np.sin(2 * np.pi * drift_freq * t / signal.shape[-1] + np.random.rand() * np.pi)
            signal_with_phys += drift

        sigma_rician = base_noise_level / np.sqrt(2)
        noise_real = np.random.normal(0, sigma_rician, signal.shape)
        noise_imag = np.random.normal(0, sigma_rician, signal.shape)
        noisy_signal = np.sqrt((signal_with_phys + noise_real)**2 + noise_imag**2)

        if include_spike_artifacts and signal.ndim > 0:
            num_spikes = np.random.poisson(signal.shape[-1] * spike_probability)
            spike_indices = np.random.choice(signal.shape[-1], num_spikes, replace=False)
            for i in spike_indices:
                spike = (np.random.choice([-1, 1])) * spike_magnitude_factor * base_noise_level
                noisy_signal[i] += spike
        
        return noisy_signal

    def generate_diverse_dataset(self, plds: np.ndarray, n_subjects: int = 100,
                               conditions: List[str] = ['healthy', 'stroke', 'tumor', 'elderly'],
                               noise_levels: List[float] = [3.0, 5.0, 10.0]) -> Dict:
        """
        Generates a fixed-size dataset with maximal realism for validation or testing.
        This function now correctly uses the unified, layered noise model for every generated sample,
        ensuring the validation data is as realistic as the training data.
        """
        dataset = {'signals': [], 'parameters': [], 'conditions': [], 'noise_levels': [], 'perturbed_params': []}
        base_params = self.params

        condition_map = {
            'healthy': (self.physio_var.cbf_range, self.physio_var.att_range, self.physio_var.t1_artery_range),
            'stroke': (self.physio_var.stroke_cbf_range, self.physio_var.stroke_att_range, (self.physio_var.t1_artery_range[0]-100, self.physio_var.t1_artery_range[1]+100)),
            'tumor': (self.physio_var.tumor_cbf_range, self.physio_var.tumor_att_range, (self.physio_var.t1_artery_range[0]-150, self.physio_var.t1_artery_range[1]+150)),
            'elderly': (self.physio_var.elderly_cbf_range, self.physio_var.elderly_att_range, (self.physio_var.t1_artery_range[0]+50, self.physio_var.t1_artery_range[1]+150))
        }

        for _ in tqdm(range(n_subjects), desc="Generating Fixed Diverse Dataset"):
            condition = np.random.choice(conditions)
            cbf_range, att_range, t1_range = condition_map.get(condition, (self.physio_var.cbf_range, self.physio_var.att_range, self.physio_var.t1_artery_range))

            cbf = np.random.uniform(*cbf_range)
            att = np.random.uniform(*att_range)
            t1_a = np.random.uniform(*t1_range)
            abv = np.random.uniform(*self.physio_var.arterial_blood_volume_range) if np.random.rand() > 0.5 else 0.0
            slice_idx = np.random.randint(0, 20) # Simulating 20 slices
            slice_delay_factor = np.exp(-(slice_idx * 45.0)/1000.0)
            
            perturbed_t_tau = base_params.T_tau * (1 + np.random.uniform(*self.physio_var.t_tau_perturb_range))
            perturbed_alpha_pcasl = np.clip(base_params.alpha_PCASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range)), 0.1, 1.1)
            perturbed_alpha_vsasl = np.clip(base_params.alpha_VSASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range)), 0.1, 1.0)
            
            # Apply slice timing to alphas
            eff_alpha_pcasl = perturbed_alpha_pcasl * slice_delay_factor
            eff_alpha_vsasl = perturbed_alpha_vsasl * slice_delay_factor

            vsasl_clean = self._generate_vsasl_signal(plds, att, cbf, t1_a, eff_alpha_vsasl)
            pcasl_clean = self._generate_pcasl_signal(plds, att, cbf, t1_a, perturbed_t_tau, eff_alpha_pcasl)
            art_sig = self._generate_arterial_signal(plds, att, abv, t1_a, eff_alpha_pcasl)
            
            pcasl_clean += art_sig # Add macrovascular component
            
            for snr in noise_levels:
                vsasl_noisy = self.add_realistic_noise(vsasl_clean, snr=snr)
                pcasl_noisy = self.add_realistic_noise(pcasl_clean, snr=snr)
                
                multiverse_signal_flat = np.concatenate([pcasl_noisy, vsasl_noisy])
                
                dataset['signals'].append(multiverse_signal_flat)
                dataset['parameters'].append([cbf, att, t1_a, float(slice_idx)]) # Store slice index
                dataset['conditions'].append(condition)
                dataset['noise_levels'].append(snr)
                dataset['perturbed_params'].append({
                    't1_artery': t1_a, 't_tau': perturbed_t_tau, 
                    'alpha_pcasl': perturbed_alpha_pcasl, 'alpha_vsasl': perturbed_alpha_vsasl
                })

        dataset['signals'] = np.array(dataset['signals'])
        dataset['parameters'] = np.array(dataset['parameters'])
        return dataset

if __name__ == "__main__":
    simulator = RealisticASLSimulator()
    logger.info("Enhanced ASL Simulator initialized. `generate_balanced_dataset` has been removed.")
    logger.info("For training, please use the `generate_offline_dataset.py` script.")
    logger.info("\nTesting `generate_diverse_dataset` for fixed validation/test sets...")
    test_data = simulator.generate_diverse_dataset(plds=np.arange(500, 3001, 500), n_subjects=10)
    if test_data['signals'].size > 0:
        logger.info(f"\nGenerated test dataset shape: {test_data['signals'].shape}")
    else: logger.info("No test data generated.")
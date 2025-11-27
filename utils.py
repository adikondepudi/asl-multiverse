# FILE: utils.py
import numpy as np
from typing import Optional, Dict
import multiprocessing as mp
import time
import torch
from asl_simulation import _generate_pcasl_signal_jit, _generate_vsasl_signal_jit

def engineer_signal_features_torch(raw_signal: torch.Tensor, num_plds: int) -> torch.Tensor:
    """
    GPU-accelerated feature engineering.
    Returns: [pcasl_ttp, vsasl_ttp, pcasl_com, vsasl_com, pcasl_peak, vsasl_peak]
    """
    pcasl_curves = raw_signal[:, :num_plds]
    vsasl_curves = raw_signal[:, num_plds:]
    
    device = raw_signal.device
    plds_indices = torch.arange(num_plds, device=device, dtype=raw_signal.dtype)

    # Feature 1: Time to peak
    pcasl_ttp = torch.argmax(pcasl_curves, dim=1).float()
    vsasl_ttp = torch.argmax(vsasl_curves, dim=1).float()

    # Feature 2: Center of mass (Safe division)
    pcasl_sum = torch.sum(pcasl_curves, dim=1)
    vsasl_sum = torch.sum(vsasl_curves, dim=1)
    
    pcasl_sum_safe = torch.where(torch.abs(pcasl_sum) < 1e-4, torch.ones_like(pcasl_sum), pcasl_sum)
    vsasl_sum_safe = torch.where(torch.abs(vsasl_sum) < 1e-4, torch.ones_like(vsasl_sum), vsasl_sum)
    
    pcasl_com = torch.sum(pcasl_curves * plds_indices, dim=1) / pcasl_sum_safe
    vsasl_com = torch.sum(vsasl_curves * plds_indices, dim=1) / vsasl_sum_safe
    
    # Feature 3: Peak Height (NEW)
    pcasl_peak = torch.max(pcasl_curves, dim=1).values
    vsasl_peak = torch.max(vsasl_curves, dim=1).values

    # Stack all 6 features
    return torch.stack([pcasl_ttp, vsasl_ttp, pcasl_com, vsasl_com, pcasl_peak, vsasl_peak], dim=1)

# ... (Keep the rest of the file: engineer_signal_features (numpy), _worker_generate_sample, ParallelStreamingStatsCalculator as they were) ...
def engineer_signal_features(raw_signal: np.ndarray, num_plds: int) -> np.ndarray:
    """Legacy Numpy version updated with Peak Height."""
    is_1d = raw_signal.ndim == 1
    if is_1d:
        raw_signal = raw_signal.reshape(1, -1)

    pcasl_curves = raw_signal[:, :num_plds]
    vsasl_curves = raw_signal[:, num_plds:]
    plds_indices = np.arange(num_plds)

    pcasl_ttp = np.argmax(pcasl_curves, axis=1)
    vsasl_ttp = np.argmax(vsasl_curves, axis=1)

    pcasl_sum = np.sum(pcasl_curves, axis=1) + 1e-6
    vsasl_sum = np.sum(vsasl_curves, axis=1) + 1e-6
    pcasl_com = np.sum(pcasl_curves * plds_indices, axis=1) / pcasl_sum
    vsasl_com = np.sum(vsasl_curves * plds_indices, axis=1) / vsasl_sum
    
    # NEW: Peak Height
    pcasl_peak = np.max(pcasl_curves, axis=1)
    vsasl_peak = np.max(vsasl_curves, axis=1)
    
    engineered_features = np.stack([pcasl_ttp, vsasl_ttp, pcasl_com, vsasl_com, pcasl_peak, vsasl_peak], axis=1)

    if is_1d:
        return engineered_features.flatten().astype(np.float32)
    else:
        return engineered_features.astype(np.float32)

def _worker_generate_sample(args_tuple):
    simulator, plds, seed = args_tuple
    np.random.seed(seed)

    physio_var = simulator.physio_var
    true_att = np.random.uniform(*physio_var.att_range)
    true_cbf = np.random.uniform(*physio_var.cbf_range)
    true_t1_artery = np.random.uniform(*physio_var.t1_artery_range)
    
    # Generate clean signals with specific T1
    vsasl_clean = simulator._generate_vsasl_signal(plds, true_att, true_cbf, true_t1_artery, simulator.params.alpha_VSASL)
    pcasl_clean = simulator._generate_pcasl_signal(plds, true_att, true_cbf, true_t1_artery, simulator.params.T_tau, simulator.params.alpha_PCASL)
    
    pcasl_mu, pcasl_sigma = np.mean(pcasl_clean), np.std(pcasl_clean)
    vsasl_mu, vsasl_sigma = np.mean(vsasl_clean), np.std(vsasl_clean)
    pcasl_shape = (pcasl_clean - pcasl_mu) / (pcasl_sigma + 1e-6)
    vsasl_shape = (vsasl_clean - vsasl_mu) / (vsasl_sigma + 1e-6)
    shape_vector = np.concatenate([pcasl_shape, vsasl_shape])

    raw_signal_vector = np.concatenate([pcasl_clean, vsasl_clean])
    
    # Calculate features including new Peaks
    eng_features = engineer_signal_features(raw_signal_vector, len(plds))
    
    # scalars: [mu_p, sig_p, mu_v, sig_v, ttp_p, ttp_v, com_p, com_v, peak_p, peak_v]
    scalar_features = np.array([pcasl_mu, pcasl_sigma, vsasl_mu, vsasl_sigma, *eng_features])
    
    # Return t1_artery as well so we can calculate stats for it
    return shape_vector, true_cbf, true_att, true_t1_artery, scalar_features

class ParallelStreamingStatsCalculator:
    def __init__(self, simulator, plds, num_samples, num_workers):
        self.simulator = simulator
        self.plds = plds
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.num_plds = len(plds)
        self.num_plds = len(plds)
        self.num_scalar_features = None # Will be determined dynamically

        self.count = 0
        self.shape_vector_mean = np.zeros(self.num_plds * 2)
        self.shape_vector_m2 = np.zeros(self.num_plds * 2)
        self.cbf_mean, self.cbf_m2 = 0.0, 0.0
        self.att_mean, self.att_m2 = 0.0, 0.0
        self.scalar_mean = None
        self.scalar_m2 = None

    def _update_stats(self, existing_agg, new_value):
        mean, m2 = existing_agg
        delta = new_value - mean
        mean += delta / self.count
        delta2 = new_value - mean
        m2 += delta * delta2
        return mean, m2

    def calculate(self):
        print(f"Calculating normalization stats over {self.num_samples} samples using {self.num_workers} workers...")
        base_seed = int(time.time())
        worker_args = [(self.simulator, self.plds, base_seed + i) for i in range(self.num_samples)]
        
        base_seed = int(time.time())
        worker_args = [(self.simulator, self.plds, base_seed + i) for i in range(self.num_samples)]
        
        self.t1_mean, self.t1_m2 = 0.0, 0.0 # New T1 trackers

        with mp.Pool(processes=self.num_workers) as pool:
            results_iterator = pool.imap_unordered(_worker_generate_sample, worker_args)
            # Unpack new return tuple including t1
            for shape_vec, cbf, att, t1, scalars in results_iterator:
                if self.num_scalar_features is None:
                    self.num_scalar_features = len(scalars)
                    self.scalar_mean = np.zeros(self.num_scalar_features)
                    self.scalar_m2 = np.zeros(self.num_scalar_features)
                
                self.count += 1
                self.shape_vector_mean, self.shape_vector_m2 = self._update_stats((self.shape_vector_mean, self.shape_vector_m2), shape_vec)
                self.cbf_mean, self.cbf_m2 = self._update_stats((self.cbf_mean, self.cbf_m2), cbf)
                self.att_mean, self.att_m2 = self._update_stats((self.att_mean, self.att_m2), att)
                self.t1_mean, self.t1_m2 = self._update_stats((self.t1_mean, self.t1_m2), t1) # Update T1 stats
                self.scalar_mean, self.scalar_m2 = self._update_stats((self.scalar_mean, self.scalar_m2), scalars)

        if self.count < 2: return {}

        shape_vector_std = np.sqrt(self.shape_vector_m2 / self.count)
        cbf_std = np.sqrt(self.cbf_m2 / self.count)
        att_std = np.sqrt(self.att_m2 / self.count)
        t1_std = np.sqrt(self.t1_m2 / self.count)
        scalar_std = np.sqrt(self.scalar_m2 / self.count)

        return {
            'shape_vector_mean': self.shape_vector_mean.tolist(),
            'shape_vector_std': np.clip(shape_vector_std, 1e-6, None).tolist(),
            'y_mean_cbf': self.cbf_mean,
            'y_std_cbf': max(float(cbf_std), 1e-6),
            'y_mean_att': self.att_mean,
            'y_std_att': max(float(att_std), 1e-6),
            'y_mean_t1': self.t1_mean,
            'y_std_t1': max(float(t1_std), 1e-6),
            'scalar_features_mean': self.scalar_mean.tolist(),
            'scalar_features_std': np.clip(scalar_std, 1e-6, None).tolist()
        }

def process_signals_cpu(signals_unnorm: np.ndarray, norm_stats: dict, num_plds: int, t1_values: Optional[np.ndarray] = None) -> np.ndarray:
    """CPU version of preprocessing for validation data."""
    raw_curves = signals_unnorm[:, :num_plds * 2]
    eng_ttp_com = signals_unnorm[:, num_plds * 2:]

    pcasl_raw = raw_curves[:, :num_plds]
    vsasl_raw = raw_curves[:, num_plds:]

    pcasl_mu = np.mean(pcasl_raw, axis=1, keepdims=True)
    pcasl_sigma = np.std(pcasl_raw, axis=1, keepdims=True)
    pcasl_shape = (pcasl_raw - pcasl_mu) / (pcasl_sigma + 1e-6)

    vsasl_mu = np.mean(vsasl_raw, axis=1, keepdims=True)
    vsasl_sigma = np.std(vsasl_raw, axis=1, keepdims=True)
    vsasl_shape = (vsasl_raw - vsasl_mu) / (vsasl_sigma + 1e-6)

    shape_vector = np.concatenate([pcasl_shape, vsasl_shape], axis=1)
    scalar_features_unnorm = np.concatenate([pcasl_mu, pcasl_sigma, vsasl_mu, vsasl_sigma, eng_ttp_com], axis=1)
    
    s_mean = np.array(norm_stats['scalar_features_mean'])
    s_std = np.array(norm_stats['scalar_features_std']) + 1e-6
    scalar_features_norm = (scalar_features_unnorm - s_mean) / s_std

    if t1_values is not None:
        t1_mean = norm_stats.get('y_mean_t1', 1850.0)
        t1_std = norm_stats.get('y_std_t1', 200.0)
        t1_norm = (t1_values - t1_mean) / (t1_std + 1e-6)
        scalar_features_norm = np.concatenate([scalar_features_norm, t1_norm], axis=1)

    return np.concatenate([shape_vector, scalar_features_norm], axis=1)
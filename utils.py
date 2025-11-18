# utils.py
import numpy as np
import multiprocessing as mp
import time
import torch
from asl_simulation import _generate_pcasl_signal_jit, _generate_vsasl_signal_jit

def engineer_signal_features_torch(raw_signal: torch.Tensor, num_plds: int) -> torch.Tensor:
    """
    GPU-accelerated version of feature engineering using PyTorch.
    Args:
        raw_signal: Tensor of shape (Batch, num_plds * 2)
        num_plds: Number of PLDs per modality
    Returns:
        Tensor of shape (Batch, 4) with [pcasl_ttp, vsasl_ttp, pcasl_com, vsasl_com]
    """
    pcasl_curves = raw_signal[:, :num_plds]
    vsasl_curves = raw_signal[:, num_plds:]
    
    # Create PLD indices tensor on the same device
    device = raw_signal.device
    plds_indices = torch.arange(num_plds, device=device, dtype=raw_signal.dtype)

    # Feature 1: Time to peak (argmax is not differentiable, but fine for input features)
    # We cast to float to match network input types
    pcasl_ttp = torch.argmax(pcasl_curves, dim=1).float()
    vsasl_ttp = torch.argmax(vsasl_curves, dim=1).float()

    # Feature 2: Center of mass
    # Add epsilon to avoid division by zero
    pcasl_sum = torch.sum(pcasl_curves, dim=1) + 1e-6
    vsasl_sum = torch.sum(vsasl_curves, dim=1) + 1e-6
    
    pcasl_com = torch.sum(pcasl_curves * plds_indices, dim=1) / pcasl_sum
    vsasl_com = torch.sum(vsasl_curves * plds_indices, dim=1) / vsasl_sum
    
    return torch.stack([pcasl_ttp, vsasl_ttp, pcasl_com, vsasl_com], dim=1)

def engineer_signal_features(raw_signal: np.ndarray, num_plds: int) -> np.ndarray:
    """
    Legacy Numpy version for offline generation/cpu validation.
    """
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
    
    engineered_features = np.stack([pcasl_ttp, vsasl_ttp, pcasl_com, vsasl_com], axis=1)

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
    
    vsasl_clean = simulator._generate_vsasl_signal(plds, true_att, true_cbf, true_t1_artery, simulator.params.alpha_VSASL)
    pcasl_clean = simulator._generate_pcasl_signal(plds, true_att, true_cbf, true_t1_artery, simulator.params.T_tau, simulator.params.alpha_PCASL)
    
    pcasl_mu, pcasl_sigma = np.mean(pcasl_clean), np.std(pcasl_clean)
    vsasl_mu, vsasl_sigma = np.mean(vsasl_clean), np.std(vsasl_clean)
    pcasl_shape = (pcasl_clean - pcasl_mu) / (pcasl_sigma + 1e-6)
    vsasl_shape = (vsasl_clean - vsasl_mu) / (vsasl_sigma + 1e-6)
    shape_vector = np.concatenate([pcasl_shape, vsasl_shape])

    raw_signal_vector = np.concatenate([pcasl_clean, vsasl_clean])
    eng_features = engineer_signal_features(raw_signal_vector, len(plds))
    
    scalar_features = np.array([pcasl_mu, pcasl_sigma, vsasl_mu, vsasl_sigma, *eng_features])
    
    return shape_vector, true_cbf, true_att, scalar_features

class ParallelStreamingStatsCalculator:
    def __init__(self, simulator, plds, num_samples, num_workers):
        self.simulator = simulator
        self.plds = plds
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.num_plds = len(plds)
        self.num_scalar_features = 8

        self.count = 0
        self.shape_vector_mean = np.zeros(self.num_plds * 2)
        self.shape_vector_m2 = np.zeros(self.num_plds * 2)
        self.cbf_mean, self.cbf_m2 = 0.0, 0.0
        self.att_mean, self.att_m2 = 0.0, 0.0
        self.scalar_mean = np.zeros(self.num_scalar_features)
        self.scalar_m2 = np.zeros(self.num_scalar_features)

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
        
        with mp.Pool(processes=self.num_workers) as pool:
            results_iterator = pool.imap_unordered(_worker_generate_sample, worker_args)
            for shape_vec, cbf, att, scalars in results_iterator:
                self.count += 1
                self.shape_vector_mean, self.shape_vector_m2 = self._update_stats((self.shape_vector_mean, self.shape_vector_m2), shape_vec)
                self.cbf_mean, self.cbf_m2 = self._update_stats((self.cbf_mean, self.cbf_m2), cbf)
                self.att_mean, self.att_m2 = self._update_stats((self.att_mean, self.att_m2), att)
                self.scalar_mean, self.scalar_m2 = self._update_stats((self.scalar_mean, self.scalar_m2), scalars)

        if self.count < 2: return {}

        shape_vector_std = np.sqrt(self.shape_vector_m2 / self.count)
        cbf_std = np.sqrt(self.cbf_m2 / self.count)
        att_std = np.sqrt(self.att_m2 / self.count)
        scalar_std = np.sqrt(self.scalar_m2 / self.count)

        return {
            'shape_vector_mean': self.shape_vector_mean.tolist(),
            'shape_vector_std': np.clip(shape_vector_std, 1e-6, None).tolist(),
            'y_mean_cbf': self.cbf_mean,
            'y_std_cbf': max(float(cbf_std), 1e-6),
            'y_mean_att': self.att_mean,
            'y_std_att': max(float(att_std), 1e-6),
            'scalar_features_mean': self.scalar_mean.tolist(),
            'scalar_features_std': np.clip(scalar_std, 1e-6, None).tolist()
        }
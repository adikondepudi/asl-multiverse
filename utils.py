# FILE: utils.py
import numpy as np
from typing import Optional, Dict
import multiprocessing as mp
import time
import torch
from asl_simulation import _generate_pcasl_signal_jit, _generate_vsasl_signal_jit
from feature_registry import FeatureRegistry

def process_signals_dynamic(raw_signals, norm_stats, config, t1_values=None, z_values=None):
    """
    Dynamically constructs input vectors based on config['active_features'].
    Uses FeatureRegistry.compute_feature_vector() as single source of truth.

    Supports two normalization modes (config['normalization_mode']):
    - 'per_curve' (default): Normalize each curve by its mean/std (shape vectors)
    - 'global_scale': Multiply by global_scale_factor to get signals into ~0-1 range.
      This preserves absolute magnitude information, similar to IVIM-NET approach.
    """
    num_plds = len(config['pld_values'])
    normalization_mode = config.get('normalization_mode', 'per_curve')
    global_scale_factor = config.get('global_scale_factor', 10.0)

    pcasl_raw = raw_signals[:, :num_plds]
    vsasl_raw = raw_signals[:, num_plds:]

    if normalization_mode == 'global_scale':
        # Global scaling: multiply signals by factor to get into ~0-1 range
        # This preserves relative magnitudes between curves (like S(b)/S(b=0) in IVIM)
        pcasl_scaled = pcasl_raw * global_scale_factor
        vsasl_scaled = vsasl_raw * global_scale_factor
        input_parts = [pcasl_scaled, vsasl_scaled]
    else:
        # Per-curve normalization (legacy behavior)
        # Creates "shape vectors" that are SNR-invariant
        pcasl_mu = np.mean(pcasl_raw, axis=1, keepdims=True)
        pcasl_std = np.std(pcasl_raw, axis=1, keepdims=True) + 1e-6
        pcasl_shape = (pcasl_raw - pcasl_mu) / pcasl_std

        vsasl_mu = np.mean(vsasl_raw, axis=1, keepdims=True)
        vsasl_std = np.std(vsasl_raw, axis=1, keepdims=True) + 1e-6
        vsasl_shape = (vsasl_raw - vsasl_mu) / vsasl_std

        input_parts = [pcasl_shape, vsasl_shape]
    
    s_mean = np.array(norm_stats['scalar_features_mean'])
    s_std = np.array(norm_stats['scalar_features_std']) + 1e-6
    
    active_list = config.get('active_features', ['mean', 'std'])
    
    # Calculate raw features using Single Source of Truth
    raw_features = FeatureRegistry.compute_feature_vector(raw_signals, num_plds, active_list)
    
    # We need to normalize. We assume norm_stats corresponds to ALL supported features in Registry order.
    # We need to pick the correct indices from s_mean/s_std based on active_list.
    
    current_idx = 0
    for feat_name in active_list:
        if feat_name in FeatureRegistry.NORM_STATS_INDICES:
            indices = FeatureRegistry.NORM_STATS_INDICES[feat_name]
            width = len(indices)
            
            # Extract this feature's columns from raw_features
            # (Assumes FeatureRegistry.compute_feature_vector returns them in order of active_list)
            feat_vals = raw_features[:, current_idx : current_idx + width]
            current_idx += width
            
            mu = s_mean[indices]
            std = s_std[indices]
            feat_norm = (feat_vals - mu) / std
            input_parts.append(feat_norm)
        
        elif feat_name == 't1_artery' and t1_values is not None:
            mu = norm_stats.get('y_mean_t1', 1650.0)  # 3T consensus (Alsop 2015)
            std = norm_stats.get('y_std_t1', 200.0) + 1e-6
            input_parts.append((t1_values - mu) / std)
            
        elif feat_name == 'z_coord' and z_values is not None:
            mu = norm_stats.get('y_mean_z', 15.0)
            std = norm_stats.get('y_std_z', 8.0) + 1e-6
            input_parts.append((z_values - mu) / std)

    return np.concatenate(input_parts, axis=1)

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
    
    # Calculate ALL features for norm stats population
    # Order: mean, std, ttp, com, peak, weighted_sum... matching NORM_STATS_INDICES keys order implies stability
    all_supported_feats = ['mean', 'std', 'ttp', 'com', 'peak', 'weighted_sum']
    scalar_features = FeatureRegistry.compute_feature_vector(raw_signal_vector, len(plds), all_supported_feats)
    
    return shape_vector, true_cbf, true_att, true_t1_artery, scalar_features.flatten()

class ParallelStreamingStatsCalculator:
    def __init__(self, simulator, plds, num_samples, num_workers):
        self.simulator = simulator
        self.plds = plds
        self.num_samples = num_samples
        self.num_workers = num_workers
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

def process_signals_cpu(signals_unnorm: np.ndarray, norm_stats: dict, num_plds: int,
                        t1_values: Optional[np.ndarray] = None, z_values: Optional[np.ndarray] = None,
                        normalization_mode: str = 'per_curve', global_scale_factor: float = 10.0) -> np.ndarray:
    """
    CPU version of preprocessing for validation data.
    Gracefully handles dimension mismatch between new engineered features (10) and old stats (8).

    Args:
        signals_unnorm: Raw signals with optional engineered features appended
        norm_stats: Normalization statistics dictionary
        num_plds: Number of PLDs
        t1_values: Optional T1 values for conditioning
        z_values: Optional Z slice values for conditioning
        normalization_mode: 'per_curve' (default) or 'global_scale'
        global_scale_factor: Scale factor for global_scale mode (default: 10.0)
    """
    raw_curves = signals_unnorm[:, :num_plds * 2]
    eng_ttp_com = signals_unnorm[:, num_plds * 2:]

    pcasl_raw = raw_curves[:, :num_plds]
    vsasl_raw = raw_curves[:, num_plds:]

    if normalization_mode == 'global_scale':
        # Global scaling: multiply signals by factor to get into ~0-1 range
        pcasl_scaled = pcasl_raw * global_scale_factor
        vsasl_scaled = vsasl_raw * global_scale_factor
        shape_vector = np.concatenate([pcasl_scaled, vsasl_scaled], axis=1)
    else:
        # Per-curve normalization (legacy behavior)
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
    
    # --- LEGACY FIX ---
    # Check dimensions. If unnorm has more features than stats, slice unnorm.
    # The new features (Peaks) are at the END of eng_ttp_com (indices 8, 9).
    # The stats (8 features) expect indices 0-7.
    # We truncate unnorm to match s_mean.
    
    if scalar_features_unnorm.shape[1] > s_mean.shape[0]:
        # Truncate input features to match the normalization stats
        scalar_features_unnorm = scalar_features_unnorm[:, :s_mean.shape[0]]
    
    # Now shapes match (8,8) or (10,10)
    scalar_features_norm = (scalar_features_unnorm - s_mean) / s_std

    if t1_values is not None:
        t1_mean = norm_stats.get('y_mean_t1', 1650.0)  # 3T consensus (Alsop 2015)
        t1_std = norm_stats.get('y_std_t1', 200.0)
        t1_norm = (t1_values - t1_mean) / (t1_std + 1e-6)
        scalar_features_norm = np.concatenate([scalar_features_norm, t1_norm], axis=1)

    if z_values is not None:
        z_mean = norm_stats.get('y_mean_z', 15.0)
        z_std = norm_stats.get('y_std_z', 8.0)
        z_norm = (z_values - z_mean) / (z_std + 1e-6)
        scalar_features_norm = np.concatenate([scalar_features_norm, z_norm], axis=1)

    return np.concatenate([shape_vector, scalar_features_norm], axis=1)

def get_grid_search_initial_guess(
    observed_signal: np.ndarray,
    plds: np.ndarray,
    asl_params: dict
) -> list:
    """
    Performs a coarse grid search for a single voxel to find a robust
    initial guess for NLLS fitting.

    Grid bounds must match the LS optimizer bounds in fit_PCVSASL_misMatchPLD_vectInit_pep:
    - CBF: [0, 200] ml/100g/min  (internal: [0/6000, 200/6000] ml/g/s)
    - ATT: [100, 4000] ms
    """
    # --- 1. Define the search grid ---
    # IMPORTANT: Must stay within LS optimizer bounds: CBF [0, 200], ATT [100, 4000]
    cbf_values_grid = np.linspace(1, 200, 20)  # 20 steps for CBF, extended to 200
    att_values_grid = np.linspace(100, 4000, 22) # 22 steps for ATT, within [100, 4000]

    # --- 2. Pre-calculate model parameters ---
    t1_artery = asl_params['T1_artery']
    t_tau = asl_params['T_tau']
    t2_factor = asl_params.get('T2_factor', 1.0)
    alpha_bs1 = asl_params.get('alpha_BS1', 1.0)
    t_sat_vs = asl_params.get('T_sat_vs', 2000.0)
    alpha_pcasl = asl_params['alpha_PCASL'] * (alpha_bs1**4)
    alpha_vsasl = asl_params['alpha_VSASL'] * (alpha_bs1**3)

    num_plds = len(plds)
    observed_pcasl = observed_signal[:num_plds]
    observed_vsasl = observed_signal[num_plds:]

    best_mse = float('inf')
    best_params = [50.0 / 6000.0, 1500.0] # Default fallback

    # --- 3. Multi-start: coarse ATT grid with analytic CBF solve ---
    # For each candidate ATT, evaluate a range of CBF values and pick the best.
    # This places the solver in the correct basin of attraction and avoids
    # the topology trap where ATT→6000ms compensates for amplitude mismatch.
    coarse_att_grid = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500])

    for att in coarse_att_grid:
        for cbf in cbf_values_grid:
            cbf_cgs = cbf / 6000.0
            pcasl_pred = _generate_pcasl_signal_jit(
                plds, att, cbf_cgs, t1_artery, t_tau, alpha_pcasl, t2_factor
            )
            vsasl_pred = _generate_vsasl_signal_jit(
                plds, att, cbf_cgs, t1_artery, alpha_vsasl, t2_factor, t_sat_vs
            )
            mse = np.mean((observed_pcasl - pcasl_pred)**2) + \
                  np.mean((observed_vsasl - vsasl_pred)**2)
            if mse < best_mse:
                best_mse = mse
                best_params = [cbf_cgs, att]

    # --- 4. Fine grid around the coarse best ---
    coarse_cbf = best_params[0] * 6000.0  # Back to ml/100g/min
    coarse_att = best_params[1]
    fine_cbf_grid = np.linspace(max(0, coarse_cbf - 20), min(200, coarse_cbf + 20), 10)
    fine_att_grid = np.linspace(max(100, coarse_att - 300), min(4000, coarse_att + 300), 10)

    for cbf in fine_cbf_grid:
        cbf_cgs = cbf / 6000.0
        for att in fine_att_grid:
            pcasl_pred = _generate_pcasl_signal_jit(
                plds, att, cbf_cgs, t1_artery, t_tau, alpha_pcasl, t2_factor
            )
            vsasl_pred = _generate_vsasl_signal_jit(
                plds, att, cbf_cgs, t1_artery, alpha_vsasl, t2_factor, t_sat_vs
            )
            mse = np.mean((observed_pcasl - pcasl_pred)**2) + \
                  np.mean((observed_vsasl - vsasl_pred)**2)
            if mse < best_mse:
                best_mse = mse
                best_params = [cbf_cgs, att]

    return best_params
# utils.py
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
from asl_simulation import _generate_pcasl_signal_jit, _generate_vsasl_signal_jit

def engineer_signal_features(raw_signal: np.ndarray, num_plds: int) -> np.ndarray:
    """
    Engineers explicit shape-based features from raw ASL signal curves to
    make timing information more salient for the neural network.

    Args:
        raw_signal: A numpy array of shape (N, num_plds * 2) or (num_plds * 2,)
                    containing concatenated PCASL and VSASL signals.
        num_plds: The number of Post-Labeling Delays per modality.

    Returns:
        A numpy array of shape (N, 4) with the engineered features:
        - PCASL time-to-peak index
        - VSASL time-to-peak index
        - PCASL center-of-mass
        - VSASL center-of-mass
    """
    is_1d = raw_signal.ndim == 1
    if is_1d:
        raw_signal = raw_signal.reshape(1, -1)  # Ensure 2D for processing

    pcasl_curves = raw_signal[:, :num_plds]
    vsasl_curves = raw_signal[:, num_plds:]
    plds_indices = np.arange(num_plds)

    # Feature 1: Time to peak (vectorized)
    pcasl_ttp = np.argmax(pcasl_curves, axis=1)
    vsasl_ttp = np.argmax(vsasl_curves, axis=1)

    # Feature 2: Center of mass (vectorized)
    pcasl_sum = np.sum(pcasl_curves, axis=1) + 1e-6
    vsasl_sum = np.sum(vsasl_curves, axis=1) + 1e-6
    pcasl_com = np.sum(pcasl_curves * plds_indices, axis=1) / pcasl_sum
    vsasl_com = np.sum(vsasl_curves * plds_indices, axis=1) / vsasl_sum
    
    engineered_features = np.stack([pcasl_ttp, vsasl_ttp, pcasl_com, vsasl_com], axis=1)

    if is_1d:
        return engineered_features.flatten().astype(np.float32)
    else:
        return engineered_features.astype(np.float32)

# Top-level worker function for multiprocessing to be able to pickle it.
def _worker_generate_sample(args_tuple):
    """
    Generates a single raw data sample for statistics calculation.
    This is run in a separate process.
    """
    simulator, plds, seed = args_tuple
    np.random.seed(seed) # Ensure each worker has a different random stream

    # Generate one subject with random physiological parameters
    # The specific condition ('healthy', 'stroke', etc.) doesn't matter as much as
    # covering the full range of CBF/ATT values for robust stats.
    physio_var = simulator.physio_var
    true_att = np.random.uniform(*physio_var.att_range)
    true_cbf = np.random.uniform(*physio_var.cbf_range)
    true_t1_artery = np.random.uniform(*physio_var.t1_artery_range)
    
    # Generate the clean signal (noise will be added later if needed, but for stats we care about the signal distribution)
    vsasl_clean = simulator._generate_vsasl_signal(plds, true_att, true_cbf, true_t1_artery, simulator.params.alpha_VSASL)
    pcasl_clean = simulator._generate_pcasl_signal(plds, true_att, true_cbf, true_t1_artery, simulator.params.T_tau, simulator.params.alpha_PCASL)
    raw_signal_vector = np.concatenate([pcasl_clean, vsasl_clean])
    
    # Calculate amplitude from the raw signal vector
    amplitude = np.linalg.norm(raw_signal_vector)
    
    return pcasl_clean, vsasl_clean, true_cbf, true_att, amplitude


class ParallelStreamingStatsCalculator:
    """
    Calculates normalization statistics in parallel using a streaming algorithm
    (Welford's algorithm) to minimize memory usage.
    """
    def __init__(self, simulator, plds, num_samples, num_workers):
        self.simulator = simulator
        self.plds = plds
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.num_plds = len(plds)

        # Initialize statistics using Welford's algorithm components
        self.count = 0
        self.pcasl_mean = np.zeros(self.num_plds)
        self.pcasl_m2 = np.zeros(self.num_plds)
        self.vsasl_mean = np.zeros(self.num_plds)
        self.vsasl_m2 = np.zeros(self.num_plds)
        self.cbf_mean, self.cbf_m2 = 0.0, 0.0
        self.att_mean, self.att_m2 = 0.0, 0.0
        self.amp_mean, self.amp_m2 = 0.0, 0.0

    def _update_stats(self, existing_agg, new_value):
        """Welford's algorithm for single pass variance calculation."""
        mean, m2 = existing_agg
        delta = new_value - mean
        mean += delta / self.count
        delta2 = new_value - mean
        m2 += delta * delta2
        return mean, m2

    def calculate(self):
        """Runs the parallel calculation and returns the final norm_stats dict."""
        print(f"Calculating normalization stats over {self.num_samples} samples using {self.num_workers} workers...")
        
        # Prepare arguments for all worker processes
        base_seed = int(time.time())
        worker_args = [(self.simulator, self.plds, base_seed + i) for i in range(self.num_samples)]
        
        with mp.Pool(processes=self.num_workers) as pool:
            # Use imap_unordered for efficiency, as order doesn't matter for stats
            results_iterator = pool.imap_unordered(_worker_generate_sample, worker_args)
            
            for pcasl, vsasl, cbf, att, amp in tqdm(results_iterator, total=len(worker_args), desc="Streaming Stats"):
                self.count += 1
                self.pcasl_mean, self.pcasl_m2 = self._update_stats((self.pcasl_mean, self.pcasl_m2), pcasl)
                self.vsasl_mean, self.vsasl_m2 = self._update_stats((self.vsasl_mean, self.vsasl_m2), vsasl)
                self.cbf_mean, self.cbf_m2 = self._update_stats((self.cbf_mean, self.cbf_m2), cbf)
                self.att_mean, self.att_m2 = self._update_stats((self.att_mean, self.att_m2), att)
                self.amp_mean, self.amp_m2 = self._update_stats((self.amp_mean, self.amp_m2), amp)

        if self.count < 2:
            return {} # Not enough data to calculate variance

        # Finalize calculations: compute variance from M2 and then std dev
        pcasl_std = np.sqrt(self.pcasl_m2 / self.count)
        vsasl_std = np.sqrt(self.vsasl_m2 / self.count)
        cbf_std = np.sqrt(self.cbf_m2 / self.count)
        att_std = np.sqrt(self.att_m2 / self.count)
        amp_std = np.sqrt(self.amp_m2 / self.count)

        return {
            'pcasl_mean': self.pcasl_mean.tolist(),
            'pcasl_std': np.clip(pcasl_std, 1e-6, None).tolist(),
            'vsasl_mean': self.vsasl_mean.tolist(),
            'vsasl_std': np.clip(vsasl_std, 1e-6, None).tolist(),
            'y_mean_cbf': self.cbf_mean,
            'y_std_cbf': max(cbf_std, 1e-6),
            'y_mean_att': self.att_mean,
            'y_std_att': max(att_std, 1e-6),
            'amplitude_mean': self.amp_mean,
            'amplitude_std': max(amp_std, 1e-6)
        }
    
def get_grid_search_initial_guess(
    observed_signal: np.ndarray,
    plds: np.ndarray,
    asl_params: dict
) -> list:
    """
    Performs a coarse grid search for a single voxel to find a robust
    initial guess for NLLS fitting.

    Args:
        observed_signal: A numpy array of shape (num_plds * 2,)
                         containing concatenated PCASL and VSASL signals.
        plds: A numpy array of the PLD values.
        asl_params: A dictionary containing the physical parameters needed for
                    the kinetic model (T1_artery, T_tau, alpha_PCASL, etc.).

    Returns:
        A list containing the best-fit [cbf_init, att_init] in units of
        [ml/g/s, ms].
    """
    # --- 1. Define the search grid ---
    # These ranges should be wide enough to cover all plausible physiological scenarios.
    cbf_values_grid = np.linspace(1, 150, 15)  # 15 steps for CBF
    att_values_grid = np.linspace(100, 4500, 22) # 22 steps for ATT

    # --- 2. Pre-calculate model parameters ---
    # This logic is copied from ASLSimulator to match the model exactly
    t1_artery = asl_params['T1_artery']
    t_tau = asl_params['T_tau']
    t2_factor = asl_params.get('T2_factor', 1.0)
    alpha_bs1 = asl_params.get('alpha_BS1', 1.0)
    alpha_pcasl = asl_params['alpha_PCASL'] * (alpha_bs1**4)
    alpha_vsasl = asl_params['alpha_VSASL'] * (alpha_bs1**3)
    
    num_plds = len(plds)
    observed_pcasl = observed_signal[:num_plds]
    observed_vsasl = observed_signal[num_plds:]

    best_mse = float('inf')
    best_params = [50.0 / 6000.0, 1500.0] # Default fallback

    # --- 3. Iterate through the grid to find the best fit ---
    for cbf in cbf_values_grid:
        cbf_cgs = cbf / 6000.0  # Convert CBF to ml/g/s for the model
        for att in att_values_grid:
            # Predict the signal for this grid point
            pcasl_pred = _generate_pcasl_signal_jit(
                plds, att, cbf_cgs, t1_artery, t_tau, alpha_pcasl, t2_factor
            )
            vsasl_pred = _generate_vsasl_signal_jit(
                plds, att, cbf_cgs, t1_artery, alpha_vsasl, t2_factor
            )

            # Calculate Mean Squared Error
            mse = np.mean((observed_pcasl - pcasl_pred)**2) + \
                  np.mean((observed_vsasl - vsasl_pred)**2)

            if mse < best_mse:
                best_mse = mse
                best_params = [cbf_cgs, att]

    return best_params
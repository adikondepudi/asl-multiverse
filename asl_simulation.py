import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import multiprocessing as mp
from itertools import product
import numba

@dataclass
class ASLParameters:
    """ASL acquisition and physiological parameters"""
    # Physiological parameters
    CBF: float = 20.0  # ml/100g/min
    T1_artery: float = 1850.0  # ms
    T2_factor: float = 1.0 # T2 decay effect on ASL signal (e.g., from crusher gradients or long TEs)
    alpha_BS1: float = 1.0 # Background suppression efficiency factor for the first BS pulse. Overall effect is alpha_BS1^n_bs_pulses.
    
    # Labeling parameters
    T_tau: float = 1800.0  # PCASL labeling duration (ms)
    T_sat_vs: float = 2000.0  # VSASL saturation delay (ms)
    alpha_PCASL: float = 0.85 # PCASL labeling efficiency
    alpha_VSASL: float = 0.56 # VSASL labeling efficiency (velocity-selective inversion/saturation efficiency)
    
    # Timing parameters for noise scaling
    TR_VSASL: float = 3936.0  # ms, Repetition time for one VSASL pair
    TR_PCASL: float = 4000.0  # ms, Repetition time for one PCASL pair (e.g. at PLD=1.5s, T_tau=1.8s)
    basetime_VS: float = 436.0  # ms (GRASE readout + VS module + extra dead time in TR for VSASL)
    basetime_PC: float = 700.0  # ms (GRASE readout + extra dead time in TR for PCASL)

class ASLSimulator:
    """ASL signal simulation with realistic noise scaling"""
    
    def __init__(self, params: ASLParameters = ASLParameters()):
        self.params = params # This holds the 'base' parameters
        self.cbf = self.params.CBF / 6000  # Convert to ml/g/s
        
    def generate_grid_points(self, 
                           cbf_range: Tuple[float, float, float] = (10, 90, 10),
                           att_range: Tuple[float, float, float] = (100, 5400, 100)) -> np.ndarray:
        """Generate grid points for CBF and ATT"""
        cbf_values = np.arange(*cbf_range)
        att_values = np.arange(*att_range)
        return np.array(list(product(cbf_values, att_values)))
    
    def compute_tr_noise_scaling(self, plds: np.ndarray) -> Dict[str, float]:
        """Compute noise scaling factors based on TR and total scan time"""
        # Total scan times for a multi-PLD acquisition
        # Assumes PLD is the variable part of TR. Other components are fixed per PLD.
        total_time_VSASL = np.sum(plds) + (self.params.T_sat_vs + self.params.basetime_VS) * len(plds)
        total_time_PCASL = np.sum(plds) + (self.params.T_tau + self.params.basetime_PC) * len(plds)
        total_time_MULTIVERSE = total_time_VSASL + total_time_PCASL
        
        # Reference TR for noise normalization (e.g. single PLD PCASL)
        # Using TR_PCASL as the reference TR for a single PLD acquisition, as in original code.
        reference_scan_time_single_pld = self.params.TR_PCASL 

        return {
            'VSASL': np.sqrt(reference_scan_time_single_pld / total_time_VSASL),
            'PCASL': np.sqrt(reference_scan_time_single_pld / total_time_PCASL),
            'MULTIVERSE': np.sqrt(reference_scan_time_single_pld / total_time_MULTIVERSE)
        }
    
    def generate_synthetic_data(self, 
                              plds: np.ndarray,
                              att_values: np.ndarray, # Array of ATT values to simulate for
                              n_noise: int = 1000, # Number of noise realizations per ATT
                              tsnr: float = 5.0,
                              # Optional parameters to override self.params for this specific generation run
                              # This is useful if RealisticASLSimulator wants to pass perturbed params
                              cbf_val: Optional[float] = None,
                              t1_artery_val: Optional[float] = None,
                              t_tau_val: Optional[float] = None,
                              alpha_pcasl_val: Optional[float] = None,
                              alpha_vsasl_val: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Generate synthetic ASL data with realistic noise"""
        
        # Use provided parameters or fall back to instance parameters
        current_cbf = cbf_val if cbf_val is not None else self.params.CBF
        current_t1_artery = t1_artery_val if t1_artery_val is not None else self.params.T1_artery
        current_t_tau = t_tau_val if t_tau_val is not None else self.params.T_tau
        current_alpha_pcasl = alpha_pcasl_val if alpha_pcasl_val is not None else self.params.alpha_PCASL
        current_alpha_vsasl = alpha_vsasl_val if alpha_vsasl_val is not None else self.params.alpha_VSASL

        # Reference signal level for noise calculation. 
        # MUST be calculated using the FIXED, base parameters from self.params
        sig_level = self._compute_reference_signal() # No arguments passed!
        noise_sd = sig_level / tsnr
        
        # Compute noise scaling factors (uses instance params for TR definitions)
        noise_scaling = self.compute_tr_noise_scaling(plds)
        
        # Initialize arrays for signals
        n_plds_len = len(plds)
        n_att_len = len(att_values)
        
        signals = {
            'VSASL': np.zeros((n_noise, n_att_len, n_plds_len)),
            'PCASL': np.zeros((n_noise, n_att_len, n_plds_len)),
            'MULTIVERSE': np.zeros((n_noise, n_att_len, n_plds_len, 2)) # Last dim for PCASL, VSASL
        }
        
        # Generate base signals and add noise
        for i, att in enumerate(att_values):
            # Generate clean signals using potentially overridden parameters
            vsasl_sig = self._generate_vsasl_signal(plds, att, cbf_ml_100g_min=current_cbf, t1_artery=current_t1_artery, alpha_vsasl=current_alpha_vsasl)
            pcasl_sig = self._generate_pcasl_signal(plds, att, cbf_ml_100g_min=current_cbf, t1_artery=current_t1_artery, t_tau=current_t_tau, alpha_pcasl=current_alpha_pcasl)
            
            # Add scaled noise
            for n in range(n_noise):
                signals['VSASL'][n,i] = vsasl_sig + noise_sd * noise_scaling['VSASL'] * np.random.randn(n_plds_len)
                signals['PCASL'][n,i] = pcasl_sig + noise_sd * noise_scaling['PCASL'] * np.random.randn(n_plds_len)
                # For MULTIVERSE, noise is added to PCASL and VSASL components using their respective, correct scaling factors.
                signals['MULTIVERSE'][n,i,:,0] = pcasl_sig + noise_sd * noise_scaling['PCASL'] * np.random.randn(n_plds_len)
                signals['MULTIVERSE'][n,i,:,1] = vsasl_sig + noise_sd * noise_scaling['VSASL'] * np.random.randn(n_plds_len)
        
        return signals
    
    def _compute_reference_signal(self, cbf_val: Optional[float] = None, t1_artery_val: Optional[float] = None, t_tau_val: Optional[float] = None, alpha_pcasl_val: Optional[float] = None) -> float:
        """Compute reference signal level for noise calculation (PCASL at PLD=2s, ATT=1.5s)"""
        ref_cbf = cbf_val if cbf_val is not None else self.params.CBF
        ref_t1_artery = t1_artery_val if t1_artery_val is not None else self.params.T1_artery
        ref_t_tau = t_tau_val if t_tau_val is not None else self.params.T_tau
        ref_alpha_pcasl = alpha_pcasl_val if alpha_pcasl_val is not None else self.params.alpha_PCASL

        return self._generate_pcasl_signal(
            plds=np.array([2000]), att=1500,
            cbf_ml_100g_min=ref_cbf, t1_artery=ref_t1_artery,
            t_tau=ref_t_tau, alpha_pcasl=ref_alpha_pcasl
        )[0]
    
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _generate_vsasl_signal_jit(plds, att, cbf_ml_g_s, t1_artery, alpha2, T2_factor):
        """JIT-compiled worker for VSASL signal generation."""
        M0_b = 1.0
        lambda_blood = 0.90
        signal = np.zeros_like(plds, dtype=np.float64)

        # Condition 1: TI <= ATT
        for i in range(plds.shape[0]):
            if plds[i] <= att:
                signal[i] = (2 * M0_b * cbf_ml_g_s * alpha2 / lambda_blood * 
                             (plds[i] / 1000.0) *
                             np.exp(-plds[i] / t1_artery) * 
                             T2_factor)
            # Condition 2: TI > ATT
            else:
                signal[i] = (2 * M0_b * cbf_ml_g_s * alpha2 / lambda_blood * 
                             (att / 1000.0) *
                             np.exp(-plds[i] / t1_artery) * 
                             T2_factor)
        return signal

    def _generate_vsasl_signal(self, plds: np.ndarray, att: float, 
                               cbf_ml_100g_min: float, t1_artery: float, alpha_vsasl: float) -> np.ndarray:
        """Wrapper for the JIT-compiled VSASL function."""
        alpha2 = alpha_vsasl * (self.params.alpha_BS1**3)
        cbf_ml_g_s = cbf_ml_100g_min / 6000.0
        return self._generate_vsasl_signal_jit(plds, att, cbf_ml_g_s, t1_artery, alpha2, self.params.T2_factor)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _generate_pcasl_signal_jit(plds, att, cbf_ml_g_s, t1_artery, t_tau, alpha1, T2_factor):
        """JIT-compiled worker for PCASL signal generation."""
        M0_b = 1.0
        lambda_blood = 0.90
        signal = np.zeros_like(plds, dtype=np.float64)
        
        for i in range(plds.shape[0]):
            # Condition 1: ATT - T_tau <= PLD < ATT
            if plds[i] >= (att - t_tau) and plds[i] < att:
                signal[i] = (2 * M0_b * cbf_ml_g_s * alpha1 / lambda_blood * 
                             (t1_artery / 1000.0) *
                             (np.exp(-att / t1_artery) - np.exp(-(t_tau + plds[i]) / t1_artery)) * 
                             T2_factor)
            # Condition 2: PLD >= ATT
            elif plds[i] >= att:
                signal[i] = (2 * M0_b * cbf_ml_g_s * alpha1 / lambda_blood * 
                             (t1_artery / 1000.0) *
                             np.exp(-plds[i] / t1_artery) *
                             (1 - np.exp(-t_tau / t1_artery)) * 
                             T2_factor)
            # Condition 0 (PLD < ATT - T_tau) is implicitly handled by signal being initialized to zeros.
        return signal

    def _generate_pcasl_signal(self, plds: np.ndarray, att: float,
                               cbf_ml_100g_min: float, t1_artery: float, t_tau: float, alpha_pcasl: float) -> np.ndarray:
        """Wrapper for the JIT-compiled PCASL function."""
        alpha1 = alpha_pcasl * (self.params.alpha_BS1**4)
        cbf_ml_g_s = cbf_ml_100g_min / 6000.0
        return self._generate_pcasl_signal_jit(plds, att, cbf_ml_g_s, t1_artery, t_tau, alpha1, self.params.T2_factor)

    def parallel_grid_search(self, 
                           observed_signal: np.ndarray,
                           grid_points: np.ndarray,
                           signal_type: str = 'PCASL',
                           plds: Optional[np.ndarray] = None,
                           # Allow passing specific parameters for generation during grid search
                           # These would typically be the *fixed* known parameters for the simulation
                           t1_artery_val: Optional[float] = None,
                           t_tau_val: Optional[float] = None,
                           alpha_pcasl_val: Optional[float] = None,
                           alpha_vsasl_val: Optional[float] = None
                           ) -> Tuple[float, float]:
        """
        Perform parallel grid search to find initial CBF and ATT estimates.
        CBF in grid_points should be in ml/100g/min.
        """
        if plds is None:
            plds = np.arange(500, 3001, 500)

        # Use instance params as default if specific values not provided
        gs_t1_artery = t1_artery_val if t1_artery_val is not None else self.params.T1_artery
        gs_t_tau = t_tau_val if t_tau_val is not None else self.params.T_tau
        gs_alpha_pcasl = alpha_pcasl_val if alpha_pcasl_val is not None else self.params.alpha_PCASL
        gs_alpha_vsasl = alpha_vsasl_val if alpha_vsasl_val is not None else self.params.alpha_VSASL
            
        def compute_error(params_cbf_att): # params_cbf_att is [CBF_ml/100g/min, ATT_ms]
            cbf_search_val, att_search_val = params_cbf_att
            if signal_type == 'PCASL':
                predicted = self._generate_pcasl_signal(plds, att_search_val, 
                                                        cbf_ml_100g_min=cbf_search_val, t1_artery=gs_t1_artery, 
                                                        t_tau=gs_t_tau, alpha_pcasl=gs_alpha_pcasl)
            elif signal_type == 'VSASL':
                predicted = self._generate_vsasl_signal(plds, att_search_val,
                                                        cbf_ml_100g_min=cbf_search_val, t1_artery=gs_t1_artery,
                                                        alpha_vsasl=gs_alpha_vsasl)
            else:  # MULTIVERSE, observed_signal is expected to be (n_plds, 2) or flattened
                pcasl_pred = self._generate_pcasl_signal(plds, att_search_val,
                                                        cbf_ml_100g_min=cbf_search_val, t1_artery=gs_t1_artery, 
                                                        t_tau=gs_t_tau, alpha_pcasl=gs_alpha_pcasl)
                vsasl_pred = self._generate_vsasl_signal(plds, att_search_val,
                                                        cbf_ml_100g_min=cbf_search_val, t1_artery=gs_t1_artery,
                                                        alpha_vsasl=gs_alpha_vsasl)
                if observed_signal.ndim == 1: # Flattened (PCASL_part, VSASL_part)
                    predicted = np.concatenate((pcasl_pred, vsasl_pred))
                else: # Stacked (n_plds, 2)
                    predicted = np.column_stack((pcasl_pred, vsasl_pred))
            
            return np.mean((predicted - observed_signal)**2)
        
        # Using context manager for Pool is good practice
        with mp.Pool() as pool:
            errors = pool.map(compute_error, grid_points)
            
        best_idx = np.argmin(errors)
        return tuple(grid_points[best_idx])
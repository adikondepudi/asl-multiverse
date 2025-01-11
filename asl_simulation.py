import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import multiprocessing as mp
from itertools import product

@dataclass
class ASLParameters:
    """ASL acquisition and physiological parameters"""
    # Physiological parameters
    CBF: float = 20.0  # ml/100g/min
    T1_artery: float = 1850.0  # ms
    T2_factor: float = 1.0
    alpha_BS1: float = 1.0
    
    # Labeling parameters
    T_tau: float = 1800.0  # PCASL labeling duration (ms)
    T_sat_vs: float = 2000.0  # VSASL saturation delay (ms)
    alpha_PCASL: float = 0.85
    alpha_VSASL: float = 0.56
    
    # Timing parameters for noise scaling
    TR_VSASL: float = 3936.0  # ms
    TR_PCASL: float = 4000.0  # ms
    basetime_VS: float = 436.0  # ms (GRASE + VS module + extra)
    basetime_PC: float = 700.0  # ms (GRASE + extra)

class ASLSimulator:
    """ASL signal simulation with realistic noise scaling"""
    
    def __init__(self, params: ASLParameters = ASLParameters()):
        self.params = params
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
        # Total scan times
        total_time_VSASL = np.sum(plds) + (self.params.T_sat_vs + self.params.basetime_VS) * len(plds)
        total_time_PCASL = np.sum(plds) + (self.params.T_tau + self.params.basetime_PC) * len(plds)
        total_time_MULTIVERSE = total_time_VSASL + total_time_PCASL
        
        # Noise scaling factors (sqrt of time ratio)
        return {
            'VSASL': np.sqrt(total_time_VSASL / self.params.TR_PCASL),
            'PCASL': np.sqrt(total_time_PCASL / self.params.TR_PCASL),
            'MULTIVERSE': np.sqrt(total_time_MULTIVERSE / self.params.TR_PCASL)
        }
    
    def generate_synthetic_data(self, 
                              plds: np.ndarray,
                              att_values: np.ndarray,
                              n_noise: int = 1000,
                              tsnr: float = 5.0) -> Dict[str, np.ndarray]:
        """Generate synthetic ASL data with realistic noise"""
        # Reference signal level for noise calculation
        sig_level = self._compute_reference_signal()
        noise_sd = sig_level / tsnr
        
        # Compute noise scaling factors
        noise_scaling = self.compute_tr_noise_scaling(plds)
        
        # Initialize arrays for signals
        n_plds = len(plds)
        n_att = len(att_values)
        
        signals = {
            'VSASL': np.zeros((n_noise, n_att, n_plds)),
            'PCASL': np.zeros((n_noise, n_att, n_plds)),
            'MULTIVERSE': np.zeros((n_noise, n_att, n_plds, 2))
        }
        
        # Generate base signals and add noise
        for i, att in enumerate(att_values):
            # Generate clean signals
            vsasl_sig = self._generate_vsasl_signal(plds, att)
            pcasl_sig = self._generate_pcasl_signal(plds, att)
            
            # Add scaled noise
            for n in range(n_noise):
                signals['VSASL'][n,i] = vsasl_sig + noise_sd * noise_scaling['VSASL'] * np.random.randn(n_plds)
                signals['PCASL'][n,i] = pcasl_sig + noise_sd * noise_scaling['PCASL'] * np.random.randn(n_plds)
                signals['MULTIVERSE'][n,i,:,0] = pcasl_sig + noise_sd * noise_scaling['MULTIVERSE'] * np.random.randn(n_plds)
                signals['MULTIVERSE'][n,i,:,1] = vsasl_sig + noise_sd * noise_scaling['MULTIVERSE'] * np.random.randn(n_plds)
        
        return signals
    
    def _compute_reference_signal(self) -> float:
        """Compute reference signal level for noise calculation"""
        return self._generate_pcasl_signal(np.array([2000]), 1500)[0]
    
    def _generate_vsasl_signal(self, plds: np.ndarray, att: float) -> np.ndarray:
        """Generate VSASL signal without noise"""
        M0_b = 1
        lambda_blood = 0.90
        alpha2 = self.params.alpha_VSASL * (self.params.alpha_BS1**3)
        
        signal = np.zeros_like(plds, dtype=float)
        index_1 = plds <= att
        
        if np.any(~index_1):
            signal[~index_1] = (2 * M0_b * self.cbf * alpha2 / lambda_blood * att / 1000 *
                              np.exp(-plds[~index_1]/self.params.T1_artery) * 
                              self.params.T2_factor)
            
        if np.any(index_1):
            signal[index_1] = (2 * M0_b * self.cbf * alpha2 / lambda_blood * 
                             plds[index_1] / 1000 *
                             np.exp(-plds[index_1]/self.params.T1_artery) * 
                             self.params.T2_factor)
            
        return signal
    
    def _generate_pcasl_signal(self, plds: np.ndarray, att: float) -> np.ndarray:
        """Generate PCASL signal without noise"""
        M0_b = 1
        lambda_blood = 0.90
        alpha1 = self.params.alpha_PCASL * (self.params.alpha_BS1**4)
        
        signal = np.zeros_like(plds, dtype=float)
        
        index_0 = plds < (att - self.params.T_tau)
        index_1 = (plds < att) & (plds >= (att - self.params.T_tau))
        index_2 = plds >= att
        
        if np.any(index_1):
            signal[index_1] = (2 * M0_b * self.cbf * alpha1 / lambda_blood * 
                             self.params.T1_artery / 1000 *
                             (np.exp(-att/self.params.T1_artery) - 
                              np.exp(-(self.params.T_tau + plds[index_1])/self.params.T1_artery)) * 
                             self.params.T2_factor)
            
        if np.any(index_2):
            signal[index_2] = (2 * M0_b * self.cbf * alpha1 / lambda_blood * 
                             self.params.T1_artery / 1000 *
                             np.exp(-plds[index_2]/self.params.T1_artery) *
                             (1 - np.exp(-self.params.T_tau/self.params.T1_artery)) * 
                             self.params.T2_factor)
            
        return signal

    def parallel_grid_search(self, 
                           observed_signal: np.ndarray,
                           grid_points: np.ndarray,
                           signal_type: str = 'PCASL',
                           plds: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Perform parallel grid search to find initial CBF and ATT estimates
        
        Parameters
        ----------
        observed_signal : array_like
            Measured signal values
        grid_points : array_like
            Array of [CBF, ATT] pairs to test
        signal_type : str
            Type of ASL signal ('PCASL', 'VSASL', or 'MULTIVERSE')
        plds : array_like, optional
            PLDs corresponding to the observed signal
            
        Returns
        -------
        Tuple[float, float]
            Best fitting [CBF, ATT] pair
        """
        if plds is None:
            plds = np.arange(500, 3001, 500)
            
        def compute_error(params):
            cbf, att = params
            if signal_type == 'PCASL':
                predicted = self._generate_pcasl_signal(plds, att)
            elif signal_type == 'VSASL':
                predicted = self._generate_vsasl_signal(plds, att)
            else:  # MULTIVERSE
                pcasl = self._generate_pcasl_signal(plds, att)
                vsasl = self._generate_vsasl_signal(plds, att)
                predicted = np.column_stack((pcasl, vsasl))
            return np.mean((predicted - observed_signal)**2)
        
        with mp.Pool() as pool:
            errors = pool.map(compute_error, grid_points)
            
        best_idx = np.argmin(errors)
        return tuple(grid_points[best_idx])
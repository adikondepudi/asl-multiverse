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

def _generate_single_balanced_subject(args: Tuple) -> Tuple[np.ndarray, List[float], Dict]:
    """Worker function for parallel generation of a single subject."""
    att_min_bin, att_max_bin, cbf_full_range, t1_artery_full_range, base_params, plds, noise_levels, physio_var, seed = args

    np.random.seed(seed)

    true_att = np.random.uniform(att_min_bin, att_max_bin)
    true_cbf = np.random.uniform(*cbf_full_range)
    true_t1_artery = np.random.uniform(*t1_artery_full_range)
    current_snr = np.random.choice(noise_levels)

    perturbed_t_tau = base_params.T_tau * (1 + np.random.uniform(*physio_var.t_tau_perturb_range))
    perturbed_alpha_pcasl = np.clip(base_params.alpha_PCASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.1)
    perturbed_alpha_vsasl = np.clip(base_params.alpha_VSASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.0)

    simulator = RealisticASLSimulator(base_params)

    data_dict = simulator.generate_synthetic_data(
        plds,
        att_values=np.array([true_att]),
        n_noise=1,
        tsnr=current_snr,
        cbf_val=true_cbf,
        t1_artery_val=true_t1_artery,
        t_tau_val=perturbed_t_tau,
        alpha_pcasl_val=perturbed_alpha_pcasl,
        alpha_vsasl_val=perturbed_alpha_vsasl
    )

    pcasl_noisy = data_dict['MULTIVERSE'][0, 0, :, 0]
    vsasl_noisy = data_dict['MULTIVERSE'][0, 0, :, 1]
    multiverse_signal_flat = np.concatenate([pcasl_noisy, vsasl_noisy])

    params = [true_cbf, true_att]
    perturbed_params_dict = {'t1_artery': true_t1_artery, 't_tau': perturbed_t_tau, 'alpha_pcasl': perturbed_alpha_pcasl, 'alpha_vsasl': perturbed_alpha_vsasl}

    return multiverse_signal_flat, params, perturbed_params_dict

class RealisticASLSimulator(ASLSimulator):
    def __init__(self, params: ASLParameters = ASLParameters()):
        super().__init__(params)
        self.physio_var = PhysiologicalVariation()

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
            
            perturbed_t_tau = base_params.T_tau * (1 + np.random.uniform(*self.physio_var.t_tau_perturb_range))
            perturbed_alpha_pcasl = np.clip(base_params.alpha_PCASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range)), 0.1, 1.1)
            perturbed_alpha_vsasl = np.clip(base_params.alpha_VSASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range)), 0.1, 1.0)
            
            vsasl_clean = self._generate_vsasl_signal(plds, att, cbf, t1_a, perturbed_alpha_vsasl)
            pcasl_clean = self._generate_pcasl_signal(plds, att, cbf, t1_a, perturbed_t_tau, perturbed_alpha_pcasl)
            
            for snr in noise_levels:
                vsasl_noisy = self.add_realistic_noise(vsasl_clean, snr=snr)
                pcasl_noisy = self.add_realistic_noise(pcasl_clean, snr=snr)
                
                multiverse_signal_flat = np.concatenate([pcasl_noisy, vsasl_noisy])
                
                dataset['signals'].append(multiverse_signal_flat)
                dataset['parameters'].append([cbf, att])
                dataset['conditions'].append(condition)
                dataset['noise_levels'].append(snr)
                dataset['perturbed_params'].append({
                    't1_artery': t1_a, 't_tau': perturbed_t_tau, 
                    'alpha_pcasl': perturbed_alpha_pcasl, 'alpha_vsasl': perturbed_alpha_vsasl
                })

        dataset['signals'] = np.array(dataset['signals'])
        dataset['parameters'] = np.array(dataset['parameters'])
        return dataset

    def generate_spatial_data(self, matrix_size: Tuple[int, int] = (64, 64), n_slices: int = 20,
                            plds: Optional[np.ndarray] = None) -> np.ndarray:
        if plds is None: plds = np.arange(500, 3001, 500)
        x, y = np.meshgrid(np.linspace(-1,1,matrix_size[0]), np.linspace(-1,1,matrix_size[1]))
        cbf_map = 60 + 20*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
        cbf_map[(x**2+y**2)<0.7] += 20; cbf_map[(x**2+y**2)>0.7] *= 0.5
        cbf_map = np.clip(cbf_map, 5, 150)
        att_map = 1000 + 500*np.sqrt(x**2+y**2); att_map = np.clip(att_map, 300, 4000)
        cbf_map_smooth, att_map_smooth = gaussian_filter(cbf_map, sigma=2), gaussian_filter(att_map, sigma=2)
        data_4d_pcasl = np.zeros((matrix_size[0], matrix_size[1], n_slices, len(plds)))
        for z in range(n_slices):
            slice_cbf_factor, slice_att_factor = 0.9+0.2*np.random.rand(), 0.95+0.1*np.random.rand()
            slice_cbf_map, slice_att_map = np.clip(cbf_map_smooth*slice_cbf_factor,5,150), np.clip(att_map_smooth*slice_att_factor,300,4000)
            for i in range(matrix_size[0]):
                for j in range(matrix_size[1]):
                    signal = self._generate_pcasl_signal(plds, slice_att_map[i,j], slice_cbf_map[i,j],
                                                        self.params.T1_artery, self.params.T_tau, self.params.alpha_PCASL)
                    data_4d_pcasl[i,j,z,:] = self.add_realistic_noise(signal, snr=np.random.uniform(5,15))
        return data_4d_pcasl, cbf_map_smooth, att_map_smooth

def generate_training_data_parallel(n_cores: int = None):
    if n_cores is None: n_cores = max(1, mp.cpu_count() - 1)
    base_asl_params = ASLParameters(); plds = np.arange(500, 3001, 500)
    total_subjects_to_generate = 1000
    n_subjects_per_core = total_subjects_to_generate // n_cores
    extra_subjects_list = [1]*(total_subjects_to_generate%n_cores) + [0]*(n_cores-(total_subjects_to_generate%n_cores)) if total_subjects_to_generate % n_cores != 0 else [0]*n_cores
    worker_args_list = [(i, n_subjects_per_core+extra_subjects_list[i], base_asl_params, plds) for i in range(n_cores) if (n_subjects_per_core+extra_subjects_list[i]) > 0]
    if not worker_args_list:
        logger.warning("No subjects to generate. Exiting parallel generation.")
        return {'signals': np.array([]), 'parameters': np.array([]), 'conditions': [], 'noise_levels': [], 'perturbed_params': []}
    def worker_process(args_tuple):
        core_id, n_subj_worker, b_params, p_lds = args_tuple
        np.random.seed(42 + core_id)
        sim = RealisticASLSimulator(params=b_params)
        return sim.generate_diverse_dataset(p_lds, n_subj_worker, ['healthy','stroke','tumor','elderly'], [3.0,5.0,10.0,15.0])
    logger.info(f"Generating data using {len(worker_args_list)} cores for {total_subjects_to_generate} total effective subjects...")
    with mp.Pool(len(worker_args_list)) as pool:
        results = list(tqdm(pool.imap(worker_process, worker_args_list), total=len(worker_args_list), desc="Parallel Generation"))
    if not results:
        logger.warning("No results from workers.")
        return {'signals': np.array([]), 'parameters': np.array([]), 'conditions': [], 'noise_levels': [], 'perturbed_params': []}
    combined_data = {'signals': np.vstack([r['signals'] for r in results if r['signals'].size > 0]),
                     'parameters': np.vstack([r['parameters'] for r in results if r['parameters'].size > 0]),
                     'conditions': sum([r['conditions'] for r in results], []),
                     'noise_levels': sum([r['noise_levels'] for r in results], []),
                     'perturbed_params': sum([r['perturbed_params'] for r in results], [])}
    logger.info(f"Generated {len(combined_data['signals'])} total samples.")
    return combined_data

if __name__ == "__main__":
    simulator = RealisticASLSimulator()
    logger.info("Enhanced ASL Simulator initialized. `generate_balanced_dataset` has been removed.")
    logger.info("For training, please use the `ASLIterableDataset` class.")
    logger.info("\nTesting `generate_diverse_dataset` for fixed validation/test sets...")
    test_data = simulator.generate_diverse_dataset(plds=np.arange(500, 3001, 500), n_subjects=10)
    if test_data['signals'].size > 0:
        logger.info(f"\nGenerated test dataset shape: {test_data['signals'].shape}")
    else: logger.info("No test data generated.")
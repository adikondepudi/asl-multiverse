import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from asl_simulation import ASLSimulator, ASLParameters
import multiprocessing as mp
from tqdm import tqdm

@dataclass
class PhysiologicalVariation:
    """Physiological parameter variations for realistic simulation"""
    cbf_range: Tuple[float, float] = (20.0, 100.0)  # ml/100g/min
    att_range: Tuple[float, float] = (500.0, 4000.0)  # ms
    t1_artery_range: Tuple[float, float] = (1650.0, 2050.0)  # ms
    
    # Disease-specific variations
    stroke_cbf_range: Tuple[float, float] = (5.0, 30.0)
    stroke_att_range: Tuple[float, float] = (1500.0, 4500.0) # Typically prolonged

    tumor_cbf_range: Tuple[float, float] = (10.0, 150.0) # Can be hypo or hyperperfused
    tumor_att_range: Tuple[float, float] = (700.0, 3500.0) # Can be variable
    
    # Age-related variations
    young_cbf_range: Tuple[float, float] = (60.0, 120.0)
    young_att_range: Tuple[float, float] = (500.0, 1500.0)

    elderly_cbf_range: Tuple[float, float] = (30.0, 70.0) # Generally reduced CBF
    elderly_att_range: Tuple[float, float] = (1500.0, 3500.0) # Generally prolonged ATT

    # Sequence parameter variation ranges (percentage deviation)
    t_tau_perturb_range: Tuple[float, float] = (-0.05, 0.05) # +/- 5%
    alpha_perturb_range: Tuple[float, float] = (-0.10, 0.10) # +/- 10%


class RealisticASLSimulator(ASLSimulator):
    """Enhanced ASL simulator with realistic noise and physiological variations"""
    
    def __init__(self, params: ASLParameters = ASLParameters()):
        super().__init__(params)
        self.physio_var = PhysiologicalVariation()
        
    def add_realistic_noise(self, 
                          signal: np.ndarray, # Clean signal
                          noise_type: str = 'gaussian',
                          snr: float = 5.0,
                          temporal_correlation: float = 0.3,
                          include_spike_artifacts: bool = False,
                          spike_probability: float = 0.01, # Probability of a spike per PLD
                          spike_magnitude_factor: float = 5.0, # Spike magnitude relative to noise_level
                          include_baseline_drift: bool = False,
                          drift_magnitude_factor: float = 0.1 # Drift magnitude relative to signal max
                          ) -> np.ndarray:
        """Add realistic noise with temporal correlations, spikes, and drifts."""
        
        base_noise_level = np.mean(np.abs(signal)) / snr if snr > 0 else 0
        if base_noise_level == 0 and np.all(signal == 0): # Handle zero signal case
            base_noise_level = 1e-5 # A very small noise floor if signal is all zeros

        noise = np.zeros_like(signal)

        if noise_type == 'gaussian':
            noise = np.random.normal(0, base_noise_level, signal.shape)
            
        elif noise_type == 'rician':
            signal_magnitude = np.abs(signal)
            # For Rician, sigma is the std dev of Gaussian components
            sigma_rician = base_noise_level / np.sqrt(2) # Approximation, actual Rician SNR def is complex
            
            noise_real = np.random.normal(0, sigma_rician, signal.shape)
            noise_imag = np.random.normal(0, sigma_rician, signal.shape)
            
            # Add to signal and compute magnitude
            # Note: If signal is already a difference signal, adding Rician noise this way might be less direct.
            # A more common approach is to simulate complex M0, add complex noise, then subtract.
            # For simplicity here, applying to the difference signal magnitude.
            noisy_signal_mag = np.sqrt((signal_magnitude + noise_real)**2 + noise_imag**2)
            return noisy_signal_mag # Return directly as noise is incorporated
            
        elif noise_type == 'physiological':
            # Simplified physiological noise components
            if signal.ndim > 0 and signal.shape[-1] > 1: # Need at least 2 points for t
                t = np.linspace(0, signal.shape[-1], signal.shape[-1], endpoint=False) # Time points for each PLD
                # Cardiac pulsation (~1 Hz, assuming TR ~1s per PLD for scaling)
                cardiac = (base_noise_level * 0.5) * np.sin(2 * np.pi * 1.0 * t / (signal.shape[-1]/5) + np.random.rand()*np.pi) # Random phase
                # Respiratory (~0.2-0.3 Hz)
                respiratory = (base_noise_level * 0.3) * np.sin(2 * np.pi * 0.25 * t / (signal.shape[-1]/5) + np.random.rand()*np.pi)
                noise += cardiac + respiratory
            
            noise += np.random.normal(0, base_noise_level, signal.shape) # Add base Gaussian
            
        # Add temporal correlation if specified (to the generated noise)
        if temporal_correlation > 0 and signal.ndim > 0 and signal.shape[-1] > 1:
            from scipy.ndimage import gaussian_filter1d # Import here to avoid global dep
            noise = gaussian_filter1d(noise, sigma=temporal_correlation, axis=-1)

        noisy_signal_intermediate = signal + noise

        # Add sporadic spike artifacts
        if include_spike_artifacts and signal.ndim > 0:
            num_plds = signal.shape[-1]
            for i in range(num_plds):
                if np.random.rand() < spike_probability:
                    spike = (np.random.choice([-1,1])) * spike_magnitude_factor * base_noise_level
                    if signal.ndim == 1:
                        noisy_signal_intermediate[i] += spike
                    elif signal.ndim == 2: # (batch, plds)
                        noisy_signal_intermediate[:, i] += spike # Add same spike across batch for this PLD, or make random per batch
                    # Add more cases if higher dims
        
        # Add low-frequency baseline drift
        if include_baseline_drift and signal.ndim > 0 and signal.shape[-1] > 1:
            max_signal_val = np.max(np.abs(signal)) if np.any(signal) else base_noise_level
            drift_amp = drift_magnitude_factor * max_signal_val
            # Simple sinusoidal drift
            drift = drift_amp * np.sin(2 * np.pi * np.random.uniform(0.05, 0.2) * np.arange(signal.shape[-1]) / signal.shape[-1] + np.random.rand()*np.pi)
            noisy_signal_intermediate += drift
            
        return noisy_signal_intermediate
    
    def generate_diverse_dataset(self,
                               plds: np.ndarray,
                               n_subjects: int = 100,
                               conditions: List[str] = ['healthy', 'stroke', 'tumor', 'elderly'],
                               noise_levels: List[float] = [3.0, 5.0, 10.0], # SNR levels
                               noise_artifact_options: Optional[Dict] = None
                               ) -> Dict:
        """Generate diverse dataset with various physiological conditions and perturbed parameters."""
        
        dataset = {
            'signals': [],      # Stores the final MULTIVERSE signal (PCASL+VSASL flattened)
            'parameters': [],   # Stores [CBF_true, ATT_true]
            'conditions': [],   # Stores the condition string
            'noise_levels': [], # Stores the SNR level used
            'perturbed_params': [] # Stores dict of perturbed T1a, T_tau, alphas
        }

        default_artifact_options = {
            'temporal_correlation': 0.2, 'include_spike_artifacts': True, 
            'spike_probability': 0.01, 'spike_magnitude_factor': 3.0,
            'include_baseline_drift': True, 'drift_magnitude_factor': 0.05
        }
        if noise_artifact_options is None:
            noise_artifact_options = default_artifact_options
        else: # Update defaults with any provided options
            default_artifact_options.update(noise_artifact_options)
            noise_artifact_options = default_artifact_options

        base_params = self.params # Keep a copy of original base parameters

        for _ in tqdm(range(n_subjects), desc="Generating Subjects"):
            condition = np.random.choice(conditions) # Randomly pick a condition for this subject
            
            # Sample physiological parameters based on condition
            if condition == 'healthy':
                cbf = np.random.uniform(*self.physio_var.cbf_range)
                att = np.random.uniform(*self.physio_var.att_range)
                t1_a = np.random.uniform(*self.physio_var.t1_artery_range)
            elif condition == 'stroke':
                cbf = np.random.uniform(*self.physio_var.stroke_cbf_range)
                att = np.random.uniform(*self.physio_var.stroke_att_range)
                t1_a = np.random.uniform(self.physio_var.t1_artery_range[0]-100, self.physio_var.t1_artery_range[1]+100) # Wider T1a for pathology
            elif condition == 'tumor':
                cbf = np.random.uniform(*self.physio_var.tumor_cbf_range)
                att = np.random.uniform(*self.physio_var.tumor_att_range)
                t1_a = np.random.uniform(self.physio_var.t1_artery_range[0]-150, self.physio_var.t1_artery_range[1]+150)
            elif condition == 'elderly':
                cbf = np.random.uniform(*self.physio_var.elderly_cbf_range)
                att = np.random.uniform(*self.physio_var.elderly_att_range)
                t1_a = np.random.uniform(self.physio_var.t1_artery_range[0]+50, self.physio_var.t1_artery_range[1]+150) # Tend to be longer
            else: # Default to healthy if unknown condition
                cbf = np.random.uniform(*self.physio_var.cbf_range)
                att = np.random.uniform(*self.physio_var.att_range)
                t1_a = np.random.uniform(*self.physio_var.t1_artery_range)

            # Perturb sequence parameters for this subject
            perturbed_t_tau = base_params.T_tau * (1 + np.random.uniform(*self.physio_var.t_tau_perturb_range))
            perturbed_alpha_pcasl = base_params.alpha_PCASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range))
            perturbed_alpha_vsasl = base_params.alpha_VSASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range))
            
            # Ensure alphas are within reasonable bounds (e.g., 0 to 1 or 0 to 1.2)
            perturbed_alpha_pcasl = np.clip(perturbed_alpha_pcasl, 0.1, 1.1)
            perturbed_alpha_vsasl = np.clip(perturbed_alpha_vsasl, 0.1, 1.0)

            # Generate clean signals using these subject-specific (and perturbed) parameters
            # The _generate_xxx_signal methods in ASLSimulator were updated to accept these
            vsasl_clean = self._generate_vsasl_signal(plds, att, cbf_ml_100g_min=cbf, t1_artery=t1_a, alpha_vsasl=perturbed_alpha_vsasl)
            pcasl_clean = self._generate_pcasl_signal(plds, att, cbf_ml_100g_min=cbf, t1_artery=t1_a, t_tau=perturbed_t_tau, alpha_pcasl=perturbed_alpha_pcasl)
            
            # Add noise at different levels and types
            for snr in noise_levels:
                # Try different noise types for more robustness
                for noise_type in ['gaussian', 'rician', 'physiological']:
                    vsasl_noisy = self.add_realistic_noise(vsasl_clean, noise_type, snr, **noise_artifact_options)
                    pcasl_noisy = self.add_realistic_noise(pcasl_clean, noise_type, snr, **noise_artifact_options)
                    
                    # Store combined MULTIVERSE signal (PCASL first, then VSASL, flattened)
                    multiverse_signal_flat = np.concatenate([pcasl_noisy, vsasl_noisy])
                    
                    dataset['signals'].append(multiverse_signal_flat)
                    dataset['parameters'].append([cbf, att]) # CBF in ml/100g/min, ATT in ms
                    dataset['conditions'].append(condition)
                    dataset['noise_levels'].append(snr)
                    dataset['perturbed_params'].append({
                        't1_artery': t1_a, 't_tau': perturbed_t_tau, 
                        'alpha_pcasl': perturbed_alpha_pcasl, 'alpha_vsasl': perturbed_alpha_vsasl
                    })
        
        # Convert to numpy arrays
        dataset['signals'] = np.array(dataset['signals'])
        dataset['parameters'] = np.array(dataset['parameters'])
        # Conditions, noise_levels, perturbed_params remain lists
        
        return dataset
    
    def generate_spatial_data(self,
                            matrix_size: Tuple[int, int] = (64, 64),
                            n_slices: int = 20,
                            plds: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate spatially varying ASL data for CNN training"""
        
        if plds is None:
            plds = np.arange(500, 3001, 500)
            
        # Create spatial CBF and ATT maps with smooth variations
        x, y = np.meshgrid(np.linspace(-1, 1, matrix_size[0]),
                          np.linspace(-1, 1, matrix_size[1]))
        
        # Base CBF map with spatial variations
        cbf_map = 60 + 20 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
        
        # Add some anatomical structure
        # Gray matter regions (higher CBF)
        gm_mask = (x**2 + y**2) < 0.7
        cbf_map[gm_mask] += 20
        
        # White matter regions (lower CBF)
        wm_mask = (x**2 + y**2) > 0.7
        cbf_map[wm_mask] *= 0.5
        cbf_map = np.clip(cbf_map, 5, 150) # Clip CBF to realistic physiological range
        
        # ATT map with gradual increase from center to periphery
        att_map = 1000 + 500 * np.sqrt(x**2 + y**2)
        att_map = np.clip(att_map, 300, 4000) # Clip ATT
        
        # Apply smoothing
        cbf_map_smooth = gaussian_filter(cbf_map, sigma=2)
        att_map_smooth = gaussian_filter(att_map, sigma=2)
        
        # Generate 4D data (x, y, z, pld) for PCASL
        # For MULTIVERSE, this would be (x,y,z, pld, 2) or (x,y,z, pld*2)
        data_4d_pcasl = np.zeros((matrix_size[0], matrix_size[1], n_slices, len(plds)))
        # If generating MULTIVERSE, you'd also generate VSASL and combine them.
        # For simplicity, this example focuses on PCASL spatial data.
        
        for z in range(n_slices):
            # Add some slice-to-slice variation (e.g. global scaling for this slice)
            slice_cbf_factor = 0.9 + 0.2 * np.random.rand()
            slice_att_factor = 0.95 + 0.1 * np.random.rand()
            
            slice_cbf_map = np.clip(cbf_map_smooth * slice_cbf_factor, 5, 150)
            slice_att_map = np.clip(att_map_smooth * slice_att_factor, 300, 4000)
            
            for i in range(matrix_size[0]):
                for j in range(matrix_size[1]):
                    # Use instance parameters for T1a, T_tau, alphas unless they also vary spatially
                    # Here, self.params are the base parameters from ASLParameters.
                    signal = self._generate_pcasl_signal(plds, slice_att_map[i, j],
                                                        cbf_ml_100g_min=slice_cbf_map[i,j],
                                                        t1_artery=self.params.T1_artery,
                                                        t_tau=self.params.T_tau,
                                                        alpha_pcasl=self.params.alpha_PCASL)
                    
                    # Add spatially correlated noise (simplified: add independent noise per voxel for now)
                    # Proper spatially correlated noise would involve filtering a noise field.
                    # Using add_realistic_noise for more complex noise per voxel's timeseries
                    noisy_signal = self.add_realistic_noise(signal, noise_type='rician', snr=np.random.uniform(5,15))
                    data_4d_pcasl[i, j, z, :] = noisy_signal
        
        return data_4d_pcasl, cbf_map_smooth, att_map_smooth # Return the smoothed maps used for generation


def visualize_simulation_quality(simulator: RealisticASLSimulator):
    """Visualize simulation data quality"""
    
    plds = np.arange(500, 3001, 500)
    
    # Generate sample data
    fig, axes = plt.subplots(2, 3, figsize=(18, 12)) # Increased figure size
    
    # Different noise types
    true_att = 1600
    true_cbf = 60
    
    # Use specific parameters for generating the clean signal
    clean_signal_pcasl = simulator._generate_pcasl_signal(plds, true_att, 
                                                          cbf_ml_100g_min=true_cbf, 
                                                          t1_artery=simulator.params.T1_artery,
                                                          t_tau=simulator.params.T_tau,
                                                          alpha_pcasl=simulator.params.alpha_PCASL)
    
    noise_types_config = [
        ('gaussian', {'temporal_correlation': 0}),
        ('rician', {'temporal_correlation': 0.2}),
        ('physiological', {'temporal_correlation': 0.3, 'include_spike_artifacts': True, 'include_baseline_drift': True})
    ]

    for i, (noise_type, kwargs) in enumerate(noise_types_config):
        noisy_signal = simulator.add_realistic_noise(clean_signal_pcasl, noise_type, snr=8, **kwargs)
        
        axes[0, i].plot(plds, clean_signal_pcasl, 'b-', label='Clean PCASL', linewidth=2)
        axes[0, i].plot(plds, noisy_signal, 'r.-', label=f'{noise_type.capitalize()} noise', markersize=8, alpha=0.7)
        axes[0, i].set_xlabel('PLD (ms)')
        axes[0, i].set_ylabel('Signal')
        axes[0, i].set_title(f'{noise_type.capitalize()} Noise (SNR=8)')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    # Different physiological conditions
    conditions_params = { # CBF (ml/100g/min), ATT (ms), T1_artery (ms)
        'Healthy': (60, 1200, 1850),
        'Stroke': (20, 2800, 1950), # Prolonged ATT, potentially altered T1a
        'Elderly': (40, 2200, 1900) # Reduced CBF, prolonged ATT
    }
    
    for i, (condition_name, (cbf_val, att_val, t1a_val)) in enumerate(conditions_params.items()):
        # Generate clean signal with condition-specific params
        signal_cond_pcasl = simulator._generate_pcasl_signal(plds, att_val, 
                                                             cbf_ml_100g_min=cbf_val, 
                                                             t1_artery=t1a_val, # Use condition T1a
                                                             t_tau=simulator.params.T_tau,
                                                             alpha_pcasl=simulator.params.alpha_PCASL)
        
        noisy_cond_pcasl = simulator.add_realistic_noise(signal_cond_pcasl, 'rician', snr=5, temporal_correlation=0.2)
        
        axes[1, i].plot(plds, signal_cond_pcasl, 'b-', label='Clean PCASL', linewidth=2)
        axes[1, i].plot(plds, noisy_cond_pcasl, 'r.-', label='Noisy (Rician, SNR=5)', markersize=8, alpha=0.7)
        axes[1, i].set_xlabel('PLD (ms)')
        axes[1, i].set_ylabel('Signal')
        axes[1, i].set_title(f'{condition_name}: CBF={cbf_val}, ATT={att_val}, T1a={t1a_val}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_quality_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_training_data_parallel(n_cores: int = None):
    """Generate training data using parallel processing"""
    
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)
    
    # Base ASLParameters for the simulator instances in each worker
    # Perturbations will happen within generate_diverse_dataset
    base_asl_params = ASLParameters() 
    plds = np.arange(500, 3001, 500)
    
    # Example: Generate data for 1000 total subjects, split across cores
    total_subjects_to_generate = 1000
    n_subjects_per_core = total_subjects_to_generate // n_cores
    if total_subjects_to_generate % n_cores != 0: # Distribute remainder
         extra_subjects_list = [1] * (total_subjects_to_generate % n_cores) + [0] * (n_cores - (total_subjects_to_generate % n_cores))
    else:
         extra_subjects_list = [0] * n_cores

    worker_args_list = []
    for i in range(n_cores):
        subjects_for_this_core = n_subjects_per_core + extra_subjects_list[i]
        if subjects_for_this_core > 0:
            worker_args_list.append((i, subjects_for_this_core, base_asl_params, plds))

    if not worker_args_list:
        print("No subjects to generate. Exiting parallel generation.")
        return {'signals': np.array([]), 'parameters': np.array([]), 
                'conditions': [], 'noise_levels': [], 'perturbed_params': []}

    def worker_process(args_tuple):
        core_id, n_subjects_for_worker, b_params, p_lds = args_tuple
        np.random.seed(42 + core_id)  # Ensure reproducibility per worker
        
        sim = RealisticASLSimulator(params=b_params)
        data = sim.generate_diverse_dataset(
            p_lds,
            n_subjects=n_subjects_for_worker,
            conditions=['healthy', 'stroke', 'tumor', 'elderly'],
            noise_levels=[3.0, 5.0, 10.0, 15.0] # Diverse SNRs
            # Noise artifact options will use defaults in generate_diverse_dataset
        )
        return data
    
    print(f"Generating data using {len(worker_args_list)} cores for {total_subjects_to_generate} total effective subjects...")
    with mp.Pool(len(worker_args_list)) as pool:
        # Wrap pool.map with tqdm for progress bar
        results = list(tqdm(pool.imap(worker_process, worker_args_list), total=len(worker_args_list), desc="Parallel Generation"))

    # Combine results
    if not results: # Should not happen if worker_args_list was populated
        print("No results from workers.")
        return {'signals': np.array([]), 'parameters': np.array([]), 
                'conditions': [], 'noise_levels': [], 'perturbed_params': []}

    combined_data = {
        'signals': np.vstack([r['signals'] for r in results if r['signals'].size > 0]),
        'parameters': np.vstack([r['parameters'] for r in results if r['parameters'].size > 0]),
        'conditions': sum([r['conditions'] for r in results], []), # Concatenate lists
        'noise_levels': sum([r['noise_levels'] for r in results], []),
        'perturbed_params': sum([r['perturbed_params'] for r in results], [])
    }
    
    print(f"Generated {len(combined_data['signals'])} total samples (subject*SNR*noise_type combinations)")
    return combined_data


if __name__ == "__main__":
    # Test the enhanced simulator
    simulator = RealisticASLSimulator()
    
    # Visualize simulation quality
    visualize_simulation_quality(simulator)
    
    # Generate a small test dataset using the parallel function
    print("\nTesting parallel data generation...")
    training_data = generate_training_data_parallel(n_cores=2) # Use 2 cores for test
    
    if training_data['signals'].size > 0:
        print(f"\nGenerated training dataset shape: {training_data['signals'].shape}")
        print(f"Parameter ranges: CBF [{training_data['parameters'][:, 0].min():.1f}, {training_data['parameters'][:, 0].max():.1f}]")
        print(f"                 ATT [{training_data['parameters'][:, 1].min():.1f}, {training_data['parameters'][:, 1].max():.1f}]")
        print(f"Number of unique conditions in generated data: {len(set(training_data['conditions']))}")
        print(f"Example perturbed params for one sample: {training_data['perturbed_params'][0] if training_data['perturbed_params'] else 'N/A'}")
    else:
        print("No training data generated in the test.")

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
    tumor_cbf_range: Tuple[float, float] = (10.0, 150.0)
    
    # Age-related variations
    young_att_range: Tuple[float, float] = (500.0, 1500.0)
    elderly_att_range: Tuple[float, float] = (1500.0, 3500.0)

class RealisticASLSimulator(ASLSimulator):
    """Enhanced ASL simulator with realistic noise and physiological variations"""
    
    def __init__(self, params: ASLParameters = ASLParameters()):
        super().__init__(params)
        self.physio_var = PhysiologicalVariation()
        
    def add_realistic_noise(self, 
                          signal: np.ndarray,
                          noise_type: str = 'gaussian',
                          snr: float = 5.0,
                          temporal_correlation: float = 0.3) -> np.ndarray:
        """Add realistic noise with temporal correlations"""
        
        if noise_type == 'gaussian':
            # Basic Gaussian noise
            noise_level = np.mean(np.abs(signal)) / snr
            noise = np.random.normal(0, noise_level, signal.shape)
            
        elif noise_type == 'rician':
            # Rician noise (more realistic for MRI)
            signal_magnitude = np.abs(signal)
            noise_level = np.mean(signal_magnitude) / snr
            
            # Generate complex noise
            noise_real = np.random.normal(0, noise_level/np.sqrt(2), signal.shape)
            noise_imag = np.random.normal(0, noise_level/np.sqrt(2), signal.shape)
            
            # Add to signal and compute magnitude
            noisy_signal = np.sqrt((signal_magnitude + noise_real)**2 + noise_imag**2)
            return noisy_signal
            
        elif noise_type == 'physiological':
            # Add physiological noise components
            # Cardiac pulsation (~1 Hz)
            t = np.linspace(0, signal.shape[-1], signal.shape[-1])
            cardiac = 0.02 * np.sin(2 * np.pi * 1.0 * t)
            
            # Respiratory (~0.3 Hz)
            respiratory = 0.01 * np.sin(2 * np.pi * 0.3 * t)
            
            # Combine with Gaussian noise
            noise_level = np.mean(np.abs(signal)) / snr
            gaussian_noise = np.random.normal(0, noise_level, signal.shape)
            
            noise = gaussian_noise + cardiac + respiratory
            
        # Add temporal correlation if specified
        if temporal_correlation > 0:
            # Apply low-pass filter for temporal smoothing
            from scipy.ndimage import gaussian_filter1d
            noise = gaussian_filter1d(noise, sigma=temporal_correlation, axis=-1)
            
        return signal + noise
    
    def generate_diverse_dataset(self,
                               plds: np.ndarray,
                               n_subjects: int = 100,
                               conditions: List[str] = ['healthy', 'stroke', 'tumor', 'elderly'],
                               noise_levels: List[float] = [3.0, 5.0, 10.0]) -> Dict:
        """Generate diverse dataset with various physiological conditions"""
        
        dataset = {
            'signals': [],
            'parameters': [],
            'conditions': [],
            'noise_levels': []
        }
        
        for condition in conditions:
            print(f"Generating {condition} data...")
            
            for _ in tqdm(range(n_subjects)):
                # Sample physiological parameters based on condition
                if condition == 'healthy':
                    cbf = np.random.uniform(*self.physio_var.cbf_range)
                    att = np.random.uniform(*self.physio_var.att_range)
                elif condition == 'stroke':
                    cbf = np.random.uniform(*self.physio_var.stroke_cbf_range)
                    att = np.random.uniform(1500, 3500)  # Typically prolonged
                elif condition == 'tumor':
                    cbf = np.random.uniform(*self.physio_var.tumor_cbf_range)
                    att = np.random.uniform(*self.physio_var.att_range)
                elif condition == 'elderly':
                    cbf = np.random.uniform(30, 70)  # Reduced CBF
                    att = np.random.uniform(*self.physio_var.elderly_att_range)
                
                # Generate clean signals
                self.cbf = cbf / 6000  # Convert to ml/g/s
                vsasl_clean = self._generate_vsasl_signal(plds, att)
                pcasl_clean = self._generate_pcasl_signal(plds, att)
                
                # Add noise at different levels
                for snr in noise_levels:
                    # Try different noise types
                    for noise_type in ['gaussian', 'rician', 'physiological']:
                        vsasl_noisy = self.add_realistic_noise(vsasl_clean, noise_type, snr)
                        pcasl_noisy = self.add_realistic_noise(pcasl_clean, noise_type, snr)
                        
                        # Store combined MULTIVERSE signal
                        multiverse_signal = np.column_stack([pcasl_noisy, vsasl_noisy])
                        
                        dataset['signals'].append(multiverse_signal.flatten())
                        dataset['parameters'].append([cbf, att])
                        dataset['conditions'].append(condition)
                        dataset['noise_levels'].append(snr)
        
        # Convert to numpy arrays
        dataset['signals'] = np.array(dataset['signals'])
        dataset['parameters'] = np.array(dataset['parameters'])
        
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
        
        # ATT map with gradual increase from center to periphery
        att_map = 1000 + 500 * np.sqrt(x**2 + y**2)
        
        # Apply smoothing
        cbf_map = gaussian_filter(cbf_map, sigma=2)
        att_map = gaussian_filter(att_map, sigma=2)
        
        # Generate 4D data (x, y, z, pld)
        data_4d = np.zeros((matrix_size[0], matrix_size[1], n_slices, len(plds)))
        
        for z in range(n_slices):
            # Add some slice-to-slice variation
            slice_cbf = cbf_map * (0.9 + 0.2 * np.random.rand())
            slice_att = att_map * (0.95 + 0.1 * np.random.rand())
            
            for i in range(matrix_size[0]):
                for j in range(matrix_size[1]):
                    self.cbf = slice_cbf[i, j] / 6000
                    signal = self._generate_pcasl_signal(plds, slice_att[i, j])
                    
                    # Add spatially correlated noise
                    noise = np.random.normal(0, 0.001, len(plds))
                    data_4d[i, j, z, :] = signal + noise
        
        return data_4d, cbf_map, att_map


def visualize_simulation_quality(simulator: RealisticASLSimulator):
    """Visualize simulation data quality"""
    
    plds = np.arange(500, 3001, 500)
    
    # Generate sample data
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Different noise types
    att = 1600
    simulator.cbf = 60 / 6000
    clean_signal = simulator._generate_pcasl_signal(plds, att)
    
    noise_types = ['gaussian', 'rician', 'physiological']
    for i, noise_type in enumerate(noise_types):
        noisy_signal = simulator.add_realistic_noise(clean_signal, noise_type, snr=5)
        
        axes[0, i].plot(plds, clean_signal, 'b-', label='Clean', linewidth=2)
        axes[0, i].plot(plds, noisy_signal, 'r.', label=f'{noise_type.capitalize()} noise', markersize=8)
        axes[0, i].set_xlabel('PLD (ms)')
        axes[0, i].set_ylabel('Signal')
        axes[0, i].set_title(f'{noise_type.capitalize()} Noise')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    # Different physiological conditions
    conditions = {
        'Healthy': (60, 1200),
        'Stroke': (20, 2500),
        'Elderly': (40, 2800)
    }
    
    for i, (condition, (cbf, att)) in enumerate(conditions.items()):
        simulator.cbf = cbf / 6000
        signal = simulator._generate_pcasl_signal(plds, att)
        noisy = simulator.add_realistic_noise(signal, 'rician', snr=5)
        
        axes[1, i].plot(plds, signal, 'b-', label='Clean', linewidth=2)
        axes[1, i].plot(plds, noisy, 'r.', label='Noisy', markersize=8)
        axes[1, i].set_xlabel('PLD (ms)')
        axes[1, i].set_ylabel('Signal')
        axes[1, i].set_title(f'{condition}: CBF={cbf}, ATT={att}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_quality.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_training_data_parallel(n_cores: int = None):
    """Generate training data using parallel processing"""
    
    if n_cores is None:
        n_cores = mp.cpu_count() - 1
    
    simulator = RealisticASLSimulator()
    plds = np.arange(500, 3001, 500)
    
    # Split work across cores
    n_subjects_per_core = 25
    
    def worker(args):
        core_id, n_subjects = args
        np.random.seed(42 + core_id)  # Ensure reproducibility
        
        sim = RealisticASLSimulator()
        data = sim.generate_diverse_dataset(
            plds,
            n_subjects=n_subjects,
            conditions=['healthy', 'stroke', 'tumor', 'elderly'],
            noise_levels=[3.0, 5.0, 10.0]
        )
        return data
    
    print(f"Generating data using {n_cores} cores...")
    with mp.Pool(n_cores) as pool:
        results = pool.map(worker, [(i, n_subjects_per_core) for i in range(n_cores)])
    
    # Combine results
    combined_data = {
        'signals': np.vstack([r['signals'] for r in results]),
        'parameters': np.vstack([r['parameters'] for r in results]),
        'conditions': sum([r['conditions'] for r in results], []),
        'noise_levels': sum([r['noise_levels'] for r in results], [])
    }
    
    print(f"Generated {len(combined_data['signals'])} samples")
    return combined_data


if __name__ == "__main__":
    # Test the enhanced simulator
    simulator = RealisticASLSimulator()
    
    # Visualize simulation quality
    visualize_simulation_quality(simulator)
    
    # Generate a small test dataset
    plds = np.arange(500, 3001, 500)
    test_data = simulator.generate_diverse_dataset(
        plds,
        n_subjects=10,
        conditions=['healthy', 'stroke'],
        noise_levels=[5.0]
    )
    
    print(f"Test dataset shape: {test_data['signals'].shape}")
    print(f"Parameter ranges: CBF [{test_data['parameters'][:, 0].min():.1f}, {test_data['parameters'][:, 0].max():.1f}]")
    print(f"                 ATT [{test_data['parameters'][:, 1].min():.1f}, {test_data['parameters'][:, 1].max():.1f}]")
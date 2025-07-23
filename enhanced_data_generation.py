# enhanced_data_generation.py
# --- WARNING ---
# This script is DEPRECATED for production training. It generates a full dataset in memory,
# which is not scalable. It is kept for reference or for generating small, fixed test sets.
# For scalable training, please use the ASLIterableDataset class in asl_trainer.py,
# which generates data on-the-fly.
# --- END WARNING ---

import numpy as np
from enhanced_simulation import RealisticASLSimulator
from pathlib import Path
import pickle

def generate_comprehensive_dataset():
    """Generate comprehensive training dataset matching proposal requirements"""
    
    simulator = RealisticASLSimulator()
    plds = np.arange(500, 3001, 500)  # 6 PLDs as in MULTIVERSE
    
    # Create diverse physiological conditions as mentioned in proposal
    datasets = {}
    
    # 1. Healthy subjects with varying SNR levels
    print("Generating healthy subject data...")
    for snr in [3, 5, 10, 15, 20]:  # Range of SNR levels
        datasets[f'healthy_snr_{snr}'] = simulator.generate_diverse_dataset(
            plds=plds,
            n_subjects=500,  # Large dataset for robust training
            conditions=['healthy'],
            noise_levels=[snr]
        )
    
    # 2. Pathological conditions
    print("Generating pathological condition data...")
    for condition in ['stroke', 'tumor', 'elderly']:
        datasets[f'{condition}_mixed_snr'] = simulator.generate_diverse_dataset(
            plds=plds,
            n_subjects=300,
            conditions=[condition],
            noise_levels=[3, 5, 10]  # Mixed noise levels
        )
    
    # 3. Transit time focused dataset (key for MULTIVERSE validation)
    print("Generating ATT-focused dataset...")
    att_ranges = [
        (500, 1500, 'short_att'),
        (1500, 2500, 'medium_att'), 
        (2500, 4000, 'long_att')
    ]
    
    for att_min, att_max, name in att_ranges:
        # Generate specific ATT values for detailed analysis
        att_values = np.linspace(att_min, att_max, 50)
        signals = simulator.generate_synthetic_data(plds, att_values, n_noise=200)
        
        datasets[name] = {
            'signals': signals['MULTIVERSE'].reshape(-1, len(plds)*2),
            'parameters': np.column_stack([
                np.full(len(att_values)*200, 60),  # CBF
                np.repeat(att_values, 200)  # ATT
            ])
        }
    
    # Save datasets
    data_dir = Path('data/comprehensive_datasets')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for name, data in datasets.items():
        with open(data_dir / f'{name}.pkl', 'wb') as f:
            pickle.dump(data, f)
    
    print(f"Generated {len(datasets)} datasets")
    return datasets

if __name__ == "__main__":
    datasets = generate_comprehensive_dataset()
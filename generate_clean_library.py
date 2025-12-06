# FILE: generate_clean_library.py
"""
Generates a library of CLEAN (noise-free) ASL signals for training.
Noise is added dynamically during training by the NoiseInjector.
"""
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import os
import argparse
import time

from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator, PhysiologicalVariation

def generate_and_save_chunk(args):
    """Worker function to generate one chunk of CLEAN signals."""
    chunk_id, num_samples_per_chunk, plds, output_dir, config_dict = args
    
    np.random.seed(int(time.time()) + chunk_id)
    
    asl_params = ASLParameters(**{k:v for k,v in config_dict.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params)
    physio_var = PhysiologicalVariation()
    num_plds = len(plds)
    
    signals_clean_chunk = []
    params_chunk = []
        
    for _ in range(num_samples_per_chunk):
        true_cbf = np.random.uniform(*physio_var.cbf_range)
        true_att = np.random.uniform(*physio_var.att_range)
        true_t1_artery = np.random.uniform(*physio_var.t1_artery_range)
        true_abv = np.random.uniform(*physio_var.arterial_blood_volume_range) if np.random.rand() > 0.6 else 0.0
        true_slice_idx = np.random.randint(0, 30) # Random Z-position
        
        # Enhancement B: Slice timing effect
        slice_delay_factor = np.exp(-(true_slice_idx * 45.0)/1000.0)

        perturbed_t_tau = simulator.params.T_tau * (1 + np.random.uniform(*physio_var.t_tau_perturb_range))
        perturbed_alpha_pcasl = np.clip(simulator.params.alpha_PCASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.1)
        perturbed_alpha_vsasl = np.clip(simulator.params.alpha_VSASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.0)

        eff_alpha_p = perturbed_alpha_pcasl * slice_delay_factor
        eff_alpha_v = perturbed_alpha_vsasl * slice_delay_factor

        vsasl_clean = simulator._generate_vsasl_signal(plds, true_att, true_cbf, true_t1_artery, eff_alpha_v)
        pcasl_clean = simulator._generate_pcasl_signal(plds, true_att, true_cbf, true_t1_artery, perturbed_t_tau, eff_alpha_p)
        art_sig = simulator._generate_arterial_signal(plds, true_att, true_abv, true_t1_artery, eff_alpha_p)
        
        pcasl_clean += art_sig # Enhancement A: Macrovascular
        clean_signal_vector = np.concatenate([pcasl_clean, vsasl_clean])

        # SAVE ONLY CLEAN SIGNALS - noise is added dynamically during training
        signals_clean_chunk.append(clean_signal_vector.astype(np.float32))
        # Save Slice Index as param 3 (0-indexed)
        params_chunk.append(np.array([true_cbf, true_att, true_t1_artery, float(true_slice_idx)]).astype(np.float32))
        
    np.savez_compressed(
        output_dir / f'dataset_chunk_{chunk_id:04d}.npz',
        signals_clean=np.array(signals_clean_chunk),
        params=np.array(params_chunk)
    )
    return len(signals_clean_chunk)

if __name__ == '__main__':
    # Import FeatureRegistry for default values
    from feature_registry import FeatureRegistry
    
    parser = argparse.ArgumentParser(description="Generate a large offline dataset for ASL training.")
    parser.add_argument("output_dir", type=str, help="Directory to save the dataset chunks.")
    parser.add_argument("--total_samples", type=int, default=10_000_000, help="Total number of samples to generate.")
    parser.add_argument("--chunk_size", type=int, default=25_000, help="Number of samples per output file.")
    parser.add_argument("--pld-values", type=int, nargs='+', default=None,
                        help="PLD values in ms. Default: 500 1000 1500 2000 2500 3000")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_chunks = args.total_samples // args.chunk_size
    
    # Use CLI PLDs or FeatureRegistry default (no more hardcoding)
    if args.pld_values is not None:
        plds_np = np.array(args.pld_values)
    else:
        plds_np = np.array(FeatureRegistry.DEFAULT_PLDS)
    
    print(f"Using PLDs: {plds_np}")
    
    # Use FeatureRegistry default physics
    sim_config = FeatureRegistry.DEFAULT_PHYSICS.copy()
    
    worker_args = [(i, args.chunk_size, plds_np, output_path, sim_config) for i in range(num_chunks)]
    
    num_cpus = os.cpu_count()
    print(f"Generating {args.total_samples:,} samples in {num_chunks} chunks using {num_cpus} workers...")
    
    with mp.Pool(processes=num_cpus) as pool:
        list(tqdm(pool.imap_unordered(generate_and_save_chunk, worker_args), total=num_chunks))
        
    print(f"\nDataset generation complete. Files saved in: {output_path}")
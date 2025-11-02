# FILE: generate_offline_dataset.py
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import os
import argparse
import time

from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator, PhysiologicalVariation
from utils import engineer_signal_features

def generate_and_save_chunk(args):
    """Worker function to generate one chunk of the dataset."""
    chunk_id, num_samples_per_chunk, plds, output_dir, config_dict = args
    
    np.random.seed(int(time.time()) + chunk_id)
    
    asl_params = ASLParameters(**{k:v for k,v in config_dict.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params)
    physio_var = PhysiologicalVariation()
    num_plds = len(plds)
    
    signals_noisy_chunk = []
    signals_clean_chunk = []
    params_chunk = []
        
    for _ in range(num_samples_per_chunk):
        true_cbf = np.random.uniform(*physio_var.cbf_range)
        true_att = np.random.uniform(*physio_var.att_range)
        true_t1_artery = np.random.uniform(*physio_var.t1_artery_range)
        current_snr = np.random.choice([3.0, 5.0, 10.0, 15.0, 20.0])

        perturbed_t_tau = simulator.params.T_tau * (1 + np.random.uniform(*physio_var.t_tau_perturb_range))
        perturbed_alpha_pcasl = np.clip(simulator.params.alpha_PCASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.1)
        perturbed_alpha_vsasl = np.clip(simulator.params.alpha_VSASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.0)

        vsasl_clean = simulator._generate_vsasl_signal(plds, true_att, true_cbf, true_t1_artery, perturbed_alpha_vsasl)
        pcasl_clean = simulator._generate_pcasl_signal(plds, true_att, true_cbf, true_t1_artery, perturbed_t_tau, perturbed_alpha_pcasl)
        clean_signal_vector = np.concatenate([pcasl_clean, vsasl_clean])

        pcasl_noisy = simulator.add_realistic_noise(pcasl_clean, snr=current_snr)
        vsasl_noisy = simulator.add_realistic_noise(vsasl_clean, snr=current_snr)
        noisy_signal_vector = np.concatenate([pcasl_noisy, vsasl_noisy])
        
        eng_features = engineer_signal_features(noisy_signal_vector.reshape(1, -1), num_plds)
        final_noisy_input = np.concatenate([noisy_signal_vector, eng_features.flatten()])
        
        signals_noisy_chunk.append(final_noisy_input.astype(np.float32))
        signals_clean_chunk.append(clean_signal_vector.astype(np.float32))
        params_chunk.append(np.array([true_cbf, true_att]).astype(np.float32))
        
    np.savez_compressed(
        output_dir / f'dataset_chunk_{chunk_id:04d}.npz',
        signals_noisy=np.array(signals_noisy_chunk),
        signals_clean=np.array(signals_clean_chunk),
        params=np.array(params_chunk)
    )
    return len(signals_noisy_chunk)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a large offline dataset for ASL training.")
    parser.add_argument("output_dir", type=str, help="Directory to save the dataset chunks.")
    parser.add_argument("--total_samples", type=int, default=10_000_000, help="Total number of samples to generate.")
    parser.add_argument("--chunk_size", type=int, default=25_000, help="Number of samples per output file.")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_chunks = args.total_samples // args.chunk_size
    plds_np = np.arange(500, 3001, 500)
    
    sim_config = {
        'T1_artery': 1850.0, 'T_tau': 1800.0, 'alpha_PCASL': 0.85, 'alpha_VSASL': 0.56,
        'alpha_BS1': 1.0, 'T2_factor': 1.0
    }
    
    worker_args = [(i, args.chunk_size, plds_np, output_path, sim_config) for i in range(num_chunks)]
    
    num_cpus = os.cpu_count()
    print(f"Generating {args.total_samples:,} samples in {num_chunks} chunks using {num_cpus} workers...")
    
    with mp.Pool(processes=num_cpus) as pool:
        list(tqdm(pool.imap_unordered(generate_and_save_chunk, worker_args), total=num_chunks))
        
    print(f"\nDataset generation complete. Files saved in: {output_path}")
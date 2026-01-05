# FILE: generate_clean_library.py
"""
Generates a library of CLEAN (noise-free) ASL signals for training.
Supports both 1D voxel-wise and 2D spatial data generation.
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
from enhanced_simulation import RealisticASLSimulator, PhysiologicalVariation, SpatialPhantomGenerator


def generate_and_save_chunk(args):
    """Worker function to generate one chunk of CLEAN signals (1D voxel-wise)."""
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
        true_slice_idx = np.random.randint(0, 30)
        
        slice_delay_factor = np.exp(-(true_slice_idx * 45.0)/1000.0)

        perturbed_t_tau = simulator.params.T_tau * (1 + np.random.uniform(*physio_var.t_tau_perturb_range))
        perturbed_alpha_pcasl = np.clip(simulator.params.alpha_PCASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.1)
        perturbed_alpha_vsasl = np.clip(simulator.params.alpha_VSASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.0)

        eff_alpha_p = perturbed_alpha_pcasl * slice_delay_factor
        eff_alpha_v = perturbed_alpha_vsasl * slice_delay_factor

        vsasl_clean = simulator._generate_vsasl_signal(plds, true_att, true_cbf, true_t1_artery, eff_alpha_v)
        pcasl_clean = simulator._generate_pcasl_signal(plds, true_att, true_cbf, true_t1_artery, perturbed_t_tau, eff_alpha_p)
        art_sig = simulator._generate_arterial_signal(plds, true_att, true_abv, true_t1_artery, eff_alpha_p)
        
        pcasl_clean += art_sig
        clean_signal_vector = np.concatenate([pcasl_clean, vsasl_clean])

        signals_clean_chunk.append(clean_signal_vector.astype(np.float32))
        params_chunk.append(np.array([true_cbf, true_att, true_t1_artery, float(true_slice_idx)]).astype(np.float32))
        
    np.savez_compressed(
        output_dir / f'dataset_chunk_{chunk_id:04d}.npz',
        signals_clean=np.array(signals_clean_chunk),
        params=np.array(params_chunk)
    )
    return len(signals_clean_chunk)


def generate_spatial_chunk(args):
    """
    Worker function to generate one chunk of SPATIAL (2D) ASL data.
    
    Output format: 
        signals: (batch_per_chunk, 2*n_plds, H, W) - PCASL + VSASL channels
        targets: (batch_per_chunk, 2, H, W) - CBF and ATT maps
    """
    chunk_id, samples_per_chunk, plds, output_dir, config_dict, size = args
    
    np.random.seed(int(time.time()) + chunk_id * 1000)
    
    asl_params = ASLParameters(**{k:v for k,v in config_dict.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params)
    phantom_gen = SpatialPhantomGenerator(size=size, pve_sigma=1.0)
    
    n_plds = len(plds)
    signals_batch = []
    targets_batch = []
    
    # Pre-compute physics constants for vectorized signal generation
    plds_bc = plds[:, np.newaxis, np.newaxis].astype(np.float32)
    t1_b = asl_params.T1_artery
    tau = asl_params.T_tau
    lambda_b = 0.90
    t2_f = asl_params.T2_factor
    alpha_p = asl_params.alpha_PCASL * (asl_params.alpha_BS1**4)
    alpha_v = asl_params.alpha_VSASL * (asl_params.alpha_BS1**3)
    
    for _ in range(samples_per_chunk):
        # Generate phantom
        cbf_map, att_map, _ = phantom_gen.generate_phantom(include_pathology=True)
        
        # Vectorized signal generation using NumPy broadcasting
        att_bc = att_map[np.newaxis, :, :].astype(np.float32)
        cbf_bc = cbf_map[np.newaxis, :, :].astype(np.float32)
        
        # --- PCASL Signal ---
        mask_arrived = (plds_bc >= att_bc)
        mask_transit = (plds_bc < att_bc) & (plds_bc >= (att_bc - tau))
        
        sig_p_arrived = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                        np.exp(-plds_bc / t1_b) *
                        (1 - np.exp(-tau / t1_b)) * t2_f) / lambda_b
        
        sig_p_transit = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                        (np.exp(-att_bc / t1_b) - np.exp(-(tau + plds_bc) / t1_b)) *
                        t2_f) / lambda_b
        
        pcasl_sig = np.zeros_like(plds_bc * cbf_bc)
        pcasl_sig[mask_arrived] = sig_p_arrived[mask_arrived]
        pcasl_sig[mask_transit] = sig_p_transit[mask_transit]
        
        # --- VSASL Signal ---
        mask_vs_arrived = (plds_bc > att_bc)
        
        sig_v_early = (2 * alpha_v * cbf_bc * (plds_bc / 1000.0) *
                      np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
        
        sig_v_late = (2 * alpha_v * cbf_bc * (att_bc / 1000.0) *
                     np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
        
        vsasl_sig = np.where(mask_vs_arrived, sig_v_late, sig_v_early)
        
        # Stack: (2*n_plds, H, W)
        clean_signal = np.concatenate([pcasl_sig, vsasl_sig], axis=0).astype(np.float32)
        
        # Target: (2, H, W) - [CBF, ATT]
        target = np.stack([cbf_map, att_map], axis=0).astype(np.float32)
        
        signals_batch.append(clean_signal)
        targets_batch.append(target)
    
    # Save chunk: (batch, 2*n_plds, H, W), (batch, 2, H, W)
    np.savez_compressed(
        output_dir / f'spatial_chunk_{chunk_id:04d}.npz',
        signals=np.array(signals_batch),
        targets=np.array(targets_batch)
    )
    
    return len(signals_batch)


if __name__ == '__main__':
    from feature_registry import FeatureRegistry
    
    parser = argparse.ArgumentParser(description="Generate a large offline dataset for ASL training.")
    parser.add_argument("output_dir", type=str, help="Directory to save the dataset chunks.")
    parser.add_argument("--total_samples", type=int, default=10_000_000, help="Total number of samples to generate.")
    parser.add_argument("--chunk_size", type=int, default=25_000, help="Number of samples per output file.")
    parser.add_argument("--pld-values", type=int, nargs='+', default=None,
                        help="PLD values in ms. Default: 500 1000 1500 2000 2500 3000")
    
    # Spatial mode arguments
    parser.add_argument("--spatial", action='store_true', help="Generate 2D spatial data instead of 1D voxels.")
    parser.add_argument("--image-size", type=int, default=64, help="Image size for spatial mode (default: 64).")
    parser.add_argument("--spatial-chunk-size", type=int, default=500, help="Samples per chunk in spatial mode.")
    
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use CLI PLDs or FeatureRegistry default
    if args.pld_values is not None:
        plds_np = np.array(args.pld_values)
    else:
        plds_np = np.array(FeatureRegistry.DEFAULT_PLDS)
    
    print(f"Using PLDs: {plds_np}")
    
    # Use FeatureRegistry default physics
    sim_config = FeatureRegistry.DEFAULT_PHYSICS.copy()
    
    if args.spatial:
        # Spatial mode: Generate 2D slices
        chunk_size = args.spatial_chunk_size
        num_chunks = args.total_samples // chunk_size
        
        print(f"[SPATIAL MODE] Generating {args.total_samples:,} samples ({args.image_size}x{args.image_size}) in {num_chunks} chunks...")
        
        worker_args = [
            (i, chunk_size, plds_np, output_path, sim_config, args.image_size) 
            for i in range(num_chunks)
        ]
        
        num_cpus = min(os.cpu_count(), 16)  # Limit for memory
        with mp.Pool(processes=num_cpus) as pool:
            list(tqdm(pool.imap_unordered(generate_spatial_chunk, worker_args), total=num_chunks))
    else:
        # Standard mode: 1D voxels
        num_chunks = args.total_samples // args.chunk_size
        
        print(f"[VOXEL MODE] Generating {args.total_samples:,} samples in {num_chunks} chunks...")
        
        worker_args = [(i, args.chunk_size, plds_np, output_path, sim_config) for i in range(num_chunks)]
        
        num_cpus = os.cpu_count()
        with mp.Pool(processes=num_cpus) as pool:
            list(tqdm(pool.imap_unordered(generate_and_save_chunk, worker_args), total=num_chunks))
        
    print(f"\nDataset generation complete. Files saved in: {output_path}")
#!/usr/bin/env python3
"""
Generate Realistic Test Phantoms for V7 Evaluation
====================================================
Generates 100 realistic brain phantoms with known ground truth CBF/ATT maps,
converts them to clean ASL signals, adds Rician noise at multiple SNRs,
and saves everything as .npz files.

Usage:
    python amplitude_ablation_v7/generate_test_phantoms.py [--n-phantoms 100] [--output-dir ...]
"""

import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.enhanced_simulation import SpatialPhantomGenerator
from simulation.asl_simulation import _generate_pcasl_signal_jit, _generate_vsasl_signal_jit
from simulation.noise_engine import SpatialNoiseEngine

# Physics constants
PLDS = np.array([500, 1000, 1500, 2000, 2500], dtype=np.float64)
T1_ARTERY = 1650.0
T_TAU = 1800.0
ALPHA_PCASL = 0.85
ALPHA_VSASL = 0.56
ALPHA_BS1 = 1.0
T2_FACTOR = 1.0
T_SAT_VS = 2000.0

# Effective labeling efficiencies (BS-corrected)
ALPHA1 = ALPHA_PCASL * (ALPHA_BS1 ** 4)
ALPHA2 = ALPHA_VSASL * (ALPHA_BS1 ** 3)

SNR_LEVELS = [2, 3, 5, 10, 15, 25]
PHANTOM_SIZE = 64
N_PLDS = len(PLDS)
N_CHANNELS = N_PLDS * 2  # 5 PCASL + 5 VSASL


def generate_clean_signals(cbf_map, att_map):
    """Generate clean 10-channel ASL signals from CBF/ATT maps.

    Iterates over every voxel and calls JIT-compiled signal functions.

    Args:
        cbf_map: (64, 64) CBF in ml/100g/min
        att_map: (64, 64) ATT in ms

    Returns:
        clean_signals: (10, 64, 64) float32 array
    """
    h, w = cbf_map.shape
    clean_signals = np.zeros((N_CHANNELS, h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            cbf_val = cbf_map[i, j]
            att_val = att_map[i, j]

            if cbf_val <= 0 and att_val <= 0:
                continue

            cbf_cgs = cbf_val / 6000.0
            pcasl_sig = _generate_pcasl_signal_jit(
                PLDS, att_val, cbf_cgs, T1_ARTERY, T_TAU, ALPHA1, T2_FACTOR
            )
            vsasl_sig = _generate_vsasl_signal_jit(
                PLDS, att_val, cbf_cgs, T1_ARTERY, ALPHA2, T2_FACTOR, T_SAT_VS
            )
            clean_signals[:N_PLDS, i, j] = pcasl_sig
            clean_signals[N_PLDS:, i, j] = vsasl_sig

    return clean_signals


def main():
    parser = argparse.ArgumentParser(description="Generate realistic test phantoms for V7 evaluation")
    parser.add_argument("--n-phantoms", type=int, default=100, help="Number of phantoms to generate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: amplitude_ablation_v7/test_phantoms)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "amplitude_ablation_v7" / "test_phantoms")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_phantoms = args.n_phantoms
    n_with_pathology = int(n_phantoms * 0.7)  # 70% with pathology

    phantom_gen = SpatialPhantomGenerator(size=PHANTOM_SIZE, pve_sigma=1.0)
    noise_engine = SpatialNoiseEngine(config={})

    print(f"Generating {n_phantoms} phantoms ({n_with_pathology} with pathology)")
    print(f"SNR levels: {SNR_LEVELS}")
    print(f"Output: {output_dir}")
    print()

    t_start = time.time()

    for idx in range(n_phantoms):
        np.random.seed(7000 + idx)

        include_pathology = idx < n_with_pathology
        cbf_map, att_map, metadata = phantom_gen.generate_phantom(include_pathology=include_pathology)
        tissue_map = metadata["tissue_map"]

        # Generate clean signals
        clean_signals = generate_clean_signals(cbf_map, att_map)

        # Build save dict
        save_dict = {
            "clean_signals": clean_signals,
            "cbf_map": cbf_map,
            "att_map": att_map,
            "tissue_map": tissue_map,
        }

        # Add noisy versions at each SNR
        for snr in SNR_LEVELS:
            noisy = noise_engine.add_rician_noise(clean_signals, snr=float(snr))
            save_dict[f"noisy_snr_{snr}"] = noisy

        # Save
        out_path = output_dir / f"phantom_{idx:03d}.npz"
        np.savez_compressed(out_path, **save_dict)

        elapsed = time.time() - t_start
        rate = (idx + 1) / elapsed
        eta = (n_phantoms - idx - 1) / rate if rate > 0 else 0
        print(f"  [{idx+1:3d}/{n_phantoms}] phantom_{idx:03d}.npz  "
              f"({elapsed:.1f}s elapsed, ETA {eta:.0f}s)")

    # Save metadata JSON
    meta = {
        "n_phantoms": n_phantoms,
        "n_with_pathology": n_with_pathology,
        "phantom_size": PHANTOM_SIZE,
        "snr_levels": SNR_LEVELS,
        "n_plds": N_PLDS,
        "plds": PLDS.tolist(),
        "physics": {
            "T1_artery": T1_ARTERY,
            "T_tau": T_TAU,
            "alpha_PCASL": ALPHA_PCASL,
            "alpha_VSASL": ALPHA_VSASL,
            "alpha_BS1": ALPHA_BS1,
            "T2_factor": T2_FACTOR,
            "T_sat_vs": T_SAT_VS,
        },
        "seed_base": 7000,
    }
    meta_path = output_dir / "phantom_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nDone. {n_phantoms} phantoms saved to {output_dir}")
    print(f"Total time: {total_time:.1f}s ({total_time/n_phantoms:.2f}s per phantom)")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()

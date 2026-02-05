#!/usr/bin/env python3
"""
Debug LS fitting by examining actual signal values vs model expectations.

Key question: Are the in-vivo signal magnitudes consistent with what the kinetic model predicts?
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import re

from multiverse_functions import fun_PCVSASL_misMatchPLD_vect_pep


def calculate_expected_signal(cbf_ml_100g_min: float, att_ms: float, plds_ms: np.ndarray) -> dict:
    """Calculate expected ΔM/M0 signal for given CBF and ATT."""

    params = {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T2_factor': 1.0,
        'alpha_BS1': 1.0,
    }

    cbf_internal = cbf_ml_100g_min / 6000.0
    pldti = np.column_stack([plds_ms, plds_ms])

    signal = fun_PCVSASL_misMatchPLD_vect_pep(
        [cbf_internal, att_ms], pldti,
        params['T1_artery'], params['T_tau'], params['T2_factor'],
        params['alpha_BS1'], params['alpha_PCASL'], params['alpha_VSASL']
    )

    return {
        'pcasl': signal[:, 0],
        'vsasl': signal[:, 1],
        'plds': plds_ms,
    }


def analyze_invivo_signals(subject_dir: Path):
    """Analyze actual signal values from in-vivo data."""

    def find_files(pattern):
        files = list(subject_dir.glob(pattern))
        def get_pld(p):
            m = re.search(r'_(\d+)', p.name)
            return int(m.group(1)) if m else -1
        return sorted(files, key=get_pld)

    pcasl_files = find_files('r_normdiff_alldyn_PCASL_*.nii*')
    vsasl_files = find_files('r_normdiff_alldyn_VSASL_*.nii*')

    if not pcasl_files or not vsasl_files:
        print(f"  No ASL files found in {subject_dir}")
        return None

    # Get PLDs
    plds = [int(re.search(r'_(\d+)', f.name).group(1)) for f in pcasl_files]

    # Load M0 for brain mask
    m0_files = list(subject_dir.glob('r_M0.nii*'))
    if m0_files:
        m0_data = np.nan_to_num(nib.load(m0_files[0]).get_fdata())
        threshold = np.percentile(m0_data[m0_data > 0], 50) * 0.3
        brain_mask = m0_data > threshold
    else:
        print("  No M0 file found")
        return None

    print(f"  PLDs: {plds}")
    print(f"  Brain voxels: {brain_mask.sum():,}")

    # Analyze signal at each PLD
    print(f"\n  In-vivo PCASL signal statistics (ΔM/M0):")
    pcasl_means = []
    for f in pcasl_files:
        pld = int(re.search(r'_(\d+)', f.name).group(1))
        data = np.nan_to_num(nib.load(f).get_fdata())
        if data.ndim == 4:
            data = np.mean(data, axis=-1)

        brain_values = data[brain_mask]
        pcasl_means.append(np.mean(brain_values))

        print(f"    PLD={pld}ms: mean={np.mean(brain_values):.6f}, "
              f"std={np.std(brain_values):.6f}, "
              f"median={np.median(brain_values):.6f}, "
              f"range=[{np.min(brain_values):.6f}, {np.max(brain_values):.6f}]")

    print(f"\n  In-vivo VSASL signal statistics (ΔM/M0):")
    vsasl_means = []
    for f in vsasl_files:
        pld = int(re.search(r'_(\d+)', f.name).group(1))
        data = np.nan_to_num(nib.load(f).get_fdata())
        if data.ndim == 4:
            data = np.mean(data, axis=-1)

        brain_values = data[brain_mask]
        vsasl_means.append(np.mean(brain_values))

        print(f"    PLD={pld}ms: mean={np.mean(brain_values):.6f}, "
              f"std={np.std(brain_values):.6f}, "
              f"median={np.median(brain_values):.6f}, "
              f"range=[{np.min(brain_values):.6f}, {np.max(brain_values):.6f}]")

    return {
        'plds': plds,
        'pcasl_means': pcasl_means,
        'vsasl_means': vsasl_means,
    }


def main():
    print("=" * 70)
    print("DEBUG: LS FITTING SIGNAL VALUES")
    print("=" * 70)

    # Expected signal values for typical CBF/ATT
    print("\n" + "=" * 70)
    print("EXPECTED SIGNAL VALUES FROM KINETIC MODEL")
    print("=" * 70)

    plds = np.array([500, 1000, 1500, 2000, 2500], dtype=np.float64)

    # Test case 1: Typical gray matter
    print("\nCase 1: Gray matter (CBF=60 ml/100g/min, ATT=1200ms)")
    expected = calculate_expected_signal(60, 1200, plds)
    for i, pld in enumerate(plds):
        print(f"  PLD={int(pld)}ms: PCASL={expected['pcasl'][i]:.6f}, VSASL={expected['vsasl'][i]:.6f}")

    # Test case 2: Typical white matter
    print("\nCase 2: White matter (CBF=25 ml/100g/min, ATT=1800ms)")
    expected = calculate_expected_signal(25, 1800, plds)
    for i, pld in enumerate(plds):
        print(f"  PLD={int(pld)}ms: PCASL={expected['pcasl'][i]:.6f}, VSASL={expected['vsasl'][i]:.6f}")

    # Test case 3: What CBF=23 (our LS result) would produce
    print("\nCase 3: Our LS result (CBF=23 ml/100g/min, ATT=1400ms)")
    expected = calculate_expected_signal(23, 1400, plds)
    for i, pld in enumerate(plds):
        print(f"  PLD={int(pld)}ms: PCASL={expected['pcasl'][i]:.6f}, VSASL={expected['vsasl'][i]:.6f}")

    # Test case 4: Paper's PCASL result
    print("\nCase 4: Paper PCASL result (CBF=35 ml/100g/min, ATT=1400ms)")
    expected = calculate_expected_signal(35, 1400, plds)
    for i, pld in enumerate(plds):
        print(f"  PLD={int(pld)}ms: PCASL={expected['pcasl'][i]:.6f}, VSASL={expected['vsasl'][i]:.6f}")

    # Analyze actual in-vivo data
    print("\n" + "=" * 70)
    print("ACTUAL IN-VIVO SIGNAL VALUES")
    print("=" * 70)

    # Try to find in-vivo data
    possible_paths = [
        Path("/Users/adikondepudi/MULTIVERSE_data/raw_invivo"),
        Path("/Users/adikondepudi/Desktop/asl-multiverse-archive/prototype invivo archive/data/raw_invivo_validated"),
    ]

    invivo_dir = None
    for p in possible_paths:
        if p.exists():
            invivo_dir = p
            break

    if invivo_dir is None:
        print("  In-vivo data directory not found")
        return

    print(f"\nUsing data from: {invivo_dir}")

    # Analyze first subject
    subject_dirs = sorted([d for d in invivo_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

    if subject_dirs:
        print(f"\nAnalyzing: {subject_dirs[0].name}")
        result = analyze_invivo_signals(subject_dirs[0])

        if result:
            print("\n" + "=" * 70)
            print("COMPARISON: In-vivo vs Model")
            print("=" * 70)

            print("\n  If in-vivo signal matches model expectation:")
            print("  - Gray matter (CBF=60, ATT=1200): PCASL ~0.006-0.009")
            print("  - White matter (CBF=25, ATT=1800): PCASL ~0.002-0.004")

            mean_pcasl = np.mean(result['pcasl_means'])
            mean_vsasl = np.mean(result['vsasl_means'])

            print(f"\n  Actual in-vivo mean (whole brain): PCASL={mean_pcasl:.6f}, VSASL={mean_vsasl:.6f}")

            # Estimate CBF from signal
            # For PLD=1500ms, ATT=1400ms: signal ≈ 0.97 * CBF_internal
            # So CBF ≈ signal / 0.97 * 6000
            estimated_cbf = mean_pcasl / 0.00016 * 6000  # Rough approximation
            print(f"\n  Rough CBF estimate from PCASL signal: {estimated_cbf:.1f} ml/100g/min")


if __name__ == "__main__":
    main()

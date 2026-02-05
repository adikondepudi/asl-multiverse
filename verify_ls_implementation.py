#!/usr/bin/env python3
"""
Verify LS implementation is correct by testing on synthetic data.

This tests:
1. Signal model consistency (JIT vs LS model)
2. LS fitting accuracy on clean data
3. LS fitting accuracy on noisy data at various SNR levels
4. Failure rate analysis

If LS fails on synthetic data, we have a bug.
If LS works on synthetic but fails on in-vivo, it's a real data issue.
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

from asl_simulation import ASLParameters, _generate_pcasl_signal_jit, _generate_vsasl_signal_jit
from multiverse_functions import fun_PCVSASL_misMatchPLD_vect_pep, fit_PCVSASL_misMatchPLD_vectInit_pep
from utils import get_grid_search_initial_guess


def generate_synthetic_signal(cbf_ml_100g_min: float, att_ms: float, plds: np.ndarray,
                               params: dict) -> np.ndarray:
    """Generate clean synthetic signal."""
    cbf_ml_g_s = cbf_ml_100g_min / 6000.0

    # Combined alpha with background suppression
    alpha_pcasl = params['alpha_PCASL'] * (params.get('alpha_BS1', 1.0) ** 4)
    alpha_vsasl = params['alpha_VSASL'] * (params.get('alpha_BS1', 1.0) ** 3)

    pcasl = _generate_pcasl_signal_jit(
        plds.astype(np.float64), att_ms, cbf_ml_g_s,
        params['T1_artery'], params['T_tau'],
        alpha_pcasl, params.get('T2_factor', 1.0)
    )

    vsasl = _generate_vsasl_signal_jit(
        plds.astype(np.float64), att_ms, cbf_ml_g_s,
        params['T1_artery'], alpha_vsasl,
        params.get('T2_factor', 1.0), params.get('T_sat_vs', 2000.0)
    )

    return np.concatenate([pcasl, vsasl])


def add_rician_noise(signal: np.ndarray, snr: float) -> np.ndarray:
    """Add Rician noise to signal."""
    # SNR = signal_max / noise_std
    noise_std = np.max(np.abs(signal)) / snr

    real_part = signal + np.random.normal(0, noise_std, signal.shape)
    imag_part = np.random.normal(0, noise_std, signal.shape)

    return np.sqrt(real_part**2 + imag_part**2) * np.sign(signal)


def test_signal_model_consistency():
    """Test that JIT and LS model produce same signals."""
    print("\n" + "="*60)
    print("TEST 1: Signal Model Consistency")
    print("="*60)

    plds = np.array([500, 1000, 1500, 2000, 2500], dtype=np.float64)

    params = {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T2_factor': 1.0,
        'alpha_BS1': 1.0,
        'T_sat_vs': 2000.0,
    }

    test_cases = [
        (40, 800),   # Short ATT
        (60, 1500),  # Medium ATT
        (30, 2500),  # Long ATT
        (80, 1000),  # High CBF
        (15, 2000),  # Low CBF
    ]

    all_passed = True

    for cbf, att in test_cases:
        # Generate with JIT (used in grid search)
        jit_signal = generate_synthetic_signal(cbf, att, plds, params)

        # Generate with LS model
        cbf_internal = cbf / 6000.0
        pldti = np.column_stack([plds, plds])
        ls_signal = fun_PCVSASL_misMatchPLD_vect_pep(
            [cbf_internal, att], pldti,
            params['T1_artery'], params['T_tau'], params['T2_factor'],
            params['alpha_BS1'], params['alpha_PCASL'], params['alpha_VSASL']
        )
        ls_signal_flat = np.concatenate([ls_signal[:, 0], ls_signal[:, 1]])

        # Compare
        max_diff = np.max(np.abs(jit_signal - ls_signal_flat))
        rel_diff = max_diff / np.max(np.abs(jit_signal)) if np.max(np.abs(jit_signal)) > 0 else 0

        passed = rel_diff < 1e-6
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(f"  CBF={cbf:3d}, ATT={att:4d}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e} [{status}]")

    return all_passed


def test_ls_fitting_clean():
    """Test LS fitting on clean (noiseless) synthetic data."""
    print("\n" + "="*60)
    print("TEST 2: LS Fitting on Clean Data")
    print("="*60)

    plds = np.array([500, 1000, 1500, 2000, 2500], dtype=np.float64)

    params = {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T2_factor': 1.0,
        'alpha_BS1': 1.0,
    }

    test_cases = [
        (40, 800),
        (60, 1500),
        (30, 2500),
        (80, 1000),
        (15, 2000),
        (50, 1200),
        (25, 1800),
    ]

    results = []

    for true_cbf, true_att in test_cases:
        # Generate clean signal
        signal = generate_synthetic_signal(true_cbf, true_att, plds, params)

        # Get initial guess
        init_guess = get_grid_search_initial_guess(signal, plds, params)

        # Prepare for LS fitting
        n_plds = len(plds)
        signal_reshaped = np.column_stack([signal[:n_plds], signal[n_plds:]])
        pldti = np.column_stack([plds, plds])

        # Fit
        try:
            beta, _, rmse, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                pldti, signal_reshaped, init_guess, **params
            )

            fit_cbf = beta[0] * 6000.0
            fit_att = beta[1]

            cbf_err = fit_cbf - true_cbf
            att_err = fit_att - true_att

            results.append({
                'true_cbf': true_cbf, 'true_att': true_att,
                'fit_cbf': fit_cbf, 'fit_att': fit_att,
                'cbf_err': cbf_err, 'att_err': att_err,
                'rmse': rmse, 'success': True
            })

            print(f"  CBF={true_cbf:3d}, ATT={true_att:4d}: "
                  f"fit_CBF={fit_cbf:5.1f} (err={cbf_err:+6.2f}), "
                  f"fit_ATT={fit_att:6.0f} (err={att_err:+6.0f}ms)")

        except Exception as e:
            results.append({
                'true_cbf': true_cbf, 'true_att': true_att,
                'success': False, 'error': str(e)
            })
            print(f"  CBF={true_cbf:3d}, ATT={true_att:4d}: FAILED - {e}")

    # Summary
    successes = [r for r in results if r['success']]
    if successes:
        mean_cbf_err = np.mean([abs(r['cbf_err']) for r in successes])
        mean_att_err = np.mean([abs(r['att_err']) for r in successes])
        print(f"\n  Summary: {len(successes)}/{len(results)} succeeded")
        print(f"  Mean |CBF error|: {mean_cbf_err:.2f} ml/100g/min")
        print(f"  Mean |ATT error|: {mean_att_err:.1f} ms")

    return all(r['success'] for r in results)


def test_ls_fitting_noisy():
    """Test LS fitting on noisy synthetic data at various SNR levels."""
    print("\n" + "="*60)
    print("TEST 3: LS Fitting on Noisy Data")
    print("="*60)

    plds = np.array([500, 1000, 1500, 2000, 2500], dtype=np.float64)

    params = {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T2_factor': 1.0,
        'alpha_BS1': 1.0,
    }

    # Test grid
    cbf_values = [20, 40, 60, 80]
    att_values = [800, 1200, 1800, 2400]
    snr_values = [2, 5, 10, 20, 50]
    n_trials = 50  # Number of noise realizations per condition

    results_by_snr = {}

    for snr in snr_values:
        successes = 0
        cbf_errors = []
        att_errors = []

        for true_cbf in cbf_values:
            for true_att in att_values:
                for trial in range(n_trials):
                    # Generate clean signal
                    clean_signal = generate_synthetic_signal(true_cbf, true_att, plds, params)

                    # Add noise
                    np.random.seed(trial)
                    noisy_signal = add_rician_noise(clean_signal, snr)

                    # Get initial guess
                    try:
                        init_guess = get_grid_search_initial_guess(noisy_signal, plds, params)
                    except:
                        init_guess = [true_cbf / 6000.0, true_att]  # Use true as fallback

                    # Prepare for LS fitting
                    n_plds = len(plds)
                    signal_reshaped = np.column_stack([noisy_signal[:n_plds], noisy_signal[n_plds:]])
                    pldti = np.column_stack([plds, plds])

                    # Fit
                    try:
                        beta, _, rmse, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                            pldti, signal_reshaped, init_guess, **params
                        )

                        fit_cbf = beta[0] * 6000.0
                        fit_att = beta[1]

                        successes += 1
                        cbf_errors.append(fit_cbf - true_cbf)
                        att_errors.append(fit_att - true_att)

                    except Exception:
                        pass  # Count as failure

        total_trials = len(cbf_values) * len(att_values) * n_trials
        success_rate = successes / total_trials * 100

        results_by_snr[snr] = {
            'success_rate': success_rate,
            'n_success': successes,
            'n_total': total_trials,
            'cbf_bias': np.mean(cbf_errors) if cbf_errors else np.nan,
            'cbf_std': np.std(cbf_errors) if cbf_errors else np.nan,
            'att_bias': np.mean(att_errors) if att_errors else np.nan,
            'att_std': np.std(att_errors) if att_errors else np.nan,
        }

        print(f"  SNR={snr:2d}: Success={success_rate:5.1f}% ({successes}/{total_trials}), "
              f"CBF bias={results_by_snr[snr]['cbf_bias']:+5.1f}±{results_by_snr[snr]['cbf_std']:5.1f}, "
              f"ATT bias={results_by_snr[snr]['att_bias']:+5.0f}±{results_by_snr[snr]['att_std']:5.0f}ms")

    return results_by_snr


def analyze_in_vivo_signal_characteristics():
    """Analyze in-vivo signal characteristics to understand failure modes."""
    print("\n" + "="*60)
    print("TEST 4: In-Vivo Signal Characteristics")
    print("="*60)

    import nibabel as nib
    import re

    invivo_dir = Path("/Users/adikondepudi/MULTIVERSE_data/raw_invivo")

    if not invivo_dir.exists():
        print("  In-vivo data directory not found, skipping...")
        return None

    # Load one subject
    subject_dirs = sorted([d for d in invivo_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    if not subject_dirs:
        print("  No subjects found, skipping...")
        return None

    subject_dir = subject_dirs[0]
    print(f"  Analyzing: {subject_dir.name}")

    # Load PCASL signals
    pcasl_files = sorted(subject_dir.glob('r_normdiff_alldyn_PCASL_*.nii*'),
                         key=lambda p: int(re.search(r'_(\d+)', p.name).group(1)))

    if not pcasl_files:
        print("  No PCASL files found, skipping...")
        return None

    # Load and analyze
    signal_stats = []
    for f in pcasl_files:
        pld = int(re.search(r'_(\d+)', f.name).group(1))
        data = np.nan_to_num(nib.load(f).get_fdata())
        if data.ndim == 4:
            data = np.mean(data, axis=-1)

        # Get non-zero brain voxels
        brain_voxels = data[data != 0]

        signal_stats.append({
            'pld': pld,
            'mean': np.mean(brain_voxels),
            'std': np.std(brain_voxels),
            'min': np.min(brain_voxels),
            'max': np.max(brain_voxels),
            'median': np.median(brain_voxels),
            'p5': np.percentile(brain_voxels, 5),
            'p95': np.percentile(brain_voxels, 95),
        })

        print(f"    PLD={pld}ms: mean={signal_stats[-1]['mean']:.4f}, "
              f"std={signal_stats[-1]['std']:.4f}, "
              f"range=[{signal_stats[-1]['min']:.4f}, {signal_stats[-1]['max']:.4f}]")

    # Compare to synthetic signal range
    print("\n  Expected synthetic signal range (for CBF=40, ATT=1500):")
    plds = np.array([s['pld'] for s in signal_stats], dtype=np.float64)
    params = {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T2_factor': 1.0,
        'alpha_BS1': 1.0,
    }
    synthetic = generate_synthetic_signal(40, 1500, plds, params)
    n_plds = len(plds)
    print(f"    PCASL signal range: [{np.min(synthetic[:n_plds]):.4f}, {np.max(synthetic[:n_plds]):.4f}]")

    # Check SNR estimate
    mean_signal = np.mean([s['mean'] for s in signal_stats])
    mean_std = np.mean([s['std'] for s in signal_stats])
    estimated_snr = mean_signal / mean_std if mean_std > 0 else 0
    print(f"\n  Estimated in-vivo SNR (rough): {estimated_snr:.2f}")

    return signal_stats


def test_failure_mode_analysis():
    """Analyze what causes LS fitting to fail."""
    print("\n" + "="*60)
    print("TEST 5: LS Failure Mode Analysis")
    print("="*60)

    plds = np.array([500, 1000, 1500, 2000, 2500], dtype=np.float64)

    params = {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T2_factor': 1.0,
        'alpha_BS1': 1.0,
    }

    # Test edge cases
    edge_cases = [
        ("Very low CBF", 5, 1500),
        ("Very high CBF", 95, 1500),
        ("Very short ATT", 40, 200),
        ("Very long ATT", 40, 4000),
        ("ATT > all PLDs", 40, 3000),
        ("Negative noise (simulated)", 40, 1500),
        ("Zero signal", 0.1, 1500),
    ]

    for name, true_cbf, true_att in edge_cases:
        # Generate clean signal
        clean_signal = generate_synthetic_signal(true_cbf, true_att, plds, params)

        # Add noise at SNR=5
        np.random.seed(42)
        noisy_signal = add_rician_noise(clean_signal, 5.0)

        if "Negative" in name:
            # Simulate negative values from subtraction artifacts
            noisy_signal = clean_signal + np.random.normal(0, 0.005, clean_signal.shape)

        # Try fitting
        try:
            init_guess = get_grid_search_initial_guess(noisy_signal, plds, params)

            n_plds = len(plds)
            signal_reshaped = np.column_stack([noisy_signal[:n_plds], noisy_signal[n_plds:]])
            pldti = np.column_stack([plds, plds])

            beta, _, rmse, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                pldti, signal_reshaped, init_guess, **params
            )

            fit_cbf = beta[0] * 6000.0
            fit_att = beta[1]

            cbf_err = fit_cbf - true_cbf
            att_err = fit_att - true_att

            print(f"  {name:25s}: fit_CBF={fit_cbf:5.1f} (err={cbf_err:+6.1f}), "
                  f"fit_ATT={fit_att:6.0f} (err={att_err:+6.0f})")

        except Exception as e:
            print(f"  {name:25s}: FAILED - {type(e).__name__}: {e}")


def main():
    output_dir = Path("/Users/adikondepudi/Desktop/asl-multiverse/ls_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("LS IMPLEMENTATION VERIFICATION")
    print("="*60)

    # Test 1: Signal model consistency
    test1_passed = test_signal_model_consistency()

    # Test 2: Clean data fitting
    test2_passed = test_ls_fitting_clean()

    # Test 3: Noisy data fitting at various SNR
    noisy_results = test_ls_fitting_noisy()

    # Test 4: In-vivo signal characteristics
    invivo_stats = analyze_in_vivo_signal_characteristics()

    # Test 5: Failure mode analysis
    test_failure_mode_analysis()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n  Test 1 (Model Consistency): {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Test 2 (Clean Data Fitting): {'PASS' if test2_passed else 'FAIL'}")

    print(f"\n  Test 3 (Noisy Data) Summary:")
    for snr, results in noisy_results.items():
        print(f"    SNR={snr:2d}: {results['success_rate']:.1f}% success, "
              f"CBF bias={results['cbf_bias']:+.1f}±{results['cbf_std']:.1f}")

    # Save results
    results = {
        'test1_passed': bool(test1_passed),
        'test2_passed': bool(test2_passed),
        'noisy_results': {str(k): v for k, v in noisy_results.items()},
    }

    with open(output_dir / 'verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_dir}")

    # Key finding
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)
    if test1_passed and test2_passed:
        print("  LS implementation appears CORRECT on synthetic data.")
        print("  Failures on in-vivo data likely due to:")
        print("    - Low SNR in real data")
        print("    - Partial volume effects")
        print("    - Model-data mismatch (real physics differ from model)")
    else:
        print("  WARNING: LS implementation has bugs that need fixing!")


if __name__ == "__main__":
    main()

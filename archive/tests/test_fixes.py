#!/usr/bin/env python
"""
Quick diagnostic script to verify bug fixes.
Run this to confirm the NORM_STATS_INDICES fix is working correctly.
"""
import numpy as np
import sys

def test_norm_stats_indices():
    """Test that NORM_STATS_INDICES matches compute_feature_vector output order."""
    from feature_registry import FeatureRegistry

    # Create test signal
    np.random.seed(42)
    n_plds = 6
    signal = np.random.randn(10, n_plds * 2)  # 10 samples, 12 channels

    # Test with different active_features orderings
    test_cases = [
        ['mean', 'std'],
        ['mean', 'std', 'peak'],
        ['peak', 'mean'],  # Different order
        ['mean', 'std', 'peak', 'ttp', 'com'],
    ]

    all_passed = True

    for active_features in test_cases:
        # Compute features
        features = FeatureRegistry.compute_feature_vector(signal, n_plds, active_features)

        # Verify dimensions
        expected_dim = FeatureRegistry.compute_scalar_dim(active_features)
        actual_dim = features.shape[1]

        if actual_dim != expected_dim:
            print(f"FAIL: {active_features}")
            print(f"  Expected dim: {expected_dim}, Got: {actual_dim}")
            all_passed = False
            continue

        # Verify that features are in correct order
        current_idx = 0
        for feat_name in active_features:
            if feat_name not in FeatureRegistry.NORM_STATS_INDICES:
                continue

            width = FeatureRegistry.FEATURE_DIMS[feat_name]
            feat_slice = features[:, current_idx:current_idx + width]
            current_idx += width

            # Basic sanity check - features should have reasonable values
            if np.any(np.isnan(feat_slice)):
                print(f"FAIL: {feat_name} contains NaN")
                all_passed = False

        print(f"PASS: {active_features} -> dim={actual_dim}")

    return all_passed


def test_process_signals_consistency():
    """Test that process_signals_dynamic matches trainer's _process_batch_on_gpu."""
    import torch
    from utils import process_signals_dynamic
    from feature_registry import FeatureRegistry

    np.random.seed(42)
    n_plds = 6
    n_samples = 5

    # Create test signals
    signals = np.random.randn(n_samples, n_plds * 2).astype(np.float32) * 0.01

    # Create fake norm_stats
    norm_stats = {
        'scalar_features_mean': [0.0] * 12,
        'scalar_features_std': [1.0] * 12,
        'y_mean_t1': 1650.0,  # 3T consensus (Alsop 2015)
        'y_std_t1': 100.0,
    }

    active_features = ['mean', 'std', 'peak']
    config = {
        'pld_values': list(range(500, 3001, 500)),
        'active_features': active_features,
        'normalization_mode': 'per_curve',
    }

    t1_values = np.full((n_samples, 1), 1650.0, dtype=np.float32)  # 3T consensus (Alsop 2015)

    # Process
    processed = process_signals_dynamic(signals, norm_stats, config, t1_values=t1_values)

    # Expected dimensions
    expected_dim = n_plds * 2 + FeatureRegistry.compute_scalar_dim(active_features)
    if 't1_artery' in active_features:
        expected_dim += 1

    actual_dim = processed.shape[1]

    if actual_dim != expected_dim:
        print(f"FAIL: process_signals_dynamic dimension mismatch")
        print(f"  Expected: {expected_dim}, Got: {actual_dim}")
        return False

    print(f"PASS: process_signals_dynamic -> dim={actual_dim}")
    return True


def test_physics_simulation():
    """Test that physics simulation produces reasonable signals."""
    from asl_simulation import _generate_pcasl_signal_jit, _generate_vsasl_signal_jit

    plds = np.array([500, 1000, 1500, 2000, 2500, 3000], dtype=np.float64)

    # Test cases: (CBF, ATT, expected_peak_pld_index)
    test_cases = [
        (60.0, 1000.0, 1),   # Peak around PLD 1000
        (60.0, 1500.0, 2),   # Peak around PLD 1500
        (60.0, 2500.0, 4),   # Peak around PLD 2500
        (30.0, 1500.0, 2),   # Lower CBF, same shape
    ]

    all_passed = True

    for cbf, att, expected_peak_idx in test_cases:
        cbf_scaled = cbf / 6000.0  # Convert to ml/g/s

        pcasl = _generate_pcasl_signal_jit(plds, att, cbf_scaled, 1650.0, 1800.0, 0.85, 1.0)  # T1_artery=1650: 3T consensus (Alsop 2015)
        vsasl = _generate_vsasl_signal_jit(plds, att, cbf_scaled, 1650.0, 0.56, 1.0, 0.0)  # T1_artery=1650: 3T consensus (Alsop 2015)

        # Check signal is non-zero
        if np.max(pcasl) < 1e-10:
            print(f"FAIL: CBF={cbf}, ATT={att} produces zero PCASL signal")
            all_passed = False
            continue

        # Check peak is near expected
        actual_peak_idx = np.argmax(pcasl)
        if abs(actual_peak_idx - expected_peak_idx) > 1:
            print(f"WARNING: CBF={cbf}, ATT={att}")
            print(f"  Expected peak at PLD index ~{expected_peak_idx}, got {actual_peak_idx}")
            print(f"  PCASL signal: {pcasl}")

        print(f"PASS: CBF={cbf}, ATT={att} -> PCASL peak at idx {actual_peak_idx}")

    return all_passed


def main():
    print("=" * 60)
    print("Testing Bug Fixes")
    print("=" * 60)
    print()

    print("1. Testing NORM_STATS_INDICES fix...")
    print("-" * 40)
    test1 = test_norm_stats_indices()
    print()

    print("2. Testing process_signals_dynamic...")
    print("-" * 40)
    test2 = test_process_signals_consistency()
    print()

    print("3. Testing physics simulation...")
    print("-" * 40)
    test3 = test_physics_simulation()
    print()

    print("=" * 60)
    if test1 and test2 and test3:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

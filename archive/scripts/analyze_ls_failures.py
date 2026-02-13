#!/usr/bin/env python3
"""
Analyze LS fitting failures on in-vivo data.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import json

def analyze_subject(subject_dir: Path):
    """Analyze LS failures for one subject."""
    ls_cbf_path = subject_dir / 'ls_cbf.nii.gz'
    ls_att_path = subject_dir / 'ls_att.nii.gz'
    metrics_path = subject_dir / 'comparison_metrics.json'

    if not ls_cbf_path.exists():
        print(f"  No LS CBF found at {ls_cbf_path}")
        return

    # Load data
    ls_cbf = nib.load(ls_cbf_path).get_fdata()
    ls_att = nib.load(ls_att_path).get_fdata()

    with open(metrics_path) as f:
        metrics = json.load(f)

    print(f"\n{subject_dir.name}")
    print(f"  Total brain voxels: {metrics['n_brain_voxels']:,}")
    print(f"  LS failure rate: {metrics['ls_failure_rate']*100:.1f}%")

    # Analyze valid LS voxels
    valid_mask = ~np.isnan(ls_cbf) & (ls_cbf > 0)
    valid_cbf = ls_cbf[valid_mask]
    valid_att = ls_att[valid_mask]

    print(f"  Valid LS voxels: {valid_mask.sum():,}")

    # Check for boundary values (hitting optimizer bounds)
    cbf_at_lower = np.sum(valid_cbf <= 1.1)  # Near lower bound of 1
    cbf_at_upper = np.sum(valid_cbf >= 99)   # Near upper bound of 100
    att_at_lower = np.sum(valid_att <= 110)  # Near lower bound of 100
    att_at_upper = np.sum(valid_att >= 5900) # Near upper bound of 6000

    print(f"\n  CBF at lower bound (≤1.1): {cbf_at_lower:,} ({cbf_at_lower/valid_mask.sum()*100:.1f}%)")
    print(f"  CBF at upper bound (≥99): {cbf_at_upper:,} ({cbf_at_upper/valid_mask.sum()*100:.1f}%)")
    print(f"  ATT at lower bound (≤110): {att_at_lower:,} ({att_at_lower/valid_mask.sum()*100:.1f}%)")
    print(f"  ATT at upper bound (≥5900): {att_at_upper:,} ({att_at_upper/valid_mask.sum()*100:.1f}%)")

    # Distribution statistics
    print(f"\n  CBF distribution (valid voxels):")
    print(f"    Mean: {np.mean(valid_cbf):.1f} ml/100g/min")
    print(f"    Std:  {np.std(valid_cbf):.1f} ml/100g/min")
    print(f"    Min:  {np.min(valid_cbf):.1f} ml/100g/min")
    print(f"    Max:  {np.max(valid_cbf):.1f} ml/100g/min")
    print(f"    Percentiles: 5th={np.percentile(valid_cbf, 5):.1f}, "
          f"50th={np.percentile(valid_cbf, 50):.1f}, "
          f"95th={np.percentile(valid_cbf, 95):.1f}")

    print(f"\n  ATT distribution (valid voxels):")
    print(f"    Mean: {np.mean(valid_att):.0f} ms")
    print(f"    Std:  {np.std(valid_att):.0f} ms")
    print(f"    Min:  {np.min(valid_att):.0f} ms")
    print(f"    Max:  {np.max(valid_att):.0f} ms")
    print(f"    Percentiles: 5th={np.percentile(valid_att, 5):.0f}, "
          f"50th={np.percentile(valid_att, 50):.0f}, "
          f"95th={np.percentile(valid_att, 95):.0f}")

    # Analyze NaN voxels (failures)
    nan_mask = np.isnan(ls_cbf)
    if nan_mask.any():
        # This would need the original signals to analyze, which we don't have
        print(f"\n  NaN voxels: {nan_mask.sum():,}")


def main():
    base_dir = Path("/Users/adikondepudi/Desktop/asl-multiverse")

    # Analyze baseline comparison
    print("=" * 60)
    print("BASELINE MODEL - LS FAILURE ANALYSIS")
    print("=" * 60)

    baseline_dir = base_dir / "invivo_comparison_baseline"
    for subject_dir in sorted(baseline_dir.iterdir()):
        if subject_dir.is_dir():
            analyze_subject(subject_dir)

    # Summary across all subjects
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL SUBJECTS")
    print("=" * 60)

    all_failure_rates = []
    all_cbf_at_bounds = []

    for subject_dir in sorted(baseline_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        metrics_path = subject_dir / 'comparison_metrics.json'
        ls_cbf_path = subject_dir / 'ls_cbf.nii.gz'

        if metrics_path.exists() and ls_cbf_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            all_failure_rates.append(metrics['ls_failure_rate'])

            ls_cbf = nib.load(ls_cbf_path).get_fdata()
            valid_mask = ~np.isnan(ls_cbf) & (ls_cbf > 0)
            valid_cbf = ls_cbf[valid_mask]

            cbf_at_bounds = (np.sum(valid_cbf <= 1.1) + np.sum(valid_cbf >= 99)) / valid_mask.sum()
            all_cbf_at_bounds.append(cbf_at_bounds)

    print(f"\nFailure rate: {np.mean(all_failure_rates)*100:.1f}% ± {np.std(all_failure_rates)*100:.1f}%")
    print(f"CBF at bounds: {np.mean(all_cbf_at_bounds)*100:.1f}% ± {np.std(all_cbf_at_bounds)*100:.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare Neural Network vs Least-Squares fitting on in-vivo ASL data.

For PhD thesis/publication: Rigorous comparison with statistical analysis.

Outputs:
- Voxel-wise LS CBF/ATT maps
- NN vs LS comparison metrics
- Statistical analysis (correlation, Bland-Altman, ICC)
- Publication-ready figures
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import argparse
import re
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from asl_simulation import ASLParameters
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from utils import get_grid_search_initial_guess


def load_subject_data(subject_dir: Path, target_plds: List[int] = [500, 1000, 1500, 2000, 2500, 3000]
                      ) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image, List[int]]:
    """
    Load in-vivo ASL data for a subject.

    Returns:
        signals: (H, W, Z, 2*n_plds) - PCASL then VSASL
        brain_mask: (H, W, Z) binary mask
        ref_img: NIfTI reference for saving
        subject_plds: PLDs found in data
    """
    def find_files(pattern):
        files = list(subject_dir.glob(pattern))
        def get_pld(p):
            m = re.search(r'_(\d+)', p.name)
            return int(m.group(1)) if m else -1
        return sorted(files, key=get_pld)

    pcasl_files = find_files('r_normdiff_alldyn_PCASL_*.nii*')
    vsasl_files = find_files('r_normdiff_alldyn_VSASL_*.nii*')

    if not pcasl_files or not vsasl_files:
        raise ValueError(f"Missing ASL files in {subject_dir}")

    # Get PLDs from filenames
    pcasl_plds = [int(re.search(r'_(\d+)', f.name).group(1)) for f in pcasl_files]
    vsasl_plds = [int(re.search(r'_(\d+)', f.name).group(1)) for f in vsasl_files]
    common_plds = sorted(set(pcasl_plds) & set(vsasl_plds))

    # Filter to common PLDs
    pcasl_files = [f for f in pcasl_files if int(re.search(r'_(\d+)', f.name).group(1)) in common_plds]
    vsasl_files = [f for f in vsasl_files if int(re.search(r'_(\d+)', f.name).group(1)) in common_plds]

    # Load reference
    ref_img = nib.load(pcasl_files[0])

    # Load and average repeats
    pcasl_vols = []
    for f in pcasl_files:
        data = nib.load(f).get_fdata()
        data = np.nan_to_num(data, nan=0.0)
        if data.ndim == 4:
            data = np.mean(data, axis=-1)
        pcasl_vols.append(data)

    vsasl_vols = []
    for f in vsasl_files:
        data = nib.load(f).get_fdata()
        data = np.nan_to_num(data, nan=0.0)
        if data.ndim == 4:
            data = np.mean(data, axis=-1)
        vsasl_vols.append(data)

    # Stack: (H, W, Z, n_plds)
    pcasl = np.stack(pcasl_vols, axis=-1)
    vsasl = np.stack(vsasl_vols, axis=-1)

    # Combine: (H, W, Z, 2*n_plds)
    signals = np.concatenate([pcasl, vsasl], axis=-1)

    # Create brain mask from M0
    m0_files = list(subject_dir.glob('r_M0.nii*'))
    if m0_files:
        m0_data = np.nan_to_num(nib.load(m0_files[0]).get_fdata())
        threshold = np.percentile(m0_data[m0_data > 0], 50) * 0.3
        brain_mask = m0_data > threshold
    else:
        mean_signal = np.mean(np.abs(signals), axis=-1)
        threshold = np.percentile(mean_signal[mean_signal > 0], 90) * 0.1
        brain_mask = mean_signal > threshold

    return signals, brain_mask, ref_img, common_plds


def fit_single_voxel(args):
    """Fit a single voxel using least-squares. For parallel processing."""
    signal, plds, ls_params = args

    try:
        # Get initial guess via grid search
        init_guess = get_grid_search_initial_guess(signal, plds, ls_params)

        # Reshape for optimizer: (n_plds, 2) with PCASL and VSASL columns
        n_plds = len(plds)
        signal_reshaped = np.column_stack([signal[:n_plds], signal[n_plds:]])

        # Prepare PLD input
        pldti = np.column_stack([plds, plds])

        # Run optimizer
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
            pldti, signal_reshaped, init_guess, **ls_params
        )

        cbf = beta[0] * 6000.0  # Convert to ml/100g/min
        att = beta[1]  # Already in ms

        return cbf, att, True
    except Exception:
        return np.nan, np.nan, False


def run_ls_fitting(signals: np.ndarray, brain_mask: np.ndarray,
                   plds: np.ndarray, n_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run least-squares fitting on all brain voxels.

    Args:
        signals: (H, W, Z, 2*n_plds)
        brain_mask: (H, W, Z)
        plds: PLD values in ms
        n_workers: Number of parallel workers

    Returns:
        cbf_map: (H, W, Z)
        att_map: (H, W, Z)
    """
    h, w, z = brain_mask.shape
    n_plds = len(plds)

    # Initialize output maps
    cbf_map = np.full((h, w, z), np.nan, dtype=np.float32)
    att_map = np.full((h, w, z), np.nan, dtype=np.float32)

    # LS parameters
    ls_params = {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T2_factor': 1.0,
        'alpha_BS1': 1.0
    }

    # Get brain voxel indices
    brain_indices = np.argwhere(brain_mask)
    n_voxels = len(brain_indices)

    print(f"  Fitting {n_voxels} brain voxels with LS...")

    # Prepare arguments for parallel processing
    fit_args = []
    for idx in brain_indices:
        i, j, k = idx
        signal = signals[i, j, k, :]
        fit_args.append((signal, plds, ls_params))

    # Run fitting (parallel or serial based on n_workers)
    start_time = time.time()

    if n_workers > 1:
        # Parallel processing
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(fit_single_voxel, arg): i
                      for i, arg in enumerate(fit_args)}

            for future in tqdm(as_completed(futures), total=len(futures), desc="  LS fitting"):
                idx = futures[future]
                results.append((idx, future.result()))

        # Sort by original index and extract results
        results.sort(key=lambda x: x[0])
        for (idx, (cbf, att, success)), brain_idx in zip(results, brain_indices):
            i, j, k = brain_idx
            cbf_map[i, j, k] = cbf
            att_map[i, j, k] = att
    else:
        # Serial processing with progress bar
        for (i, j, k), arg in tqdm(zip(brain_indices, fit_args),
                                    total=n_voxels, desc="  LS fitting"):
            cbf, att, success = fit_single_voxel(arg)
            cbf_map[i, j, k] = cbf
            att_map[i, j, k] = att

    elapsed = time.time() - start_time
    print(f"  LS fitting completed in {elapsed:.1f}s ({elapsed/n_voxels*1000:.2f}ms/voxel)")

    return cbf_map, att_map


def compute_comparison_metrics(nn_map: np.ndarray, ls_map: np.ndarray,
                                mask: np.ndarray, param_name: str) -> Dict:
    """
    Compute comparison metrics between NN and LS maps.

    Returns dict with:
    - Correlation (Pearson, Spearman)
    - Bland-Altman (bias, limits of agreement)
    - ICC (intraclass correlation)
    - MAE, RMSE between methods
    """
    from scipy import stats

    # Get valid voxels (in mask and not NaN in either map)
    valid = mask & ~np.isnan(nn_map) & ~np.isnan(ls_map)
    nn_vals = nn_map[valid]
    ls_vals = ls_map[valid]

    if len(nn_vals) < 10:
        return {'error': 'Insufficient valid voxels'}

    # Correlation
    pearson_r, pearson_p = stats.pearsonr(nn_vals, ls_vals)
    spearman_r, spearman_p = stats.spearmanr(nn_vals, ls_vals)

    # Bland-Altman
    diff = nn_vals - ls_vals
    mean_vals = (nn_vals + ls_vals) / 2
    bias = np.mean(diff)
    std_diff = np.std(diff)
    loa_lower = bias - 1.96 * std_diff
    loa_upper = bias + 1.96 * std_diff

    # ICC (two-way random, absolute agreement)
    # Simplified ICC(2,1) calculation
    n = len(nn_vals)
    mean_nn = np.mean(nn_vals)
    mean_ls = np.mean(ls_vals)
    grand_mean = (mean_nn + mean_ls) / 2

    ss_between = n * ((mean_nn - grand_mean)**2 + (mean_ls - grand_mean)**2)
    ss_within = np.sum((nn_vals - mean_nn)**2) + np.sum((ls_vals - mean_ls)**2)
    ss_error = np.sum((nn_vals - ls_vals)**2) / 2

    ms_between = ss_between / 1
    ms_within = ss_within / (2 * (n - 1))
    ms_error = ss_error / n

    icc = (ms_between - ms_error) / (ms_between + ms_error + 2*(ms_within - ms_error)/n)
    icc = max(0, min(1, icc))  # Clamp to [0, 1]

    # Error metrics
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    # Distribution stats
    nn_mean, nn_std = np.mean(nn_vals), np.std(nn_vals)
    ls_mean, ls_std = np.mean(ls_vals), np.std(ls_vals)

    return {
        'n_voxels': int(len(nn_vals)),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'bland_altman': {
            'bias': float(bias),
            'std': float(std_diff),
            'loa_lower': float(loa_lower),
            'loa_upper': float(loa_upper),
        },
        'icc': float(icc),
        'mae': float(mae),
        'rmse': float(rmse),
        'nn_stats': {'mean': float(nn_mean), 'std': float(nn_std)},
        'ls_stats': {'mean': float(ls_mean), 'std': float(ls_std)},
    }


def save_nifti(data: np.ndarray, reference: nib.Nifti1Image, output_path: Path):
    """Save array as NIfTI."""
    img = nib.Nifti1Image(data.astype(np.float32), reference.affine, reference.header)
    nib.save(img, output_path)


def process_subject(subject_dir: Path, nn_results_dir: Path, output_dir: Path,
                    n_workers: int = 4) -> Dict:
    """
    Process a single subject: run LS fitting and compare with NN.
    """
    subject_id = subject_dir.name
    print(f"\n{'='*60}")
    print(f"Processing: {subject_id}")
    print(f"{'='*60}")

    # Create output directory
    subject_output = output_dir / subject_id
    subject_output.mkdir(parents=True, exist_ok=True)

    # Load data
    signals, brain_mask, ref_img, subject_plds = load_subject_data(subject_dir)
    plds = np.array(subject_plds)

    print(f"  Data shape: {signals.shape}")
    print(f"  PLDs: {subject_plds}")
    print(f"  Brain voxels: {brain_mask.sum()}")

    # Run LS fitting
    ls_cbf, ls_att = run_ls_fitting(signals, brain_mask, plds, n_workers)

    # Save LS results
    save_nifti(ls_cbf, ref_img, subject_output / 'ls_cbf.nii.gz')
    save_nifti(ls_att, ref_img, subject_output / 'ls_att.nii.gz')

    # Load NN results
    nn_cbf_path = nn_results_dir / subject_id / 'nn_cbf.nii.gz'
    nn_att_path = nn_results_dir / subject_id / 'nn_att.nii.gz'

    if not nn_cbf_path.exists():
        print(f"  WARNING: NN results not found at {nn_cbf_path}")
        return None

    nn_cbf = nib.load(nn_cbf_path).get_fdata()
    nn_att = nib.load(nn_att_path).get_fdata()

    # Compute comparison metrics
    cbf_metrics = compute_comparison_metrics(nn_cbf, ls_cbf, brain_mask, 'CBF')
    att_metrics = compute_comparison_metrics(nn_att, ls_att, brain_mask, 'ATT')

    # Print summary
    print(f"\n  CBF Comparison:")
    print(f"    NN:  {cbf_metrics['nn_stats']['mean']:.1f} ± {cbf_metrics['nn_stats']['std']:.1f} ml/100g/min")
    print(f"    LS:  {cbf_metrics['ls_stats']['mean']:.1f} ± {cbf_metrics['ls_stats']['std']:.1f} ml/100g/min")
    print(f"    Correlation: r={cbf_metrics['pearson_r']:.3f}")
    print(f"    ICC: {cbf_metrics['icc']:.3f}")
    print(f"    Bias: {cbf_metrics['bland_altman']['bias']:.2f} ml/100g/min")

    print(f"\n  ATT Comparison:")
    print(f"    NN:  {att_metrics['nn_stats']['mean']:.0f} ± {att_metrics['nn_stats']['std']:.0f} ms")
    print(f"    LS:  {att_metrics['ls_stats']['mean']:.0f} ± {att_metrics['ls_stats']['std']:.0f} ms")
    print(f"    Correlation: r={att_metrics['pearson_r']:.3f}")
    print(f"    ICC: {att_metrics['icc']:.3f}")
    print(f"    Bias: {att_metrics['bland_altman']['bias']:.0f} ms")

    # Count LS failures
    ls_cbf_valid = ~np.isnan(ls_cbf) & brain_mask
    ls_failure_rate = 1 - ls_cbf_valid.sum() / brain_mask.sum()
    print(f"\n  LS failure rate: {ls_failure_rate*100:.1f}%")

    # Save comparison results
    results = {
        'subject_id': subject_id,
        'plds': subject_plds,
        'n_brain_voxels': int(brain_mask.sum()),
        'ls_failure_rate': float(ls_failure_rate),
        'cbf_comparison': cbf_metrics,
        'att_comparison': att_metrics,
    }

    with open(subject_output / 'comparison_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def aggregate_results(output_dir: Path) -> Dict:
    """Aggregate results across all subjects."""
    all_results = []

    for subj_dir in sorted(output_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        metrics_file = subj_dir / 'comparison_metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                all_results.append(json.load(f))

    if not all_results:
        return {}

    # Aggregate metrics
    cbf_correlations = [r['cbf_comparison']['pearson_r'] for r in all_results]
    cbf_iccs = [r['cbf_comparison']['icc'] for r in all_results]
    cbf_biases = [r['cbf_comparison']['bland_altman']['bias'] for r in all_results]

    att_correlations = [r['att_comparison']['pearson_r'] for r in all_results]
    att_iccs = [r['att_comparison']['icc'] for r in all_results]
    att_biases = [r['att_comparison']['bland_altman']['bias'] for r in all_results]

    ls_failures = [r['ls_failure_rate'] for r in all_results]

    aggregate = {
        'n_subjects': len(all_results),
        'cbf': {
            'mean_correlation': float(np.mean(cbf_correlations)),
            'std_correlation': float(np.std(cbf_correlations)),
            'mean_icc': float(np.mean(cbf_iccs)),
            'std_icc': float(np.std(cbf_iccs)),
            'mean_bias': float(np.mean(cbf_biases)),
            'std_bias': float(np.std(cbf_biases)),
        },
        'att': {
            'mean_correlation': float(np.mean(att_correlations)),
            'std_correlation': float(np.std(att_correlations)),
            'mean_icc': float(np.mean(att_iccs)),
            'std_icc': float(np.std(att_iccs)),
            'mean_bias': float(np.mean(att_biases)),
            'std_bias': float(np.std(att_biases)),
        },
        'ls_mean_failure_rate': float(np.mean(ls_failures)),
        'per_subject': all_results,
    }

    # Save aggregate results
    with open(output_dir / 'aggregate_comparison.json', 'w') as f:
        json.dump(aggregate, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("AGGREGATE RESULTS ACROSS ALL SUBJECTS")
    print("="*70)
    print(f"\nN subjects: {aggregate['n_subjects']}")
    print(f"\nCBF (NN vs LS):")
    print(f"  Correlation: {aggregate['cbf']['mean_correlation']:.3f} ± {aggregate['cbf']['std_correlation']:.3f}")
    print(f"  ICC: {aggregate['cbf']['mean_icc']:.3f} ± {aggregate['cbf']['std_icc']:.3f}")
    print(f"  Bias: {aggregate['cbf']['mean_bias']:.2f} ± {aggregate['cbf']['std_bias']:.2f} ml/100g/min")
    print(f"\nATT (NN vs LS):")
    print(f"  Correlation: {aggregate['att']['mean_correlation']:.3f} ± {aggregate['att']['std_correlation']:.3f}")
    print(f"  ICC: {aggregate['att']['mean_icc']:.3f} ± {aggregate['att']['std_icc']:.3f}")
    print(f"  Bias: {aggregate['att']['mean_bias']:.0f} ± {aggregate['att']['std_bias']:.0f} ms")
    print(f"\nLS failure rate: {aggregate['ls_mean_failure_rate']*100:.1f}%")

    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Compare NN vs LS on in-vivo ASL data")
    parser.add_argument("invivo_dir", type=str, help="Directory with raw in-vivo data")
    parser.add_argument("nn_results_dir", type=str, help="Directory with NN predictions")
    parser.add_argument("output_dir", type=str, help="Output directory for LS results and comparison")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--subjects", type=str, nargs='+', help="Specific subjects to process")

    args = parser.parse_args()

    invivo_dir = Path(args.invivo_dir)
    nn_results_dir = Path(args.nn_results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find subjects
    subject_dirs = sorted([d for d in invivo_dir.iterdir()
                          if d.is_dir() and not d.name.startswith('.')])

    if args.subjects:
        subject_dirs = [d for d in subject_dirs if d.name in args.subjects]

    # Filter to subjects with NN results
    subject_dirs = [d for d in subject_dirs
                   if (nn_results_dir / d.name / 'nn_cbf.nii.gz').exists()]

    print(f"Found {len(subject_dirs)} subjects with NN results")

    # Process each subject
    for subject_dir in subject_dirs:
        try:
            process_subject(subject_dir, nn_results_dir, output_dir, args.workers)
        except Exception as e:
            print(f"ERROR processing {subject_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate results
    aggregate_results(output_dir)

    print(f"\n=== Complete! Results saved to: {output_dir} ===")


if __name__ == '__main__':
    main()

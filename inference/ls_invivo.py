#!/usr/bin/env python3
"""
Least-squares fitting on in-vivo ASL data for direct comparison with NN models.

Runs voxel-wise combined PCASL+VSASL fitting using the same LS fitter used in
bias/CoV evaluation. Outputs CBF and ATT maps as NIfTI files.

Usage:
    python -m inference.ls_invivo --invivo-dir data/invivo_validated \
        --output-dir invivo_results_v3/LS_baseline

    python -m inference.ls_invivo --invivo-dir data/invivo_validated \
        --output-dir invivo_results_v3/LS_baseline --subjects 20231002_MR1_A144
"""

import argparse
import json
import re
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

from baselines.multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from simulation.asl_simulation import _generate_pcasl_signal_jit, _generate_vsasl_signal_jit
from utils.helpers import get_grid_search_initial_guess

warnings.filterwarnings('ignore')

# Physics parameters — must match training data generation (FeatureRegistry defaults)
# NOTE: ALPHA_BS1 default changed from 1.0 to 0.93 for in-vivo data which has
# background suppression. Use --alpha-bs1 CLI arg to override.
T1_ARTERY = 1650.0
T_TAU = 1800.0
ALPHA_PCASL = 0.85
ALPHA_VSASL = 0.56
ALPHA_BS1 = 0.93
T2_FACTOR = 1.0
T_SAT_VS = 2000.0


def find_and_sort_files_by_pld(subject_dir: Path, pattern: str) -> List[Path]:
    def get_pld(path: Path) -> int:
        match = re.search(r'_(\d+)', path.name)
        return int(match.group(1)) if match else -1
    files = list(subject_dir.glob(pattern))
    return sorted(files, key=get_pld)


def load_nifti(file_path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(file_path)
    data = img.get_fdata(dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, img


def fit_single_voxel(args):
    """Worker for parallel LS fitting of a single voxel."""
    signal_1d, plds, ls_params = args
    pldti = np.column_stack([plds, plds])
    try:
        init = get_grid_search_initial_guess(signal_1d, plds, ls_params)
        signal_reshaped = signal_1d.reshape((len(plds), 2), order='F')
        beta, conintval, rmse, df = fit_PCVSASL_misMatchPLD_vectInit_pep(
            pldti, signal_reshaped, init, **ls_params
        )
        cbf = beta[0] * 6000.0  # Convert from ml/g/s to ml/100g/min
        att = beta[1]           # Already in ms
        if np.isfinite(cbf) and np.isfinite(att):
            return cbf, att, rmse
    except Exception:
        pass
    return np.nan, np.nan, np.nan


def process_subject(subject_dir: Path, output_dir: Path, n_workers: int = 8):
    """Run LS fitting on all brain voxels for one subject."""
    subject_id = subject_dir.name
    subject_output = output_dir / subject_id
    subject_output.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {subject_id}")

    # Find PCASL and VSASL files
    pcasl_files = find_and_sort_files_by_pld(subject_dir, 'r_normdiff_alldyn_PCASL_*.nii*')
    vsasl_files = find_and_sort_files_by_pld(subject_dir, 'r_normdiff_alldyn_VSASL_*.nii*')

    if not pcasl_files or not vsasl_files:
        print(f"  SKIP: Missing PCASL or VSASL files")
        return

    # Extract PLDs
    pcasl_plds = [int(re.search(r'_(\d+)', f.name).group(1)) for f in pcasl_files]
    vsasl_plds = [int(re.search(r'_(\d+)', f.name).group(1)) for f in vsasl_files]
    common_plds = sorted(set(pcasl_plds) & set(vsasl_plds))

    pcasl_files = [f for f in pcasl_files if int(re.search(r'_(\d+)', f.name).group(1)) in common_plds]
    vsasl_files = [f for f in vsasl_files if int(re.search(r'_(\d+)', f.name).group(1)) in common_plds]

    plds = np.array(common_plds, dtype=np.float64)
    n_plds = len(plds)
    print(f"  PLDs: {common_plds}")

    # Load reference image
    _, ref_img = load_nifti(pcasl_files[0])

    # Load and average repeats
    pcasl_volumes = []
    for f in pcasl_files:
        data, _ = load_nifti(f)
        if data.ndim == 4:
            data = np.mean(data, axis=-1)
        pcasl_volumes.append(data)

    vsasl_volumes = []
    for f in vsasl_files:
        data, _ = load_nifti(f)
        if data.ndim == 4:
            data = np.mean(data, axis=-1)
        vsasl_volumes.append(data)

    # Stack: (H, W, Z, n_plds)
    pcasl_stack = np.stack(pcasl_volumes, axis=-1)
    vsasl_stack = np.stack(vsasl_volumes, axis=-1)
    H, W, Z, _ = pcasl_stack.shape

    # Brain mask from M0
    m0_files = list(subject_dir.glob('r_M0.nii*'))
    if m0_files:
        m0_data, _ = load_nifti(m0_files[0])
        threshold = np.percentile(m0_data[m0_data > 0], 50) * 0.3
        brain_mask = m0_data > threshold
    else:
        mean_signal = np.mean(np.abs(pcasl_stack), axis=-1)
        threshold = np.percentile(mean_signal[mean_signal > 0], 90) * 0.1
        brain_mask = mean_signal > threshold

    # Build voxel-wise fitting tasks
    # Signal format for LS: concatenated [PCASL_pld1, ..., PCASL_pldN, VSASL_pld1, ..., VSASL_pldN]
    brain_indices = np.argwhere(brain_mask)
    n_voxels = len(brain_indices)
    print(f"  Brain voxels: {n_voxels}")

    tasks = []
    for idx in brain_indices:
        i, j, k = idx
        pcasl_signal = pcasl_stack[i, j, k, :]  # (n_plds,)
        vsasl_signal = vsasl_stack[i, j, k, :]  # (n_plds,)
        signal_1d = np.concatenate([pcasl_signal, vsasl_signal])
        tasks.append((signal_1d, plds, LS_PARAMS))

    # Run parallel fitting
    print(f"  Fitting {n_voxels} voxels with {n_workers} workers...")
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(fit_single_voxel, tasks, chunksize=100),
            total=n_voxels,
            desc=f"  {subject_id}",
            leave=False
        ))

    # Reconstruct volumes
    cbf_vol = np.zeros((H, W, Z), dtype=np.float32)
    att_vol = np.zeros((H, W, Z), dtype=np.float32)
    rmse_vol = np.zeros((H, W, Z), dtype=np.float32)
    fit_success = np.zeros((H, W, Z), dtype=bool)

    for idx, (cbf, att, rmse) in zip(brain_indices, results):
        i, j, k = idx
        if np.isfinite(cbf) and np.isfinite(att):
            cbf_vol[i, j, k] = np.clip(cbf, 0, 200)
            att_vol[i, j, k] = np.clip(att, 0, 5000)
            rmse_vol[i, j, k] = rmse
            fit_success[i, j, k] = True

    n_success = int(fit_success.sum())
    n_fail = n_voxels - n_success
    print(f"  Fit success: {n_success}/{n_voxels} ({100*n_success/n_voxels:.1f}%), failed: {n_fail}")

    # Save outputs
    def save_nifti(data, path):
        img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
        nib.save(img, path)

    save_nifti(cbf_vol, subject_output / 'ls_cbf.nii.gz')
    save_nifti(att_vol, subject_output / 'ls_att.nii.gz')
    save_nifti(rmse_vol, subject_output / 'ls_rmse.nii.gz')
    save_nifti(brain_mask.astype(np.float32), subject_output / 'brain_mask.nii.gz')

    # Stats
    cbf_masked = cbf_vol[brain_mask & fit_success]
    att_masked = att_vol[brain_mask & fit_success]

    metadata = {
        'method': 'Least Squares (multiverse combined PCASL+VSASL)',
        'physics': LS_PARAMS,
        'plds': common_plds,
        'n_brain_voxels': n_voxels,
        'n_fit_success': n_success,
        'n_fit_failed': n_fail,
        'cbf_stats': {
            'mean': float(cbf_masked.mean()),
            'std': float(cbf_masked.std()),
            'median': float(np.median(cbf_masked)),
            'min': float(cbf_masked.min()),
            'max': float(cbf_masked.max()),
        },
        'att_stats': {
            'mean': float(att_masked.mean()),
            'std': float(att_masked.std()),
            'median': float(np.median(att_masked)),
            'min': float(att_masked.min()),
            'max': float(att_masked.max()),
        }
    }
    with open(subject_output / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  CBF: {metadata['cbf_stats']['mean']:.1f} ± {metadata['cbf_stats']['std']:.1f} ml/100g/min")
    print(f"  ATT: {metadata['att_stats']['mean']:.0f} ± {metadata['att_stats']['std']:.0f} ms")
    print(f"  Saved to: {subject_output}")


def main():
    parser = argparse.ArgumentParser(description="Run LS fitting on in-vivo ASL data")
    parser.add_argument("--invivo-dir", type=str, required=True,
                        help="Directory containing subject folders")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--subjects", type=str, nargs='+', default=None,
                        help="Specific subjects to process (default: all)")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Number of parallel workers for fitting")
    parser.add_argument("--alpha-bs1", type=float, default=ALPHA_BS1,
                        help="Background suppression efficiency (default: 0.93). "
                             "Use 1.0 for no BS, 0.85-0.95 for typical in-vivo BS.")
    args = parser.parse_args()

    # Build LS_PARAMS with CLI-overridable alpha_BS1
    global LS_PARAMS
    LS_PARAMS = {
        'T1_artery': T1_ARTERY,
        'T_tau': T_TAU,
        'T2_factor': T2_FACTOR,
        'alpha_BS1': args.alpha_bs1,
        'alpha_PCASL': ALPHA_PCASL,
        'alpha_VSASL': ALPHA_VSASL,
        'T_sat_vs': T_SAT_VS,
    }

    invivo_dir = Path(args.invivo_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([d for d in invivo_dir.iterdir()
                          if d.is_dir() and not d.name.startswith('.')])

    if args.subjects:
        subject_dirs = [d for d in subject_dirs if d.name in args.subjects]

    print(f"Found {len(subject_dirs)} subjects")
    print(f"Physics: T1={T1_ARTERY}, T_tau={T_TAU}, alpha_PCASL={ALPHA_PCASL}, "
          f"alpha_VSASL={ALPHA_VSASL}, alpha_BS1={args.alpha_bs1}")

    for subject_dir in subject_dirs:
        try:
            process_subject(subject_dir, output_dir, args.n_workers)
        except Exception as e:
            print(f"  ERROR processing {subject_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Complete! Results saved to: {output_dir} ===")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
In-Vivo Repeat Comparison: NN(1-rep) vs LS(1-rep) vs LS(4-rep)
===============================================================
Compares neural network and least-squares fitting on in-vivo ASL data
using single-repeat (noisy) and multi-repeat (averaged) signals.

LS(4-rep) averaged signal serves as the reference "ground truth".

Usage:
    python amplitude_ablation_v7/evaluate_invivo_repeats.py \
        --model-dir amplitude_ablation_v7/B_AmplitudeAware \
        --output-dir amplitude_ablation_v7/v7_evaluation_results
"""

import sys
import os
import json
import time
import re
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
import yaml

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from models.spatial_asl_network import SpatialASLNet
from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
from utils.helpers import get_grid_search_initial_guess
from baselines.multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel required. Install with: pip install nibabel")
    sys.exit(1)


# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_PLDS = [500, 1000, 1500, 2000, 2500]
T1_ARTERY = 1650.0
T_TAU = 1800.0
ALPHA_PCASL = 0.85
ALPHA_VSASL = 0.56
T2_FACTOR = 1.0
T_SAT_VS = 2000.0
M0_SCALE_FACTOR = 100.0


# ── Model Loading ─────────────────────────────────────────────────────────

def load_model(run_dir, device):
    """Load spatial model ensemble from a run directory."""
    run_dir = Path(run_dir)

    with open(run_dir / 'config.yaml') as f:
        full_config = yaml.safe_load(f)
    training_config = full_config.get('training', {})
    data_config = full_config.get('data', {})

    with open(run_dir / 'norm_stats.json') as f:
        norm_stats = json.load(f)

    model_class_name = training_config.get('model_class_name', 'SpatialASLNet')
    hidden_sizes = training_config.get('hidden_sizes', [32, 64, 128, 256])
    pld_values = data_config.get('pld_values', DEFAULT_PLDS)
    n_plds = len(pld_values)
    gsf = data_config.get('global_scale_factor', 10.0)

    model_files = sorted((run_dir / 'trained_models').glob('ensemble_model_*.pt'))
    if not model_files:
        raise FileNotFoundError(f"No ensemble models found in {run_dir / 'trained_models'}")

    models = []
    for mf in model_files:
        if model_class_name == 'AmplitudeAwareSpatialASLNet':
            model = AmplitudeAwareSpatialASLNet(
                n_plds=n_plds, features=hidden_sizes,
                use_film_at_bottleneck=training_config.get('use_film_at_bottleneck', True),
                use_film_at_decoder=training_config.get('use_film_at_decoder', True),
                use_amplitude_output_modulation=training_config.get('use_amplitude_output_modulation', True),
            )
        else:
            model = SpatialASLNet(n_plds=n_plds, features=hidden_sizes)

        sd = torch.load(mf, map_location=device, weights_only=False)
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        model.load_state_dict(sd)
        model.to(device).eval()
        models.append(model)

    print(f"  Loaded {len(models)} ensemble members ({model_class_name})")
    return models, norm_stats, gsf, pld_values


# ── Spatial Utilities ──────────────────────────────────────────────────────

def pad_to_multiple(tensor, multiple=16):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(tensor, padding):
    pad_top, pad_bottom, pad_left, pad_right = padding
    _, _, h, w = tensor.shape
    return tensor[:, :,
                  pad_top:h - pad_bottom if pad_bottom else h,
                  pad_left:w - pad_right if pad_right else w]


# ── In-Vivo Data Loading ──────────────────────────────────────────────────

def find_nifti_files(subject_dir, modality, plds):
    """Find NIfTI files for a modality (PCASL/VSASL) sorted by PLD."""
    files = {}
    for pld in plds:
        patterns = [
            f'r_normdiff_alldyn_{modality}_{pld}.nii.gz',
            f'r_normdiff_alldyn_{modality}_{pld}.nii',
        ]
        for pat in patterns:
            path = subject_dir / pat
            if path.exists():
                files[pld] = path
                break
        if pld not in files:
            # Try glob
            matches = list(subject_dir.glob(f'r_normdiff_alldyn_{modality}_{pld}.nii*'))
            if matches:
                files[pld] = matches[0]
    return files


def load_subject_data(subject_dir, model_plds, alpha_bs1):
    """
    Load in-vivo NIfTI data for a subject.

    Returns:
        signals_4rep: (H, W, Z, 2*n_plds) - averaged over all repeats
        signals_1rep: (H, W, Z, 2*n_plds) - first repeat only
        brain_mask: (H, W, Z) boolean mask
        ref_img: NIfTI reference image
        n_repeats_pcasl: number of PCASL repeats
        n_repeats_vsasl: number of VSASL repeats
    """
    # Find available PLDs
    pcasl_files = find_nifti_files(subject_dir, 'PCASL', model_plds)
    vsasl_files = find_nifti_files(subject_dir, 'VSASL', model_plds)

    if not pcasl_files or not vsasl_files:
        raise ValueError(f"Missing PCASL or VSASL files in {subject_dir}")

    common_plds = sorted(set(pcasl_files.keys()) & set(vsasl_files.keys()))
    if len(common_plds) < 3:
        raise ValueError(f"Only {len(common_plds)} common PLDs found, need at least 3")

    print(f"  Found PLDs: {common_plds}")

    ref_img = nib.load(pcasl_files[common_plds[0]])

    # Load PCASL data - shape (H, W, Z, n_repeats)
    pcasl_4rep = []
    pcasl_1rep = []
    n_repeats_pcasl = None
    for pld in common_plds:
        data = nib.load(pcasl_files[pld]).get_fdata(dtype=np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if data.ndim == 4:
            n_repeats_pcasl = data.shape[-1]
            pcasl_4rep.append(np.mean(data, axis=-1))
            pcasl_1rep.append(data[..., 0])  # first repeat
        else:
            n_repeats_pcasl = 1
            pcasl_4rep.append(data)
            pcasl_1rep.append(data)

    # Load VSASL data
    vsasl_4rep = []
    vsasl_1rep = []
    n_repeats_vsasl = None
    for pld in common_plds:
        data = nib.load(vsasl_files[pld]).get_fdata(dtype=np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if data.ndim == 4:
            n_repeats_vsasl = data.shape[-1]
            vsasl_4rep.append(np.mean(data, axis=-1))
            vsasl_1rep.append(data[..., 0])
        else:
            n_repeats_vsasl = 1
            vsasl_4rep.append(data)
            vsasl_1rep.append(data)

    print(f"  Repeats: PCASL={n_repeats_pcasl}, VSASL={n_repeats_vsasl}")

    # Stack: (H, W, Z, n_plds)
    pcasl_4rep_stack = np.stack(pcasl_4rep, axis=-1)
    pcasl_1rep_stack = np.stack(pcasl_1rep, axis=-1)
    vsasl_4rep_stack = np.stack(vsasl_4rep, axis=-1)
    vsasl_1rep_stack = np.stack(vsasl_1rep, axis=-1)

    # BS correction
    if alpha_bs1 != 1.0:
        pcasl_corr = alpha_bs1 ** 4
        vsasl_corr = alpha_bs1 ** 3
        pcasl_4rep_stack /= pcasl_corr
        pcasl_1rep_stack /= pcasl_corr
        vsasl_4rep_stack /= vsasl_corr
        vsasl_1rep_stack /= vsasl_corr

    # Zero-pad missing PLDs
    missing_plds = sorted(set(model_plds) - set(common_plds))
    if missing_plds:
        print(f"  Zero-padding missing PLDs: {missing_plds}")
        H, W, Z = pcasl_4rep_stack.shape[:3]
        n_missing = len(missing_plds)
        pad = np.zeros((H, W, Z, n_missing), dtype=np.float64)
        pcasl_4rep_stack = np.concatenate([pcasl_4rep_stack, pad], axis=-1)
        pcasl_1rep_stack = np.concatenate([pcasl_1rep_stack, pad], axis=-1)
        vsasl_4rep_stack = np.concatenate([vsasl_4rep_stack, pad], axis=-1)
        vsasl_1rep_stack = np.concatenate([vsasl_1rep_stack, pad], axis=-1)

    # Concatenate PCASL + VSASL: (H, W, Z, 2*n_plds)
    signals_4rep = np.concatenate([pcasl_4rep_stack, vsasl_4rep_stack], axis=-1)
    signals_1rep = np.concatenate([pcasl_1rep_stack, vsasl_1rep_stack], axis=-1)

    # Brain mask from M0
    m0_files = list(subject_dir.glob('r_M0.nii*'))
    wb_mask_files = list(subject_dir.glob('M0_WBmask*.nii*'))
    if wb_mask_files:
        brain_mask = nib.load(wb_mask_files[0]).get_fdata(dtype=np.float64) > 0
    elif m0_files:
        m0_data = nib.load(m0_files[0]).get_fdata(dtype=np.float64)
        m0_data = np.nan_to_num(m0_data)
        threshold = np.percentile(m0_data[m0_data > 0], 50) * 0.3
        brain_mask = m0_data > threshold
    else:
        mean_sig = np.mean(np.abs(signals_4rep), axis=-1)
        threshold = np.percentile(mean_sig[mean_sig > 0], 10)
        brain_mask = mean_sig > threshold

    return signals_4rep, signals_1rep, brain_mask, ref_img, n_repeats_pcasl, n_repeats_vsasl


# ── NN Inference ──────────────────────────────────────────────────────────

def nn_predict_volume(signals, models, norm_stats, gsf, device, n_plds):
    """
    Run NN ensemble inference on a volume.

    Args:
        signals: (H, W, Z, 2*n_plds) - raw normdiff signals (BS-corrected)

    Returns:
        cbf_volume: (H, W, Z)
        att_volume: (H, W, Z)
    """
    # Transpose to (Z, 2*n_plds, H, W) and apply scaling
    spatial_stack = np.transpose(signals, (2, 3, 0, 1))  # (Z, 2*n_plds, H, W)
    spatial_stack = (spatial_stack * M0_SCALE_FACTOR * gsf).astype(np.float32)

    n_slices = spatial_stack.shape[0]

    all_cbf = []
    all_att = []

    for model in models:
        cbf_slices = []
        att_slices = []

        with torch.no_grad():
            for s in range(n_slices):
                inp = torch.from_numpy(spatial_stack[s:s + 1]).float().to(device)
                inp_padded, padding = pad_to_multiple(inp, 16)
                cbf_pred, att_pred, _, _ = model(inp_padded)
                cbf_pred = unpad(cbf_pred, padding)
                att_pred = unpad(att_pred, padding)
                cbf_slices.append(cbf_pred.cpu().numpy())
                att_slices.append(att_pred.cpu().numpy())

        cbf_vol = np.concatenate(cbf_slices, axis=0)  # (Z, 1, H, W)
        att_vol = np.concatenate(att_slices, axis=0)
        all_cbf.append(cbf_vol)
        all_att.append(att_vol)

    # Ensemble average
    cbf_ens = np.mean(all_cbf, axis=0)
    att_ens = np.mean(all_att, axis=0)

    # Denormalize
    cbf_denorm = cbf_ens * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att_denorm = att_ens * norm_stats['y_std_att'] + norm_stats['y_mean_att']

    cbf_denorm = np.clip(cbf_denorm, 0, 250)
    att_denorm = np.clip(att_denorm, 0, 5000)

    # Transpose to (H, W, Z)
    cbf_volume = np.transpose(cbf_denorm[:, 0, :, :], (1, 2, 0))
    att_volume = np.transpose(att_denorm[:, 0, :, :], (1, 2, 0))

    return cbf_volume, att_volume


# ── LS Fitting ────────────────────────────────────────────────────────────

def _fit_single_voxel_invivo(args):
    """Fit a single voxel using multi-start LS."""
    signal_1d, plds, alpha_bs1 = args
    n_plds = len(plds)
    pcasl = signal_1d[:n_plds]
    vsasl = signal_1d[n_plds:]

    pldti = np.column_stack([plds, plds])
    observed = np.column_stack([pcasl, vsasl])

    ls_params = {
        'T1_artery': T1_ARTERY, 'T_tau': T_TAU,
        'alpha_PCASL': ALPHA_PCASL, 'alpha_VSASL': ALPHA_VSASL,
        'T2_factor': T2_FACTOR, 'alpha_BS1': alpha_bs1,
        'T_sat_vs': T_SAT_VS,
    }

    try:
        init = get_grid_search_initial_guess(signal_1d, plds, ls_params)
        beta, _, rmse, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
            pldti, observed, init,
            T1_ARTERY, T_TAU, T2_FACTOR, alpha_bs1, ALPHA_PCASL, ALPHA_VSASL,
            T_SAT_VS
        )
        cbf = beta[0] * 6000.0
        att = beta[1]
        if np.isfinite(cbf) and np.isfinite(att) and 0 <= cbf <= 300 and 0 <= att <= 5000:
            return cbf, att, rmse
    except Exception:
        pass
    return np.nan, np.nan, np.nan


def ls_fit_volume(signals, brain_mask, plds, alpha_bs1, subsample_frac=0.1):
    """
    Run LS fitting on brain voxels (with optional subsampling).

    Args:
        signals: (H, W, Z, 2*n_plds) - raw normdiff signals (BS-corrected)
        brain_mask: (H, W, Z)
        plds: array of PLD values
        alpha_bs1: BS efficiency for LS fitting
        subsample_frac: fraction of brain voxels to fit (1.0 = all)

    Returns:
        cbf_volume: (H, W, Z) - NaN where not fitted
        att_volume: (H, W, Z) - NaN where not fitted
    """
    H, W, Z = brain_mask.shape
    n_plds_val = len(plds)
    plds_arr = np.array(plds, dtype=np.float64)

    brain_idx = np.argwhere(brain_mask > 0)
    n_brain = len(brain_idx)

    # Subsample
    if subsample_frac < 1.0:
        n_sample = max(100, int(n_brain * subsample_frac))
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(n_brain, size=min(n_sample, n_brain), replace=False)
        brain_idx_sub = brain_idx[sample_indices]
    else:
        brain_idx_sub = brain_idx

    print(f"    LS fitting {len(brain_idx_sub)} of {n_brain} brain voxels "
          f"({len(brain_idx_sub)/n_brain*100:.0f}%)")

    # Prepare per-voxel tasks
    tasks = []
    for idx in brain_idx_sub:
        i, j, k = idx
        voxel_signal = signals[i, j, k, :].astype(np.float64)
        tasks.append((voxel_signal, plds_arr, alpha_bs1))

    # Parallel fitting
    n_workers = min(8, os.cpu_count() or 1)
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_fit_single_voxel_invivo, tasks)
    elapsed = time.time() - t0
    print(f"    LS fitting complete: {elapsed:.1f}s ({elapsed/len(tasks)*1000:.1f} ms/voxel)")

    # Fill volumes
    cbf_volume = np.full((H, W, Z), np.nan)
    att_volume = np.full((H, W, Z), np.nan)

    n_valid = 0
    for idx, (cbf, att, rmse) in zip(brain_idx_sub, results):
        i, j, k = idx
        if np.isfinite(cbf):
            cbf_volume[i, j, k] = cbf
            att_volume[i, j, k] = att
            n_valid += 1

    print(f"    Valid fits: {n_valid}/{len(brain_idx_sub)} ({n_valid/len(brain_idx_sub)*100:.0f}%)")
    return cbf_volume, att_volume


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_comparison_metrics(pred, ref, mask):
    """
    Compare pred vs ref within mask where both are valid.

    Returns dict with MAE, RMSE, bias, Pearson r.
    """
    valid = mask & np.isfinite(pred) & np.isfinite(ref)
    if valid.sum() < 10:
        return {'mae': np.nan, 'rmse': np.nan, 'bias': np.nan,
                'pearson_r': np.nan, 'n_voxels': int(valid.sum())}

    p = pred[valid]
    r = ref[valid]

    mae = np.mean(np.abs(p - r))
    rmse = np.sqrt(np.mean((p - r) ** 2))
    bias = np.mean(p - r)

    if np.std(p) > 1e-10 and np.std(r) > 1e-10:
        pearson_r = np.corrcoef(p, r)[0, 1]
    else:
        pearson_r = np.nan

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'bias': float(bias),
        'pearson_r': float(pearson_r),
        'n_voxels': int(valid.sum()),
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────

def process_subject(subject_dir, models, norm_stats, gsf, model_plds,
                    device, alpha_bs1, ls_subsample):
    """Process a single subject: NN and LS on 1-rep and 4-rep data."""
    sid = subject_dir.name
    print(f"\n--- Subject: {sid} ---")

    # Load data
    signals_4rep, signals_1rep, brain_mask, ref_img, n_rep_pc, n_rep_vs = \
        load_subject_data(subject_dir, model_plds, alpha_bs1)

    n_plds = len(model_plds)
    results = {'subject': sid, 'n_repeats_pcasl': n_rep_pc, 'n_repeats_vsasl': n_rep_vs}

    # 1. NN on 1-rep
    print("  NN(1-rep)...")
    t0 = time.time()
    nn_1rep_cbf, nn_1rep_att = nn_predict_volume(
        signals_1rep, models, norm_stats, gsf, device, n_plds)
    results['nn_1rep_time'] = time.time() - t0

    # 2. NN on 4-rep
    print("  NN(4-rep)...")
    t0 = time.time()
    nn_4rep_cbf, nn_4rep_att = nn_predict_volume(
        signals_4rep, models, norm_stats, gsf, device, n_plds)
    results['nn_4rep_time'] = time.time() - t0

    # 3. LS on 4-rep (reference)
    print("  LS(4-rep) [reference]...")
    t0 = time.time()
    ls_4rep_cbf, ls_4rep_att = ls_fit_volume(
        signals_4rep, brain_mask, model_plds, alpha_bs1, subsample_frac=ls_subsample)
    results['ls_4rep_time'] = time.time() - t0

    # 4. LS on 1-rep
    print("  LS(1-rep)...")
    t0 = time.time()
    ls_1rep_cbf, ls_1rep_att = ls_fit_volume(
        signals_1rep, brain_mask, model_plds, alpha_bs1, subsample_frac=ls_subsample)
    results['ls_1rep_time'] = time.time() - t0

    # Comparison mask: brain voxels where LS(4rep) reference is valid
    ref_mask = brain_mask & np.isfinite(ls_4rep_cbf) & np.isfinite(ls_4rep_att)
    results['n_ref_voxels'] = int(ref_mask.sum())

    # Compute metrics vs LS(4-rep) reference
    for method_name, cbf_pred, att_pred in [
        ('nn_1rep', nn_1rep_cbf, nn_1rep_att),
        ('nn_4rep', nn_4rep_cbf, nn_4rep_att),
        ('ls_1rep', ls_1rep_cbf, ls_1rep_att),
    ]:
        cbf_metrics = compute_comparison_metrics(cbf_pred, ls_4rep_cbf, ref_mask)
        att_metrics = compute_comparison_metrics(att_pred, ls_4rep_att, ref_mask)
        results[f'{method_name}_cbf'] = cbf_metrics
        results[f'{method_name}_att'] = att_metrics

    # Win rates: NN(1rep) vs LS(1rep), using LS(4rep) as reference
    valid = ref_mask & np.isfinite(ls_1rep_cbf) & np.isfinite(nn_1rep_cbf)
    if valid.sum() > 0:
        nn_cbf_err = np.abs(nn_1rep_cbf[valid] - ls_4rep_cbf[valid])
        ls_cbf_err = np.abs(ls_1rep_cbf[valid] - ls_4rep_cbf[valid])
        nn_att_err = np.abs(nn_1rep_att[valid] - ls_4rep_att[valid])
        ls_att_err = np.abs(ls_1rep_att[valid] - ls_4rep_att[valid])

        cbf_win_rate = float(np.mean(nn_cbf_err < ls_cbf_err) * 100)
        att_win_rate = float(np.mean(nn_att_err < ls_att_err) * 100)
        results['win_rate_cbf'] = cbf_win_rate
        results['win_rate_att'] = att_win_rate
        results['n_win_comparisons'] = int(valid.sum())
    else:
        results['win_rate_cbf'] = np.nan
        results['win_rate_att'] = np.nan
        results['n_win_comparisons'] = 0

    # Summary stats for reference CBF/ATT
    ref_cbf_vals = ls_4rep_cbf[ref_mask]
    ref_att_vals = ls_4rep_att[ref_mask]
    results['ref_cbf_stats'] = {
        'mean': float(np.mean(ref_cbf_vals)),
        'std': float(np.std(ref_cbf_vals)),
        'median': float(np.median(ref_cbf_vals)),
    }
    results['ref_att_stats'] = {
        'mean': float(np.mean(ref_att_vals)),
        'std': float(np.std(ref_att_vals)),
        'median': float(np.median(ref_att_vals)),
    }

    # Save maps for figure generation
    maps = {
        'nn_1rep_cbf': nn_1rep_cbf,
        'nn_1rep_att': nn_1rep_att,
        'nn_4rep_cbf': nn_4rep_cbf,
        'nn_4rep_att': nn_4rep_att,
        'ls_4rep_cbf': ls_4rep_cbf,
        'ls_4rep_att': ls_4rep_att,
        'ls_1rep_cbf': ls_1rep_cbf,
        'ls_1rep_att': ls_1rep_att,
        'brain_mask': brain_mask.astype(np.float32),
    }

    print(f"  Reference: CBF={results['ref_cbf_stats']['mean']:.1f} +/- {results['ref_cbf_stats']['std']:.1f}, "
          f"ATT={results['ref_att_stats']['mean']:.0f} +/- {results['ref_att_stats']['std']:.0f}")

    if not np.isnan(results.get('win_rate_cbf', np.nan)):
        print(f"  Win rates (NN vs LS, 1-rep): CBF={results['win_rate_cbf']:.1f}%, "
              f"ATT={results['win_rate_att']:.1f}% (n={results['n_win_comparisons']})")

    for method in ['nn_1rep', 'nn_4rep', 'ls_1rep']:
        cbf_m = results[f'{method}_cbf']
        att_m = results[f'{method}_att']
        print(f"  {method}: CBF MAE={cbf_m['mae']:.2f}, r={cbf_m['pearson_r']:.3f} | "
              f"ATT MAE={att_m['mae']:.0f}, r={att_m['pearson_r']:.3f}")

    return results, maps


def main():
    parser = argparse.ArgumentParser(
        description="In-vivo repeat comparison: NN(1-rep) vs LS(1-rep) vs LS(4-rep)")
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--invivo-dir', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'invivo_validated'),
                        help='Directory containing subject folders with NIfTI data')
    parser.add_argument('--output-dir', type=str,
                        default=str(PROJECT_ROOT / 'amplitude_ablation_v7' / 'v7_evaluation_results'),
                        help='Output directory for results')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='Specific subject IDs to process (default: all)')
    parser.add_argument('--alpha-bs1', type=float, default=0.93,
                        help='Background suppression efficiency (default: 0.93)')
    parser.add_argument('--ls-subsample', type=float, default=0.1,
                        help='Fraction of brain voxels for LS fitting (default: 0.1)')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device: 'cuda', 'mps', 'cpu', or 'auto'")

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model_dir = Path(args.model_dir)
    models, norm_stats, gsf, model_plds = load_model(model_dir, device)

    # Find subjects
    invivo_dir = Path(args.invivo_dir)
    subject_dirs = sorted([d for d in invivo_dir.iterdir()
                           if d.is_dir() and not d.name.startswith('.')])

    if args.subjects:
        subject_dirs = [d for d in subject_dirs if d.name in args.subjects]

    print(f"Found {len(subject_dirs)} subjects in {invivo_dir}")
    print(f"Model PLDs: {model_plds}")
    print(f"alpha_BS1: {args.alpha_bs1}")
    print(f"LS subsample: {args.ls_subsample}")

    # Process subjects
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for subject_dir in subject_dirs:
        try:
            results, maps = process_subject(
                subject_dir, models, norm_stats, gsf, model_plds,
                device, args.alpha_bs1, args.ls_subsample)
            all_results.append(results)

            # Save per-subject maps
            subject_output = output_dir / 'invivo_maps' / results['subject']
            subject_output.mkdir(parents=True, exist_ok=True)
            for name, arr in maps.items():
                np.save(subject_output / f'{name}.npy', arr)

        except Exception as e:
            print(f"\nERROR processing {subject_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No subjects processed successfully.")
        return

    # Aggregate summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    summary = {'subjects': [], 'aggregate': {}}

    for r in all_results:
        summary['subjects'].append(r)

    # Compute mean metrics across subjects
    methods = ['nn_1rep', 'nn_4rep', 'ls_1rep']
    for method in methods:
        for param in ['cbf', 'att']:
            key = f'{method}_{param}'
            maes = [r[key]['mae'] for r in all_results if not np.isnan(r[key].get('mae', np.nan))]
            rs = [r[key]['pearson_r'] for r in all_results if not np.isnan(r[key].get('pearson_r', np.nan))]
            biases = [r[key]['bias'] for r in all_results if not np.isnan(r[key].get('bias', np.nan))]

            summary['aggregate'][key] = {
                'mean_mae': float(np.mean(maes)) if maes else np.nan,
                'std_mae': float(np.std(maes)) if maes else np.nan,
                'mean_pearson_r': float(np.mean(rs)) if rs else np.nan,
                'mean_bias': float(np.mean(biases)) if biases else np.nan,
            }

    # Win rates
    cbf_wrs = [r['win_rate_cbf'] for r in all_results if not np.isnan(r.get('win_rate_cbf', np.nan))]
    att_wrs = [r['win_rate_att'] for r in all_results if not np.isnan(r.get('win_rate_att', np.nan))]
    summary['aggregate']['mean_cbf_win_rate'] = float(np.mean(cbf_wrs)) if cbf_wrs else np.nan
    summary['aggregate']['mean_att_win_rate'] = float(np.mean(att_wrs)) if att_wrs else np.nan

    # Timing
    for method in ['nn_1rep', 'nn_4rep', 'ls_1rep', 'ls_4rep']:
        times = [r[f'{method}_time'] for r in all_results if f'{method}_time' in r]
        summary['aggregate'][f'{method}_mean_time'] = float(np.mean(times)) if times else np.nan

    # Print summary
    print(f"\nMethods compared to LS(4-rep) reference:")
    for method in methods:
        cbf = summary['aggregate'][f'{method}_cbf']
        att = summary['aggregate'][f'{method}_att']
        print(f"  {method}: CBF MAE={cbf['mean_mae']:.2f}+/-{cbf['std_mae']:.2f}, "
              f"r={cbf['mean_pearson_r']:.3f} | "
              f"ATT MAE={att['mean_mae']:.0f}+/-{att['std_mae']:.0f}, "
              f"r={att['mean_pearson_r']:.3f}")

    if not np.isnan(summary['aggregate'].get('mean_cbf_win_rate', np.nan)):
        print(f"\nWin rates (NN(1rep) vs LS(1rep)):")
        print(f"  CBF: {summary['aggregate']['mean_cbf_win_rate']:.1f}%")
        print(f"  ATT: {summary['aggregate']['mean_att_win_rate']:.1f}%")

    print(f"\nTiming (mean per subject):")
    for method in ['nn_1rep', 'nn_4rep', 'ls_1rep', 'ls_4rep']:
        t = summary['aggregate'].get(f'{method}_mean_time', np.nan)
        if not np.isnan(t):
            print(f"  {method}: {t:.1f}s")

    # Save results
    results_path = output_dir / 'invivo_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()

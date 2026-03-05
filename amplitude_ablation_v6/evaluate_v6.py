#!/usr/bin/env python3
"""
Comprehensive V6 Evaluation Pipeline
=====================================
Runs simulated bias/CoV analysis, in-vivo inference, and generates publication figures.

Usage:
    python3 amplitude_ablation_v6/evaluate_v6.py
"""

import sys
import os
import json
import time
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.asl_simulation import ASLParameters, ASLSimulator, _generate_pcasl_signal_jit, _generate_vsasl_signal_jit
from models.spatial_asl_network import SpatialASLNet
from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
from utils.helpers import get_grid_search_initial_guess
from baselines.multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

# ── Constants ──────────────────────────────────────────────────────────────
PLDS = np.array([500, 1000, 1500, 2000, 2500], dtype=np.float64)
T1_ARTERY = 1650.0
T_TAU = 1800.0
ALPHA_PCASL = 0.85
ALPHA_VSASL = 0.56
ALPHA_BS1 = 1.0
T2_FACTOR = 1.0
T_SAT_VS = 2000.0
PHANTOM_SIZE = 64
MASK_RADIUS = 28

OUTPUT_DIR = PROJECT_ROOT / 'amplitude_ablation_v6' / 'v6_evaluation_results'
V6_DIR = PROJECT_ROOT / 'amplitude_ablation_v6'

# ── Device ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


# ═══════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(run_dir, device=DEVICE):
    """Load a spatial model from a run directory."""
    run_dir = Path(run_dir)

    with open(run_dir / 'config.yaml') as f:
        full_config = yaml.safe_load(f)
    training_config = full_config.get('training', {})
    data_config = full_config.get('data', {})

    with open(run_dir / 'norm_stats.json') as f:
        norm_stats = json.load(f)

    model_class_name = training_config.get('model_class_name', 'SpatialASLNet')
    hidden_sizes = training_config.get('hidden_sizes', [32, 64, 128, 256])
    n_plds = len(data_config.get('pld_values', PLDS))

    model_files = sorted((run_dir / 'trained_models').glob('ensemble_model_*.pt'))

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

    gsf = data_config.get('global_scale_factor', 10.0)
    return models, norm_stats, gsf, model_class_name


# ═══════════════════════════════════════════════════════════════════════════
# Signal Generation & Phantom
# ═══════════════════════════════════════════════════════════════════════════

def make_brain_mask(size=PHANTOM_SIZE, radius=MASK_RADIUS):
    yy, xx = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    return ((xx - cx)**2 + (yy - cy)**2 <= radius**2).astype(np.float32)


def generate_clean_phantom(cbf, att, mask):
    """Generate clean 10-channel phantom (5 PCASL + 5 VSASL)."""
    cbf_cgs = cbf / 6000.0
    alpha1 = ALPHA_PCASL * (ALPHA_BS1**4)
    alpha2 = ALPHA_VSASL * (ALPHA_BS1**3)

    pcasl = _generate_pcasl_signal_jit(PLDS, att, cbf_cgs, T1_ARTERY, T_TAU, alpha1, T2_FACTOR)
    vsasl = _generate_vsasl_signal_jit(PLDS, att, cbf_cgs, T1_ARTERY, alpha2, T2_FACTOR, T_SAT_VS)

    n_plds = len(PLDS)
    signals = np.zeros((n_plds * 2, PHANTOM_SIZE, PHANTOM_SIZE), dtype=np.float32)
    for c in range(n_plds):
        signals[c] = pcasl[c] * mask
        signals[n_plds + c] = vsasl[c] * mask
    return signals


def generate_clean_1d(cbf, att):
    """Generate clean 1D signal for LS fitting."""
    cbf_cgs = cbf / 6000.0
    alpha1 = ALPHA_PCASL * (ALPHA_BS1**4)
    alpha2 = ALPHA_VSASL * (ALPHA_BS1**3)
    pcasl = _generate_pcasl_signal_jit(PLDS, att, cbf_cgs, T1_ARTERY, T_TAU, alpha1, T2_FACTOR)
    vsasl = _generate_vsasl_signal_jit(PLDS, att, cbf_cgs, T1_ARTERY, alpha2, T2_FACTOR, T_SAT_VS)
    return np.concatenate([pcasl, vsasl])


# ═══════════════════════════════════════════════════════════════════════════
# NN Inference
# ═══════════════════════════════════════════════════════════════════════════

def nn_inference(clean_signals, mask, models, norm_stats, gsf, snr, n_phantoms, rng):
    """Run NN on n_phantoms noise realizations, return (cbf_preds, att_preds) per brain voxel."""
    sim = ASLSimulator(ASLParameters(T1_artery=T1_ARTERY, T_tau=T_TAU,
                                     alpha_PCASL=ALPHA_PCASL, alpha_VSASL=ALPHA_VSASL))
    noise_sd = sim._compute_reference_signal() / snr
    brain_idx = mask > 0

    all_cbf, all_att = [], []

    for _ in range(n_phantoms):
        noise = noise_sd * rng.randn(*clean_signals.shape).astype(np.float32)
        noisy = clean_signals + noise
        noisy_norm = noisy * 100.0 * gsf  # M0_SCALE * global_scale

        inp = torch.from_numpy(noisy_norm[np.newaxis]).float().to(DEVICE)

        with torch.no_grad():
            cbf_maps, att_maps = [], []
            for model in models:
                cbf_pred, att_pred, _, _ = model(inp)
                cbf_d = cbf_pred * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
                att_d = att_pred * norm_stats['y_std_att'] + norm_stats['y_mean_att']
                cbf_d = torch.clamp(cbf_d, 0.0, 250.0)
                att_d = torch.clamp(att_d, 0.0, 5000.0)
                cbf_maps.append(cbf_d.cpu().numpy())
                att_maps.append(att_d.cpu().numpy())

            cbf_ens = np.mean(cbf_maps, axis=0)[0, 0]
            att_ens = np.mean(att_maps, axis=0)[0, 0]

        all_cbf.append(cbf_ens[brain_idx])
        all_att.append(att_ens[brain_idx])

    return np.concatenate(all_cbf), np.concatenate(all_att)


# ═══════════════════════════════════════════════════════════════════════════
# LS Fitting
# ═══════════════════════════════════════════════════════════════════════════

def _fit_single_voxel(args):
    signal_1d, plds_flat, pldti = args
    ls_params = {
        'T1_artery': T1_ARTERY, 'T_tau': T_TAU,
        'alpha_PCASL': ALPHA_PCASL, 'alpha_VSASL': ALPHA_VSASL,
        'T2_factor': T2_FACTOR, 'alpha_BS1': ALPHA_BS1,
    }
    try:
        init = get_grid_search_initial_guess(signal_1d, plds_flat, ls_params)
        signal_reshaped = signal_1d.reshape((len(plds_flat), 2), order='F')
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
            pldti, signal_reshaped, init,
            T1_ARTERY, T_TAU, T2_FACTOR, ALPHA_BS1, ALPHA_PCASL, ALPHA_VSASL
        )
        cbf = beta[0] * 6000.0
        att = beta[1]
        if np.isfinite(cbf) and np.isfinite(att):
            return cbf, att
    except Exception:
        pass
    return None


def ls_inference(clean_1d, snr, n_realizations, rng):
    """Run LS fitting over n_realizations noise realizations."""
    sim = ASLSimulator(ASLParameters(T1_artery=T1_ARTERY, T_tau=T_TAU,
                                     alpha_PCASL=ALPHA_PCASL, alpha_VSASL=ALPHA_VSASL))
    noise_sd = sim._compute_reference_signal() / snr
    plds_flat = PLDS.copy()
    pldti = np.column_stack([plds_flat, plds_flat])

    tasks = []
    for _ in range(n_realizations):
        noise = noise_sd * rng.randn(len(clean_1d)).astype(np.float64)
        tasks.append((clean_1d + noise, plds_flat, pldti))

    n_workers = min(8, os.cpu_count() or 1)
    with Pool(n_workers) as pool:
        results = pool.map(_fit_single_voxel, tasks)

    cbf_list, att_list = [], []
    for r in results:
        if r is not None:
            cbf_list.append(r[0])
            att_list.append(r[1])
    return np.array(cbf_list), np.array(att_list)


def compute_bias_cov(preds, true_val):
    if len(preds) == 0:
        return np.nan, np.nan
    bias = np.mean(preds) - true_val
    cov = np.std(preds) / abs(true_val) * 100.0 if true_val != 0 else np.nan
    return bias, cov


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Bias/CoV Analysis
# ═══════════════════════════════════════════════════════════════════════════

def run_bias_cov_analysis(model_info_list, snr_levels=[3.0, 5.0, 10.0],
                          n_phantoms=5, n_ls_realizations=200, seed=42):
    """Run bias/CoV sweeps across ATT and CBF for all models + LS."""
    print("\n" + "="*70)
    print("PART 1: BIAS / COEFFICIENT OF VARIATION ANALYSIS")
    print("="*70)

    mask = make_brain_mask()

    att_sweep = np.arange(500, 3001, 250)
    cbf_sweep = np.arange(20, 201, 10)
    fixed_cbf = 50.0
    fixed_att = 1500.0

    all_results = {}

    for snr in snr_levels:
        print(f"\n--- SNR = {snr} ---")
        rng = np.random.RandomState(seed + int(snr * 10))
        snr_key = f'snr_{snr}'
        all_results[snr_key] = {'sweep_att': {}, 'sweep_cbf': {}}

        # Sweep A: vary ATT, fixed CBF
        print(f"  Sweep A: CBF={fixed_cbf}, ATT={att_sweep[0]}-{att_sweep[-1]}ms")
        for minfo in model_info_list:
            name = minfo['name']
            print(f"    NN: {name}...", end='', flush=True)
            t0 = time.time()
            cbf_b, cbf_c, att_b, att_c = [], [], [], []
            for att_val in att_sweep:
                clean = generate_clean_phantom(fixed_cbf, att_val, mask)
                cbf_p, att_p = nn_inference(clean, mask, minfo['models'], minfo['norm_stats'],
                                            minfo['gsf'], snr, n_phantoms, rng)
                cb, cc = compute_bias_cov(cbf_p, fixed_cbf)
                ab, ac = compute_bias_cov(att_p, att_val)
                cbf_b.append(cb); cbf_c.append(cc)
                att_b.append(ab); att_c.append(ac)
            all_results[snr_key]['sweep_att'][name] = {
                'cbf_bias': cbf_b, 'cbf_cov': cbf_c,
                'att_bias': att_b, 'att_cov': att_c,
            }
            print(f" ({time.time()-t0:.0f}s)")

        # LS sweep A
        print(f"    LS: Least Squares...", end='', flush=True)
        t0 = time.time()
        ls_cb, ls_cc, ls_ab, ls_ac = [], [], [], []
        for att_val in att_sweep:
            clean_1d = generate_clean_1d(fixed_cbf, att_val)
            cbf_p, att_p = ls_inference(clean_1d, snr, n_ls_realizations, rng)
            cb, cc = compute_bias_cov(cbf_p, fixed_cbf)
            ab, ac = compute_bias_cov(att_p, att_val)
            ls_cb.append(cb); ls_cc.append(cc)
            ls_ab.append(ab); ls_ac.append(ac)
        all_results[snr_key]['sweep_att']['Least Squares'] = {
            'cbf_bias': ls_cb, 'cbf_cov': ls_cc,
            'att_bias': ls_ab, 'att_cov': ls_ac,
        }
        print(f" ({time.time()-t0:.0f}s)")

        # Sweep B: vary CBF, fixed ATT
        print(f"  Sweep B: ATT={fixed_att}, CBF={cbf_sweep[0]}-{cbf_sweep[-1]}")
        for minfo in model_info_list:
            name = minfo['name']
            print(f"    NN: {name}...", end='', flush=True)
            t0 = time.time()
            cbf_b, cbf_c, att_b, att_c = [], [], [], []
            for cbf_val in cbf_sweep:
                clean = generate_clean_phantom(cbf_val, fixed_att, mask)
                cbf_p, att_p = nn_inference(clean, mask, minfo['models'], minfo['norm_stats'],
                                            minfo['gsf'], snr, n_phantoms, rng)
                cb, cc = compute_bias_cov(cbf_p, cbf_val)
                ab, ac = compute_bias_cov(att_p, fixed_att)
                cbf_b.append(cb); cbf_c.append(cc)
                att_b.append(ab); att_c.append(ac)
            all_results[snr_key]['sweep_cbf'][name] = {
                'cbf_bias': cbf_b, 'cbf_cov': cbf_c,
                'att_bias': att_b, 'att_cov': att_c,
            }
            print(f" ({time.time()-t0:.0f}s)")

        # LS sweep B
        print(f"    LS: Least Squares...", end='', flush=True)
        t0 = time.time()
        ls_cb, ls_cc, ls_ab, ls_ac = [], [], [], []
        for cbf_val in cbf_sweep:
            clean_1d = generate_clean_1d(cbf_val, fixed_att)
            cbf_p, att_p = ls_inference(clean_1d, snr, n_ls_realizations, rng)
            cb, cc = compute_bias_cov(cbf_p, cbf_val)
            ab, ac = compute_bias_cov(att_p, fixed_att)
            ls_cb.append(cb); ls_cc.append(cc)
            ls_ab.append(ab); ls_ac.append(ac)
        all_results[snr_key]['sweep_cbf']['Least Squares'] = {
            'cbf_bias': ls_cb, 'cbf_cov': ls_cc,
            'att_bias': ls_ab, 'att_cov': ls_ac,
        }
        print(f" ({time.time()-t0:.0f}s)")

    return all_results, att_sweep, cbf_sweep, fixed_cbf, fixed_att


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Win Rate Analysis
# ═══════════════════════════════════════════════════════════════════════════

def run_win_rate_analysis(model_info_list, snr_levels=[3.0, 5.0, 10.0, 15.0, 25.0],
                          n_test_phantoms=20, seed=123):
    """Generate diverse phantoms and compare NN vs LS voxel-by-voxel."""
    print("\n" + "="*70)
    print("PART 2: WIN RATE ANALYSIS (NN vs LS)")
    print("="*70)

    mask = make_brain_mask()
    brain_idx = mask > 0
    n_brain = int(brain_idx.sum())

    # Generate diverse ground truth values
    rng = np.random.RandomState(seed)
    test_cbf_values = rng.uniform(20, 120, n_test_phantoms)
    test_att_values = rng.uniform(500, 2500, n_test_phantoms)

    results = {}

    for snr in snr_levels:
        print(f"\n--- SNR = {snr} ---")
        rng_noise = np.random.RandomState(seed + int(snr * 100))
        sim = ASLSimulator(ASLParameters(T1_artery=T1_ARTERY, T_tau=T_TAU,
                                         alpha_PCASL=ALPHA_PCASL, alpha_VSASL=ALPHA_VSASL))
        noise_sd = sim._compute_reference_signal() / snr

        snr_results = {}

        for minfo in model_info_list:
            name = minfo['name']
            nn_cbf_wins = 0
            nn_att_wins = 0
            total = 0

            for i in range(n_test_phantoms):
                cbf_true = test_cbf_values[i]
                att_true = test_att_values[i]

                # NN inference
                clean = generate_clean_phantom(cbf_true, att_true, mask)
                noise = noise_sd * rng_noise.randn(*clean.shape).astype(np.float32)
                noisy = clean + noise
                noisy_norm = noisy * 100.0 * minfo['gsf']

                inp = torch.from_numpy(noisy_norm[np.newaxis]).float().to(DEVICE)
                with torch.no_grad():
                    cbf_pred, att_pred, _, _ = minfo['models'][0](inp)
                    nn_cbf = (cbf_pred * minfo['norm_stats']['y_std_cbf'] + minfo['norm_stats']['y_mean_cbf']).cpu().numpy()[0, 0]
                    nn_att = (att_pred * minfo['norm_stats']['y_std_att'] + minfo['norm_stats']['y_mean_att']).cpu().numpy()[0, 0]
                    nn_cbf = np.clip(nn_cbf, 0, 250)
                    nn_att = np.clip(nn_att, 0, 5000)

                # LS inference on same noisy data (subsample brain voxels)
                noisy_flat = noisy[:, brain_idx]
                pldti = np.column_stack([PLDS, PLDS])
                ls_params = {'T1_artery': T1_ARTERY, 'T_tau': T_TAU,
                             'alpha_PCASL': ALPHA_PCASL, 'alpha_VSASL': ALPHA_VSASL,
                             'T2_factor': T2_FACTOR, 'alpha_BS1': ALPHA_BS1}

                # Subsample for speed
                subsample = max(1, n_brain // 50)
                for vi in range(0, n_brain, subsample):
                    voxel_signal = noisy_flat[:, vi].astype(np.float64)
                    try:
                        init = get_grid_search_initial_guess(voxel_signal, PLDS, ls_params)
                        sig_r = voxel_signal.reshape((len(PLDS), 2), order='F')
                        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                            pldti, sig_r, init,
                            T1_ARTERY, T_TAU, T2_FACTOR, ALPHA_BS1, ALPHA_PCASL, ALPHA_VSASL
                        )
                        ls_cbf = beta[0] * 6000.0
                        ls_att = beta[1]
                    except Exception:
                        continue

                    if not (np.isfinite(ls_cbf) and np.isfinite(ls_att)):
                        continue

                    nn_cbf_err = abs(nn_cbf[brain_idx][vi] - cbf_true)
                    nn_att_err = abs(nn_att[brain_idx][vi] - att_true)
                    ls_cbf_err = abs(ls_cbf - cbf_true)
                    ls_att_err = abs(ls_att - att_true)

                    if nn_cbf_err < ls_cbf_err:
                        nn_cbf_wins += 1
                    if nn_att_err < ls_att_err:
                        nn_att_wins += 1
                    total += 1

            cbf_wr = nn_cbf_wins / total * 100 if total > 0 else 0
            att_wr = nn_att_wins / total * 100 if total > 0 else 0
            snr_results[name] = {'cbf_win_rate': cbf_wr, 'att_win_rate': att_wr, 'n_comparisons': total}
            print(f"  {name}: CBF WR={cbf_wr:.1f}%, ATT WR={att_wr:.1f}% (n={total})")

        results[f'snr_{snr}'] = snr_results

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: In-Vivo Inference
# ═══════════════════════════════════════════════════════════════════════════

def run_invivo_inference(model_info_list, alpha_bs1=0.93):
    """Run in-vivo inference using preprocessed numpy data."""
    print("\n" + "="*70)
    print("PART 3: IN-VIVO INFERENCE")
    print("="*70)

    invivo_dir = PROJECT_ROOT / 'data' / 'invivo_validated'
    subjects = sorted([d for d in invivo_dir.iterdir()
                       if d.is_dir() and not d.name.startswith('.')])
    print(f"Found {len(subjects)} subjects in {invivo_dir}")

    import nibabel as nib
    import re

    all_invivo_results = {}

    for minfo in model_info_list:
        name = minfo['name']
        model_plds = [500, 1000, 1500, 2000, 2500]
        print(f"\n--- {name} ---")

        subject_results = {}
        for subject_dir in subjects:
            sid = subject_dir.name
            try:
                # Find PCASL and VSASL normdiff files
                pcasl_files = sorted(subject_dir.glob('r_normdiff_alldyn_PCASL_*.nii*'),
                                     key=lambda f: int(re.search(r'_(\d+)', f.name).group(1)))
                vsasl_files = sorted(subject_dir.glob('r_normdiff_alldyn_VSASL_*.nii*'),
                                     key=lambda f: int(re.search(r'_(\d+)', f.name).group(1)))

                if not pcasl_files or not vsasl_files:
                    print(f"  {sid}: SKIP (missing files)")
                    continue

                pcasl_plds = [int(re.search(r'_(\d+)', f.name).group(1)) for f in pcasl_files]
                vsasl_plds = [int(re.search(r'_(\d+)', f.name).group(1)) for f in vsasl_files]
                common_plds = sorted(set(pcasl_plds) & set(vsasl_plds) & set(model_plds))

                if len(common_plds) < 3:
                    print(f"  {sid}: SKIP (only {len(common_plds)} matching PLDs)")
                    continue

                # Load and average data
                pcasl_vols = []
                for pld in common_plds:
                    f = [x for x in pcasl_files if f'_{pld}.' in x.name or x.name.endswith(f'_{pld}')][0]
                    data = nib.load(f).get_fdata(dtype=np.float64)
                    data = np.nan_to_num(data)
                    if data.ndim == 4:
                        data = np.mean(data, axis=-1)
                    pcasl_vols.append(data)

                vsasl_vols = []
                for pld in common_plds:
                    f = [x for x in vsasl_files if f'_{pld}.' in x.name or x.name.endswith(f'_{pld}')][0]
                    data = nib.load(f).get_fdata(dtype=np.float64)
                    data = np.nan_to_num(data)
                    if data.ndim == 4:
                        data = np.mean(data, axis=-1)
                    vsasl_vols.append(data)

                ref_img = nib.load(pcasl_files[0])

                # BS correction
                pcasl_stack = np.stack(pcasl_vols, axis=-1) / (alpha_bs1 ** 4)
                vsasl_stack = np.stack(vsasl_vols, axis=-1) / (alpha_bs1 ** 3)

                # Zero-pad missing PLDs
                missing = sorted(set(model_plds) - set(common_plds))
                if missing:
                    H, W, Z = pcasl_stack.shape[:3]
                    n_m = len(missing)
                    pcasl_stack = np.concatenate([pcasl_stack, np.zeros((H,W,Z,n_m))], axis=-1)
                    vsasl_stack = np.concatenate([vsasl_stack, np.zeros((H,W,Z,n_m))], axis=-1)

                combined = np.concatenate([pcasl_stack, vsasl_stack], axis=-1)  # (H,W,Z, 2*n_plds)
                spatial_stack = np.transpose(combined, (2, 3, 0, 1))  # (Z, 2*n_plds, H, W)

                # Apply normalization
                spatial_stack = (spatial_stack * 100.0 * minfo['gsf']).astype(np.float32)

                # Brain mask from M0
                m0_files = list(subject_dir.glob('r_M0.nii*'))
                if m0_files:
                    m0_data = nib.load(m0_files[0]).get_fdata(dtype=np.float64)
                    m0_data = np.nan_to_num(m0_data)
                    brain_mask = m0_data > (np.percentile(m0_data[m0_data > 0], 50) * 0.3)
                else:
                    mean_sig = np.mean(np.abs(spatial_stack), axis=(0, 1))
                    brain_mask = mean_sig > np.percentile(mean_sig[mean_sig > 0], 10)

                # Run inference slice-by-slice
                n_slices = spatial_stack.shape[0]
                cbf_slices, att_slices = [], []

                with torch.no_grad():
                    for s in range(n_slices):
                        inp = torch.from_numpy(spatial_stack[s:s+1]).float().to(DEVICE)
                        # Pad to multiple of 16
                        _, _, h, w = inp.shape
                        ph = (16 - h % 16) % 16
                        pw = (16 - w % 16) % 16
                        if ph > 0 or pw > 0:
                            inp = torch.nn.functional.pad(inp, (pw//2, pw-pw//2, ph//2, ph-ph//2), mode='reflect')

                        cbf_pred, att_pred, _, _ = minfo['models'][0](inp)

                        # Unpad
                        if ph > 0 or pw > 0:
                            cbf_pred = cbf_pred[:, :, ph//2:ph//2+h, pw//2:pw//2+w]
                            att_pred = att_pred[:, :, ph//2:ph//2+h, pw//2:pw//2+w]

                        cbf_slices.append(cbf_pred.cpu().numpy())
                        att_slices.append(att_pred.cpu().numpy())

                cbf_vol = np.concatenate(cbf_slices, axis=0)  # (Z,1,H,W)
                att_vol = np.concatenate(att_slices, axis=0)

                # Denormalize
                cbf_denorm = cbf_vol * minfo['norm_stats']['y_std_cbf'] + minfo['norm_stats']['y_mean_cbf']
                att_denorm = att_vol * minfo['norm_stats']['y_std_att'] + minfo['norm_stats']['y_mean_att']

                cbf_denorm = np.clip(cbf_denorm, 0, 200)
                att_denorm = np.clip(att_denorm, 0, 5000)

                # Transpose to (H, W, Z)
                cbf_map = np.transpose(cbf_denorm[:, 0, :, :], (1, 2, 0))
                att_map = np.transpose(att_denorm[:, 0, :, :], (1, 2, 0))

                cbf_masked = cbf_map * brain_mask
                att_masked = att_map * brain_mask

                brain_voxels = brain_mask > 0
                stats = {
                    'cbf_mean': float(cbf_masked[brain_voxels].mean()),
                    'cbf_std': float(cbf_masked[brain_voxels].std()),
                    'cbf_median': float(np.median(cbf_masked[brain_voxels])),
                    'att_mean': float(att_masked[brain_voxels].mean()),
                    'att_std': float(att_masked[brain_voxels].std()),
                    'att_median': float(np.median(att_masked[brain_voxels])),
                    'n_brain_voxels': int(brain_voxels.sum()),
                    'common_plds': common_plds,
                }
                subject_results[sid] = {
                    'stats': stats,
                    'cbf_map': cbf_masked,
                    'att_map': att_masked,
                    'brain_mask': brain_mask,
                    'affine': ref_img.affine,
                    'header': ref_img.header,
                }
                print(f"  {sid}: CBF={stats['cbf_mean']:.1f}+/-{stats['cbf_std']:.1f}, "
                      f"ATT={stats['att_mean']:.0f}+/-{stats['att_std']:.0f}")

            except Exception as e:
                print(f"  {sid}: ERROR - {e}")
                import traceback
                traceback.print_exc()

        all_invivo_results[name] = subject_results

    return all_invivo_results


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: Linearity Test
# ═══════════════════════════════════════════════════════════════════════════

def run_linearity_test(model_info_list, snr=10.0, seed=42):
    """Test CBF linearity: predicted vs true CBF across wide range."""
    print("\n" + "="*70)
    print("PART 4: CBF LINEARITY TEST")
    print("="*70)

    mask = make_brain_mask()
    brain_idx = mask > 0
    rng = np.random.RandomState(seed)

    cbf_values = np.arange(10, 201, 10)
    att_fixed = 1500.0
    n_reps = 3

    results = {}
    for minfo in model_info_list:
        name = minfo['name']
        print(f"  {name}...")
        pred_means = []
        pred_stds = []

        for cbf_true in cbf_values:
            preds = []
            for _ in range(n_reps):
                clean = generate_clean_phantom(cbf_true, att_fixed, mask)
                sim = ASLSimulator(ASLParameters(T1_artery=T1_ARTERY, T_tau=T_TAU,
                                                 alpha_PCASL=ALPHA_PCASL, alpha_VSASL=ALPHA_VSASL))
                noise_sd = sim._compute_reference_signal() / snr
                noise = noise_sd * rng.randn(*clean.shape).astype(np.float32)
                noisy = clean + noise
                noisy_norm = noisy * 100.0 * minfo['gsf']

                inp = torch.from_numpy(noisy_norm[np.newaxis]).float().to(DEVICE)
                with torch.no_grad():
                    cbf_pred, _, _, _ = minfo['models'][0](inp)
                    cbf_d = (cbf_pred * minfo['norm_stats']['y_std_cbf'] + minfo['norm_stats']['y_mean_cbf']).cpu().numpy()[0, 0]
                    cbf_d = np.clip(cbf_d, 0, 250)
                preds.append(np.mean(cbf_d[brain_idx]))

            pred_means.append(np.mean(preds))
            pred_stds.append(np.std(preds))

        pred_means = np.array(pred_means)
        pred_stds = np.array(pred_stds)

        # Linear fit
        slope, intercept = np.polyfit(cbf_values, pred_means, 1)
        ss_res = np.sum((pred_means - (slope * cbf_values + intercept))**2)
        ss_tot = np.sum((pred_means - np.mean(pred_means))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # R2 vs identity
        ss_res_id = np.sum((pred_means - cbf_values)**2)
        ss_tot_id = np.sum((cbf_values - np.mean(cbf_values))**2)
        r2_identity = 1 - ss_res_id / ss_tot_id

        results[name] = {
            'cbf_true': cbf_values.tolist(),
            'pred_means': pred_means.tolist(),
            'pred_stds': pred_stds.tolist(),
            'slope': slope,
            'intercept': intercept,
            'r2_fit': r2,
            'r2_identity': r2_identity,
        }
        print(f"    slope={slope:.3f}, intercept={intercept:.1f}, R2_identity={r2_identity:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    'Baseline SpatialASL': '#1f77b4',
    'AmplitudeAware': '#d62728',
    'Least Squares': '#7f7f7f',
}
MARKERS = {
    'Baseline SpatialASL': 'o',
    'AmplitudeAware': 's',
    'Least Squares': 'x',
}


def plot_training_curves(output_dir):
    """Parse training logs and plot learning curves."""
    print("Generating training curves...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    logs = {
        'Baseline SpatialASL': V6_DIR / 'A_Baseline_SpatialASL' / 'slurm_2813342.out',
        'AmplitudeAware': V6_DIR / 'B_AmplitudeAware' / 'slurm_2813343.out',
    }

    for name, log_path in logs.items():
        epochs, train_loss, val_loss = [], [], []
        with open(log_path) as f:
            for line in f:
                if 'Epoch' in line and 'Train Loss' in line:
                    parts = line.strip()
                    ep = int(parts.split('Epoch ')[1].split('/')[0])
                    tl = float(parts.split('Train Loss = ')[1].split(',')[0])
                    vl = float(parts.split('Val Loss = ')[1])
                    epochs.append(ep)
                    train_loss.append(tl)
                    val_loss.append(vl)

        color = COLORS[name]
        axes[0].plot(epochs, train_loss, '-', color=color, alpha=0.7, label=f'{name} (train)')
        axes[0].plot(epochs, val_loss, '--', color=color, label=f'{name} (val)')
        axes[1].plot(epochs[1:], val_loss[1:], '-o', color=color, markersize=3, label=name)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss (Zoomed)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_bias_cov(results, att_sweep, cbf_sweep, fixed_cbf, fixed_att, output_dir):
    """Plot bias/CoV results."""
    print("Generating bias/CoV figures...")

    for snr_key, snr_data in results.items():
        snr = float(snr_key.split('_')[1])

        for sweep_name, x_vals, x_label, x_param in [
            ('sweep_att', att_sweep, 'True ATT (ms)', 'ATT'),
            ('sweep_cbf', cbf_sweep, 'True CBF (ml/100g/min)', 'CBF'),
        ]:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
            fig.suptitle(f'Bias & CoV vs {x_param} (SNR = {snr})', fontsize=14, fontweight='bold')

            panels = [
                (axes[0, 0], 'cbf_bias', 'CBF Bias (ml/100g/min)', True),
                (axes[0, 1], 'cbf_cov', 'CBF CoV (%)', False),
                (axes[1, 0], 'att_bias', 'ATT Bias (ms)', True),
                (axes[1, 1], 'att_cov', 'ATT CoV (%)', False),
            ]

            for ax, key, ylabel, show_zero in panels:
                for model_name, model_data in snr_data[sweep_name].items():
                    color = COLORS.get(model_name, '#333333')
                    marker = MARKERS.get(model_name, 'D')
                    ls = '--' if model_name == 'Least Squares' else '-'
                    ax.plot(x_vals, model_data[key], color=color, marker=marker,
                            markersize=4, linewidth=1.5, linestyle=ls, label=model_name)
                if show_zero:
                    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
                ax.set_xlabel(x_label, fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            slug = x_param.lower()
            plt.savefig(output_dir / f'fig2_bias_cov_vs_{slug}_snr{snr}.png', dpi=200)
            plt.close()


def plot_win_rates(win_results, output_dir):
    """Plot win rate bar chart."""
    print("Generating win rate figure...")

    snr_levels = sorted([float(k.split('_')[1]) for k in win_results.keys()])
    model_names = list(win_results[f'snr_{snr_levels[0]}'].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(snr_levels))
    width = 0.35

    for mi, mname in enumerate(model_names):
        cbf_wrs = [win_results[f'snr_{s}'][mname]['cbf_win_rate'] for s in snr_levels]
        att_wrs = [win_results[f'snr_{s}'][mname]['att_win_rate'] for s in snr_levels]
        color = COLORS.get(mname, '#333333')

        axes[0].bar(x + mi * width - width/2, cbf_wrs, width, label=mname, color=color, alpha=0.8)
        axes[1].bar(x + mi * width - width/2, att_wrs, width, label=mname, color=color, alpha=0.8)

    for ax, title in [(axes[0], 'CBF Win Rate (NN vs LS)'), (axes[1], 'ATT Win Rate (NN vs LS)')]:
        ax.axhline(50, color='gray', linestyle='--', linewidth=1, label='Chance (50%)')
        ax.set_xlabel('SNR')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(s)) for s in snr_levels])
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_win_rates.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_linearity(linearity_results, output_dir):
    """Plot CBF linearity results."""
    print("Generating linearity figure...")

    fig, axes = plt.subplots(1, len(linearity_results), figsize=(6*len(linearity_results), 5))
    if len(linearity_results) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, linearity_results.items()):
        cbf_true = np.array(data['cbf_true'])
        pred_means = np.array(data['pred_means'])
        pred_stds = np.array(data['pred_stds'])

        color = COLORS.get(name, '#333333')
        ax.errorbar(cbf_true, pred_means, yerr=pred_stds, fmt='o-', color=color,
                     markersize=4, linewidth=1.5, capsize=3, label=f'{name}')
        ax.plot([0, 210], [0, 210], 'k--', linewidth=1, label='Identity', alpha=0.5)

        # Linear fit line
        ax.plot(cbf_true, data['slope'] * cbf_true + data['intercept'],
                ':', color=color, linewidth=1,
                label=f'Fit: slope={data["slope"]:.2f}, R2id={data["r2_identity"]:.3f}')

        ax.set_xlabel('True CBF (ml/100g/min)')
        ax.set_ylabel('Predicted CBF (ml/100g/min)')
        ax.set_title(f'{name} CBF Linearity')
        ax.set_xlim(0, 210)
        ax.set_ylim(0, 210)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_linearity.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_invivo_results(invivo_results, output_dir):
    """Plot in-vivo CBF and ATT maps for each model and subject."""
    print("Generating in-vivo figures...")

    model_names = list(invivo_results.keys())
    # Find common subjects
    all_subjects = set()
    for mn in model_names:
        all_subjects.update(invivo_results[mn].keys())
    subjects = sorted(all_subjects)

    if not subjects:
        print("  No in-vivo results to plot.")
        return

    for sid in subjects[:4]:  # Limit to first 4 subjects
        n_models = len(model_names)
        # Pick middle slice
        first_model = model_names[0]
        if sid not in invivo_results[first_model]:
            continue

        cbf_map = invivo_results[first_model][sid]['cbf_map']
        n_slices = cbf_map.shape[2]
        mid_slice = n_slices // 2

        fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'In-Vivo Results: {sid} (Slice {mid_slice})', fontsize=14, fontweight='bold')

        for mi, mn in enumerate(model_names):
            if sid not in invivo_results[mn]:
                continue
            data = invivo_results[mn][sid]
            cbf = data['cbf_map'][:, :, mid_slice]
            att = data['att_map'][:, :, mid_slice]
            mask = data['brain_mask'][:, :, mid_slice] > 0

            cbf_disp = np.where(mask, cbf, np.nan)
            att_disp = np.where(mask, att, np.nan)

            im1 = axes[0, mi].imshow(cbf_disp.T, cmap='hot', vmin=0, vmax=100, origin='lower')
            axes[0, mi].set_title(f'{mn}\nCBF (ml/100g/min)')
            axes[0, mi].axis('off')
            plt.colorbar(im1, ax=axes[0, mi], fraction=0.046)

            im2 = axes[1, mi].imshow(att_disp.T, cmap='viridis', vmin=500, vmax=3000, origin='lower')
            axes[1, mi].set_title(f'ATT (ms)')
            axes[1, mi].axis('off')
            plt.colorbar(im2, ax=axes[1, mi], fraction=0.046)

        plt.tight_layout()
        plt.savefig(output_dir / f'fig5_invivo_{sid}.png', dpi=200, bbox_inches='tight')
        plt.close()

    # Summary stats comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_pos = np.arange(len(subjects))
    width = 0.35

    for mi, mn in enumerate(model_names):
        cbf_means = []
        att_means = []
        for sid in subjects:
            if sid in invivo_results[mn]:
                cbf_means.append(invivo_results[mn][sid]['stats']['cbf_mean'])
                att_means.append(invivo_results[mn][sid]['stats']['att_mean'])
            else:
                cbf_means.append(0)
                att_means.append(0)

        color = COLORS.get(mn, '#333333')
        axes[0].bar(x_pos + mi * width - width/2, cbf_means, width, label=mn, color=color, alpha=0.8)
        axes[1].bar(x_pos + mi * width - width/2, att_means, width, label=mn, color=color, alpha=0.8)

    axes[0].set_ylabel('Mean CBF (ml/100g/min)')
    axes[0].set_title('In-Vivo CBF by Subject')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([s.split('_')[-1] for s in subjects], rotation=45, ha='right', fontsize=7)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].set_ylabel('Mean ATT (ms)')
    axes[1].set_title('In-Vivo ATT by Subject')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([s.split('_')[-1] for s in subjects], rotation=45, ha='right', fontsize=7)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_invivo_summary.png', dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(bias_cov_results, win_results, invivo_results, linearity_results,
                    att_sweep, cbf_sweep, fixed_cbf, fixed_att, output_dir):
    """Generate comprehensive Markdown report."""
    print("\nGenerating report...")

    lines = []
    lines.append("# V6 Evaluation Report")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # V6 changes summary
    lines.append("## V6 Configuration Changes")
    lines.append("")
    lines.append("| Parameter | V5 (Previous) | V6 (Current) |")
    lines.append("|-----------|---------------|--------------|")
    lines.append("| Loss | NLL | **L1** (reverted) |")
    lines.append("| variance_weight | 0.1 | **0.5** |")
    lines.append("| PLDs | 6 [500-3000] | **5 [500-2500]** (matches in-vivo) |")
    lines.append("| CBF range (GM) | 10-120 | **10-200** |")
    lines.append("| ATT range (GM) | 500-2500 | **500-3000** |")
    lines.append("| CBF clip | 200 | **250** |")
    lines.append("| T1_artery | 1650 | 1650 (unchanged) |")
    lines.append("| att_scale | 1.0 | 1.0 (unchanged) |")
    lines.append("")

    # Training summary
    lines.append("## Training Summary")
    lines.append("")
    lines.append("| Model | Epochs Completed | Best Val Loss | Parameters |")
    lines.append("|-------|-----------------|---------------|------------|")
    lines.append("| Baseline SpatialASL | 19/30 (time limit) | 0.328 | 1,929,060 |")
    lines.append("| AmplitudeAware | 29/40 (time limit) | 0.333 | 2,035,493 |")
    lines.append("")
    lines.append("Note: Both jobs hit SLURM time limits. Models were saved at best checkpoint.")
    lines.append("")

    # Linearity
    if linearity_results:
        lines.append("## CBF Linearity Test (SNR=10)")
        lines.append("")
        lines.append("| Model | Slope | Intercept | R2 (identity) | R2 (fit) |")
        lines.append("|-------|-------|-----------|---------------|----------|")
        for name, data in linearity_results.items():
            lines.append(f"| {name} | {data['slope']:.3f} | {data['intercept']:.1f} | "
                         f"{data['r2_identity']:.3f} | {data['r2_fit']:.3f} |")
        lines.append("")
        ideal_note = "Ideal: slope=1.0, intercept=0, R2_identity=1.0"
        lines.append(f"*{ideal_note}*")
        lines.append("")

    # Win rates
    if win_results:
        lines.append("## Win Rate Analysis (NN vs Corrected LS)")
        lines.append("")
        snr_levels = sorted([float(k.split('_')[1]) for k in win_results.keys()])
        model_names = list(win_results[f'snr_{snr_levels[0]}'].keys())

        lines.append("### CBF Win Rate (%)")
        lines.append("")
        header = "| SNR | " + " | ".join(model_names) + " |"
        lines.append(header)
        lines.append("|-----|" + "|".join(["---" for _ in model_names]) + "|")
        for snr in snr_levels:
            row = f"| {int(snr)} | "
            for mn in model_names:
                wr = win_results[f'snr_{snr}'][mn]['cbf_win_rate']
                row += f"{wr:.1f}% | "
            lines.append(row)
        lines.append("")

        lines.append("### ATT Win Rate (%)")
        lines.append("")
        lines.append(header)
        lines.append("|-----|" + "|".join(["---" for _ in model_names]) + "|")
        for snr in snr_levels:
            row = f"| {int(snr)} | "
            for mn in model_names:
                wr = win_results[f'snr_{snr}'][mn]['att_win_rate']
                row += f"{wr:.1f}% | "
            lines.append(row)
        lines.append("")

    # Bias/CoV summary
    if bias_cov_results:
        lines.append("## Bias/CoV Summary")
        lines.append("")
        lines.append("### CBF Bias at Typical Values (CBF=50, ATT=1500)")
        lines.append("")
        lines.append("| SNR | Model | CBF Bias | CBF CoV (%) | ATT Bias (ms) | ATT CoV (%) |")
        lines.append("|-----|-------|----------|-------------|---------------|-------------|")

        for snr_key in sorted(bias_cov_results.keys()):
            snr = snr_key.split('_')[1]
            sweep_data = bias_cov_results[snr_key]['sweep_att']
            # Find index closest to ATT=1500
            att_idx = np.argmin(np.abs(att_sweep - 1500))
            for model_name, data in sweep_data.items():
                cb = data['cbf_bias'][att_idx]
                cc = data['cbf_cov'][att_idx]
                ab = data['att_bias'][att_idx]
                ac = data['att_cov'][att_idx]
                lines.append(f"| {snr} | {model_name} | {cb:.2f} | {cc:.1f} | {ab:.0f} | {ac:.1f} |")
        lines.append("")

    # In-vivo results
    if invivo_results:
        lines.append("## In-Vivo Results")
        lines.append("")
        for model_name, subjects_data in invivo_results.items():
            lines.append(f"### {model_name}")
            lines.append("")
            lines.append("| Subject | CBF Mean | CBF Std | CBF Median | ATT Mean | ATT Std | ATT Median |")
            lines.append("|---------|----------|---------|------------|----------|---------|------------|")
            for sid, data in sorted(subjects_data.items()):
                s = data['stats']
                lines.append(f"| {sid} | {s['cbf_mean']:.1f} | {s['cbf_std']:.1f} | {s['cbf_median']:.1f} | "
                             f"{s['att_mean']:.0f} | {s['att_std']:.0f} | {s['att_median']:.0f} |")
            lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    if linearity_results:
        baseline_slope = linearity_results.get('Baseline SpatialASL', {}).get('slope', None)
        ampaware_slope = linearity_results.get('AmplitudeAware', {}).get('slope', None)

        if baseline_slope is not None:
            if baseline_slope < 0.1:
                lines.append("1. **Baseline SpatialASL still shows variance collapse** "
                             f"(slope={baseline_slope:.3f}). V6 fixes did NOT resolve this.")
            elif baseline_slope > 1.3:
                lines.append(f"1. **Baseline SpatialASL shows super-linearity** (slope={baseline_slope:.3f})")
            else:
                lines.append(f"1. **Baseline SpatialASL linearity improved** (slope={baseline_slope:.3f})")

        if ampaware_slope is not None:
            if ampaware_slope > 1.3:
                lines.append(f"2. **AmplitudeAware still shows super-linearity** (slope={ampaware_slope:.3f}). "
                             "Expanded CBF range partially helped but did not fully resolve.")
            elif 0.8 <= ampaware_slope <= 1.2:
                lines.append(f"2. **AmplitudeAware linearity is good** (slope={ampaware_slope:.3f}). "
                             "Expanded CBF range in V6 helped.")
            else:
                lines.append(f"2. **AmplitudeAware linearity** (slope={ampaware_slope:.3f})")

    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `fig1_training_curves.png` - Training and validation loss curves")
    lines.append("- `fig2_bias_cov_*.png` - Bias and CoV across ATT/CBF sweeps per SNR")
    lines.append("- `fig3_win_rates.png` - NN vs LS win rate by SNR")
    lines.append("- `fig4_linearity.png` - CBF predicted vs true (linearity test)")
    lines.append("- `fig5_invivo_*.png` - In-vivo CBF/ATT maps per subject")
    lines.append("- `fig6_invivo_summary.png` - In-vivo summary statistics")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = output_dir / 'v6_evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"Report saved to: {report_path}")
    return report_text


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("V6 COMPREHENSIVE EVALUATION PIPELINE")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    print("\nLoading models...")
    models_A, ns_A, gsf_A, class_A = load_model(V6_DIR / 'A_Baseline_SpatialASL')
    models_B, ns_B, gsf_B, class_B = load_model(V6_DIR / 'B_AmplitudeAware')
    print(f"  Model A ({class_A}): {len(models_A)} members, gsf={gsf_A}")
    print(f"  Model B ({class_B}): {len(models_B)} members, gsf={gsf_B}")

    model_info_list = [
        {'name': 'Baseline SpatialASL', 'models': models_A, 'norm_stats': ns_A, 'gsf': gsf_A},
        {'name': 'AmplitudeAware', 'models': models_B, 'norm_stats': ns_B, 'gsf': gsf_B},
    ]

    t_start = time.time()

    # Part 1: Bias/CoV analysis
    bias_cov_results, att_sweep, cbf_sweep, fixed_cbf, fixed_att = run_bias_cov_analysis(
        model_info_list, snr_levels=[3.0, 5.0, 10.0],
        n_phantoms=5, n_ls_realizations=200, seed=42
    )

    # Part 2: Win rate analysis
    win_results = run_win_rate_analysis(
        model_info_list, snr_levels=[3.0, 5.0, 10.0, 15.0, 25.0],
        n_test_phantoms=20, seed=123
    )

    # Part 3: In-vivo inference
    invivo_results = run_invivo_inference(model_info_list, alpha_bs1=0.93)

    # Part 4: Linearity test
    linearity_results = run_linearity_test(model_info_list, snr=10.0)

    # Generate figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    plot_training_curves(OUTPUT_DIR)
    plot_bias_cov(bias_cov_results, att_sweep, cbf_sweep, fixed_cbf, fixed_att, OUTPUT_DIR)
    plot_win_rates(win_results, OUTPUT_DIR)
    plot_linearity(linearity_results, OUTPUT_DIR)
    plot_invivo_results(invivo_results, OUTPUT_DIR)

    # Generate report
    report_text = generate_report(
        bias_cov_results, win_results, invivo_results, linearity_results,
        att_sweep, cbf_sweep, fixed_cbf, fixed_att, OUTPUT_DIR
    )

    # Save all numeric results
    json_results = {
        'bias_cov': {k: {kk: {kkk: vvv for kkk, vvv in vv.items()}
                         for kk, vv in v.items()} for k, v in bias_cov_results.items()},
        'win_rates': win_results,
        'linearity': linearity_results,
        'invivo_stats': {
            mn: {sid: data['stats'] for sid, data in sd.items()}
            for mn, sd in invivo_results.items()
        },
    }
    with open(OUTPUT_DIR / 'v6_all_results.json', 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Results: {OUTPUT_DIR}")
    print(f"{'='*70}")

    # Print key findings
    print("\n" + report_text.split("## Key Findings")[1] if "## Key Findings" in report_text else "")


if __name__ == '__main__':
    main()

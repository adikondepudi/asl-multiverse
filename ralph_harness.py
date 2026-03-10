#!/usr/bin/env python3
"""
Ralph Harness — Combined synthetic + in-vivo evaluation.

One script that trains a model, tests on synthetic phantoms, AND tests on
real patient data. The goal: make synthetic evaluation predictive of in-vivo
performance. If synthetic says "great" but in-vivo says "bad", the synthetic
data generation is broken and needs fixing.

Usage:
    python ralph_harness.py --device mps
    python ralph_harness.py --device mps --n-epochs 15 --n-samples 1500  # faster
"""
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

from simulation.asl_simulation import ASLParameters
from simulation.enhanced_simulation import RealisticASLSimulator, SpatialPhantomGenerator
from simulation.noise_engine import NoiseInjector
from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
from models.spatial_asl_network import MaskedSpatialLoss
from baselines.multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from utils.helpers import get_grid_search_initial_guess


# =============================================================================
# Config
# =============================================================================

def load_config():
    """Load config from config/invivo_experiment.yaml."""
    config_path = Path('config/invivo_experiment.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = {}
    for section in ['simulation', 'data', 'training']:
        if section in raw:
            cfg.update(raw[section])
    if 'simulation' in raw and 'domain_randomization' in raw['simulation']:
        cfg['domain_randomization'] = raw['simulation']['domain_randomization']
    if 'noise_config' in raw:
        cfg['noise_config'] = raw['noise_config']
    return cfg


# =============================================================================
# Signal generation (shared between training and synthetic test)
# =============================================================================

def generate_asl_signal(cbf_map, att_map, plds_bc, t1_b, alpha_p, alpha_v, tau, t2_f, lambda_b=0.90):
    """Generate combined PCASL+VSASL signal from parameter maps."""
    att_bc = att_map[np.newaxis, :, :].astype(np.float32)
    cbf_bc = (cbf_map / 6000.0)[np.newaxis, :, :].astype(np.float32)

    # PCASL
    mask_arrived = (plds_bc >= att_bc)
    mask_transit = (plds_bc < att_bc) & (plds_bc >= (att_bc - tau))
    sig_p_arrived = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                     np.exp(-plds_bc / t1_b) * (1 - np.exp(-tau / t1_b)) * t2_f) / lambda_b
    sig_p_transit = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                     (np.exp(-att_bc / t1_b) - np.exp(-(tau + plds_bc) / t1_b)) * t2_f) / lambda_b
    pcasl_sig = np.zeros_like(plds_bc * cbf_bc)
    pcasl_sig[mask_arrived] = sig_p_arrived[mask_arrived]
    pcasl_sig[mask_transit] = sig_p_transit[mask_transit]

    # VSASL
    sib = 1.0
    mask_vs_arrived = (plds_bc > att_bc)
    sig_v_early = (2 * alpha_v * cbf_bc * sib * (plds_bc / 1000.0) *
                   np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
    sig_v_late = (2 * alpha_v * cbf_bc * sib * (att_bc / 1000.0) *
                  np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
    vsasl_sig = np.where(mask_vs_arrived, sig_v_late, sig_v_early)

    return np.concatenate([pcasl_sig, vsasl_sig], axis=0).astype(np.float32)


def sample_physics(cfg, base_params, use_domain_rand=True):
    """Sample physics parameters (with or without domain randomization)."""
    if use_domain_rand:
        dr = cfg.get('domain_randomization', {})
        t1_b = np.random.uniform(*dr.get('T1_artery_range', [1550, 2150]))
        alpha_bs1 = np.random.uniform(*dr.get('alpha_BS1_range', [0.85, 1.0]))
        alpha_p = np.random.uniform(*dr.get('alpha_PCASL_range', [0.75, 0.95])) * (alpha_bs1 ** 4)
        alpha_v = np.random.uniform(*dr.get('alpha_VSASL_range', [0.40, 0.70])) * (alpha_bs1 ** 3)
        tau = base_params.T_tau * (1 + np.random.uniform(
            -dr.get('T_tau_perturb', 0.10), dr.get('T_tau_perturb', 0.10)))
    else:
        t1_b = base_params.T1_artery
        alpha_p = base_params.alpha_PCASL * (base_params.alpha_BS1 ** 4)
        alpha_v = base_params.alpha_VSASL * (base_params.alpha_BS1 ** 3)
        tau = base_params.T_tau
    return t1_b, alpha_p, alpha_v, tau


# =============================================================================
# Phase 1: Generate training data + train model
# =============================================================================

def make_brain_mask_2d(size=64):
    """Create a random elliptical brain-like mask with ~50-60% coverage."""
    # Random ellipse parameters (matching in-vivo brain coverage ~50-60%)
    cx = size // 2 + np.random.randint(-3, 4)
    cy = size // 2 + np.random.randint(-3, 4)
    rx = np.random.uniform(0.35, 0.45) * size  # semi-axis x
    ry = np.random.uniform(0.38, 0.48) * size  # semi-axis y
    angle = np.random.uniform(-15, 15) * np.pi / 180  # slight rotation
    y, x = np.ogrid[:size, :size]
    # Rotated ellipse
    dx = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
    dy = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    mask = (dx**2 / rx**2 + dy**2 / ry**2) <= 1.0
    return mask


def generate_training_data(cfg, n_samples, seed):
    """Generate spatial phantom training data with realistic brain masks."""
    np.random.seed(seed)
    plds = np.array(cfg.get('pld_values', [500, 1000, 1500, 2000, 2500]), dtype=np.float32)
    asl_dict = {k: v for k, v in cfg.items() if k in ASLParameters.__annotations__}
    base_params = ASLParameters(**asl_dict)
    dr_cfg = cfg.get('domain_randomization', {})
    use_dr = dr_cfg.get('enabled', True)

    phantom_gen = SpatialPhantomGenerator(size=64, pve_sigma=3.0)
    plds_bc = plds[:, np.newaxis, np.newaxis]

    signals_list, targets_list = [], []
    for _ in range(n_samples):
        cbf_map, att_map, _ = phantom_gen.generate_phantom(include_pathology=False)
        t1_b, alpha_p, alpha_v, tau = sample_physics(cfg, base_params, use_dr)
        sig = generate_asl_signal(cbf_map, att_map, plds_bc, t1_b, alpha_p, alpha_v, tau, base_params.T2_factor)

        # Apply random brain-shaped mask (~55% coverage, matching in-vivo)
        brain_mask = make_brain_mask_2d(64)
        sig[:, ~brain_mask] = 0.0
        cbf_map[~brain_mask] = 0.0
        att_map[~brain_mask] = 0.0

        signals_list.append(sig)
        targets_list.append(np.stack([cbf_map, att_map], axis=0).astype(np.float32))

    signals = np.array(signals_list, dtype=np.float32) * 100.0  # M0 scaling
    targets = np.array(targets_list, dtype=np.float32)

    cbf_all = targets[:, 0]; att_all = targets[:, 1]
    brain = cbf_all > 1.0
    norm_stats = {
        'y_mean_cbf': float(np.mean(cbf_all[brain])), 'y_std_cbf': float(np.std(cbf_all[brain])),
        'y_mean_att': float(np.mean(att_all[brain])), 'y_std_att': float(np.std(att_all[brain])),
    }

    # Compute training signal P90 (brain voxels) for per-subject calibration at inference
    # This enables matching in-vivo signal distributions to training
    brain_signals = signals[:, :, :, :][np.broadcast_to(brain[:, np.newaxis, :, :], signals.shape)]
    norm_stats['signal_p90'] = float(np.percentile(np.abs(brain_signals), 90))

    return signals, targets, norm_stats


def create_brain_mask(signals_np):
    m = np.mean(np.abs(signals_np), axis=0)
    return (m > np.percentile(m, 5)).astype(np.float32)


def train_model(cfg, signals, targets, norm_stats, device, n_epochs, seed):
    """Train AmplitudeAwareSpatialASLNet with periodic data regeneration."""
    torch.manual_seed(seed)
    np.random.seed(seed + 100)
    plds = np.array(cfg.get('pld_values', [500, 1000, 1500, 2000, 2500]), dtype=np.float32)
    n_plds = len(plds)

    features = cfg.get('hidden_sizes', [32, 64, 128, 256])
    model = AmplitudeAwareSpatialASLNet(
        n_plds=n_plds, features=features,
        use_film_at_bottleneck=cfg.get('use_film_at_bottleneck', True),
        use_film_at_decoder=cfg.get('use_film_at_decoder', True),
        use_amplitude_output_modulation=cfg.get('use_amplitude_output_modulation', True),
    ).to(device)

    loss_fn = MaskedSpatialLoss(
        loss_type=cfg.get('loss_type', 'l1'), dc_weight=cfg.get('dc_weight', 0.0),
        att_scale=cfg.get('att_scale', 1.0), cbf_weight=cfg.get('cbf_weight', 1.0),
        att_weight=cfg.get('att_weight', 1.0), norm_stats=norm_stats,
        variance_weight=cfg.get('variance_weight', 0.01),
    )

    asl_dict = {k: v for k, v in cfg.items() if k in ASLParameters.__annotations__}
    base_params = ASLParameters(**asl_dict)
    noise_cfg = {
        'noise_config': cfg.get('noise_config', {'snr_range': [2.0, 25.0]}),
        'data_noise_components': cfg.get('data_noise_components', ['thermal']),
        'noise_type': cfg.get('noise_type', 'gaussian'),
    }
    noise_injector = NoiseInjector(noise_cfg)
    simulator = RealisticASLSimulator(params=base_params)
    ref_signal = simulator._compute_reference_signal()
    scalings = simulator.compute_tr_noise_scaling(plds)
    pld_scaling = {'PCASL': scalings['PCASL'], 'VSASL': scalings['VSASL']}

    global_scale = cfg.get('global_scale_factor', 10.0)
    batch_size = cfg.get('batch_size', 64)
    lr = cfg.get('learning_rate', 0.002)
    n_samples = len(signals)

    # Initial train/val split
    n = len(signals)
    n_train = n - max(1, int(n * 0.1))
    perm = np.random.permutation(n)
    train_idx = perm[:n_train]
    train_signals = torch.from_numpy(signals[train_idx]).float()
    train_targets = targets[train_idx]

    # Online data regeneration: regenerate phantoms every regen_interval epochs
    # This gives 3x phantom diversity (9000 unique anatomies across 30 epochs)
    regen_interval = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get('weight_decay', 0.0001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    clean_fraction = 0.0  # I1: skip clean epochs, all epochs get noise
    snr_min_base = noise_cfg['noise_config'].get('snr_range', [2.0, 25.0])[0]
    snr_max_base = noise_cfg['noise_config'].get('snr_range', [2.0, 25.0])[1]
    loss_history = []

    # SWA: accumulate averaged weights over last swa_epochs epochs
    swa_epochs = 5
    swa_start = n_epochs - swa_epochs
    swa_state = None
    swa_count = 0

    # Gradient accumulation for effective batch size 128
    accum_steps = 2  # batch_size=64 * 2 = 128 effective

    for epoch in range(n_epochs):
        # Regenerate training data at interval boundaries (except epoch 0, which uses initial data)
        if epoch > 0 and epoch % regen_interval == 0:
            regen_seed = seed + epoch * 1000
            print(f"  Regenerating training data (seed={regen_seed})...")
            new_signals, new_targets, _ = generate_training_data(cfg, n_samples, regen_seed)
            # Keep original norm_stats for loss consistency
            n_new = len(new_signals)
            n_train = n_new - max(1, int(n_new * 0.1))
            perm = np.random.permutation(n_new)
            train_idx = perm[:n_train]
            train_signals = torch.from_numpy(new_signals[train_idx]).float()
            train_targets = new_targets[train_idx]

        model.train()
        epoch_loss, n_batches = 0.0, 0
        perm_train = np.random.permutation(n_train)
        use_noise = epoch >= int(n_epochs * clean_fraction)

        # SNR curriculum: start with high SNR (easy), gradually include low SNR (hard)
        if use_noise:
            noisy_epoch = epoch - int(n_epochs * clean_fraction)
            noisy_total = n_epochs - int(n_epochs * clean_fraction)
            progress = min(noisy_epoch / max(noisy_total * 0.6, 1), 1.0)  # reach full range at 60% of noisy phase
            curr_snr_min = snr_max_base - progress * (snr_max_base - snr_min_base)
            noise_injector.snr_range = [curr_snr_min, snr_max_base]

        optimizer.zero_grad()  # zero grads at start of epoch for accumulation
        n_batches_total = (n_train + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, n_train, batch_size)):
            end = min(start + batch_size, n_train)
            idx = perm_train[start:end]
            raw_sig = train_signals[idx].to(device)
            cbf_t = torch.from_numpy(train_targets[idx, 0:1]).float().to(device)
            att_t = torch.from_numpy(train_targets[idx, 1:2]).float().to(device)

            masks = []
            for j in range(len(idx)):
                masks.append(create_brain_mask(train_signals[idx[j]].numpy()))
            mask_t = torch.from_numpy(np.array(masks)[:, np.newaxis]).float().to(device)

            noisy_sig = noise_injector.apply_noise(raw_sig, ref_signal, pld_scaling) if use_noise else raw_sig
            normalized = torch.clamp(noisy_sig * global_scale, -30.0, 30.0)

            # Label smoothing: add small noise to targets to prevent overfitting
            cbf_t = cbf_t + 0.5 * torch.randn_like(cbf_t)
            att_t = att_t + 15.0 * torch.randn_like(att_t)

            # Random flip augmentation (applied consistently to input, targets, mask)
            if torch.rand(1).item() > 0.5:
                normalized = torch.flip(normalized, [2])  # vertical flip
                cbf_t = torch.flip(cbf_t, [2])
                att_t = torch.flip(att_t, [2])
                mask_t = torch.flip(mask_t, [2])
            if torch.rand(1).item() > 0.5:
                normalized = torch.flip(normalized, [3])  # horizontal flip
                cbf_t = torch.flip(cbf_t, [3])
                att_t = torch.flip(att_t, [3])
                mask_t = torch.flip(mask_t, [3])
            # Random 90° rotation augmentation
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                normalized = torch.rot90(normalized, k, [2, 3]).contiguous()
                cbf_t = torch.rot90(cbf_t, k, [2, 3]).contiguous()
                att_t = torch.rot90(att_t, k, [2, 3]).contiguous()
                mask_t = torch.rot90(mask_t, k, [2, 3]).contiguous()

            pred_cbf, pred_att, lv_cbf, lv_att = model(normalized)
            loss_dict = loss_fn(pred_cbf.float(), pred_att.float(), cbf_t, att_t, mask_t, normalized,
                           log_var_cbf=lv_cbf.float(), log_var_att=lv_att.float())
            loss = loss_dict['total_loss']
            # TV regularization for spatial smoothness
            tv_cbf = torch.mean(torch.abs(pred_cbf[:, :, 1:, :] - pred_cbf[:, :, :-1, :])) + \
                     torch.mean(torch.abs(pred_cbf[:, :, :, 1:] - pred_cbf[:, :, :, :-1]))
            tv_att = torch.mean(torch.abs(pred_att[:, :, 1:, :] - pred_att[:, :, :-1, :])) + \
                     torch.mean(torch.abs(pred_att[:, :, :, 1:] - pred_att[:, :, :, :-1]))
            tv_weight = cfg.get('tv_weight', 0.03)
            loss = loss + tv_weight * (tv_cbf + tv_att)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            # Gradient accumulation: scale loss and accumulate
            (loss / accum_steps).backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == n_batches_total:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item(); n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss = {avg_loss:.4f}")

        # SWA: accumulate model weights over last swa_epochs epochs
        if epoch >= swa_start:
            swa_count += 1
            current_sd = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.clone().float() for k, v in current_sd.items()}
            else:
                for k in swa_state:
                    swa_state[k] += current_sd[k].float()

    # Apply SWA averaged weights
    if swa_state is not None and swa_count > 0:
        print(f"  Applying SWA over last {swa_count} epochs")
        for k in swa_state:
            swa_state[k] /= swa_count
        model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in swa_state.items()})

    return model, loss_history


# =============================================================================
# Phase 2: Synthetic evaluation
# =============================================================================

def tta_predict_single(model, ns, inp):
    """Predict with 4-flip TTA for a single model, return averaged CBF/ATT."""
    flips = [
        (False, False),  # original
        (False, True),   # horizontal flip
        (True, False),   # vertical flip
        (True, True),    # both flips
    ]
    cbf_preds, att_preds = [], []
    for flip_v, flip_h in flips:
        aug = inp
        if flip_v:
            aug = torch.flip(aug, [2])
        if flip_h:
            aug = torch.flip(aug, [3])
        pc, pa, _, _ = model(aug)
        cbf = pc[0, 0].cpu().numpy()
        att = pa[0, 0].cpu().numpy()
        # Undo flips
        if flip_v:
            cbf = np.flip(cbf, axis=0).copy()
            att = np.flip(att, axis=0).copy()
        if flip_h:
            cbf = np.flip(cbf, axis=1).copy()
            att = np.flip(att, axis=1).copy()
        cbf_preds.append(cbf * ns['y_std_cbf'] + ns['y_mean_cbf'])
        att_preds.append(att * ns['y_std_att'] + ns['y_mean_att'])
    return np.mean(cbf_preds, axis=0), np.mean(att_preds, axis=0)


def ensemble_predict(models_and_stats, inp, device):
    """Average predictions from multiple (model, norm_stats) pairs with TTA."""
    cbf_preds, att_preds = [], []
    for model, ns in models_and_stats:
        cbf, att = tta_predict_single(model, ns, inp)
        cbf_preds.append(cbf)
        att_preds.append(att)
    return np.mean(cbf_preds, axis=0), np.mean(att_preds, axis=0)


def _ls_cache_path():
    return Path('invivo_results') / 'ls_synth_cache.npz'


def _compute_ls_cache(cfg, n_phantoms=10, snr_levels=[3, 10, 25], seed=9999):
    """Compute LS results for synthetic phantoms once and cache to disk."""
    print("  Computing LS cache (one-time cost)...")
    np.random.seed(seed)
    plds = np.array(cfg.get('pld_values', [500, 1000, 1500, 2000, 2500]), dtype=np.float32)
    asl_dict = {k: v for k, v in cfg.items() if k in ASLParameters.__annotations__}
    base_params = ASLParameters(**asl_dict)
    simulator = RealisticASLSimulator(params=base_params)
    ref_signal = simulator._compute_reference_signal()
    scalings = simulator.compute_tr_noise_scaling(plds)
    pld_scaling = {'PCASL': scalings['PCASL'], 'VSASL': scalings['VSASL']}
    n_plds = len(plds); plds_bc = plds[:, np.newaxis, np.newaxis]
    phantom_gen = SpatialPhantomGenerator(size=64, pve_sigma=3.0)
    dr_cfg = cfg.get('domain_randomization', {})

    ls_params = {
        'T1_artery': 1650.0, 'T_tau': 1800.0, 'T2_factor': 1.0,
        'alpha_BS1': 0.93, 'alpha_PCASL': 0.85, 'alpha_VSASL': 0.56,
    }

    cache = {}
    for snr in snr_levels:
        ls_errs_cbf, ls_errs_att = [], []
        voxel_coords = []  # (phantom_idx, yi, xi) for matching NN errors

        for p in range(n_phantoms):
            np.random.seed(seed + p * 100 + int(snr * 10))
            cbf_map, att_map, _ = phantom_gen.generate_phantom(include_pathology=False)
            t1_b, alpha_p, alpha_v, tau = sample_physics(cfg, base_params, dr_cfg.get('enabled', True))
            clean = generate_asl_signal(cbf_map, att_map, plds_bc, t1_b, alpha_p, alpha_v, tau, base_params.T2_factor) * 100.0

            brain_mask = make_brain_mask_2d(64)
            clean[:, ~brain_mask] = 0.0

            noise_sigma = ref_signal * 100.0 / snr
            noisy = clean.copy()
            for ch in range(clean.shape[0]):
                scale = pld_scaling['PCASL'] if ch < n_plds else pld_scaling['VSASL']
                noisy[ch] += np.random.randn(64, 64).astype(np.float32) * noise_sigma * scale

            brain_idx = np.where(brain_mask)
            n_sub = max(10, int(len(brain_idx[0]) * 0.25))
            sub = np.random.choice(len(brain_idx[0]), size=n_sub, replace=False)
            pldti = np.column_stack([plds.astype(np.float64), plds.astype(np.float64)])

            for si in sub:
                yi, xi = brain_idx[0][si], brain_idx[1][si]
                raw = noisy[:, yi, xi] / 100.0
                observed = np.zeros((n_plds, 2))
                for pp in range(n_plds):
                    observed[pp, 0] = raw[pp]
                    observed[pp, 1] = raw[n_plds + pp]
                obs_flat = np.concatenate([observed[:, 0], observed[:, 1]])
                try:
                    init = get_grid_search_initial_guess(obs_flat, plds.astype(np.float64), ls_params)
                except Exception:
                    init = [30.0 / 6000.0, 1500.0]
                try:
                    beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, observed, init, ls_params['T1_artery'], ls_params['T_tau'],
                        ls_params['T2_factor'], ls_params['alpha_BS1'], ls_params['alpha_PCASL'],
                        ls_params['alpha_VSASL'])
                    ls_cbf_val = np.clip(beta[0] * 6000.0, 0, 200)
                    ls_att_val = np.clip(beta[1], 0, 5000)
                    ls_errs_cbf.append(abs(ls_cbf_val - cbf_map[yi, xi]))
                    ls_errs_att.append(abs(ls_att_val - att_map[yi, xi]))
                    voxel_coords.append((p, yi, xi))
                except Exception:
                    pass

        snr_key = str(int(snr))
        cache[f'ls_errs_cbf_{snr_key}'] = np.array(ls_errs_cbf)
        cache[f'ls_errs_att_{snr_key}'] = np.array(ls_errs_att)
        cache[f'voxel_coords_{snr_key}'] = np.array(voxel_coords)
        print(f"    SNR={snr}: {len(ls_errs_cbf)} LS fits cached")

    np.savez(_ls_cache_path(), **cache)
    print(f"  LS cache saved to {_ls_cache_path()}")
    return cache


def synthetic_eval(model, cfg, norm_stats, device, n_phantoms=10, snr_levels=[3, 10, 25], seed=9999, models_and_stats=None):
    """Evaluate on synthetic phantoms with known ground truth. LS results are cached."""
    # Load or compute LS cache
    cache_path = _ls_cache_path()
    if cache_path.exists():
        ls_cache = dict(np.load(cache_path))
        print("  Using cached LS results")
    else:
        ls_cache = _compute_ls_cache(cfg, n_phantoms, snr_levels, seed)

    np.random.seed(seed)
    plds = np.array(cfg.get('pld_values', [500, 1000, 1500, 2000, 2500]), dtype=np.float32)
    asl_dict = {k: v for k, v in cfg.items() if k in ASLParameters.__annotations__}
    base_params = ASLParameters(**asl_dict)
    simulator = RealisticASLSimulator(params=base_params)
    ref_signal = simulator._compute_reference_signal()
    scalings = simulator.compute_tr_noise_scaling(plds)
    pld_scaling = {'PCASL': scalings['PCASL'], 'VSASL': scalings['VSASL']}
    n_plds = len(plds); plds_bc = plds[:, np.newaxis, np.newaxis]
    phantom_gen = SpatialPhantomGenerator(size=64, pve_sigma=3.0)
    global_scale = cfg.get('global_scale_factor', 10.0)
    dr_cfg = cfg.get('domain_randomization', {})

    model.eval()
    results = {}

    for snr in snr_levels:
        snr_key = str(int(snr))
        nn_errs_cbf, nn_errs_att = [], []
        cbf_slopes_data = []

        # Load cached LS errors and voxel coordinates
        ls_errs_cbf = ls_cache[f'ls_errs_cbf_{snr_key}']
        ls_errs_att = ls_cache[f'ls_errs_att_{snr_key}']
        voxel_coords = ls_cache[f'voxel_coords_{snr_key}']

        # Collect NN errors at the same voxels LS was evaluated on
        nn_sub_errs_cbf_list, nn_sub_errs_att_list = [], []

        # Regenerate phantoms with same seeds (deterministic)
        nn_cbf_maps = {}  # phantom_idx -> nn_cbf prediction
        nn_att_maps = {}

        for p in range(n_phantoms):
            np.random.seed(seed + p * 100 + int(snr * 10))
            cbf_map, att_map, _ = phantom_gen.generate_phantom(include_pathology=False)
            t1_b, alpha_p, alpha_v, tau = sample_physics(cfg, base_params, dr_cfg.get('enabled', True))
            clean = generate_asl_signal(cbf_map, att_map, plds_bc, t1_b, alpha_p, alpha_v, tau, base_params.T2_factor) * 100.0

            brain_mask = make_brain_mask_2d(64)
            clean[:, ~brain_mask] = 0.0
            cbf_map[~brain_mask] = 0.0
            att_map[~brain_mask] = 0.0

            noise_sigma = ref_signal * 100.0 / snr
            noisy = clean.copy()
            for ch in range(clean.shape[0]):
                scale = pld_scaling['PCASL'] if ch < n_plds else pld_scaling['VSASL']
                noisy[ch] += np.random.randn(64, 64).astype(np.float32) * noise_sigma * scale

            # NN prediction with 4-flip TTA
            with torch.no_grad():
                inp = torch.from_numpy(noisy[np.newaxis]).float().to(device)
                inp = torch.clamp(inp * global_scale, -30, 30)
                nn_cbf, nn_att = tta_predict_single(model, norm_stats, inp)

            # Post-processing Gaussian blur for denoising
            nn_cbf = gaussian_filter(nn_cbf, sigma=1.0)
            nn_att = gaussian_filter(nn_att, sigma=1.0)
            nn_cbf = np.clip(nn_cbf, 0, 200)
            nn_att = np.clip(nn_att, 0, 5000)

            bm = brain_mask
            nn_errs_cbf.append(np.abs(nn_cbf[bm] - cbf_map[bm]))
            nn_errs_att.append(np.abs(nn_att[bm] - att_map[bm]))
            cbf_slopes_data.append((nn_cbf[bm], cbf_map[bm]))

            nn_cbf_maps[p] = nn_cbf
            nn_att_maps[p] = nn_att
            # Store ground truth for voxel matching
            nn_cbf_maps[f'gt_cbf_{p}'] = cbf_map
            nn_att_maps[f'gt_att_{p}'] = att_map

        # Match NN errors to cached LS voxels
        for i, (p_idx, yi, xi) in enumerate(voxel_coords):
            p_idx, yi, xi = int(p_idx), int(yi), int(xi)
            if p_idx in nn_cbf_maps:
                gt_cbf = nn_cbf_maps[f'gt_cbf_{p_idx}'][yi, xi]
                gt_att = nn_att_maps[f'gt_att_{p_idx}'][yi, xi]
                nn_sub_errs_cbf_list.append(abs(nn_cbf_maps[p_idx][yi, xi] - gt_cbf))
                nn_sub_errs_att_list.append(abs(nn_att_maps[p_idx][yi, xi] - gt_att))

        nn_cbf_mae = float(np.mean(np.concatenate(nn_errs_cbf)))
        nn_att_mae = float(np.mean(np.concatenate(nn_errs_att)))
        ls_cbf_mae = float(np.mean(ls_errs_cbf)) if len(ls_errs_cbf) > 0 else 999
        ls_att_mae = float(np.mean(ls_errs_att)) if len(ls_errs_att) > 0 else 999

        # CBF slope
        all_pred = np.concatenate([d[0] for d in cbf_slopes_data])
        all_true = np.concatenate([d[1] for d in cbf_slopes_data])
        valid = np.isfinite(all_pred) & np.isfinite(all_true)
        cbf_slope = float(np.polyfit(all_true[valid], all_pred[valid], 1)[0]) if valid.sum() > 10 else float('nan')
        cbf_bias = float(np.mean(all_pred[valid] - all_true[valid]))

        # Per-voxel win rates
        nn_sub_errs_cbf = np.array(nn_sub_errs_cbf_list) if nn_sub_errs_cbf_list else np.array([])
        nn_sub_errs_att = np.array(nn_sub_errs_att_list) if nn_sub_errs_att_list else np.array([])

        n_matched = min(len(nn_sub_errs_cbf), len(ls_errs_cbf))
        if n_matched > 0:
            cbf_win = float(np.mean(nn_sub_errs_cbf[:n_matched] < ls_errs_cbf[:n_matched]) * 100)
            att_win = float(np.mean(nn_sub_errs_att[:n_matched] < ls_errs_att[:n_matched]) * 100)
        else:
            cbf_win, att_win = 0.0, 0.0

        results[snr_key] = {
            'nn_cbf_mae': nn_cbf_mae, 'nn_att_mae': nn_att_mae,
            'ls_cbf_mae': ls_cbf_mae, 'ls_att_mae': ls_att_mae,
            'cbf_slope': cbf_slope, 'cbf_bias': cbf_bias,
            'cbf_win_rate': cbf_win, 'att_win_rate': att_win,
        }

    return results


# =============================================================================
# Phase 3: In-vivo evaluation
# =============================================================================

def invivo_eval(model, norm_stats, cfg, device, data_dir, ls_cache_dir, subjects):
    """Evaluate on real patient data, compare to cached LS."""
    global_scale = cfg.get('global_scale_factor', 10.0)
    model.eval()
    all_results = {}

    for subj_id in subjects:
        subj_dir = Path(data_dir) / subj_id
        if not subj_dir.exists():
            print(f"    SKIP: {subj_id} not found")
            continue

        signals = np.load(subj_dir / 'high_snr_signals.npy')
        brain_mask = np.load(subj_dir / 'brain_mask.npy')
        gm_mask = np.load(subj_dir / 'gm_mask.npy')
        H, W, Z = brain_mask.shape
        n_plds = 5

        # NN inference
        signals_3d = signals.reshape(H, W, Z, 2 * n_plds)
        signals_spatial = np.transpose(signals_3d, (2, 3, 0, 1)).astype(np.float32)
        signals_spatial = np.clip(signals_spatial * 100.0 * global_scale, -30.0, 30.0)

        cbf_vol = np.zeros((Z, H, W), dtype=np.float32)
        att_vol = np.zeros((Z, H, W), dtype=np.float32)

        with torch.no_grad():
            for z in range(Z):
                inp = torch.from_numpy(signals_spatial[z:z + 1]).float().to(device)
                _, _, h, w = inp.shape
                pad_h = (16 - h % 16) % 16
                pad_w = (16 - w % 16) % 16
                if pad_h > 0 or pad_w > 0:
                    inp = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')
                # 4-flip TTA (same as synthetic eval)
                flips = [(False, False), (False, True), (True, False), (True, True)]
                cbf_preds, att_preds = [], []
                for flip_v, flip_h in flips:
                    aug = inp
                    if flip_v:
                        aug = torch.flip(aug, [2])
                    if flip_h:
                        aug = torch.flip(aug, [3])
                    pc, pa, _, _ = model(aug)
                    cbf_s = pc[0, 0, :h, :w].cpu().numpy()
                    att_s = pa[0, 0, :h, :w].cpu().numpy()
                    if flip_v:
                        cbf_s = np.flip(cbf_s, axis=0).copy()
                        att_s = np.flip(att_s, axis=0).copy()
                    if flip_h:
                        cbf_s = np.flip(cbf_s, axis=1).copy()
                        att_s = np.flip(att_s, axis=1).copy()
                    cbf_preds.append(cbf_s)
                    att_preds.append(att_s)
                cbf_vol[z] = np.mean(cbf_preds, axis=0) * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
                att_vol[z] = np.mean(att_preds, axis=0) * norm_stats['y_std_att'] + norm_stats['y_mean_att']

        nn_cbf = np.transpose(cbf_vol, (1, 2, 0))
        nn_att = np.transpose(att_vol, (1, 2, 0))
        # Post-processing Gaussian blur (per-slice, 2D)
        for z in range(nn_cbf.shape[2]):
            nn_cbf[:, :, z] = gaussian_filter(nn_cbf[:, :, z], sigma=1.0)
            nn_att[:, :, z] = gaussian_filter(nn_att[:, :, z], sigma=1.0)
        nn_cbf = np.clip(nn_cbf, 0, 200)
        nn_att = np.clip(nn_att, 0, 5000)

        # Load cached LS
        ls_cache = Path(ls_cache_dir) / subj_id
        ls_cbf = np.load(ls_cache / 'ls_cbf.npy')
        ls_att = np.load(ls_cache / 'ls_att.npy')

        # Metrics
        wm_mask = brain_mask & ~gm_mask
        nn_m = compute_tissue_metrics(nn_cbf, nn_att, brain_mask, gm_mask, wm_mask)
        ls_m = compute_tissue_metrics(ls_cbf, ls_att, brain_mask, gm_mask, wm_mask)

        all_results[subj_id] = {'nn': nn_m, 'ls': ls_m}

    return all_results


def compute_tissue_metrics(cbf, att, brain_mask, gm_mask, wm_mask):
    """Compute tissue-level quality metrics."""
    m = {}
    gm_cbf = cbf[gm_mask & np.isfinite(cbf)]
    if len(gm_cbf) > 10:
        m['gm_cbf_mean'] = float(np.mean(gm_cbf))
        m['gm_cbf_std'] = float(np.std(gm_cbf))
        m['gm_cbf_cov'] = float(np.std(gm_cbf) / max(np.mean(gm_cbf), 0.1) * 100)

    wm_cbf = cbf[wm_mask & np.isfinite(cbf)]
    if len(wm_cbf) > 10:
        m['wm_cbf_mean'] = float(np.mean(wm_cbf))

    if 'gm_cbf_mean' in m and 'wm_cbf_mean' in m and m['wm_cbf_mean'] > 0.1:
        m['gm_wm_ratio'] = m['gm_cbf_mean'] / m['wm_cbf_mean']

    gm_att = att[gm_mask & np.isfinite(att)]
    if len(gm_att) > 10:
        m['gm_att_mean'] = float(np.mean(gm_att))

    # Spatial smoothness
    bc = cbf.copy()
    bc[~brain_mask | ~np.isfinite(bc)] = 0
    gx = np.diff(bc, axis=0); gy = np.diff(bc, axis=1)
    mx = brain_mask[:-1, :, :] & brain_mask[1:, :, :]
    my = brain_mask[:, :-1, :] & brain_mask[:, 1:, :]
    if mx.sum() > 0 and my.sum() > 0:
        m['smoothness'] = float((np.mean(np.abs(gx[mx])) + np.mean(np.abs(gy[my]))) / 2)

    return m


# =============================================================================
# Output
# =============================================================================

def print_and_save(synth_results, invivo_results, loss_history, output_dir):
    """Print formatted output and save to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*95}")
    print(f" SYNTHETIC EVALUATION")
    print(f"{'='*95}")
    print(f"{'SNR':>5} | {'NN CBF':>7} {'LS CBF':>7} | {'NN ATT':>7} {'LS ATT':>7} | "
          f"{'CBF Win':>8} {'ATT Win':>8} | {'Slope':>6} {'Bias':>6}")
    print("-" * 95)
    for snr_key in sorted(synth_results.keys(), key=int):
        sr = synth_results[snr_key]
        print(f"{snr_key:>5} | {sr['nn_cbf_mae']:>6.1f}  {sr['ls_cbf_mae']:>6.1f}  | "
              f"{sr['nn_att_mae']:>6.0f}  {sr['ls_att_mae']:>6.0f}  | "
              f"{sr['cbf_win_rate']:>6.1f}%  {sr['att_win_rate']:>6.1f}% | "
              f"{sr['cbf_slope']:>5.2f} {sr['cbf_bias']:>+5.1f}")

    print(f"\n{'='*85}")
    print(f" IN-VIVO EVALUATION")
    print(f"{'='*85}")
    print(f"{'Subject':<22} | {'Meth':4} | {'GM CBF':>7} {'CoV%':>6} | {'WM CBF':>7} | {'GM/WM':>5} | {'GM ATT':>7} | {'Smooth':>7}")
    print("-" * 85)

    nn_covs, ls_covs, nn_smooths, ls_smooths = [], [], [], []
    nn_cbfs, nn_ratios, nn_atts = [], [], []
    physio_pass = True

    for subj in sorted(invivo_results.keys()):
        nn, ls = invivo_results[subj]['nn'], invivo_results[subj]['ls']
        def fmt(d, k, f='{:.1f}'): return f.format(d[k]) if k in d else '  N/A'

        print(f"{subj:<22} | {'NN':4} | {fmt(nn,'gm_cbf_mean'):>7} {fmt(nn,'gm_cbf_cov'):>5}% | "
              f"{fmt(nn,'wm_cbf_mean'):>7} | {fmt(nn,'gm_wm_ratio','{:.2f}'):>5} | "
              f"{fmt(nn,'gm_att_mean','{:.0f}'):>7} | {fmt(nn,'smoothness','{:.2f}'):>7}")
        print(f"{'':22} | {'LS':4} | {fmt(ls,'gm_cbf_mean'):>7} {fmt(ls,'gm_cbf_cov'):>5}% | "
              f"{fmt(ls,'wm_cbf_mean'):>7} | {fmt(ls,'gm_wm_ratio','{:.2f}'):>5} | "
              f"{fmt(ls,'gm_att_mean','{:.0f}'):>7} | {fmt(ls,'smoothness','{:.2f}'):>7}")
        print("-" * 85)

        if 'gm_cbf_cov' in nn: nn_covs.append(nn['gm_cbf_cov'])
        if 'gm_cbf_cov' in ls: ls_covs.append(ls['gm_cbf_cov'])
        if 'smoothness' in nn: nn_smooths.append(nn['smoothness'])
        if 'smoothness' in ls: ls_smooths.append(ls['smoothness'])
        if 'gm_cbf_mean' in nn: nn_cbfs.append(nn['gm_cbf_mean'])
        if 'gm_wm_ratio' in nn: nn_ratios.append(nn['gm_wm_ratio'])
        if 'gm_att_mean' in nn: nn_atts.append(nn['gm_att_mean'])

        gcbf = nn.get('gm_cbf_mean', 0)
        grat = nn.get('gm_wm_ratio', 0)
        gatt = nn.get('gm_att_mean', 0)
        if not (15 < gcbf < 120 and 1.2 < grat < 6.0 and 400 < gatt < 3000):
            physio_pass = False

    # --- Synthetic checks ---
    synth_sanity = True
    synth_nn_wins_cbf = True
    synth_nn_wins_att = True
    for sr in synth_results.values():
        if sr['cbf_slope'] < 0.3 or sr['cbf_slope'] > 1.8:
            synth_sanity = False
        if abs(sr['cbf_bias']) > 15:
            synth_sanity = False
        if sr['cbf_win_rate'] <= 50:
            synth_nn_wins_cbf = False
        if sr['att_win_rate'] <= 50:
            synth_nn_wins_att = False

    # --- In-vivo checks ---
    nn_cov_avg = np.mean(nn_covs) if nn_covs else 999
    ls_cov_avg = np.mean(ls_covs) if ls_covs else 999
    nn_smooth_avg = np.mean(nn_smooths) if nn_smooths else 999
    ls_smooth_avg = np.mean(ls_smooths) if ls_smooths else 999
    cov_wins = nn_cov_avg < ls_cov_avg
    smooth_wins = nn_smooth_avg < ls_smooth_avg

    # --- Concordance: both synthetic and in-vivo agree NN wins ---
    concordance = synth_nn_wins_cbf and cov_wins

    # --- Overall: NN beats LS across ALL metrics in BOTH domains ---
    overall = (
        synth_sanity and
        synth_nn_wins_cbf and
        synth_nn_wins_att and
        physio_pass and
        cov_wins and
        smooth_wins
    )

    print(f"\n  SYNTHETIC CHECKS:")
    print(f"    Slope/bias sanity:                {'PASS' if synth_sanity else 'FAIL'}")
    print(f"    NN CBF win >50% all SNR:          {'PASS' if synth_nn_wins_cbf else 'FAIL'}")
    for snr_key in sorted(synth_results.keys(), key=int):
        sr = synth_results[snr_key]
        print(f"      SNR={snr_key}: CBF win={sr['cbf_win_rate']:.1f}%, ATT win={sr['att_win_rate']:.1f}%")
    print(f"    NN ATT win >50% all SNR:          {'PASS' if synth_nn_wins_att else 'FAIL'}")

    print(f"\n  IN-VIVO CHECKS:")
    print(f"    Physiological plausibility:       {'PASS' if physio_pass else 'FAIL'}")
    print(f"    NN CoV < LS CoV:                  {'PASS' if cov_wins else 'FAIL'} "
          f"(NN={nn_cov_avg:.1f}% vs LS={ls_cov_avg:.1f}%)")
    print(f"    NN smoother than LS:              {'PASS' if smooth_wins else 'FAIL'} "
          f"(NN={nn_smooth_avg:.2f} vs LS={ls_smooth_avg:.2f})")
    print(f"    GM CBF avg:                       {np.mean(nn_cbfs):.1f} (expected 30-90)")
    print(f"    GM/WM ratio avg:                  {np.mean(nn_ratios):.2f} (expected 2.0-3.5)")
    print(f"    GM ATT avg:                       {np.mean(nn_atts):.0f}ms (expected 800-1800)")

    print(f"\n  CONCORDANCE:")
    print(f"    Synth + in-vivo both say NN wins: {'PASS' if concordance else 'FAIL'}")

    print(f"\n    OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"{'='*95}\n")

    # Save
    result = {
        'timestamp': datetime.now().isoformat(),
        'loss_history': loss_history,
        'synthetic': synth_results,
        'invivo': {s: invivo_results[s] for s in invivo_results},
        'checks': {
            'synth_sanity': synth_sanity,
            'synth_nn_wins_cbf': synth_nn_wins_cbf,
            'synth_nn_wins_att': synth_nn_wins_att,
            'physio_pass': physio_pass,
            'cov_wins': cov_wins, 'smooth_wins': smooth_wins,
            'concordance': concordance, 'overall': overall,
            'nn_cov_avg': nn_cov_avg, 'ls_cov_avg': ls_cov_avg,
            'nn_smooth_avg': nn_smooth_avg, 'ls_smooth_avg': ls_smooth_avg,
            'nn_cbf_avg': float(np.mean(nn_cbfs)) if nn_cbfs else 0,
            'nn_ratio_avg': float(np.mean(nn_ratios)) if nn_ratios else 0,
            'nn_att_avg': float(np.mean(nn_atts)) if nn_atts else 0,
        }
    }

    with open(output_dir / 'latest_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Append log
    log_path = output_dir / 'experiment_log.md'
    if not log_path.exists():
        log_path.write_text("# Ralph Harness Experiment Log\n\n")
    with open(log_path, 'a') as f:
        f.write(f"## {result['timestamp']}\n\n")
        f.write(f"- **Loss**: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}\n")
        for snr_key in sorted(synth_results.keys(), key=int):
            sr = synth_results[snr_key]
            f.write(f"- **Synth SNR={snr_key}**: NN CBF={sr['nn_cbf_mae']:.1f}, "
                    f"LS={sr['ls_cbf_mae']:.1f}, CBF win={sr['cbf_win_rate']:.1f}%, "
                    f"ATT win={sr['att_win_rate']:.1f}%, slope={sr['cbf_slope']:.2f}\n")
        c = result['checks']
        f.write(f"- **In-vivo CoV**: NN={c['nn_cov_avg']:.1f}% vs LS={c['ls_cov_avg']:.1f}% "
                f"({'PASS' if c['cov_wins'] else 'FAIL'})\n")
        f.write(f"- **In-vivo Smooth**: NN={c['nn_smooth_avg']:.2f} vs LS={c['ls_smooth_avg']:.2f} "
                f"({'PASS' if c['smooth_wins'] else 'FAIL'})\n")
        f.write(f"- **GM CBF**: {c['nn_cbf_avg']:.1f}, **GM/WM**: {c['nn_ratio_avg']:.2f}, "
                f"**ATT**: {c['nn_att_avg']:.0f}ms\n")
        f.write(f"- **OVERALL**: {'PASS' if c['overall'] else 'FAIL'}\n\n---\n\n")

    return overall


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ralph Harness — synthetic + in-vivo evaluation')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--n-samples', type=int, default=3000)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--data-dir', default='data/invivo_processed_npy')
    parser.add_argument('--ls-cache-dir', default='invivo_comparison_results')
    parser.add_argument('--output-dir', default='invivo_results')
    parser.add_argument('--subjects', nargs='+',
                        default=['20231030_MR1_A152', '20231003_MR1_A142', '20231016_MR1_A147'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    t_start = time.time()
    cfg = load_config()
    print(f"Device: {device}, Config: config/invivo_experiment.yaml")

    # Phase 1: Train single model (ensemble disabled for fast iteration)
    print(f"\n[Phase 1] Generating {args.n_samples} samples + training {args.n_epochs} epochs...")
    t1 = time.time()
    signals, targets, norm_stats = generate_training_data(cfg, args.n_samples, args.seed)
    print(f"  Data: {time.time()-t1:.0f}s, norm_stats: CBF={norm_stats['y_mean_cbf']:.1f}+/-{norm_stats['y_std_cbf']:.1f}")

    model, loss_history = train_model(cfg, signals, targets, norm_stats, device, args.n_epochs, args.seed)
    print(f"  Training: {time.time()-t1:.0f}s total")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'norm_stats': norm_stats},
               output_dir / 'trained_model.pt')

    # Phase 2: Synthetic eval
    print(f"\n[Phase 2] Synthetic evaluation (3 SNR levels, 10 phantoms)...")
    t2 = time.time()
    synth_results = synthetic_eval(model, cfg, norm_stats, device)
    print(f"  Synthetic eval: {time.time()-t2:.0f}s")

    # Phase 3: In-vivo eval
    print(f"\n[Phase 3] In-vivo evaluation ({len(args.subjects)} subjects)...")
    t3 = time.time()
    invivo_results = invivo_eval(model, norm_stats, cfg, device, args.data_dir, args.ls_cache_dir, args.subjects)
    print(f"  In-vivo eval: {time.time()-t3:.1f}s")

    # Output
    overall = print_and_save(synth_results, invivo_results, loss_history, args.output_dir)
    print(f"Total time: {time.time()-t_start:.0f}s")
    print(f"Results: {args.output_dir}/latest_results.json")
    print(f"Log: {args.output_dir}/experiment_log.md")


if __name__ == '__main__':
    main()

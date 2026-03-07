#!/usr/bin/env python3
"""
Fast Train & Validate — Local test harness for rapid ASL model iteration.

Generates data, trains a model, and validates in ~5 minutes on MPS/CPU.
Designed to be the inner loop of the Ralph Loop for autonomous model improvement.

Usage:
    python fast_train_validate.py --config config/fast_experiment.yaml
    python fast_train_validate.py --config config/fast_experiment.yaml --n-epochs 2 --n-samples 100 --skip-ls
    python fast_train_validate.py --config config/fast_experiment.yaml --device mps --seed 42
"""
import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings('ignore', category=UserWarning)

# ─── Project imports ─────────────────────────────────────────────────────────
from simulation.asl_simulation import ASLParameters
from simulation.enhanced_simulation import RealisticASLSimulator, SpatialPhantomGenerator
from simulation.noise_engine import NoiseInjector
from models.spatial_asl_network import SpatialASLNet, MaskedSpatialLoss
from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
from baselines.multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from utils.helpers import get_grid_search_initial_guess


# =============================================================================
# Phase 1: Config + Setup
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config and flatten into a single dict matching ResearchConfig fields."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    cfg = {}
    # Merge sections in priority order
    for section in ['simulation', 'data', 'training']:
        if section in raw:
            cfg.update(raw[section])

    # noise_config stays nested
    if 'noise_config' in raw:
        cfg['noise_config'] = raw['noise_config']

    # domain_randomization from simulation section
    if 'simulation' in raw and 'domain_randomization' in raw['simulation']:
        cfg['domain_randomization'] = raw['simulation']['domain_randomization']

    return cfg


def resolve_device(requested: str) -> torch.device:
    if requested == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if requested == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if requested in ('mps', 'cuda'):
        print(f"[WARN] Requested device '{requested}' not available, falling back to CPU")
    return torch.device('cpu')


# =============================================================================
# Phase 2: Generate Training Data On-the-Fly
# =============================================================================

def generate_training_data(cfg: dict, n_samples: int, seed: int):
    """Generate spatial phantom training data in memory.

    Returns:
        signals: (N, 12, 64, 64) float32
        targets: (N, 2, 64, 64) float32  [CBF, ATT]
        norm_stats: dict with y_mean_cbf, y_std_cbf, y_mean_att, y_std_att
    """
    np.random.seed(seed)
    plds = np.array(cfg.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000]), dtype=np.float32)
    n_plds = len(plds)

    # Physics params
    asl_dict = {k: v for k, v in cfg.items() if k in ASLParameters.__annotations__}
    asl_params = ASLParameters(**asl_dict)

    # Domain randomization config
    domain_rand = cfg.get('domain_randomization', {})
    use_domain_rand = domain_rand.get('enabled', False)
    if use_domain_rand:
        t1_range = domain_rand.get('T1_artery_range', [1550.0, 2150.0])
        alpha_pcasl_range = domain_rand.get('alpha_PCASL_range', [0.75, 0.95])
        alpha_vsasl_range = domain_rand.get('alpha_VSASL_range', [0.40, 0.70])
        alpha_bs1_range = domain_rand.get('alpha_BS1_range', [0.85, 1.0])
        t_tau_perturb = domain_rand.get('T_tau_perturb', 0.10)

    phantom_gen = SpatialPhantomGenerator(size=64, pve_sigma=1.0)
    plds_bc = plds[:, np.newaxis, np.newaxis]
    lambda_b = 0.90

    signals_list = []
    targets_list = []

    for i in range(n_samples):
        cbf_map, att_map, _ = phantom_gen.generate_phantom(include_pathology=True)

        # Sample physics per phantom
        if use_domain_rand:
            t1_b = np.random.uniform(*t1_range)
            alpha_bs1 = np.random.uniform(*alpha_bs1_range)
            alpha_p = np.random.uniform(*alpha_pcasl_range) * (alpha_bs1 ** 4)
            alpha_v = np.random.uniform(*alpha_vsasl_range) * (alpha_bs1 ** 3)
            tau = asl_params.T_tau * (1 + np.random.uniform(-t_tau_perturb, t_tau_perturb))
        else:
            t1_b = asl_params.T1_artery
            alpha_p = asl_params.alpha_PCASL * (asl_params.alpha_BS1 ** 4)
            alpha_v = asl_params.alpha_VSASL * (asl_params.alpha_BS1 ** 3)
            tau = asl_params.T_tau

        t2_f = asl_params.T2_factor
        att_bc = att_map[np.newaxis, :, :].astype(np.float32)
        cbf_bc = (cbf_map / 6000.0)[np.newaxis, :, :].astype(np.float32)

        # PCASL
        mask_arrived = (plds_bc >= att_bc)
        mask_transit = (plds_bc < att_bc) & (plds_bc >= (att_bc - tau))

        sig_p_arrived = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                         np.exp(-plds_bc / t1_b) *
                         (1 - np.exp(-tau / t1_b)) * t2_f) / lambda_b

        sig_p_transit = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                         (np.exp(-att_bc / t1_b) - np.exp(-(tau + plds_bc) / t1_b)) *
                         t2_f) / lambda_b

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

        clean_signal = np.concatenate([pcasl_sig, vsasl_sig], axis=0).astype(np.float32)
        target = np.stack([cbf_map, att_map], axis=0).astype(np.float32)

        signals_list.append(clean_signal)
        targets_list.append(target)

    signals = np.array(signals_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)

    # M0 scaling (matching SpatialDataset.M0_SCALE_FACTOR = 100)
    signals *= 100.0

    # Compute norm stats from targets (brain voxels only)
    # Use simple mean of all target pixels > threshold
    cbf_all = targets[:, 0, :, :]
    att_all = targets[:, 1, :, :]
    brain_mask = cbf_all > 1.0  # Approximate brain region

    cbf_brain = cbf_all[brain_mask]
    att_brain = att_all[brain_mask]

    norm_stats = {
        'y_mean_cbf': float(np.mean(cbf_brain)),
        'y_std_cbf': float(np.std(cbf_brain)),
        'y_mean_att': float(np.mean(att_brain)),
        'y_std_att': float(np.std(att_brain)),
    }

    return signals, targets, norm_stats


# =============================================================================
# Phase 3: Train Single Model
# =============================================================================

def create_model(cfg: dict, device: torch.device):
    """Instantiate the model from config."""
    model_class_name = cfg.get('model_class_name', 'SpatialASLNet')
    n_plds = len(cfg.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000]))
    features = cfg.get('hidden_sizes', [32, 64, 128, 256])

    model_kwargs = dict(n_plds=n_plds, features=features)

    if model_class_name == 'AmplitudeAwareSpatialASLNet':
        model_kwargs['use_film_at_bottleneck'] = cfg.get('use_film_at_bottleneck', True)
        model_kwargs['use_film_at_decoder'] = cfg.get('use_film_at_decoder', True)
        model_kwargs['use_amplitude_output_modulation'] = cfg.get('use_amplitude_output_modulation', True)
        model = AmplitudeAwareSpatialASLNet(**model_kwargs)
    elif model_class_name == 'SpatialASLNet':
        model = SpatialASLNet(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model_class_name: {model_class_name}")

    return model.to(device)


def create_brain_mask(signals: np.ndarray) -> np.ndarray:
    """Create brain mask from signal magnitude. signals: (C, H, W)."""
    mean_sig = np.mean(np.abs(signals), axis=0)
    return (mean_sig > np.percentile(mean_sig, 5)).astype(np.float32)


def train_model(cfg: dict, model, signals: np.ndarray, targets: np.ndarray,
                norm_stats: dict, device: torch.device, n_epochs: int, seed: int):
    """Train a single model (no ensemble, no wandb, no AMP).

    Returns:
        loss_history: list of epoch-average losses
    """
    torch.manual_seed(seed)
    np.random.seed(seed + 100)

    # Train/val split (90/10)
    n = len(signals)
    n_val = max(1, int(n * 0.1))
    n_train = n - n_val
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_signals = torch.from_numpy(signals[train_idx]).float()
    train_targets_np = targets[train_idx]
    val_signals = torch.from_numpy(signals[val_idx]).float()
    val_targets_np = targets[val_idx]

    # Loss function
    loss_fn = MaskedSpatialLoss(
        loss_type=cfg.get('loss_type', 'l1'),
        dc_weight=cfg.get('dc_weight', 0.0),
        att_scale=cfg.get('att_scale', 1.0),
        cbf_weight=cfg.get('cbf_weight', 1.0),
        att_weight=cfg.get('att_weight', 1.0),
        norm_stats=norm_stats,
        variance_weight=cfg.get('variance_weight', 0.1),
    )

    # Noise injector
    noise_cfg = {
        'noise_config': cfg.get('noise_config', {'snr_range': [2.0, 25.0]}),
        'data_noise_components': cfg.get('data_noise_components', ['thermal']),
        'noise_type': cfg.get('noise_type', 'rician'),
    }
    noise_injector = NoiseInjector(noise_cfg)

    # Reference signal for noise scaling
    asl_dict = {k: v for k, v in cfg.items() if k in ASLParameters.__annotations__}
    asl_params = ASLParameters(**asl_dict)
    simulator = RealisticASLSimulator(params=asl_params)
    ref_signal = simulator._compute_reference_signal()
    pld_list = cfg.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000])
    scalings = simulator.compute_tr_noise_scaling(np.array(pld_list))
    pld_scaling = {'PCASL': scalings['PCASL'], 'VSASL': scalings['VSASL']}

    global_scale = cfg.get('global_scale_factor', 10.0)
    batch_size = cfg.get('batch_size', 32)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.get('learning_rate', 0.0001),
                                  weight_decay=cfg.get('weight_decay', 0.0001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=cfg.get('learning_rate', 0.001) * 0.01)

    loss_history = []

    # Curriculum learning: fraction of epochs to train clean (no noise)
    clean_fraction = 0.15  # First 15% of epochs are clean

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        perm_train = np.random.permutation(n_train)
        use_noise = epoch >= int(n_epochs * clean_fraction)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm_train[start:end]

            raw_sig = train_signals[idx].to(device)
            cbf_t = torch.from_numpy(train_targets_np[idx, 0:1]).float().to(device)
            att_t = torch.from_numpy(train_targets_np[idx, 1:2]).float().to(device)

            # Create brain masks
            masks = []
            for j in range(len(idx)):
                m = create_brain_mask(train_signals[idx[j]].numpy())
                masks.append(m)
            mask_t = torch.from_numpy(np.array(masks)[:, np.newaxis]).float().to(device)

            # Apply noise (skip during clean pretraining phase)
            if use_noise:
                noisy_sig = noise_injector.apply_noise(raw_sig, ref_signal, pld_scaling)
            else:
                noisy_sig = raw_sig

            # Global scale normalization
            normalized = torch.clamp(noisy_sig * global_scale, -15.0, 15.0)

            optimizer.zero_grad()
            pred_cbf, pred_att, log_var_cbf, log_var_att = model(normalized)

            loss_dict = loss_fn(pred_cbf.float(), pred_att.float(),
                                cbf_t, att_t, mask_t, normalized,
                                log_var_cbf=log_var_cbf.float(),
                                log_var_att=log_var_att.float())
            loss = loss_dict['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{n_epochs}: loss = {avg_loss:.4f}")

    return loss_history


# =============================================================================
# Phase 4: Generate Test Phantoms
# =============================================================================

def generate_test_phantoms(cfg: dict, n_phantoms: int, snr_levels: list, seed: int):
    """Generate reproducible test phantoms at multiple SNR levels.

    Uses per-phantom random physics parameters to simulate real-world patient
    variability. LS fitting uses fixed consensus parameters (realistic mismatch).

    Returns:
        test_data: list of dicts with keys:
            'snr', 'clean_signals', 'noisy_signals', 'cbf_map', 'att_map', 'brain_mask'
    """
    plds = np.array(cfg.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000]), dtype=np.float32)
    asl_dict = {k: v for k, v in cfg.items() if k in ASLParameters.__annotations__}
    asl_params = ASLParameters(**asl_dict)
    simulator = RealisticASLSimulator(params=asl_params)
    ref_signal = simulator._compute_reference_signal()
    scalings = simulator.compute_tr_noise_scaling(plds)
    pld_scaling = {'PCASL': scalings['PCASL'], 'VSASL': scalings['VSASL']}

    # Domain randomization ranges for test data (matching training)
    domain_rand = cfg.get('domain_randomization', {})
    use_domain_rand = domain_rand.get('enabled', False)

    phantom_gen = SpatialPhantomGenerator(size=64, pve_sigma=1.0)
    plds_bc = plds[:, np.newaxis, np.newaxis]
    lambda_b = 0.90

    test_data = []

    for snr in snr_levels:
        for p_idx in range(n_phantoms):
            np.random.seed(seed + p_idx * 100 + int(snr * 10))
            cbf_map, att_map, _ = phantom_gen.generate_phantom(include_pathology=True)

            # Per-phantom random physics (simulates patient variability)
            if use_domain_rand:
                t1_range = domain_rand.get('T1_artery_range', [1550.0, 2150.0])
                alpha_pcasl_range = domain_rand.get('alpha_PCASL_range', [0.75, 0.95])
                alpha_vsasl_range = domain_rand.get('alpha_VSASL_range', [0.40, 0.70])
                alpha_bs1_range = domain_rand.get('alpha_BS1_range', [0.85, 1.0])
                t_tau_perturb = domain_rand.get('T_tau_perturb', 0.10)

                t1_b = np.random.uniform(*t1_range)
                alpha_bs1 = np.random.uniform(*alpha_bs1_range)
                alpha_p = np.random.uniform(*alpha_pcasl_range) * (alpha_bs1 ** 4)
                alpha_v = np.random.uniform(*alpha_vsasl_range) * (alpha_bs1 ** 3)
                tau = asl_params.T_tau * (1 + np.random.uniform(-t_tau_perturb, t_tau_perturb))
            else:
                t1_b = asl_params.T1_artery
                alpha_p = asl_params.alpha_PCASL * (asl_params.alpha_BS1 ** 4)
                alpha_v = asl_params.alpha_VSASL * (asl_params.alpha_BS1 ** 3)
                tau = asl_params.T_tau

            t2_f = asl_params.T2_factor

            att_bc = att_map[np.newaxis, :, :].astype(np.float32)
            cbf_bc = (cbf_map / 6000.0)[np.newaxis, :, :].astype(np.float32)

            # PCASL
            mask_arrived = (plds_bc >= att_bc)
            mask_transit = (plds_bc < att_bc) & (plds_bc >= (att_bc - tau))
            sig_p_arrived = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                             np.exp(-plds_bc / t1_b) *
                             (1 - np.exp(-tau / t1_b)) * t2_f) / lambda_b
            sig_p_transit = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                             (np.exp(-att_bc / t1_b) - np.exp(-(tau + plds_bc) / t1_b)) *
                             t2_f) / lambda_b
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

            clean_signal = np.concatenate([pcasl_sig, vsasl_sig], axis=0).astype(np.float32)
            clean_signal *= 100.0  # M0 scaling

            # Add noise at fixed SNR
            noisy = clean_signal.copy()
            noise_sigma = ref_signal * 100.0 / snr  # Account for M0 scaling
            n_plds = len(plds)
            for ch in range(clean_signal.shape[0]):
                if ch < n_plds:
                    scale = pld_scaling['PCASL']
                else:
                    scale = pld_scaling['VSASL']
                noisy[ch] += np.random.randn(64, 64).astype(np.float32) * noise_sigma * scale

            # Brain mask
            mean_sig = np.mean(np.abs(clean_signal), axis=0)
            brain_mask = (mean_sig > np.percentile(mean_sig, 5)).astype(np.float32)

            test_data.append({
                'snr': snr,
                'clean_signals': clean_signal,
                'noisy_signals': noisy,
                'cbf_map': cbf_map,
                'att_map': att_map,
                'brain_mask': brain_mask,
            })

    return test_data


# =============================================================================
# Phase 5: NN Inference + LS Fitting
# =============================================================================

def nn_inference(model, test_data: list, norm_stats: dict, cfg: dict, device: torch.device):
    """Run NN inference on test phantoms. Returns predictions added to test_data dicts."""
    model.eval()
    global_scale = cfg.get('global_scale_factor', 10.0)

    with torch.no_grad():
        for td in test_data:
            sig = torch.from_numpy(td['noisy_signals'][np.newaxis]).float().to(device)
            normalized = torch.clamp(sig * global_scale, -15.0, 15.0)
            pred_cbf, pred_att, _, _ = model(normalized)

            # Denormalize
            cbf_pred = pred_cbf[0, 0].cpu().numpy() * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
            att_pred = pred_att[0, 0].cpu().numpy() * norm_stats['y_std_att'] + norm_stats['y_mean_att']

            td['nn_cbf'] = cbf_pred
            td['nn_att'] = att_pred

    return test_data


def ls_fitting(test_data: list, cfg: dict, subsample_frac: float = 0.1):
    """Run LS fitting on a subsample of brain voxels."""
    plds = np.array(cfg.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000]), dtype=np.float64)
    asl_dict = {k: v for k, v in cfg.items() if k in ASLParameters.__annotations__}
    asl_params = ASLParameters(**asl_dict)

    t1_artery = asl_params.T1_artery
    t_tau = asl_params.T_tau
    t2_factor = asl_params.T2_factor
    alpha_bs1 = asl_params.alpha_BS1
    alpha_pcasl = asl_params.alpha_PCASL
    alpha_vsasl = asl_params.alpha_VSASL

    n_plds = len(plds)
    PLDTI = np.column_stack([plds, plds])  # PLD = TI for simultaneous acquisition

    asl_params_dict = {
        'T1_artery': t1_artery, 'T_tau': t_tau, 'T2_factor': t2_factor,
        'alpha_BS1': alpha_bs1, 'alpha_PCASL': alpha_pcasl, 'alpha_VSASL': alpha_vsasl,
    }

    for td in test_data:
        mask = td['brain_mask']
        brain_idx = np.where(mask > 0.5)
        n_brain = len(brain_idx[0])
        n_sub = max(10, int(n_brain * subsample_frac))
        sub_idx = np.random.choice(n_brain, size=min(n_sub, n_brain), replace=False)

        ls_cbf = np.full((64, 64), np.nan)
        ls_att = np.full((64, 64), np.nan)

        # Undo M0 scaling for LS fitting (signals were *100 scaled)
        raw_signals = td['noisy_signals'] / 100.0

        for si in sub_idx:
            y_idx, x_idx = brain_idx[0][si], brain_idx[1][si]
            observed = np.zeros((n_plds, 2))
            for p in range(n_plds):
                observed[p, 0] = raw_signals[p, y_idx, x_idx]
                observed[p, 1] = raw_signals[n_plds + p, y_idx, x_idx]

            # Grid search for initial guess
            obs_flat = np.concatenate([observed[:, 0], observed[:, 1]])
            try:
                init = get_grid_search_initial_guess(obs_flat, plds, asl_params_dict)
            except Exception:
                init = [30.0 / 6000.0, 1500.0]

            try:
                beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                    PLDTI, observed, init, t1_artery, t_tau, t2_factor,
                    alpha_bs1, alpha_pcasl, alpha_vsasl)
                ls_cbf[y_idx, x_idx] = beta[0] * 6000.0  # Convert back to ml/100g/min
                ls_att[y_idx, x_idx] = beta[1]
            except Exception:
                pass

        td['ls_cbf'] = ls_cbf
        td['ls_att'] = ls_att

    return test_data


# =============================================================================
# Phase 6: Diagnostics + Output
# =============================================================================

def compute_diagnostics(test_data: list, loss_history: list, has_ls: bool):
    """Compute per-SNR diagnostics and kill signals."""

    snr_levels = sorted(set(td['snr'] for td in test_data))
    results = {'snr_results': {}, 'loss_history': loss_history, 'kill_signals': []}

    for snr in snr_levels:
        snr_data = [td for td in test_data if td['snr'] == snr]
        snr_key = str(int(snr))

        # Gather brain-voxel predictions
        nn_cbf_all, true_cbf_all = [], []
        nn_att_all, true_att_all = [], []
        ls_cbf_all, ls_att_all = [], []

        for td in snr_data:
            mask = td['brain_mask'] > 0.5
            nn_cbf_all.append(td['nn_cbf'][mask])
            nn_att_all.append(td['nn_att'][mask])
            true_cbf_all.append(td['cbf_map'][mask])
            true_att_all.append(td['att_map'][mask])
            if has_ls:
                ls_mask = mask & ~np.isnan(td['ls_cbf'])
                if ls_mask.any():
                    ls_cbf_all.append(td['ls_cbf'][ls_mask])
                    ls_att_all.append(td['ls_att'][ls_mask])

        nn_cbf = np.concatenate(nn_cbf_all)
        nn_att = np.concatenate(nn_att_all)
        true_cbf = np.concatenate(true_cbf_all)
        true_att = np.concatenate(true_att_all)

        # Linearity (slope via simple regression)
        def lin_slope(pred, truth):
            if len(pred) < 10:
                return float('nan')
            # slope = cov(pred, truth) / var(truth)
            valid = np.isfinite(pred) & np.isfinite(truth)
            p, t = pred[valid], truth[valid]
            if len(p) < 10 or np.std(t) < 1e-6:
                return float('nan')
            return float(np.polyfit(t, p, 1)[0])

        cbf_slope = lin_slope(nn_cbf, true_cbf)
        att_slope = lin_slope(nn_att, true_att)

        # Bias and MAE
        cbf_bias = float(np.mean(nn_cbf - true_cbf))
        att_bias = float(np.mean(nn_att - true_att))
        cbf_mae = float(np.mean(np.abs(nn_cbf - true_cbf)))
        att_mae = float(np.mean(np.abs(nn_att - true_att)))

        # CoV
        cbf_cov = float(np.std(nn_cbf - true_cbf) / max(np.mean(true_cbf), 1e-6) * 100)

        # Win rate vs LS
        cbf_win = None
        att_win = None
        if has_ls and ls_cbf_all:
            ls_cbf = np.concatenate(ls_cbf_all)
            ls_att = np.concatenate(ls_att_all)
            # Need matching ground truth for LS subset
            ls_true_cbf, ls_true_att = [], []
            ls_nn_cbf, ls_nn_att = [], []
            for td in snr_data:
                mask = td['brain_mask'] > 0.5
                ls_mask = mask & ~np.isnan(td['ls_cbf'])
                if ls_mask.any():
                    ls_true_cbf.append(td['cbf_map'][ls_mask])
                    ls_true_att.append(td['att_map'][ls_mask])
                    ls_nn_cbf.append(td['nn_cbf'][ls_mask])
                    ls_nn_att.append(td['nn_att'][ls_mask])

            if ls_true_cbf:
                lt_cbf = np.concatenate(ls_true_cbf)
                lt_att = np.concatenate(ls_true_att)
                ln_cbf = np.concatenate(ls_nn_cbf)
                ln_att = np.concatenate(ls_nn_att)

                nn_err_cbf = np.abs(ln_cbf - lt_cbf)
                ls_err_cbf = np.abs(ls_cbf - lt_cbf)
                cbf_win = float(np.mean(nn_err_cbf < ls_err_cbf) * 100)

                nn_err_att = np.abs(ln_att - lt_att)
                ls_err_att = np.abs(ls_att - lt_att)
                att_win = float(np.mean(nn_err_att < ls_err_att) * 100)

                # Also compute LS MAE for comparison
                ls_cbf_mae = float(np.mean(ls_err_cbf))
                ls_att_mae = float(np.mean(ls_err_att))

        snr_result = {
            'cbf_slope': cbf_slope, 'cbf_bias': cbf_bias, 'cbf_mae': cbf_mae,
            'cbf_cov': cbf_cov,
            'att_slope': att_slope, 'att_bias': att_bias, 'att_mae': att_mae,
        }
        if cbf_win is not None:
            snr_result['cbf_win_rate'] = cbf_win
            snr_result['att_win_rate'] = att_win
            snr_result['ls_cbf_mae'] = ls_cbf_mae
            snr_result['ls_att_mae'] = ls_att_mae
        results['snr_results'][snr_key] = snr_result

    # Kill signals: check ALL SNR levels
    for snr_key, sr in results['snr_results'].items():
        if sr['cbf_slope'] < 0.3:
            results['kill_signals'].append({'check': 'cbf_variance_collapse',
                                             'detail': f"SNR={snr_key}: CBF slope={sr['cbf_slope']:.3f} < 0.3"})
        if sr['cbf_slope'] > 1.5:
            results['kill_signals'].append({'check': 'cbf_super_linearity',
                                             'detail': f"SNR={snr_key}: CBF slope={sr['cbf_slope']:.3f} > 1.5"})
        if sr['att_mae'] > 500:
            results['kill_signals'].append({'check': 'att_not_training',
                                             'detail': f"SNR={snr_key}: ATT MAE={sr['att_mae']:.1f} > 500"})
        if abs(sr['cbf_bias']) > 10.0:
            results['kill_signals'].append({'check': 'cbf_severe_bias',
                                             'detail': f"SNR={snr_key}: CBF bias={sr['cbf_bias']:.1f}"})

    # Loss trend kill signal
    if len(loss_history) >= 3 and loss_history[-1] >= loss_history[0] * 0.95:
        results['kill_signals'].append({
            'check': 'loss_not_decreasing',
            'detail': f'Loss epoch 1: {loss_history[0]:.4f}, final: {loss_history[-1]:.4f}'
        })

    results['primary_snr'] = '10' if '10' in results['snr_results'] else list(results['snr_results'].keys())[0]
    results['n_failures'] = len(results['kill_signals'])
    return results


def print_results(results: dict, cfg: dict, elapsed: float, has_ls: bool):
    """Print formatted console output."""
    model_name = cfg.get('model_class_name', 'Unknown')
    loss_hist = results['loss_history']

    loss_trend = 'PASS: decreasing' if len(loss_hist) >= 2 and loss_hist[-1] < loss_hist[0] * 0.95 else 'FAIL: not decreasing'

    print(f"\n{'='*65}")
    print(f" FAST VALIDATION ({len(loss_hist)} epochs, {elapsed:.0f}s)")
    print(f"{'='*65}")
    print(f"Model: {model_name}")
    print(f"Training:  loss {loss_hist[0]:.4f} -> {loss_hist[-1]:.4f} [{loss_trend}]")

    def status(val, lo, hi):
        if np.isnan(val):
            return 'UNKNOWN'
        if lo is not None and val < lo:
            return 'FAIL'
        if hi is not None and val > hi:
            return 'FAIL'
        return 'PASS'

    # Print ALL SNR levels
    print(f"\n{'SNR':>5} | {'CBF slope':>10} {'CBF bias':>10} {'CBF MAE':>9} | "
          f"{'ATT slope':>10} {'ATT MAE':>9} |", end="")
    if has_ls:
        print(f" {'CBF win%':>9} {'ATT win%':>9} | {'LS CBF':>7} {'LS ATT':>7}", end="")
    print()
    print("-" * (75 + (40 if has_ls else 0)))

    for snr_key in sorted(results['snr_results'].keys(), key=lambda x: int(x)):
        sr = results['snr_results'][snr_key]
        cbf_s = status(sr['cbf_slope'], 0.3, 1.5)
        cbf_b = status(abs(sr['cbf_bias']), None, 10.0)
        att_m = status(sr['att_mae'], None, 500)

        row = (f"{snr_key:>5} | "
               f"{sr['cbf_slope']:>7.2f}[{cbf_s[0]}] "
               f"{sr['cbf_bias']:>+8.1f}[{cbf_b[0]}] "
               f"{sr['cbf_mae']:>7.1f}   | "
               f"{sr['att_slope']:>7.2f}    "
               f"{sr['att_mae']:>7.0f}[{att_m[0]}] |")
        if has_ls and 'cbf_win_rate' in sr:
            row += f" {sr['cbf_win_rate']:>7.1f}%  {sr['att_win_rate']:>7.1f}% | {sr.get('ls_cbf_mae', 0):>6.1f} {sr.get('ls_att_mae', 0):>6.0f}"
        print(row)

    n_fail = results['n_failures']
    if n_fail == 0:
        verdict = 'PROMISING'
    elif n_fail <= 2:
        verdict = 'NEEDS_WORK'
    else:
        verdict = 'FAILING'

    print(f"\nRESULT: {verdict} ({n_fail} failures)")

    if results['kill_signals']:
        print("Kill signals:")
        for ks in results['kill_signals']:
            print(f"  - {ks['check']}: {ks['detail']}")

    print(f"{'='*65}\n")


def save_results(results: dict, cfg: dict, output_dir: Path, config_path: str):
    """Save JSON results and append to experiment log."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # latest_results.json
    results['config_path'] = config_path
    results['model_class_name'] = cfg.get('model_class_name', 'Unknown')
    results['timestamp'] = datetime.now().isoformat()
    json_path = output_dir / 'latest_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Append to experiment_log.md
    log_path = output_dir / 'experiment_log.md'
    is_new = not log_path.exists()

    with open(log_path, 'a') as f:
        if is_new:
            f.write("# Fast Experiment Log\n\n")

        f.write(f"## {results['timestamp']}\n\n")
        f.write(f"- **Config**: `{config_path}`\n")
        f.write(f"- **Model**: {results['model_class_name']}\n")

        loss = results['loss_history']
        f.write(f"- **Loss**: {loss[0]:.4f} -> {loss[-1]:.4f}\n")

        # Log all SNR levels
        for snr_key in sorted(results['snr_results'].keys(), key=lambda x: int(x)):
            sr = results['snr_results'][snr_key]
            f.write(f"- **SNR={snr_key}**: CBF slope={sr['cbf_slope']:.2f}, "
                    f"bias={sr['cbf_bias']:+.1f}, MAE={sr['cbf_mae']:.1f} | "
                    f"ATT slope={sr['att_slope']:.2f}, MAE={sr['att_mae']:.0f}ms")
            if 'cbf_win_rate' in sr:
                f.write(f" | CBF win={sr['cbf_win_rate']:.1f}%, ATT win={sr['att_win_rate']:.1f}%")
            f.write("\n")

        n_fail = results['n_failures']
        if n_fail == 0:
            f.write(f"- **Result**: PROMISING (0 failures)\n")
        else:
            fails = ', '.join(ks['check'] for ks in results['kill_signals'])
            f.write(f"- **Result**: {n_fail} failures: {fails}\n")

        f.write("\n---\n\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fast Train & Validate for ASL models')
    parser.add_argument('--config', required=True, help='YAML config file path')
    parser.add_argument('--n-samples', type=int, default=3000, help='Training samples (default: 3000)')
    parser.add_argument('--n-epochs', type=int, default=30, help='Training epochs (default: 30)')
    parser.add_argument('--n-test-phantoms', type=int, default=10, help='Test phantoms per SNR (default: 10)')
    parser.add_argument('--output-dir', default='fast_eval_results', help='Output directory')
    parser.add_argument('--device', default='mps', help='Device: mps, cuda, cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-ls', action='store_true', help='Skip LS fitting (faster)')
    args = parser.parse_args()

    t_start = time.time()

    # Phase 1: Config
    print(f"\n[Phase 1] Loading config: {args.config}")
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    print(f"  Device: {device}, Model: {cfg.get('model_class_name')}")

    # Phase 2: Generate data
    print(f"[Phase 2] Generating {args.n_samples} training samples...")
    t2 = time.time()
    signals, targets, norm_stats = generate_training_data(cfg, args.n_samples, args.seed)
    print(f"  Done in {time.time()-t2:.1f}s. Signals: {signals.shape}, "
          f"Norm stats: CBF mean={norm_stats['y_mean_cbf']:.1f}, std={norm_stats['y_std_cbf']:.1f}, "
          f"ATT mean={norm_stats['y_mean_att']:.0f}, std={norm_stats['y_std_att']:.0f}")

    # Phase 3: Train
    print(f"[Phase 3] Training for {args.n_epochs} epochs...")
    t3 = time.time()
    model = create_model(cfg, device)
    loss_history = train_model(cfg, model, signals, targets, norm_stats, device, args.n_epochs, args.seed)
    print(f"  Done in {time.time()-t3:.1f}s")

    # Phase 4: Test phantoms
    snr_levels = [3, 5, 10, 15, 25]
    print(f"[Phase 4] Generating {args.n_test_phantoms} test phantoms x {len(snr_levels)} SNR levels...")
    t4 = time.time()
    test_data = generate_test_phantoms(cfg, args.n_test_phantoms, snr_levels, args.seed + 5000)
    print(f"  Done in {time.time()-t4:.1f}s ({len(test_data)} total phantoms)")

    # Phase 5: Inference + LS
    print(f"[Phase 5] NN inference...")
    t5 = time.time()
    test_data = nn_inference(model, test_data, norm_stats, cfg, device)
    print(f"  NN inference: {time.time()-t5:.1f}s")

    has_ls = not args.skip_ls
    if has_ls:
        print(f"  LS fitting (10% subsample)...")
        t5b = time.time()
        test_data = ls_fitting(test_data, cfg, subsample_frac=0.2)
        print(f"  LS fitting: {time.time()-t5b:.1f}s")

    # Phase 6: Diagnostics
    print(f"[Phase 6] Computing diagnostics...")
    results = compute_diagnostics(test_data, loss_history, has_ls)
    elapsed = time.time() - t_start

    print_results(results, cfg, elapsed, has_ls)
    save_results(results, cfg, Path(args.output_dir), args.config)

    print(f"Results saved to {args.output_dir}/latest_results.json")
    print(f"Log appended to {args.output_dir}/experiment_log.md")

    # Exit code: 1 if any kill signals
    sys.exit(1 if results['kill_signals'] else 0)


if __name__ == '__main__':
    main()

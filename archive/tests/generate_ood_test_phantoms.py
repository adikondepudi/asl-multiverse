#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Test Phantom Generator
=================================================
Generates comprehensive OOD test sets for evaluating model robustness beyond
the training distribution. Tests each domain shift dimension independently
and in combination.

OOD Dimensions:
  1. CBF extremes: [5, 200] ml/100g/min (training: ~18-70)
  2. ATT extremes: [300, 3500] ms (training: ~500-3000)
  3. T1_artery: [1200, 2500] ms (training: 1550-2150)
  4. alpha_PCASL: [0.50, 0.75] (training: 0.75-0.95)
  5. alpha_VSASL: [0.25, 0.45] (training: 0.40-0.70)
  6. alpha_BS1: [0.70, 0.85] (training: 0.85-1.0 if enabled, else 1.0 only)
  7. SNR: [0.5, 2.0] and [25, 100] (training: 2-25)
  8. Pathological patterns: watershed stroke, global hypoperfusion, AVM

Usage:
    # Generate OOD test sets only (no model evaluation)
    python generate_ood_test_phantoms.py --output_dir results/ood_phantoms/

    # Generate and evaluate against a trained model
    python generate_ood_test_phantoms.py --run_dir <path> --output_dir results/ood_phantoms/

Date: February 8, 2026
"""

import sys
import os
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from asl_simulation import ASLParameters, ASLSimulator
from enhanced_simulation import SpatialPhantomGenerator


# ======================================================================
# OOD Phantom Definitions
# ======================================================================

# Training distribution bounds (from Exp 00 config + enhanced_simulation.py)
TRAINING_BOUNDS = {
    'cbf_gm': (50.0, 70.0),
    'cbf_wm': (18.0, 28.0),
    'att_gm': (1000.0, 1600.0),
    'att_wm': (1200.0, 1800.0),
    'T1_artery': (1550.0, 2150.0),
    'alpha_PCASL': (0.75, 0.95),
    'alpha_VSASL': (0.40, 0.70),
    'alpha_BS1': (0.85, 1.0),  # Only if domain rand enabled
    'snr': (2.0, 25.0),
}


@dataclass
class OODCondition:
    """Defines a single OOD test condition."""
    name: str
    description: str
    physics: Dict  # T1_artery, alpha_PCASL, alpha_VSASL, T_tau, alpha_BS1
    cbf_range: Tuple[float, float]  # Override CBF range
    att_range: Tuple[float, float]  # Override ATT range
    snr: float
    n_phantoms: int = 20
    pathology_type: Optional[str] = None  # Special phantom pattern


def get_ood_conditions(n_phantoms: int = 20) -> List[OODCondition]:
    """
    Define the full suite of OOD test conditions.

    Each condition shifts ONE dimension beyond training bounds while
    keeping others at standard values. Plus combined-shift conditions.
    """
    # Standard physics (in-distribution baseline)
    std_physics = {
        'T1_artery': 1650.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T_tau': 1800.0,
        'alpha_BS1': 1.0,
    }

    conditions = []

    # === 0. IN-DISTRIBUTION BASELINE ===
    conditions.append(OODCondition(
        name='00_baseline_in_dist',
        description='In-distribution baseline (standard physics, normal CBF/ATT)',
        physics=std_physics.copy(),
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=10.0,
        n_phantoms=n_phantoms,
    ))

    # === 1. CBF EXTREMES ===
    conditions.append(OODCondition(
        name='01a_cbf_very_low',
        description='Very low CBF: severe ischemia (5-15 ml/100g/min)',
        physics=std_physics.copy(),
        cbf_range=(5.0, 15.0),
        att_range=(2000.0, 3000.0),  # Delayed transit with ischemia
        snr=10.0,
        n_phantoms=n_phantoms,
    ))
    conditions.append(OODCondition(
        name='01b_cbf_very_high',
        description='Very high CBF: hypervascular/pediatric (120-200 ml/100g/min)',
        physics=std_physics.copy(),
        cbf_range=(120.0, 200.0),
        att_range=(400.0, 800.0),  # Fast transit with high flow
        snr=10.0,
        n_phantoms=n_phantoms,
    ))

    # === 2. ATT EXTREMES ===
    conditions.append(OODCondition(
        name='02a_att_very_fast',
        description='Very fast ATT: pediatric/AVM (300-600 ms)',
        physics=std_physics.copy(),
        cbf_range=(40.0, 80.0),
        att_range=(300.0, 600.0),
        snr=10.0,
        n_phantoms=n_phantoms,
    ))
    conditions.append(OODCondition(
        name='02b_att_very_delayed',
        description='Very delayed ATT: severe stenosis (2500-3500 ms)',
        physics=std_physics.copy(),
        cbf_range=(10.0, 30.0),
        att_range=(2500.0, 3500.0),
        snr=10.0,
        n_phantoms=n_phantoms,
    ))

    # === 3. T1_ARTERY EXTREMES ===
    conditions.append(OODCondition(
        name='03a_t1_low',
        description='Low T1_artery: polycythemia/high hematocrit (1200-1400 ms)',
        physics={**std_physics, 'T1_artery': 1300.0},
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=10.0,
        n_phantoms=n_phantoms,
    ))
    conditions.append(OODCondition(
        name='03b_t1_high',
        description='High T1_artery: anemia/low hematocrit (2200-2500 ms)',
        physics={**std_physics, 'T1_artery': 2400.0},
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=10.0,
        n_phantoms=n_phantoms,
    ))

    # === 4. LABELING EFFICIENCY EXTREMES ===
    conditions.append(OODCondition(
        name='04a_pcasl_poor',
        description='Poor PCASL labeling: patient motion/tortuous arteries (0.50-0.65)',
        physics={**std_physics, 'alpha_PCASL': 0.55},
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=10.0,
        n_phantoms=n_phantoms,
    ))
    conditions.append(OODCondition(
        name='04b_vsasl_poor',
        description='Poor VSASL efficiency: degraded gradients (0.25-0.35)',
        physics={**std_physics, 'alpha_VSASL': 0.30},
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=10.0,
        n_phantoms=n_phantoms,
    ))

    # === 5. BACKGROUND SUPPRESSION ===
    conditions.append(OODCondition(
        name='05a_bs_moderate',
        description='Moderate BS: typical in-vivo Philips (alpha_BS1=0.93)',
        physics={**std_physics, 'alpha_BS1': 0.93},
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=10.0,
        n_phantoms=n_phantoms,
    ))
    conditions.append(OODCondition(
        name='05b_bs_aggressive',
        description='Aggressive BS: strong suppression (alpha_BS1=0.75)',
        physics={**std_physics, 'alpha_BS1': 0.75},
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=10.0,
        n_phantoms=n_phantoms,
    ))

    # === 6. SNR EXTREMES ===
    conditions.append(OODCondition(
        name='06a_snr_very_low',
        description='Very low SNR: severe noise (SNR=1.0)',
        physics=std_physics.copy(),
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=1.0,
        n_phantoms=n_phantoms,
    ))
    conditions.append(OODCondition(
        name='06b_snr_very_high',
        description='Very high SNR: optimal acquisition (SNR=50)',
        physics=std_physics.copy(),
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=50.0,
        n_phantoms=n_phantoms,
    ))

    # === 7. PATHOLOGICAL PATTERNS ===
    conditions.append(OODCondition(
        name='07a_watershed_stroke',
        description='Watershed stroke: bilateral low-flow zones between vascular territories',
        physics=std_physics.copy(),
        cbf_range=(5.0, 80.0),  # Wide range: normal + ischemic
        att_range=(800.0, 3500.0),  # Wide range: normal + delayed
        snr=10.0,
        n_phantoms=n_phantoms,
        pathology_type='watershed',
    ))
    conditions.append(OODCondition(
        name='07b_global_hypoperfusion',
        description='Global hypoperfusion: heart failure/shock (all CBF reduced 50%)',
        physics=std_physics.copy(),
        cbf_range=(10.0, 35.0),  # All reduced
        att_range=(1500.0, 2500.0),  # All delayed
        snr=8.0,  # Lower SNR with lower signal
        n_phantoms=n_phantoms,
        pathology_type='global_hypo',
    ))
    conditions.append(OODCondition(
        name='07c_avm',
        description='AVM: arteriovenous malformation with very high focal CBF and fast ATT',
        physics=std_physics.copy(),
        cbf_range=(20.0, 80.0),  # Background normal
        att_range=(800.0, 2000.0),  # Background normal
        snr=10.0,
        n_phantoms=n_phantoms,
        pathology_type='avm',
    ))

    # === 8. COMBINED SHIFTS (worst-case scenarios) ===
    conditions.append(OODCondition(
        name='08a_invivo_realistic',
        description='Realistic in-vivo: BS + correct T1 + moderate noise',
        physics={
            'T1_artery': 1650.0,
            'alpha_PCASL': 0.85,
            'alpha_VSASL': 0.56,
            'T_tau': 1800.0,
            'alpha_BS1': 0.93,
        },
        cbf_range=(20.0, 80.0),
        att_range=(800.0, 2000.0),
        snr=8.0,
        n_phantoms=n_phantoms,
    ))
    conditions.append(OODCondition(
        name='08b_worst_case',
        description='Worst case: poor labeling + BS + low SNR + extreme CBF',
        physics={
            'T1_artery': 1400.0,
            'alpha_PCASL': 0.60,
            'alpha_VSASL': 0.35,
            'T_tau': 1800.0,
            'alpha_BS1': 0.80,
        },
        cbf_range=(5.0, 150.0),
        att_range=(300.0, 3500.0),
        snr=2.0,
        n_phantoms=n_phantoms,
    ))

    return conditions


# ======================================================================
# Phantom Generation Functions
# ======================================================================

def generate_ood_phantom(
    size: int,
    cbf_range: Tuple[float, float],
    att_range: Tuple[float, float],
    pathology_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single OOD phantom with specified CBF/ATT ranges.

    Args:
        size: Phantom size (size x size)
        cbf_range: (min_cbf, max_cbf) for tissue regions
        att_range: (min_att, max_att) for tissue regions
        pathology_type: Special phantom pattern (None, 'watershed', 'global_hypo', 'avm')
        seed: Random seed for reproducibility

    Returns:
        cbf_map: (size, size) CBF in ml/100g/min
        att_map: (size, size) ATT in ms
        mask: (size, size) binary tissue mask
    """
    if seed is not None:
        np.random.seed(seed)

    cbf_min, cbf_max = cbf_range
    att_min, att_max = att_range

    if pathology_type == 'watershed':
        return _generate_watershed_phantom(size, cbf_range, att_range)
    elif pathology_type == 'global_hypo':
        return _generate_global_hypo_phantom(size, cbf_range, att_range)
    elif pathology_type == 'avm':
        return _generate_avm_phantom(size, cbf_range, att_range)
    else:
        return _generate_standard_ood_phantom(size, cbf_range, att_range)


def _generate_standard_ood_phantom(
    size: int,
    cbf_range: Tuple[float, float],
    att_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard phantom with CBF/ATT sampled from specified ranges."""
    cbf_min, cbf_max = cbf_range
    att_min, att_max = att_range

    # Background tissue
    cbf_mean = (cbf_min + cbf_max) / 2
    cbf_std = (cbf_max - cbf_min) / 6  # 99.7% within range
    att_mean = (att_min + att_max) / 2
    att_std = (att_max - att_min) / 6

    cbf_map = np.random.normal(cbf_mean, max(cbf_std, 1.0), (size, size))
    att_map = np.random.normal(att_mean, max(att_std, 10.0), (size, size))

    # Add pathology blobs
    n_blobs = np.random.randint(1, 4)
    for _ in range(n_blobs):
        cx, cy = np.random.randint(10, size - 10, 2)
        r = np.random.randint(5, 15)
        y_grid, x_grid = np.ogrid[:size, :size]
        blob_mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= r**2

        # Random pathology within specified range
        cbf_map[blob_mask] = np.random.uniform(cbf_min, cbf_max)
        att_map[blob_mask] = np.random.uniform(att_min, att_max)

    # Smooth and clip
    cbf_map = gaussian_filter(cbf_map, sigma=1.0)
    att_map = gaussian_filter(att_map, sigma=1.0)
    cbf_map = np.clip(cbf_map, max(cbf_min, 0), cbf_max)
    att_map = np.clip(att_map, max(att_min, 0), att_max)

    # Circular brain mask
    mask = _circular_mask(size)

    cbf_map *= mask
    att_map *= mask
    att_map[mask == 0] = 0

    return cbf_map.astype(np.float32), att_map.astype(np.float32), mask.astype(np.float32)


def _generate_watershed_phantom(
    size: int,
    cbf_range: Tuple[float, float],
    att_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Watershed stroke phantom: bilateral low-flow zones between vascular territories.
    Simulates border-zone infarcts between ACA/MCA and MCA/PCA territories.
    """
    # Normal background
    cbf_map = np.random.normal(55.0, 5.0, (size, size))
    att_map = np.random.normal(1300.0, 100.0, (size, size))

    # Create watershed zones (vertical strips at ~1/3 and ~2/3 of image)
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    # Left watershed (ACA-MCA border)
    ws_left = np.exp(-((xx - size * 0.3)**2) / (2 * (size * 0.05)**2))
    # Right watershed (MCA-PCA border)
    ws_right = np.exp(-((xx - size * 0.7)**2) / (2 * (size * 0.05)**2))

    # Add sinusoidal variation for realism
    ws_left *= (1 + 0.3 * np.sin(2 * np.pi * yy / size * 3))
    ws_right *= (1 + 0.3 * np.sin(2 * np.pi * yy / size * 2.5 + 1.0))

    # Combine watershed masks (threshold for affected zones)
    ws_mask = np.maximum(ws_left, ws_right)
    ws_mask = np.clip(ws_mask, 0, 1)

    # Apply ischemia in watershed zones
    cbf_map -= ws_mask * 40  # Reduce CBF by up to 40 ml/100g/min
    att_map += ws_mask * 1500  # Increase ATT by up to 1500 ms

    # Smooth
    cbf_map = gaussian_filter(cbf_map, sigma=1.5)
    att_map = gaussian_filter(att_map, sigma=1.5)

    # Clip to specified ranges
    cbf_map = np.clip(cbf_map, max(cbf_range[0], 0), cbf_range[1])
    att_map = np.clip(att_map, max(att_range[0], 0), att_range[1])

    mask = _circular_mask(size)
    cbf_map *= mask
    att_map *= mask

    return cbf_map.astype(np.float32), att_map.astype(np.float32), mask.astype(np.float32)


def _generate_global_hypo_phantom(
    size: int,
    cbf_range: Tuple[float, float],
    att_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Global hypoperfusion phantom: all tissue has reduced CBF and delayed ATT.
    Simulates heart failure, cardiogenic shock, or severe dehydration.
    """
    cbf_mean = (cbf_range[0] + cbf_range[1]) / 2
    att_mean = (att_range[0] + att_range[1]) / 2

    # Base maps with reduced perfusion
    cbf_map = np.random.normal(cbf_mean, 5.0, (size, size))
    att_map = np.random.normal(att_mean, 150.0, (size, size))

    # Add slight GM/WM differentiation (WM even more reduced)
    # Use smooth noise field to create tissue-like structure
    tissue_field = gaussian_filter(np.random.randn(size, size), sigma=8)
    wm_fraction = (tissue_field > 0).astype(np.float32)

    # WM gets 60% of GM CBF
    cbf_map -= wm_fraction * cbf_mean * 0.3
    att_map += wm_fraction * 300  # WM has longer ATT

    # Smooth
    cbf_map = gaussian_filter(cbf_map, sigma=1.5)
    att_map = gaussian_filter(att_map, sigma=1.5)
    cbf_map = np.clip(cbf_map, max(cbf_range[0], 0), cbf_range[1])
    att_map = np.clip(att_map, max(att_range[0], 0), att_range[1])

    mask = _circular_mask(size)
    cbf_map *= mask
    att_map *= mask

    return cbf_map.astype(np.float32), att_map.astype(np.float32), mask.astype(np.float32)


def _generate_avm_phantom(
    size: int,
    cbf_range: Tuple[float, float],
    att_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    AVM phantom: arteriovenous malformation with very high focal CBF, fast ATT,
    and surrounding steal (reduced perfusion around the nidus).
    """
    # Normal background
    cbf_map = np.random.normal(55.0, 5.0, (size, size))
    att_map = np.random.normal(1300.0, 100.0, (size, size))

    # AVM nidus: small high-flow region
    cx, cy = np.random.randint(size // 4, 3 * size // 4, 2)
    r_nidus = np.random.randint(3, 8)
    y_grid, x_grid = np.ogrid[:size, :size]
    dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)

    # Nidus: extremely high CBF, very fast ATT
    nidus_mask = dist <= r_nidus
    cbf_map[nidus_mask] = np.random.uniform(150.0, 250.0)
    att_map[nidus_mask] = np.random.uniform(200.0, 500.0)

    # Feeding arteries: moderate high flow in a trail
    angle = np.random.uniform(0, 2 * np.pi)
    for t in np.linspace(0, 1, 20):
        ax = int(cx + t * r_nidus * 3 * np.cos(angle))
        ay = int(cy + t * r_nidus * 3 * np.sin(angle))
        if 0 <= ax < size and 0 <= ay < size:
            feeder_mask = ((x_grid - ax)**2 + (y_grid - ay)**2) <= 4
            cbf_map[feeder_mask] = np.random.uniform(100.0, 180.0)
            att_map[feeder_mask] = np.random.uniform(300.0, 600.0)

    # Steal phenomenon: reduced perfusion around AVM
    steal_mask = (dist > r_nidus) & (dist <= r_nidus * 4)
    steal_factor = 1 - 0.3 * np.exp(-(dist - r_nidus)**2 / (2 * (r_nidus * 2)**2))
    cbf_map = cbf_map * np.where(steal_mask, steal_factor, 1.0)
    att_map = att_map + np.where(steal_mask, 500 * (1 - steal_factor), 0)

    # Smooth edges (but not too much -- AVM has sharp boundaries)
    cbf_map = gaussian_filter(cbf_map, sigma=0.5)
    att_map = gaussian_filter(att_map, sigma=0.5)

    # Clip: allow very high CBF for AVM
    cbf_map = np.clip(cbf_map, 0, 300.0)
    att_map = np.clip(att_map, 100.0, 4000.0)

    mask = _circular_mask(size)
    cbf_map *= mask
    att_map *= mask

    return cbf_map.astype(np.float32), att_map.astype(np.float32), mask.astype(np.float32)


def _circular_mask(size: int, margin: float = 0.1) -> np.ndarray:
    """Generate a circular brain-like mask."""
    center = size / 2
    radius = size / 2 * (1 - margin)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    return (dist <= radius).astype(np.float32)


# ======================================================================
# Signal Generation
# ======================================================================

def generate_ood_test_set(
    condition: OODCondition,
    plds: np.ndarray,
    phantom_size: int = 64,
    seed_offset: int = 10000,
) -> Dict:
    """
    Generate a complete OOD test set for one condition.

    Args:
        condition: OODCondition defining the test parameters
        plds: PLD values in ms
        phantom_size: Spatial dimension
        seed_offset: Base seed for reproducibility

    Returns:
        Dict with 'signals', 'cbf_maps', 'att_maps', 'masks', 'condition_name'
    """
    physics = condition.physics
    params = ASLParameters(
        T1_artery=physics['T1_artery'],
        T_tau=physics.get('T_tau', 1800.0),
        alpha_PCASL=physics['alpha_PCASL'],
        alpha_VSASL=physics['alpha_VSASL'],
        alpha_BS1=physics.get('alpha_BS1', 1.0),
    )
    simulator = ASLSimulator(params=params)

    # Effective labeling efficiencies with BS
    alpha_bs1 = physics.get('alpha_BS1', 1.0)
    alpha_p_eff = physics['alpha_PCASL'] * (alpha_bs1 ** 4)
    alpha_v_eff = physics['alpha_VSASL'] * (alpha_bs1 ** 3)

    signals_list = []
    cbf_maps = []
    att_maps = []
    masks = []

    n_plds = len(plds)

    for idx in range(condition.n_phantoms):
        seed = seed_offset + idx
        cbf_map, att_map, mask = generate_ood_phantom(
            size=phantom_size,
            cbf_range=condition.cbf_range,
            att_range=condition.att_range,
            pathology_type=condition.pathology_type,
            seed=seed,
        )

        # Generate clean signals using vectorized computation
        # (same approach as generate_spatial_batch in enhanced_simulation.py)
        plds_bc = plds[:, np.newaxis, np.newaxis]
        att_bc = att_map[np.newaxis, :, :]
        cbf_bc = (cbf_map / 6000.0)[np.newaxis, :, :]  # Convert to ml/g/s

        t1_b = physics['T1_artery']
        tau = physics.get('T_tau', 1800.0)
        lambda_b = 0.90
        t2_f = 1.0

        # PCASL signal
        mask_arrived = (plds_bc >= att_bc)
        mask_transit = (plds_bc < att_bc) & (plds_bc >= (att_bc - tau))

        sig_p_arrived = (2 * alpha_p_eff * cbf_bc * t1_b / 1000.0 *
                         np.exp(-plds_bc / t1_b) *
                         (1 - np.exp(-tau / t1_b)) * t2_f) / lambda_b

        sig_p_transit = (2 * alpha_p_eff * cbf_bc * t1_b / 1000.0 *
                         (np.exp(-att_bc / t1_b) - np.exp(-(tau + plds_bc) / t1_b)) *
                         t2_f) / lambda_b

        pcasl_sig = np.zeros_like(plds_bc * cbf_bc)
        pcasl_sig[mask_arrived] = sig_p_arrived[mask_arrived]
        pcasl_sig[mask_transit] = sig_p_transit[mask_transit]

        # VSASL signal
        mask_vs_arrived = (plds_bc > att_bc)
        sig_v_early = (2 * alpha_v_eff * cbf_bc * (plds_bc / 1000.0) *
                       np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
        sig_v_late = (2 * alpha_v_eff * cbf_bc * (att_bc / 1000.0) *
                      np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
        vsasl_sig = np.where(mask_vs_arrived, sig_v_late, sig_v_early)

        # Stack: (2*n_plds, size, size)
        clean_stack = np.concatenate([pcasl_sig, vsasl_sig], axis=0)

        # Add Rician noise
        nonzero = clean_stack[clean_stack != 0]
        mean_sig = np.mean(np.abs(nonzero)) if len(nonzero) > 0 else 1e-6
        sigma = mean_sig / condition.snr
        noise_r = np.random.normal(0, sigma, clean_stack.shape).astype(np.float32)
        noise_i = np.random.normal(0, sigma, clean_stack.shape).astype(np.float32)
        noisy_stack = np.sqrt((clean_stack + noise_r)**2 + noise_i**2)

        signals_list.append(noisy_stack.astype(np.float32))
        cbf_maps.append(cbf_map)
        att_maps.append(att_map)
        masks.append(mask)

    return {
        'signals': signals_list,
        'cbf_maps': cbf_maps,
        'att_maps': att_maps,
        'masks': masks,
        'condition_name': condition.name,
        'condition_description': condition.description,
        'physics': condition.physics,
        'cbf_range': condition.cbf_range,
        'att_range': condition.att_range,
        'snr': condition.snr,
    }


# ======================================================================
# Evaluation (optional, if run_dir provided)
# ======================================================================

def evaluate_ood_test_set(
    models,
    test_set: Dict,
    norm_stats: Dict,
    config: Dict,
    device,
) -> Dict:
    """
    Evaluate a model ensemble on an OOD test set.

    Returns metrics dict with CBF/ATT MAE, Bias, RMSE, and degradation info.
    """
    import torch
    from test_domain_gap import evaluate_on_test_set
    return evaluate_on_test_set(models, test_set, norm_stats, config, device)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate OOD test phantoms and optionally evaluate model robustness."
    )
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for OOD test sets and results')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Path to trained model (optional, for evaluation)')
    parser.add_argument('--n_phantoms', type=int, default=20,
                        help='Number of phantoms per OOD condition (default: 20)')
    parser.add_argument('--phantom_size', type=int, default=64,
                        help='Spatial size of phantoms (default: 64)')
    parser.add_argument('--conditions', type=str, nargs='*', default=None,
                        help='Specific condition names to run (default: all)')
    parser.add_argument('--save_phantoms', action='store_true',
                        help='Save phantom arrays as .npz files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plds = np.array([500, 1000, 1500, 2000, 2500, 3000], dtype=np.float64)

    # Get OOD conditions
    conditions = get_ood_conditions(n_phantoms=args.n_phantoms)
    if args.conditions:
        conditions = [c for c in conditions if c.name in args.conditions]
        logger.info(f"Filtered to {len(conditions)} conditions: {[c.name for c in conditions]}")

    logger.info(f"Generating {len(conditions)} OOD test conditions, "
                f"{args.n_phantoms} phantoms each, size={args.phantom_size}")

    # Optionally load model for evaluation
    models = None
    norm_stats = None
    config = None
    device = None

    if args.run_dir:
        import torch
        from test_domain_gap import load_model_ensemble

        run_dir = Path(args.run_dir).resolve()
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        logger.info(f"Loading model from {run_dir}")
        models, config, norm_stats, _ = load_model_ensemble(run_dir, device)
        logger.info(f"Loaded {len(models)} ensemble members")

    # Generate and optionally evaluate each condition
    all_results = {}
    baseline_metrics = None

    for i, condition in enumerate(conditions):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(conditions)}] {condition.name}")
        logger.info(f"  {condition.description}")
        logger.info(f"  Physics: T1={condition.physics['T1_artery']}, "
                     f"aP={condition.physics['alpha_PCASL']}, "
                     f"aV={condition.physics['alpha_VSASL']}, "
                     f"BS1={condition.physics.get('alpha_BS1', 1.0)}")
        logger.info(f"  CBF range: {condition.cbf_range}, ATT range: {condition.att_range}")
        logger.info(f"  SNR: {condition.snr}")

        # Generate test set
        test_set = generate_ood_test_set(
            condition=condition,
            plds=plds,
            phantom_size=args.phantom_size,
        )

        # Save phantoms if requested
        if args.save_phantoms:
            phantom_dir = output_dir / 'phantoms' / condition.name
            phantom_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                phantom_dir / 'test_set.npz',
                signals=np.array(test_set['signals']),
                cbf_maps=np.array(test_set['cbf_maps']),
                att_maps=np.array(test_set['att_maps']),
                masks=np.array(test_set['masks']),
            )
            logger.info(f"  Saved phantoms to {phantom_dir}")

        # Evaluate if model loaded
        if models is not None:
            metrics = evaluate_ood_test_set(models, test_set, norm_stats, config, device)
            logger.info(f"  CBF MAE: {metrics['CBF_MAE']:.2f}, Bias: {metrics['CBF_Bias']:.2f}")
            logger.info(f"  ATT MAE: {metrics['ATT_MAE']:.0f}, Bias: {metrics['ATT_Bias']:.0f}")

            # Store baseline for degradation computation
            if condition.name == '00_baseline_in_dist':
                baseline_metrics = metrics

            # Compute degradation if baseline available
            degradation = {}
            if baseline_metrics is not None:
                degradation = {
                    'CBF_MAE_ratio': metrics['CBF_MAE'] / max(baseline_metrics['CBF_MAE'], 1e-6),
                    'ATT_MAE_ratio': metrics['ATT_MAE'] / max(baseline_metrics['ATT_MAE'], 1e-6),
                }

            all_results[condition.name] = {
                'condition': condition.description,
                'physics': condition.physics,
                'cbf_range': condition.cbf_range,
                'att_range': condition.att_range,
                'snr': condition.snr,
                'metrics': {k: v for k, v in metrics.items() if not k.startswith('per_phantom')},
                'degradation': degradation,
            }
        else:
            # Just record the condition metadata
            all_results[condition.name] = {
                'condition': condition.description,
                'physics': condition.physics,
                'cbf_range': condition.cbf_range,
                'att_range': condition.att_range,
                'snr': condition.snr,
                'metrics': 'Not evaluated (no --run_dir provided)',
            }

    # Print summary table
    if models is not None and baseline_metrics is not None:
        logger.info(f"\n{'='*80}")
        logger.info("OOD TEST RESULTS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"{'Condition':<30} {'CBF MAE':>10} {'CBF Bias':>10} "
                     f"{'ATT MAE':>10} {'CBF Deg':>10} {'ATT Deg':>10}")
        logger.info("-" * 80)

        for name, result in all_results.items():
            if isinstance(result['metrics'], dict):
                m = result['metrics']
                d = result.get('degradation', {})
                cbf_deg = f"{d.get('CBF_MAE_ratio', 0):.2f}x" if d else "N/A"
                att_deg = f"{d.get('ATT_MAE_ratio', 0):.2f}x" if d else "N/A"
                logger.info(f"{name:<30} {m['CBF_MAE']:>10.2f} {m['CBF_Bias']:>10.2f} "
                             f"{m['ATT_MAE']:>10.0f} {cbf_deg:>10} {att_deg:>10}")
        logger.info("-" * 80)

    # Save results
    results_path = output_dir / 'ood_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save condition catalog
    catalog = []
    for c in conditions:
        catalog.append({
            'name': c.name,
            'description': c.description,
            'physics': c.physics,
            'cbf_range': c.cbf_range,
            'att_range': c.att_range,
            'snr': c.snr,
            'pathology_type': c.pathology_type,
            'n_phantoms': c.n_phantoms,
            'in_training_distribution': c.name == '00_baseline_in_dist',
        })
    catalog_path = output_dir / 'ood_condition_catalog.json'
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    logger.info(f"Condition catalog saved to {catalog_path}")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n--- [CANCELLED] User stopped the script. ---")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! [FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

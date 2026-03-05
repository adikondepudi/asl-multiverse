#!/usr/bin/env python3
"""
Evaluate NN vs LS on Realistic Phantoms (V7)
=============================================
Loads generated test phantoms, runs NN inference and LS fitting,
computes bias/CoV/nRMSE/win-rate metrics per SNR level.

Usage:
    python amplitude_ablation_v7/evaluate_realistic_phantoms.py \
        --model-dir amplitude_ablation_v7/B_AmplitudeAware \
        [--phantom-dir amplitude_ablation_v7/test_phantoms] \
        [--output-dir amplitude_ablation_v7/eval_results] \
        [--max-phantoms 100] [--ls-subsample 0.2]
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from models.spatial_asl_network import SpatialASLNet
from models.amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet
from utils.helpers import get_grid_search_initial_guess
from baselines.multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

# Physics constants (must match generate_test_phantoms.py)
PLDS = np.array([500, 1000, 1500, 2000, 2500], dtype=np.float64)
T1_ARTERY = 1650.0
T_TAU = 1800.0
ALPHA_PCASL = 0.85
ALPHA_VSASL = 0.56
ALPHA_BS1 = 1.0
T2_FACTOR = 1.0
T_SAT_VS = 2000.0
N_PLDS = len(PLDS)

# NN input scaling: raw physics signals * M0_SCALE * global_scale
M0_SCALE = 100.0


# -- Device --
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ==========================================================================
# Model Loading
# ==========================================================================

def load_model(run_dir, device=DEVICE):
    """Load ensemble of spatial models from a run directory."""
    run_dir = Path(run_dir)

    with open(run_dir / "config.yaml") as f:
        full_config = yaml.safe_load(f)
    training_config = full_config.get("training", {})
    data_config = full_config.get("data", {})

    with open(run_dir / "norm_stats.json") as f:
        norm_stats = json.load(f)

    model_class_name = training_config.get("model_class_name", "SpatialASLNet")
    hidden_sizes = training_config.get("hidden_sizes", [32, 64, 128, 256])
    n_plds = len(data_config.get("pld_values", PLDS.tolist()))

    model_files = sorted((run_dir / "trained_models").glob("ensemble_model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No ensemble_model_*.pt files in {run_dir / 'trained_models'}")

    models = []
    for mf in model_files:
        if model_class_name == "AmplitudeAwareSpatialASLNet":
            model = AmplitudeAwareSpatialASLNet(
                n_plds=n_plds, features=hidden_sizes,
                use_film_at_bottleneck=training_config.get("use_film_at_bottleneck", True),
                use_film_at_decoder=training_config.get("use_film_at_decoder", True),
                use_amplitude_output_modulation=training_config.get("use_amplitude_output_modulation", True),
            )
        else:
            model = SpatialASLNet(n_plds=n_plds, features=hidden_sizes)

        sd = torch.load(mf, map_location=device, weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        model.to(device).eval()
        models.append(model)

    gsf = data_config.get("global_scale_factor", 10.0)
    return models, norm_stats, gsf, model_class_name


# ==========================================================================
# NN Inference
# ==========================================================================

def nn_predict(noisy_signals, models, norm_stats, gsf, device=DEVICE):
    """Run NN ensemble inference on a single phantom.

    Args:
        noisy_signals: (10, 64, 64) raw physics-unit signals
        models: list of loaded models
        norm_stats: dict with y_mean_cbf, y_std_cbf, etc.
        gsf: global_scale_factor from config

    Returns:
        cbf_pred: (64, 64) predicted CBF in ml/100g/min
        att_pred: (64, 64) predicted ATT in ms
    """
    # Scale: raw -> M0_SCALE * global_scale
    scaled = noisy_signals * M0_SCALE * gsf
    inp = torch.from_numpy(scaled[np.newaxis]).float().to(device)  # (1, 10, H, W)

    # Pad to multiple of 16
    _, _, h, w = inp.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        inp = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")

    cbf_maps, att_maps = [], []
    with torch.no_grad():
        for model in models:
            cbf_norm, att_norm, _, _ = model(inp)
            cbf_d = cbf_norm * norm_stats["y_std_cbf"] + norm_stats["y_mean_cbf"]
            att_d = att_norm * norm_stats["y_std_att"] + norm_stats["y_mean_att"]
            cbf_d = torch.clamp(cbf_d, 0.0, 250.0)
            att_d = torch.clamp(att_d, 0.0, 5000.0)
            cbf_maps.append(cbf_d.cpu().numpy())
            att_maps.append(att_d.cpu().numpy())

    cbf_ens = np.mean(cbf_maps, axis=0)[0, 0]  # (H_padded, W_padded)
    att_ens = np.mean(att_maps, axis=0)[0, 0]

    # Remove padding
    cbf_ens = cbf_ens[:h, :w]
    att_ens = att_ens[:h, :w]

    return cbf_ens, att_ens


# ==========================================================================
# LS Fitting (per-voxel, parallelized)
# ==========================================================================

def _fit_single_voxel(args):
    """Fit a single voxel using LS. Called by multiprocessing Pool."""
    idx, signal_1d, plds_arr, pldti = args
    asl_params = {
        "T1_artery": T1_ARTERY, "T_tau": T_TAU, "T2_factor": T2_FACTOR,
        "alpha_BS1": ALPHA_BS1, "alpha_PCASL": ALPHA_PCASL, "alpha_VSASL": ALPHA_VSASL,
        "T_sat_vs": T_SAT_VS,
    }
    try:
        init = get_grid_search_initial_guess(signal_1d, plds_arr, asl_params)
        # diff_sig shape: (n_plds, 2) with columns [PCASL, VSASL]
        pcasl_vals = signal_1d[:N_PLDS]
        vsasl_vals = signal_1d[N_PLDS:]
        diff_sig = np.column_stack([pcasl_vals, vsasl_vals])
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
            pldti, diff_sig, init,
            T1_ARTERY, T_TAU, T2_FACTOR, ALPHA_BS1, ALPHA_PCASL, ALPHA_VSASL
        )
        cbf = beta[0] * 6000.0  # Convert from ml/g/s to ml/100g/min
        att = beta[1]           # Already in ms
        if np.isfinite(cbf) and np.isfinite(att):
            return idx, cbf, att
    except Exception:
        pass
    return idx, np.nan, np.nan


def ls_predict(noisy_signals, brain_mask, subsample_frac=0.2):
    """Run LS fitting on a subsample of brain voxels.

    Args:
        noisy_signals: (10, 64, 64) raw physics-unit signals
        brain_mask: (64, 64) boolean mask (tissue_map > 0)
        subsample_frac: fraction of brain voxels to fit

    Returns:
        ls_indices: (N,) flat indices into brain_mask where LS was fit
        ls_cbf: (N,) CBF predictions
        ls_att: (N,) ATT predictions
    """
    brain_coords = np.argwhere(brain_mask)  # (M, 2)
    n_total = len(brain_coords)
    n_fit = max(1, int(n_total * subsample_frac))

    rng = np.random.RandomState(42)
    chosen = rng.choice(n_total, size=n_fit, replace=False)
    chosen_coords = brain_coords[chosen]

    plds_arr = PLDS.copy()
    pldti = np.column_stack([plds_arr, plds_arr])

    tasks = []
    for k, (i, j) in enumerate(chosen_coords):
        pcasl_vals = noisy_signals[:N_PLDS, i, j].astype(np.float64)
        vsasl_vals = noisy_signals[N_PLDS:, i, j].astype(np.float64)
        signal_1d = np.concatenate([pcasl_vals, vsasl_vals])
        tasks.append((k, signal_1d, plds_arr, pldti))

    n_workers = min(8, os.cpu_count() or 1)
    with Pool(n_workers) as pool:
        results = pool.map(_fit_single_voxel, tasks)

    # Collect results
    ls_cbf = np.full(n_fit, np.nan)
    ls_att = np.full(n_fit, np.nan)
    for k, cbf, att in results:
        ls_cbf[k] = cbf
        ls_att[k] = att

    return chosen_coords, ls_cbf, ls_att


# ==========================================================================
# Metrics
# ==========================================================================

def compute_metrics(pred, true, label=""):
    """Compute nBias, CoV, nRMSE for a set of voxels.

    All metrics are normalized by mean(true) and expressed as percentages.
    """
    valid = np.isfinite(pred) & np.isfinite(true) & (true > 0)
    pred = pred[valid]
    true = true[valid]

    if len(pred) == 0:
        return {"nBias": np.nan, "CoV": np.nan, "nRMSE": np.nan, "MAE": np.nan, "n_voxels": 0}

    errors = pred - true
    mean_true = np.mean(true)

    nBias = np.mean(errors) / mean_true * 100.0
    CoV = np.std(errors) / mean_true * 100.0
    nRMSE = np.sqrt(np.mean(errors ** 2)) / mean_true * 100.0
    MAE = np.mean(np.abs(errors))

    return {"nBias": float(nBias), "CoV": float(CoV), "nRMSE": float(nRMSE),
            "MAE": float(MAE), "n_voxels": int(len(pred))}


def compute_win_rate(nn_pred, ls_pred, true_vals):
    """Compute win rate: fraction of voxels where |nn_error| < |ls_error|."""
    valid = np.isfinite(nn_pred) & np.isfinite(ls_pred) & np.isfinite(true_vals)
    nn_err = np.abs(nn_pred[valid] - true_vals[valid])
    ls_err = np.abs(ls_pred[valid] - true_vals[valid])
    if len(nn_err) == 0:
        return np.nan, 0
    wins = np.sum(nn_err < ls_err)
    return float(wins / len(nn_err) * 100.0), int(len(nn_err))


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate NN vs LS on realistic phantoms")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to trained model directory (with config.yaml, norm_stats.json, trained_models/)")
    parser.add_argument("--phantom-dir", type=str, default=None,
                        help="Path to test phantoms (default: amplitude_ablation_v7/test_phantoms)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: <model-dir>/phantom_eval_results)")
    parser.add_argument("--max-phantoms", type=int, default=100,
                        help="Maximum number of phantoms to evaluate")
    parser.add_argument("--ls-subsample", type=float, default=0.2,
                        help="Fraction of brain voxels to fit with LS (default: 0.2)")
    parser.add_argument("--skip-ls", action="store_true",
                        help="Skip LS fitting (NN-only evaluation)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir

    phantom_dir = Path(args.phantom_dir) if args.phantom_dir else (PROJECT_ROOT / "amplitude_ablation_v7" / "test_phantoms")
    if not phantom_dir.is_absolute():
        phantom_dir = PROJECT_ROOT / phantom_dir

    output_dir = Path(args.output_dir) if args.output_dir else (model_dir / "phantom_eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load phantom metadata
    meta_path = phantom_dir / "phantom_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            phantom_meta = json.load(f)
        snr_levels = phantom_meta["snr_levels"]
    else:
        snr_levels = [2, 3, 5, 10, 15, 25]

    # Load model
    print(f"Loading model from {model_dir}")
    models, norm_stats, gsf, model_class_name = load_model(model_dir)
    print(f"  Model: {model_class_name}, ensemble size: {len(models)}, device: {DEVICE}")
    print(f"  global_scale_factor: {gsf}")

    # Find phantom files
    phantom_files = sorted(phantom_dir.glob("phantom_*.npz"))[:args.max_phantoms]
    n_phantoms = len(phantom_files)
    print(f"Found {n_phantoms} phantoms in {phantom_dir}")
    print(f"SNR levels: {snr_levels}")
    print(f"LS subsample: {args.ls_subsample:.0%}")
    print()

    # Results storage: {snr: {metric: [values_per_phantom]}}
    results = {snr: {"nn_cbf": [], "nn_att": [], "ls_cbf": [], "ls_att": [],
                      "nn_cbf_metrics": [], "nn_att_metrics": [],
                      "ls_cbf_metrics": [], "ls_att_metrics": [],
                      "cbf_win_rate": [], "att_win_rate": []}
               for snr in snr_levels}

    t_start = time.time()

    for p_idx, pf in enumerate(phantom_files):
        data = np.load(pf)
        cbf_map = data["cbf_map"]
        att_map = data["att_map"]
        tissue_map = data["tissue_map"]
        brain_mask = tissue_map > 0

        for snr in snr_levels:
            key = f"noisy_snr_{snr}"
            if key not in data:
                print(f"  WARNING: {key} not found in {pf.name}, skipping")
                continue
            noisy_signals = data[key]

            # NN inference
            nn_cbf, nn_att = nn_predict(noisy_signals, models, norm_stats, gsf)

            # Extract brain voxels for NN metrics
            nn_cbf_brain = nn_cbf[brain_mask]
            nn_att_brain = nn_att[brain_mask]
            true_cbf_brain = cbf_map[brain_mask]
            true_att_brain = att_map[brain_mask]

            nn_cbf_m = compute_metrics(nn_cbf_brain, true_cbf_brain, "NN_CBF")
            nn_att_m = compute_metrics(nn_att_brain, true_att_brain, "NN_ATT")
            results[snr]["nn_cbf_metrics"].append(nn_cbf_m)
            results[snr]["nn_att_metrics"].append(nn_att_m)

            if not args.skip_ls:
                # LS fitting on subsample
                chosen_coords, ls_cbf_vals, ls_att_vals = ls_predict(
                    noisy_signals, brain_mask, args.ls_subsample
                )

                # True values at LS-fitted locations
                true_cbf_ls = cbf_map[chosen_coords[:, 0], chosen_coords[:, 1]]
                true_att_ls = att_map[chosen_coords[:, 0], chosen_coords[:, 1]]

                ls_cbf_m = compute_metrics(ls_cbf_vals, true_cbf_ls, "LS_CBF")
                ls_att_m = compute_metrics(ls_att_vals, true_att_ls, "LS_ATT")
                results[snr]["ls_cbf_metrics"].append(ls_cbf_m)
                results[snr]["ls_att_metrics"].append(ls_att_m)

                # Win rate at LS-fitted voxels
                nn_cbf_at_ls = nn_cbf[chosen_coords[:, 0], chosen_coords[:, 1]]
                nn_att_at_ls = nn_att[chosen_coords[:, 0], chosen_coords[:, 1]]

                cbf_wr, cbf_n = compute_win_rate(nn_cbf_at_ls, ls_cbf_vals, true_cbf_ls)
                att_wr, att_n = compute_win_rate(nn_att_at_ls, ls_att_vals, true_att_ls)
                results[snr]["cbf_win_rate"].append(cbf_wr)
                results[snr]["att_win_rate"].append(att_wr)

        elapsed = time.time() - t_start
        rate = (p_idx + 1) / elapsed
        eta = (n_phantoms - p_idx - 1) / rate if rate > 0 else 0
        print(f"  [{p_idx+1:3d}/{n_phantoms}] {pf.name}  ({elapsed:.1f}s, ETA {eta:.0f}s)")

    # Aggregate and save
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    summary = {}
    for snr in snr_levels:
        r = results[snr]

        def avg_metric(metric_list, key):
            vals = [m[key] for m in metric_list if np.isfinite(m.get(key, np.nan))]
            return float(np.mean(vals)) if vals else np.nan

        s = {
            "nn_cbf_nBias": avg_metric(r["nn_cbf_metrics"], "nBias"),
            "nn_cbf_CoV": avg_metric(r["nn_cbf_metrics"], "CoV"),
            "nn_cbf_nRMSE": avg_metric(r["nn_cbf_metrics"], "nRMSE"),
            "nn_cbf_MAE": avg_metric(r["nn_cbf_metrics"], "MAE"),
            "nn_att_nBias": avg_metric(r["nn_att_metrics"], "nBias"),
            "nn_att_CoV": avg_metric(r["nn_att_metrics"], "CoV"),
            "nn_att_nRMSE": avg_metric(r["nn_att_metrics"], "nRMSE"),
            "nn_att_MAE": avg_metric(r["nn_att_metrics"], "MAE"),
        }

        if not args.skip_ls:
            s.update({
                "ls_cbf_nBias": avg_metric(r["ls_cbf_metrics"], "nBias"),
                "ls_cbf_CoV": avg_metric(r["ls_cbf_metrics"], "CoV"),
                "ls_cbf_nRMSE": avg_metric(r["ls_cbf_metrics"], "nRMSE"),
                "ls_cbf_MAE": avg_metric(r["ls_cbf_metrics"], "MAE"),
                "ls_att_nBias": avg_metric(r["ls_att_metrics"], "nBias"),
                "ls_att_CoV": avg_metric(r["ls_att_metrics"], "CoV"),
                "ls_att_nRMSE": avg_metric(r["ls_att_metrics"], "nRMSE"),
                "ls_att_MAE": avg_metric(r["ls_att_metrics"], "MAE"),
                "cbf_win_rate": float(np.nanmean(r["cbf_win_rate"])) if r["cbf_win_rate"] else np.nan,
                "att_win_rate": float(np.nanmean(r["att_win_rate"])) if r["att_win_rate"] else np.nan,
            })

        summary[f"snr_{snr}"] = s

    # Print table
    header = f"{'SNR':>5}  {'NN CBF nBias':>12}  {'NN CBF MAE':>11}  {'NN ATT MAE':>11}"
    if not args.skip_ls:
        header += f"  {'LS CBF MAE':>11}  {'LS ATT MAE':>11}  {'CBF Win%':>9}  {'ATT Win%':>9}"
    print(header)
    print("-" * len(header))

    for snr in snr_levels:
        s = summary[f"snr_{snr}"]
        row = f"{snr:>5}  {s['nn_cbf_nBias']:>12.2f}  {s['nn_cbf_MAE']:>11.2f}  {s['nn_att_MAE']:>11.2f}"
        if not args.skip_ls:
            row += (f"  {s['ls_cbf_MAE']:>11.2f}  {s['ls_att_MAE']:>11.2f}"
                    f"  {s['cbf_win_rate']:>9.1f}  {s['att_win_rate']:>9.1f}")
        print(row)

    # Save results
    out_json = output_dir / "phantom_eval_summary.json"
    with open(out_json, "w") as f:
        json.dump({
            "model_dir": str(model_dir),
            "model_class": model_class_name,
            "n_phantoms": n_phantoms,
            "ls_subsample": args.ls_subsample,
            "summary": summary,
        }, f, indent=2)

    # Save per-phantom metrics as numpy
    for snr in snr_levels:
        r = results[snr]
        np.savez_compressed(
            output_dir / f"phantom_metrics_snr_{snr}.npz",
            nn_cbf_metrics=np.array([
                [m["nBias"], m["CoV"], m["nRMSE"], m["MAE"]] for m in r["nn_cbf_metrics"]
            ]) if r["nn_cbf_metrics"] else np.array([]),
            nn_att_metrics=np.array([
                [m["nBias"], m["CoV"], m["nRMSE"], m["MAE"]] for m in r["nn_att_metrics"]
            ]) if r["nn_att_metrics"] else np.array([]),
        )

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved to {output_dir}")
    print(f"  Summary: {out_json}")


if __name__ == "__main__":
    main()

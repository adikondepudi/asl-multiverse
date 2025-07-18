# predict_on_invivo.py
import torch
import numpy as np
import nibabel as nib
import sys
from pathlib import Path
from tqdm import tqdm
import time
import argparse
from typing import Dict, Optional, List, Tuple
import json

# --- Import project modules ---
from evaluate_ensemble import load_artifacts
from main import engineer_signal_features
from pcasl_functions import fun_PCASL_1comp_vect_pep
from vsasl_functions import fun_VSASL_1comp_vect_pep
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

# --- Global cache for normalization stats to avoid repeated file I/O ---
CACHED_NORM_STATS: Optional[Dict] = None

def get_norm_stats(model_results_dir: Path) -> Dict:
    """Loads and caches the norm_stats.json file."""
    global CACHED_NORM_STATS
    if CACHED_NORM_STATS is None:
        norm_stats_path = model_results_dir / 'norm_stats.json'
        print(f"  --> Loading normalization stats from: {norm_stats_path}")
        with open(norm_stats_path, 'r') as f:
            CACHED_NORM_STATS = json.load(f)
    return CACHED_NORM_STATS

def denormalize_predictions(cbf_norm: np.ndarray, att_norm: np.ndarray, norm_stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Applies de-normalization to the NN's outputs."""
    cbf_denorm = cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att_denorm = att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    return cbf_denorm, att_denorm

def apply_normalization_vectorized(batch: np.ndarray, norm_stats: Dict, num_plds: int) -> np.ndarray:
    """Applies normalization to a batch of input signals for the NN."""
    pcasl_raw = batch[:, :num_plds]
    vsasl_raw = batch[:, num_plds:num_plds*2]
    features = batch[:, num_plds*2:]
    
    pcasl_norm = (pcasl_raw - norm_stats['pcasl_mean']) / (np.array(norm_stats['pcasl_std']) + 1e-6)
    vsasl_norm = (vsasl_raw - norm_stats['vsasl_mean']) / (np.array(norm_stats['vsasl_std']) + 1e-6)
    
    return np.concatenate([pcasl_norm, vsasl_norm, features], axis=1)

def batch_predict_nn(signals_flat: np.ndarray, subject_plds: np.ndarray, models: List, config: Dict, model_dir: Path, device: torch.device, dims: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Runs batched inference using the trained NN ensemble."""
    num_voxels = signals_flat.shape[0]
    plds_train = np.array(config['pld_values'])
    num_plds_train = len(plds_train)
    num_plds_subject = len(subject_plds)
    
    padded_signals = np.zeros((num_voxels, num_plds_train * 2))
    matching_indices = np.where(np.isin(plds_train, subject_plds))[0]
    
    padded_signals[:, matching_indices] = signals_flat[:, :num_plds_subject]
    padded_signals[:, matching_indices + num_plds_train] = signals_flat[:, num_plds_subject:]

    eng_feats = engineer_signal_features(padded_signals, num_plds_train)
    nn_input = np.concatenate([padded_signals, eng_feats], axis=1)
    
    norm_stats = get_norm_stats(model_dir)
    norm_input = apply_normalization_vectorized(nn_input, norm_stats, num_plds_train)
    
    input_tensor = torch.FloatTensor(norm_input).to(device)
    
    all_cbf_preds, all_att_preds = [], []
    with torch.no_grad():
        for model in models:
            cbf_m, att_m, _, _, _, _ = model(input_tensor)
            all_cbf_preds.append(cbf_m.cpu().numpy())
            all_att_preds.append(att_m.cpu().numpy())

    cbf_norm = np.mean(all_cbf_preds, axis=0)
    att_norm = np.mean(all_att_preds, axis=0)
    
    cbf_denorm, att_denorm = denormalize_predictions(cbf_norm, att_norm, norm_stats)
    
    return cbf_denorm.reshape(dims, order='F'), att_denorm.reshape(dims, order='F')

def fit_ls_robust(signals_flat: np.ndarray, plds: np.ndarray, config: Dict, dims: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Runs a robust LS fit for each voxel."""
    num_voxels = signals_flat.shape[0]
    pldti = np.column_stack([plds, plds])
    ls_params = {k:v for k,v in config.items() if k in ['T1_artery','T_tau','T2_factor','alpha_BS1','alpha_PCASL','alpha_VSASL']}
    
    cbf_results, att_results = np.full(num_voxels, np.nan), np.full(num_voxels, np.nan)
    signals_reshaped = signals_flat.reshape((num_voxels, len(plds), 2), order='F')

    for i in tqdm(range(num_voxels), desc="Robust LS Fitting", leave=False, ncols=80):
        # With correctly scaled and cleaned data, a simple initial guess is sufficient.
        init_guess = [50.0 / 6000.0, 1500.0]
        try:
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signals_reshaped[i], init_guess, **ls_params)
            cbf_results[i], att_results[i] = beta[0] * 6000.0, beta[1]
        except Exception:
            continue # Leave as NaN on failure
            
    return cbf_results.reshape(dims, order='F'), att_results.reshape(dims, order='F')

def predict_subject(subject_dir: Path, models: List, config: Dict, model_dir: Path, device: torch.device, output_root: Path, method: str):
    """Main processing function for a single subject."""
    subject_id = subject_dir.name
    subject_output_dir = output_root / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        low_snr_signals = np.load(subject_dir / 'low_snr_signals.npy')
        high_snr_signals = np.load(subject_dir / 'high_snr_signals.npy')
        dims = tuple(np.load(subject_dir / 'image_dims.npy'))
        affine = np.load(subject_dir / 'image_affine.npy')
        header = np.load(subject_dir / 'image_header.npy', allow_pickle=True).item()
        plds = np.load(subject_dir / 'plds.npy')
        
        if method in ['all', 'nn']:
            print(f"\n  [{subject_id}] Running NN prediction...")
            start_time = time.time()
            cbf_nn_low, att_nn_low = batch_predict_nn(low_snr_signals, plds, models, config, model_dir, device, dims)
            cbf_nn_high, att_nn_high = batch_predict_nn(high_snr_signals, plds, models, config, model_dir, device, dims)
            print(f"  --> NN prediction finished in {time.time() - start_time:.2f}s.")
            nib.save(nib.Nifti1Image(cbf_nn_low, affine, header), subject_output_dir / 'nn_cbf_low_snr.nii.gz')
            nib.save(nib.Nifti1Image(att_nn_low, affine, header), subject_output_dir / 'nn_att_low_snr.nii.gz')
            nib.save(nib.Nifti1Image(cbf_nn_high, affine, header), subject_output_dir / 'nn_cbf_high_snr.nii.gz')
            nib.save(nib.Nifti1Image(att_nn_high, affine, header), subject_output_dir / 'nn_att_high_snr.nii.gz')

        if method in ['all', 'ls']:
            print(f"  [{subject_id}] Running LS fitting...")
            start_time = time.time()
            cbf_ls_low, att_ls_low = fit_ls_robust(low_snr_signals, plds, config, dims)
            cbf_ls_high, att_ls_high = fit_ls_robust(high_snr_signals, plds, config, dims)
            print(f"  --> LS fitting finished in {time.time() - start_time:.2f}s.")
            nib.save(nib.Nifti1Image(cbf_ls_low, affine, header), subject_output_dir / 'ls_cbf_low_snr.nii.gz')
            nib.save(nib.Nifti1Image(att_ls_low, affine, header), subject_output_dir / 'ls_att_low_snr.nii.gz')
            nib.save(nib.Nifti1Image(cbf_ls_high, affine, header), subject_output_dir / 'ls_cbf_high_snr.nii.gz')
            nib.save(nib.Nifti1Image(att_ls_high, affine, header), subject_output_dir / 'ls_att_high_snr.nii.gz')

    except Exception as e:
        print(f"ERROR predicting on subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run NN and/or robust LS fitting on preprocessed in-vivo ASL data.")
    parser.add_argument("preprocessed_dir", type=str, help="Path to the parent directory of preprocessed subjects.")
    parser.add_argument("model_results_dir", type=str, help="Path to the directory containing trained models and configs.")
    parser.add_argument("output_dir", type=str, help="Path where prediction maps will be saved.")
    parser.add_argument("--subject", type=str, default=None, help="Process a single subject ID instead of all.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N subjects.")
    parser.add_argument("--method", type=str, default="all", choices=['all', 'nn', 'ls'], help="Which method to run.")
    args = parser.parse_args()

    preprocessed_root = Path(args.preprocessed_dir)
    model_results_root = Path(args.model_results_dir)
    output_root = Path(args.output_dir)

    print("--- Loading Model Artifacts ---")
    models, config, _ = load_artifacts(model_results_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device)
    print(f"Loaded {len(models)} models and config. Using device: {device}")
    
    subject_dirs = sorted([d for d in preprocessed_root.iterdir() if d.is_dir()])
    if args.subject:
        subject_dirs = [preprocessed_root / args.subject]
        if not subject_dirs[0].exists():
            print(f"Error: Specified subject directory not found: {subject_dirs[0]}"); sys.exit(1)
    if args.limit:
        subject_dirs = subject_dirs[:args.limit]

    print(f"\nFound {len(subject_dirs)} subjects to process using method: '{args.method.upper()}'")
    for sub_dir in subject_dirs:
        print(f"\n--- Processing Subject: {sub_dir.name} ---")
        predict_subject(sub_dir, models, config, model_results_root, device, output_root, args.method)

    print("\n--- In-vivo prediction pipeline complete! ---")
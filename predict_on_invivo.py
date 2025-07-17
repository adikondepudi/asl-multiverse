# predict_on_invivo.py (Final version with LS debugging and command-line controls)
import torch
import numpy as np
import nibabel as nib
import sys
from pathlib import Path
from tqdm import tqdm
import time
import argparse
from typing import Dict

from evaluate_ensemble import load_artifacts
from main import engineer_signal_features
from comparison_framework import denormalize_predictions
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

def apply_normalization_vectorized(batch_input: np.ndarray, norm_stats: Dict, num_plds: int):
    """Applies normalization to a batch of input vectors using NumPy broadcasting."""
    pcasl_signals = batch_input[:, :num_plds]
    vsasl_signals = batch_input[:, num_plds:num_plds*2]
    other_features = batch_input[:, num_plds*2:]

    pcasl_mean = np.array(norm_stats['pcasl_mean'])
    pcasl_std = np.array(norm_stats['pcasl_std'])
    vsasl_mean = np.array(norm_stats['vsasl_mean'])
    vsasl_std = np.array(norm_stats['vsasl_std'])
    
    pcasl_norm = (pcasl_signals - pcasl_mean) / (pcasl_std + 1e-6)
    vsasl_norm = (vsasl_signals - vsasl_mean) / (vsasl_std + 1e-6)
    
    return np.concatenate([pcasl_norm, vsasl_norm, other_features], axis=1)

def predict_subject(subject_processed_dir: Path, models, config, norm_stats_full, device, output_root: Path, run_method: str, ls_order: str):
    """
    Processes a single subject, running the specified fitting method(s).
    """
    subject_id = subject_processed_dir.name
    subject_output_dir = output_root / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load all the necessary data for the subject
        all_low_snr_signals = np.load(subject_processed_dir / 'low_snr_signals.npy')
        all_high_snr_signals = np.load(subject_processed_dir / 'high_snr_signals.npy')
        voxel_coords = np.load(subject_processed_dir / 'voxel_coords.npy')
        dims = np.load(subject_processed_dir / 'image_dims.npy')
        affine = np.load(subject_processed_dir / 'image_affine.npy')
        header = np.load(subject_processed_dir / 'image_header.npy', allow_pickle=True).item()
        subject_plds = np.load(subject_processed_dir / 'plds.npy')
        
        # --- Neural Network Prediction (Conditional) ---
        if run_method in ['all', 'nn']:
            print(f"\n  [{subject_id}] Running batched NN prediction...")
            start_time_nn = time.time()
            cbf_map_nn_low, att_map_nn_low = batch_predict_nn(all_low_snr_signals, subject_plds, models, config, norm_stats_full, device, dims)
            cbf_map_nn_high, att_map_nn_high = batch_predict_nn(all_high_snr_signals, subject_plds, models, config, norm_stats_full, device, dims)
            duration_nn = time.time() - start_time_nn
            print(f"  [{subject_id}] NN prediction took: {duration_nn:.2f} seconds.")
            # Save NN results
            nib.save(nib.Nifti1Image(cbf_map_nn_low, affine, header), subject_output_dir / 'nn_cbf_low_snr.nii.gz')
            nib.save(nib.Nifti1Image(att_map_nn_low, affine, header), subject_output_dir / 'nn_att_low_snr.nii.gz')
            nib.save(nib.Nifti1Image(cbf_map_nn_high, affine, header), subject_output_dir / 'nn_cbf_high_snr.nii.gz')
            nib.save(nib.Nifti1Image(att_map_nn_high, affine, header), subject_output_dir / 'nn_att_high_snr.nii.gz')

        # --- Least-Squares Fitting (Conditional) ---
        if run_method in ['all', 'ls', 'ls_debug']:
            print(f"  [{subject_id}] Running per-voxel LS fitting with order='{ls_order}'...")
            cbf_map_ls_low, att_map_ls_low = np.zeros(dims), np.zeros(dims)
            cbf_map_ls_high, att_map_ls_high = np.zeros(dims), np.zeros(dims)
            start_time_ls = time.time()
            for i in tqdm(range(len(voxel_coords)), desc=f"  LS Fitting for {subject_id}", leave=False):
                x, y, z = voxel_coords[i]
                cbf_ls, att_ls = fit_ls_voxel(all_low_snr_signals[i], subject_plds, config, order=ls_order)
                cbf_map_ls_low[x, y, z] = cbf_ls
                att_map_ls_low[x, y, z] = att_ls
                cbf_ls, att_ls = fit_ls_voxel(all_high_snr_signals[i], subject_plds, config, order=ls_order)
                cbf_map_ls_high[x, y, z] = cbf_ls
                att_map_ls_high[x, y, z] = att_ls
            duration_ls = time.time() - start_time_ls
            print(f"  [{subject_id}] All LS fitting took: {duration_ls:.2f} seconds.")
            
            # Save LS results with a descriptive suffix
            file_suffix = "F_order_debug" if ls_order == 'F' else "C_order_default"
            nib.save(nib.Nifti1Image(cbf_map_ls_low, affine, header), subject_output_dir / f'ls_cbf_low_snr_{file_suffix}.nii.gz')
            nib.save(nib.Nifti1Image(att_map_ls_low, affine, header), subject_output_dir / f'ls_att_low_snr_{file_suffix}.nii.gz')
            nib.save(nib.Nifti1Image(cbf_map_ls_high, affine, header), subject_output_dir / f'ls_cbf_high_snr_{file_suffix}.nii.gz')
            nib.save(nib.Nifti1Image(att_map_ls_high, affine, header), subject_output_dir / f'ls_att_high_snr_{file_suffix}.nii.gz')

    except Exception as e:
        print(f"ERROR predicting on subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()

def batch_predict_nn(all_signals_flat, subject_plds, models, config, norm_stats_full, device, dims):
    num_voxels = all_signals_flat.shape[0]
    num_plds_subject = len(subject_plds)
    num_plds_training = len(config['pld_values'])
    training_plds = np.array(config['pld_values'])
    
    padded_signals_batch = np.zeros((num_voxels, num_plds_training * 2))
    matching_indices = np.where(np.isin(training_plds, subject_plds))[0]
    
    pcasl_signals = all_signals_flat[:, :num_plds_subject]
    vsasl_signals = all_signals_flat[:, num_plds_subject:]
    padded_signals_batch[:, matching_indices] = pcasl_signals
    padded_signals_batch[:, matching_indices + num_plds_training] = vsasl_signals

    eng_feats_batch = engineer_signal_features(padded_signals_batch, num_plds_training)
    nn_input_batch = np.concatenate([padded_signals_batch, eng_feats_batch], axis=1)
    
    norm_input_batch = apply_normalization_vectorized(nn_input_batch, norm_stats_full, num_plds_training)
    
    input_tensor = torch.FloatTensor(norm_input_batch).to(device)
    
    all_cbf_preds_norm, all_att_preds_norm = [], []
    with torch.no_grad():
        for model in models:
            cbf_m_norm, att_m_norm, _, _, _, _ = model(input_tensor)
            all_cbf_preds_norm.append(cbf_m_norm.cpu().numpy())
            all_att_preds_norm.append(att_m_norm.cpu().numpy())

    ensemble_cbf_norm = np.mean(all_cbf_preds_norm, axis=0)
    ensemble_att_norm = np.mean(all_att_preds_norm, axis=0)
    
    cbf_denorm, att_denorm, _, _ = denormalize_predictions(
        ensemble_cbf_norm, ensemble_att_norm, None, None, norm_stats_full
    )
    
    cbf_map = cbf_denorm.reshape(dims, order='F')
    att_map = att_denorm.reshape(dims, order='F')
    
    return cbf_map, att_map

def fit_ls_voxel(signal_flat, subject_plds, config, order='C'):
    num_plds = len(subject_plds)
    # The key change is here, specifying the order
    signal_reshaped = signal_flat.reshape((num_plds, 2), order=order)
    pldti = np.column_stack([subject_plds, subject_plds])
    init = [50.0 / 6000.0, 1500.0]
    asl_params = {k: v for k, v in config.items() if k in ['T1_artery', 'T_tau', 'T2_factor', 'alpha_BS1', 'alpha_PCASL', 'alpha_VSASL']}

    try:
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signal_reshaped, init, **asl_params)
        return beta[0] * 6000.0, beta[1]
    except Exception:
        return np.nan, np.nan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run NN and/or LS fitting on preprocessed in-vivo ASL data.")
    parser.add_argument("preprocessed_dir", type=str, help="Path to the parent directory of preprocessed subjects.")
    parser.add_argument("model_results_dir", type=str, help="Path to the directory containing trained models and configs.")
    parser.add_argument("output_dir", type=str, help="Path to the parent directory where prediction maps will be saved.")
    
    parser.add_argument("--subject", type=str, default=None, help="Process a single subject ID instead of all (e.g., '20230316_MR1_A145').")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N subjects.")
    parser.add_argument("--method", type=str, default="all", choices=['all', 'nn', 'ls', 'ls_debug'], help="Which method to run: 'nn' (Neural Net), 'ls' (Least-Squares C-order), 'ls_debug' (LS F-order), or 'all'.")
    
    args = parser.parse_args()

    # Determine LS reshape order based on the selected method
    ls_order_to_use = 'F' if args.method == 'ls_debug' else 'C'
    
    preprocessed_root_dir = Path(args.preprocessed_dir)
    model_results_dir = Path(args.model_results_dir)
    output_root_dir = Path(args.output_dir)

    print("--- Loading Model Artifacts ---")
    models, config, norm_stats = load_artifacts(model_results_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device)
    print(f"Loaded {len(models)} models and config.")
    
    if args.subject:
        subject_dirs = [preprocessed_root_dir / args.subject]
        if not subject_dirs[0].exists():
            print(f"Error: Specified subject directory not found: {subject_dirs[0]}")
            sys.exit(1)
    else:
        subject_dirs = sorted([d for d in preprocessed_root_dir.iterdir() if d.is_dir()])

    if args.limit:
        subject_dirs = subject_dirs[:args.limit]

    print(f"\nFound {len(subject_dirs)} subjects to process using method: '{args.method.upper()}'")

    for sub_dir in subject_dirs:
        print(f"\n--- Processing Subject: {sub_dir.name} ---")
        predict_subject(sub_dir, models, config, norm_stats, device, output_root_dir, args.method, ls_order_to_use)

    print("\n--- In-vivo prediction pipeline complete! ---")
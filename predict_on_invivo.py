# predict_on_invivo.py (Rewritten for Robustness and Fair Comparison)
import torch
import numpy as np
import nibabel as nib
import sys
from pathlib import Path
from tqdm import tqdm
import time
import argparse
from typing import Dict

# --- Import project modules ---
from evaluate_ensemble import load_artifacts
from main import engineer_signal_features
from comparison_framework import denormalize_predictions
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

# Import the forward models needed for the robust LS grid search
from pcasl_functions import fun_PCASL_1comp_vect_pep
from vsasl_functions import fun_VSASL_1comp_vect_pep

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

def batch_predict_nn(all_signals_flat, subject_plds, models, config, norm_stats_full, device, dims):
    """
    Runs the neural network prediction on a batch of voxels.
    This function handles zero-padding for mismatched PLDs.
    """
    num_voxels = all_signals_flat.shape[0]
    num_plds_subject = len(subject_plds)
    num_plds_training = len(config['pld_values'])
    training_plds = np.array(config['pld_values'])
    
    # Create zero-padded vectors to match the network's expected input size
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
    
    # Reshape the 1D vector of predictions back into a 3D volume using Fortran order
    cbf_map = cbf_denorm.reshape(dims, order='F')
    att_map = att_denorm.reshape(dims, order='F')
    
    return cbf_map, att_map

def fit_ls_robust(all_signals_flat: np.ndarray, subject_plds: np.ndarray, config: dict):
    """
    Runs a robust conventional LS fit on a batch of voxels by first performing
    a grid search to find a good initial guess for the non-linear fitter.
    This prevents the catastrophic failures seen with noisy data.
    """
    num_voxels = all_signals_flat.shape[0]
    num_plds = len(subject_plds)
    pldti = np.column_stack([subject_plds, subject_plds])
    
    # --- Grid Search Setup ---
    grid_cbf = np.linspace(10, 90, 9)
    grid_att = np.linspace(500, 4000, 8)
    grid_cbf_cgs = grid_cbf / 6000.0
    
    pcasl_params = {k: v for k, v in config.items() if k in ['T1_artery', 'T_tau', 'T2_factor', 'alpha_BS1', 'alpha_PCASL']}
    vsasl_params = {k: v for k, v in config.items() if k in ['T1_artery', 'T2_factor', 'alpha_BS1', 'alpha_VSASL']}
    ls_params = {**pcasl_params, **vsasl_params} # For the final fit

    # Pre-calculate the grid of theoretical signals for efficiency
    theoretical_pcasl = np.zeros((len(grid_cbf), len(grid_att), num_plds))
    theoretical_vsasl = np.zeros((len(grid_cbf), len(grid_att), num_plds))
    for i, cbf in enumerate(grid_cbf_cgs):
        for j, att in enumerate(grid_att):
            theoretical_pcasl[i, j, :] = fun_PCASL_1comp_vect_pep([cbf, att], subject_plds, **pcasl_params)
            theoretical_vsasl[i, j, :] = fun_VSASL_1comp_vect_pep([cbf, att], subject_plds, **vsasl_params)
    
    theoretical_grid = np.stack([theoretical_pcasl, theoretical_vsasl], axis=-1)
    
    # --- Batch Processing ---
    cbf_results = np.full(num_voxels, np.nan)
    att_results = np.full(num_voxels, np.nan)
    
    # Reshape the entire input signal batch correctly once using Fortran order
    signals_reshaped_batch = all_signals_flat.reshape((num_voxels, num_plds, 2), order='F')

    for i in tqdm(range(num_voxels), desc="Robust LS Fitting", leave=False, ncols=80):
        try:
            voxel_signal = signals_reshaped_batch[i]

            # Find best initial guess from grid using vectorized operations
            errors = np.sum((theoretical_grid - voxel_signal)**2, axis=(2,3))
            best_cbf_idx, best_att_idx = np.unravel_index(np.argmin(errors), errors.shape)
            
            init_guess = [grid_cbf_cgs[best_cbf_idx], grid_att[best_att_idx]]

            # Run the fine-grained least-squares fit with the better initial guess
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, voxel_signal, init_guess, **ls_params)
            
            cbf_results[i] = beta[0] * 6000.0
            att_results[i] = beta[1]
        except Exception:
            continue
            
    return cbf_results, att_results

def predict_subject(subject_processed_dir: Path, models, config, norm_stats_full, device, output_root: Path, run_method: str):
    """Processes a single subject, running the specified fitting method(s)."""
    subject_id = subject_processed_dir.name
    subject_output_dir = output_root / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load all the necessary data for the subject
        all_low_snr_signals = np.load(subject_processed_dir / 'low_snr_signals.npy')
        all_high_snr_signals = np.load(subject_processed_dir / 'high_snr_signals.npy')
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

        # --- ROBUST Least-Squares Fitting (Conditional) ---
        if run_method in ['all', 'ls']:
            print(f"  [{subject_id}] Running per-voxel ROBUST LS fitting...")
            start_time_ls = time.time()

            cbf_ls_low, att_ls_low = fit_ls_robust(all_low_snr_signals, subject_plds, config)
            cbf_map_ls_low = cbf_ls_low.reshape(dims, order='F')
            att_map_ls_low = att_ls_low.reshape(dims, order='F')

            cbf_ls_high, att_ls_high = fit_ls_robust(all_high_snr_signals, subject_plds, config)
            cbf_map_ls_high = cbf_ls_high.reshape(dims, order='F')
            att_map_ls_high = att_ls_high.reshape(dims, order='F')
            
            duration_ls = time.time() - start_time_ls
            print(f"  [{subject_id}] All robust LS fitting took: {duration_ls:.2f} seconds.")
            
            nib.save(nib.Nifti1Image(cbf_map_ls_low, affine, header), subject_output_dir / 'ls_robust_cbf_low_snr.nii.gz')
            nib.save(nib.Nifti1Image(att_map_ls_low, affine, header), subject_output_dir / 'ls_robust_att_low_snr.nii.gz')
            nib.save(nib.Nifti1Image(cbf_map_ls_high, affine, header), subject_output_dir / 'ls_robust_cbf_high_snr.nii.gz')
            nib.save(nib.Nifti1Image(att_map_ls_high, affine, header), subject_output_dir / 'ls_robust_att_high_snr.nii.gz')

    except Exception as e:
        print(f"ERROR predicting on subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run NN and/or robust LS fitting on preprocessed in-vivo ASL data.")
    parser.add_argument("preprocessed_dir", type=str, help="Path to the parent directory of preprocessed subjects.")
    parser.add_argument("model_results_dir", type=str, help="Path to the directory containing trained models and configs.")
    parser.add_argument("output_dir", type=str, help="Path where prediction maps will be saved.")
    
    parser.add_argument("--subject", type=str, default=None, help="Process a single subject ID instead of all (e.g., '20230316_MR1_A145').")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N subjects.")
    parser.add_argument("--method", type=str, default="all", choices=['all', 'nn', 'ls'], help="Which method to run: 'nn', 'ls' (robust), or 'all'.")
    
    args = parser.parse_args()

    preprocessed_root_dir = Path(args.preprocessed_dir)
    model_results_dir = Path(args.model_results_dir)
    output_root_dir = Path(args.output_dir)

    print("--- Loading Model Artifacts ---")
    models, config, norm_stats = load_artifacts(model_results_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device)
    print(f"Loaded {len(models)} models and config. Using device: {device}")
    
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
        predict_subject(sub_dir, models, config, norm_stats, device, output_root_dir, args.method)

    print("\n--- In-vivo prediction pipeline complete! ---")
# FILE: predict_on_invivo.py
# THIS VERSION USES A ROBUST, GRID-SEARCH-INITIALIZED LS BASELINE.
# MODIFIED to save uncertainty maps and computational timings.

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import argparse
import time
import sys
from tqdm import tqdm
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import multiprocessing

from enhanced_asl_network import EnhancedASLNet, DisentangledASLNet
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from utils import engineer_signal_features, get_grid_search_initial_guess

def apply_normalization_vectorized(batch: np.ndarray, norm_stats: Dict, num_plds: int) -> np.ndarray:
    """Applies normalization to a batch of input signals for the NN."""
    pcasl_raw, vsasl_raw = batch[:, :num_plds], batch[:, num_plds:num_plds*2]
    features = batch[:, num_plds*2:]
    
    pcasl_norm = (pcasl_raw - norm_stats['pcasl_mean']) / (np.array(norm_stats['pcasl_std']) + 1e-6)
    vsasl_norm = (vsasl_raw - norm_stats['vsasl_mean']) / (np.array(norm_stats['vsasl_std']) + 1e-6)
    
    return np.concatenate([pcasl_norm, vsasl_norm, features], axis=1)

def apply_normalization_disentangled_vectorized(batch: np.ndarray, norm_stats: Dict, num_plds: int) -> np.ndarray:
    """Applies normalization for the DisentangledASLNet input format to a batch."""
    raw_signal = batch[:, :num_plds*2]
    eng_features = batch[:, num_plds*2:]
    
    amplitude = np.linalg.norm(raw_signal, axis=1, keepdims=True) + 1e-6
    shape_vector = raw_signal / amplitude
    
    amplitude_norm = (amplitude - norm_stats['amplitude_mean']) / (norm_stats['amplitude_std'] + 1e-6)
    
    return np.concatenate([shape_vector, eng_features, amplitude_norm], axis=1)

def denormalize_predictions(cbf_norm, att_norm, cbf_log_var_norm, att_log_var_norm, norm_stats):
    """Applies de-normalization to NN outputs, including uncertainty."""
    y_mean_cbf, y_std_cbf = norm_stats['y_mean_cbf'], norm_stats['y_std_cbf']
    y_mean_att, y_std_att = norm_stats['y_mean_att'], norm_stats['y_std_att']

    cbf_denorm = cbf_norm * y_std_cbf + y_mean_cbf
    att_denorm = att_norm * y_std_att + y_mean_att
    
    cbf_std_denorm = np.exp(cbf_log_var_norm / 2.0) * y_std_cbf
    att_std_denorm = np.exp(att_log_var_norm / 2.0) * y_std_att

    return cbf_denorm, att_denorm, cbf_std_denorm, att_std_denorm

def batch_predict_nn(signals_masked, subject_plds, models, config, norm_stats, device, is_disentangled) -> Tuple:
    """Runs batched inference and returns predictions AND uncertainty."""
    num_plds_train = len(config['pld_values'])
    eng_feats = engineer_signal_features(signals_masked, num_plds_train)
    
    # === MODIFICATION: Select correct normalization based on model type ===
    if is_disentangled:
        nn_input_unnorm = np.concatenate([signals_masked, eng_feats], axis=1)
        norm_input = apply_normalization_disentangled_vectorized(nn_input_unnorm, norm_stats, num_plds_train)
    else:
        nn_input = np.concatenate([signals_masked, eng_feats], axis=1)
        norm_input = apply_normalization_vectorized(nn_input, norm_stats, num_plds_train)
    
    input_tensor = torch.FloatTensor(norm_input).to(device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        cbf_means, att_means, cbf_log_vars, att_log_vars = [], [], [], []
        # Process in chunks to avoid GPU memory errors on large masks
        batch_size = 8192
        for i in range(0, len(input_tensor), batch_size):
            batch_tensor = input_tensor[i:i+batch_size]
            
            cbf_means_batch = [model(batch_tensor)[0].cpu().numpy() for model in models]
            att_means_batch = [model(batch_tensor)[1].cpu().numpy() for model in models]
            cbf_log_vars_batch = [model(batch_tensor)[2].cpu().numpy() for model in models]
            att_log_vars_batch = [model(batch_tensor)[3].cpu().numpy() for model in models]
            
            cbf_means.append(np.mean(cbf_means_batch, axis=0))
            att_means.append(np.mean(att_means_batch, axis=0))
            cbf_log_vars.append(np.mean(cbf_log_vars_batch, axis=0))
            att_log_vars.append(np.mean(att_log_vars_batch, axis=0))
    
    cbf_norm_ens = np.concatenate(cbf_means, axis=0)
    att_norm_ens = np.concatenate(att_means, axis=0)
    cbf_log_var_ens = np.concatenate(cbf_log_vars, axis=0)
    att_log_var_ens = np.concatenate(att_log_vars, axis=0)
    
    return denormalize_predictions(
        cbf_norm_ens.squeeze(), att_norm_ens.squeeze(),
        cbf_log_var_ens.squeeze(), att_log_var_ens.squeeze(), norm_stats
    )

def _fit_single_voxel_ls(signal_flat: np.ndarray, plds: np.ndarray, pldti: np.ndarray, ls_params: dict):
    """Helper function to fit a single voxel with robust grid-search initialization."""
    try:
        init_guess = get_grid_search_initial_guess(signal_flat, plds, ls_params)
        signal_reshaped = signal_flat.reshape((len(plds), 2), order='F')
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signal_reshaped, init_guess, **ls_params)
        return beta[0] * 6000.0, beta[1]
    except Exception:
        return np.nan, np.nan

def fit_ls_robust(signals_masked: np.ndarray, plds: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Runs a robust, parallelized LS fit for each voxel in the masked data."""
    pldti = np.column_stack([plds, plds])
    ls_params = {k:v for k,v in config.items() if k in ['T1_artery','T_tau','T2_factor','alpha_BS1','alpha_PCASL','alpha_VSASL']}
    results = Parallel(n_jobs=-1)(
        delayed(_fit_single_voxel_ls)(signals_masked[i], plds, pldti, ls_params)
        for i in tqdm(range(signals_masked.shape[0]), desc="  Fitting LS (Robust)", leave=False, ncols=100)
    )
    results_arr = np.array(results)
    return results_arr[:, 0], results_arr[:, 1]

def predict_subject(subject_dir: Path, models: List, config: Dict, norm_stats: Dict, device: torch.device, output_root: Path, is_disentangled: bool):
    subject_id = subject_dir.name
    subject_output_dir = output_root / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Predicting for Subject: {subject_id} ---")
    
    low_snr_signals = np.load(subject_dir / 'low_snr_signals.npy')
    brain_mask_flat = np.load(subject_dir / 'brain_mask.npy').flatten()
    plds = np.load(subject_dir / 'plds.npy')
    dims, affine, header = tuple(np.load(subject_dir/'image_dims.npy')), np.load(subject_dir/'image_affine.npy'), np.load(subject_dir/'image_header.npy', allow_pickle=True).item()
    
    low_snr_masked = low_snr_signals[brain_mask_flat]
    timings = {}
    
    # --- NN Prediction ---
    start_time_nn = time.time()
    nn_results = batch_predict_nn(low_snr_masked, plds, models, config, norm_stats, device, is_disentangled)
    timings['nn_total_s'] = time.time() - start_time_nn
    
    # --- LS Prediction ---
    start_time_ls = time.time()
    ls_cbf_masked, ls_att_masked = fit_ls_robust(low_snr_masked, plds, config)
    timings['ls_total_s'] = time.time() - start_time_ls
    
    # Save timings
    with open(subject_output_dir / 'timings.json', 'w') as f:
        json.dump(timings, f, indent=2)

    # Save all maps
    maps_to_save = {
        "nn_1r_cbf": nn_results[0], "nn_1r_att": nn_results[1],
        "nn_1r_cbf_uncertainty": nn_results[2], "nn_1r_att_uncertainty": nn_results[3],
        "ls_1r_cbf": ls_cbf_masked, "ls_1r_att": ls_att_masked,
    }
    for name, data_masked in maps_to_save.items():
        full_map = np.zeros(brain_mask_flat.shape, dtype=np.float32)
        full_map[brain_mask_flat] = data_masked
        nib.save(nib.Nifti1Image(full_map.reshape(dims), affine, header), subject_output_dir / f'{name}.nii.gz')
    print(f"  --> Saved all maps and timings for {subject_id}.")

def main():
    parser = argparse.ArgumentParser(description="Run inference on preprocessed in-vivo ASL data.")
    parser.add_argument("preprocessed_dir", type=str, help="Path to the directory containing the preprocessed NumPy data.")
    parser.add_argument("model_artifacts_dir", type=str, help="Path to the directory with trained models, config, and norm stats.")
    parser.add_argument("output_dir", type=str, help="Directory to save the final NIfTI parameter maps.")
    args = parser.parse_args()

    preprocessed_root = Path(args.preprocessed_dir)
    model_root = Path(args.model_artifacts_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Load Artifacts ---
    print(f"--> Loading artifacts from: {model_root}")
    try:
        with open(model_root / 'research_config.json', 'r') as f:
            config = json.load(f)
        with open(model_root / 'norm_stats.json', 'r') as f:
            norm_stats = json.load(f)

        models = []
        models_dir = model_root / 'trained_models'
        num_plds = len(config['pld_values'])
        
        is_disentangled = 'Disentangled' in config.get('model_class_name', '')
        if is_disentangled:
            model_class = DisentangledASLNet
            base_input_size = num_plds * 2 + 4 + 1
        else:
            model_class = EnhancedASLNet
            base_input_size = num_plds * 2 + 4

        for model_path in models_dir.glob('ensemble_model_*.pt'):
            model = model_class(input_size=base_input_size, **config)
            model.to(dtype=torch.bfloat16)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            models.append(model)
        
        if not models:
            raise FileNotFoundError("No models found in trained_models folder.")
    except Exception as e:
        print(f"[FATAL ERROR] Could not load artifacts: {e}. Exiting.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models:
        model.to(device)

    subject_dirs = sorted([d for d in preprocessed_root.iterdir() if d.is_dir()])
    for subject_dir in tqdm(subject_dirs, desc="Processing All Subjects"):
        predict_subject(subject_dir, models, config, norm_stats, device, output_root, is_disentangled)

    print("\n--- All subjects predicted successfully! ---")

if __name__ == '__main__':
    main()
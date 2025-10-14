# predict_on_invivo.py
# FINAL CORRECTED VERSION

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

# Import both model classes
from enhanced_asl_network import EnhancedASLNet, DisentangledASLNet
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from utils import engineer_signal_features, get_grid_search_initial_guess

def apply_normalization_vectorized(batch: np.ndarray, norm_stats: Dict, num_plds: int) -> np.ndarray:
    """Applies normalization for the original EnhancedASLNet input format."""
    raw_signal, features = batch[:, :num_plds*2], batch[:, num_plds*2:]
    pcasl_raw, vsasl_raw = raw_signal[:, :num_plds], raw_signal[:, num_plds:]
    
    pcasl_norm = (pcasl_raw - norm_stats['pcasl_mean']) / (np.array(norm_stats['pcasl_std']) + 1e-6)
    vsasl_norm = (vsasl_raw - norm_stats['vsasl_mean']) / (np.array(norm_stats['vsasl_std']) + 1e-6)
    
    return np.concatenate([pcasl_norm, vsasl_norm, features], axis=1)

def apply_normalization_disentangled(batch: np.ndarray, norm_stats: Dict, num_plds: int) -> np.ndarray:
    """Applies normalization for the new DisentangledASLNet input format."""
    raw_signal, eng_features = batch[:, :num_plds*2], batch[:, num_plds*2:]
    
    amplitude = np.linalg.norm(raw_signal, axis=1, keepdims=True) + 1e-6
    shape_vector = raw_signal / amplitude
    
    amplitude_norm = (amplitude - norm_stats['amplitude_mean']) / (norm_stats['amplitude_std'] + 1e-6)
    
    return np.concatenate([shape_vector, eng_features, amplitude_norm], axis=1)

def denormalize_predictions(cbf_norm, att_norm, cbf_log_var_norm, att_log_var_norm, norm_stats):
    """Applies de-normalization to all NN outputs, including uncertainty."""
    y_mean_cbf, y_std_cbf = norm_stats['y_mean_cbf'], norm_stats['y_std_cbf']
    y_mean_att, y_std_att = norm_stats['y_mean_att'], norm_stats['y_std_att']

    cbf_denorm = cbf_norm * y_std_cbf + y_mean_cbf
    att_denorm = att_norm * y_std_att + y_mean_att
    
    cbf_std_denorm = None
    if cbf_log_var_norm is not None:
        # Clamp log_var to prevent numerical overflow from exp()
        cbf_log_var_norm_clamped = np.clip(cbf_log_var_norm, -20, 20)
        cbf_std_denorm = np.exp(cbf_log_var_norm_clamped / 2.0) * y_std_cbf

    att_std_denorm = None
    if att_log_var_norm is not None:
        att_log_var_norm_clamped = np.clip(att_log_var_norm, -20, 20)
        att_std_denorm = np.exp(att_log_var_norm_clamped / 2.0) * y_std_att

    return cbf_denorm, att_denorm, cbf_std_denorm, att_std_denorm

def batch_predict_nn(signals_masked: np.ndarray, subject_plds: np.ndarray, models: List, config: Dict, norm_stats: Dict, device: torch.device, is_disentangled: bool) -> Tuple:
    """
    Runs batched inference, handling PLD resampling and feature engineering correctly.
    """
    model_plds_list = config['pld_values']
    num_model_plds = len(model_plds_list)
    num_subject_plds = len(subject_plds)
    
    resampled_signals = np.zeros((signals_masked.shape[0], num_model_plds * 2), dtype=np.float32)

    target_indices, source_indices = [], []
    for i, pld in enumerate(subject_plds):
        try:
            target_idx = model_plds_list.index(int(pld))
            target_indices.append(target_idx)
            source_indices.append(i)
        except ValueError:
            pass
            
    source_indices, target_indices = np.array(source_indices), np.array(target_indices)

    if source_indices.size > 0:
        resampled_signals[:, target_indices] = signals_masked[:, source_indices]
        resampled_signals[:, target_indices + num_model_plds] = signals_masked[:, source_indices + num_subject_plds]
    
    eng_feats = engineer_signal_features(resampled_signals, num_model_plds)
    
    unnormalized_input = np.concatenate([resampled_signals, eng_feats], axis=1)

    if is_disentangled:
        norm_input = apply_normalization_disentangled(unnormalized_input, norm_stats, num_model_plds)
    else:
        norm_input = apply_normalization_vectorized(unnormalized_input, norm_stats, num_model_plds)
    
    input_tensor = torch.FloatTensor(norm_input).to(device)
    
    with torch.no_grad():
        cbf_means, att_means, cbf_log_vars, att_log_vars = [], [], [], []
        batch_size = 8192
        for i in tqdm(range(0, len(input_tensor), batch_size), desc="  NN Inference", leave=False, ncols=100):
            batch_tensor = input_tensor[i:i+batch_size]
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                all_outputs = [model(batch_tensor) for model in models]

            cbf_means_batch = [out[0].cpu().float().numpy() for out in all_outputs]
            att_means_batch = [out[1].cpu().float().numpy() for out in all_outputs]
            cbf_log_vars_batch = [out[2].cpu().float().numpy() if out[2] is not None else np.zeros_like(cbf_m) for out, cbf_m in zip(all_outputs, cbf_means_batch)]
            att_log_vars_batch = [out[3].cpu().float().numpy() if out[3] is not None else np.zeros_like(att_m) for out, att_m in zip(all_outputs, att_means_batch)]

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
    try:
        init_guess = get_grid_search_initial_guess(signal_flat, plds, ls_params)
        signal_reshaped = signal_flat.reshape((len(plds), 2), order='F')
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signal_reshaped, init_guess, **ls_params)
        return beta[0] * 6000.0, beta[1]
    except Exception:
        return np.nan, np.nan

def fit_ls_robust(signals_masked: np.ndarray, plds: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    pldti = np.column_stack([plds, plds])
    ls_params = {k:v for k,v in config.items() if k in ['T1_artery','T_tau','T2_factor','alpha_BS1','alpha_PCASL','alpha_VSASL']}
    results = Parallel(n_jobs=-1)(
        delayed(_fit_single_voxel_ls)(signals_masked[i], plds, pldti, ls_params)
        for i in range(signals_masked.shape[0])
    )
    results_arr = np.array(results)
    return results_arr[:, 0], results_arr[:, 1]

def predict_subject(subject_dir: Path, models: List, config: Dict, norm_stats: Dict, device: torch.device, output_root: Path, is_disentangled: bool):
    subject_id = subject_dir.name
    subject_output_dir = output_root / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    low_snr_signals = np.load(subject_dir / 'low_snr_signals.npy')
    brain_mask_flat = np.load(subject_dir / 'brain_mask.npy').flatten()
    plds = np.load(subject_dir / 'plds.npy')
    dims, affine, header = tuple(np.load(subject_dir/'image_dims.npy')), np.load(subject_dir/'image_affine.npy'), np.load(subject_dir/'image_header.npy', allow_pickle=True).item()
    
    low_snr_masked = low_snr_signals[brain_mask_flat]
    timings = {}
    
    start_time_nn = time.time()
    nn_results = batch_predict_nn(low_snr_masked, plds, models, config, norm_stats, device, is_disentangled)
    timings['nn_total_s'] = time.time() - start_time_nn
    
    start_time_ls = time.time()
    ls_cbf_masked, ls_att_masked = fit_ls_robust(low_snr_masked, plds, config)
    timings['ls_total_s'] = time.time() - start_time_ls
    
    with open(subject_output_dir / 'timings.json', 'w') as f: json.dump(timings, f, indent=2)

    maps_to_save = {
        "nn_1r_cbf": nn_results[0], "nn_1r_att": nn_results[1],
        "nn_1r_cbf_uncertainty": nn_results[2], "nn_1r_att_uncertainty": nn_results[3],
        "ls_1r_cbf": ls_cbf_masked, "ls_1r_att": ls_att_masked,
    }
    for name, data_masked in maps_to_save.items():
        full_map = np.zeros(brain_mask_flat.shape, dtype=np.float32)
        full_map[brain_mask_flat] = data_masked
        nib.save(nib.Nifti1Image(full_map.reshape(dims), affine, header), subject_output_dir / f'{name}.nii.gz')
    tqdm.write(f"  --> Saved maps & timings for {subject_id}")

def load_artifacts(model_root: Path) -> tuple:
    print(f"--> Loading artifacts from: {model_root}")
    with open(model_root / 'research_config.json', 'r') as f: config = json.load(f)
    with open(model_root / 'norm_stats.json', 'r') as f: norm_stats = json.load(f)

    models, models_dir = [], model_root / 'trained_models'
    num_plds = len(config['pld_values'])
    
    is_disentangled = 'Disentangled' in config.get('model_class_name', '')
    if is_disentangled:
        print("  --> Detected DisentangledASLNet model type.")
        model_class = DisentangledASLNet
        base_input_size = num_plds * 2 + 4 + 1
    else:
        print("  --> Detected original EnhancedASLNet model type.")
        model_class = EnhancedASLNet
        base_input_size = num_plds * 2 + 4

    for model_path in models_dir.glob('ensemble_model_*.pt'):
        model = model_class(input_size=base_input_size, **config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        model.eval()
        models.append(model)
    
    if not models: raise FileNotFoundError("No models found.")
    return models, config, norm_stats, is_disentangled

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on preprocessed in-vivo ASL data.")
    parser.add_argument("preprocessed_dir", type=str)
    parser.add_argument("model_artifacts_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    preprocessed_root, model_root, output_root = Path(args.preprocessed_dir), Path(args.model_artifacts_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        models, config, norm_stats, is_disentangled = load_artifacts(model_root)
    except Exception as e:
        print(f"[FATAL ERROR] Could not load artifacts: {e}. Exiting.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device, dtype=torch.bfloat16)
    print(f"Loaded {len(models)} models. Using device: {device} with bfloat16.")

    subject_dirs = sorted([d for d in preprocessed_root.iterdir() if d.is_dir()])
    for subject_dir in tqdm(subject_dirs, desc="Processing All Subjects"):
        predict_subject(subject_dir, models, config, norm_stats, device, output_root, is_disentangled)

    print("\n--- All subjects predicted successfully! ---")
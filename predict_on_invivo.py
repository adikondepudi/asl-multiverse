# predict_on_invivo.py
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

from enhanced_asl_network import DisentangledASLNet
from spatial_asl_network import SpatialASLNet
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from utils import get_grid_search_initial_guess, process_signals_dynamic
from feature_registry import FeatureRegistry, validate_signals, validate_norm_stats


def pad_to_multiple(image: np.ndarray, multiple: int = 16) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad 2D/3D image to be divisible by multiple (for U-Net pooling).
    
    Args:
        image: (H, W) or (C, H, W) or (C, H, W) array
        multiple: Value dimensions should be divisible by
        
    Returns:
        padded_image: Padded array
        padding: (pad_top, pad_bottom, pad_left, pad_right) for unpadding
    """
    if image.ndim == 2:
        h, w = image.shape
    else:
        h, w = image.shape[-2:]
    
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if image.ndim == 2:
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    else:
        padded = np.pad(image, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(image: np.ndarray, padding: Tuple[int, int, int, int]) -> np.ndarray:
    """Remove padding from image."""
    pad_top, pad_bottom, pad_left, pad_right = padding
    
    if image.ndim == 2:
        return image[pad_top:image.shape[0]-pad_bottom if pad_bottom else None,
                    pad_left:image.shape[1]-pad_right if pad_right else None]
    else:
        return image[:, pad_top:image.shape[1]-pad_bottom if pad_bottom else None,
                    pad_left:image.shape[2]-pad_right if pad_right else None]


def predict_spatial_slice(slice_data: np.ndarray, model: torch.nn.Module,
                          device: torch.device, m0_slice: np.ndarray = None,
                          norm_stats: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run spatial U-Net inference on a single brain slice.

    Args:
        slice_data: (2*n_plds, H, W) - PCASL + VSASL channels
        model: SpatialASLNet model
        device: torch device
        m0_slice: (H, W) M0 calibration image for normalization
        norm_stats: dict with y_mean_cbf, y_std_cbf, y_mean_att, y_std_att for denormalization

    Returns:
        cbf_map: (H, W) CBF in ml/100g/min
        att_map: (H, W) ATT in ms
    """
    # Normalize by M0 if provided
    if m0_slice is not None:
        m0_safe = np.maximum(m0_slice, np.percentile(m0_slice, 5))
        slice_data = slice_data / m0_safe[np.newaxis, :, :]

    # CRITICAL: Per-pixel temporal normalization (must match training)
    # Z-score each pixel's temporal signal across channels
    temporal_mean = np.mean(slice_data, axis=0, keepdims=True)  # (1, H, W)
    temporal_std = np.std(slice_data, axis=0, keepdims=True) + 1e-6  # (1, H, W)
    slice_data = (slice_data - temporal_mean) / temporal_std

    # Pad to multiple of 16
    original_shape = slice_data.shape[-2:]
    padded, padding = pad_to_multiple(slice_data, multiple=16)

    # Convert to tensor and add batch dimension
    input_tensor = torch.from_numpy(padded[np.newaxis, ...]).float().to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            cbf_pred_norm, att_pred_norm, _, _ = model(input_tensor)

    # DENORMALIZE predictions - model outputs normalized z-scores
    if norm_stats is not None:
        cbf_pred = cbf_pred_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
        att_pred = att_pred_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
        # Apply physical constraints
        cbf_pred = torch.clamp(cbf_pred, min=0.0, max=200.0)
        att_pred = torch.clamp(att_pred, min=0.0, max=5000.0)
    else:
        cbf_pred = cbf_pred_norm
        att_pred = att_pred_norm

    # Convert to numpy and remove batch dim
    cbf_map = cbf_pred[0, 0].cpu().numpy()
    att_map = att_pred[0, 0].cpu().numpy()

    # Unpad to original size
    cbf_map = unpad(cbf_map, padding)
    att_map = unpad(att_map, padding)

    return cbf_map, att_map


def predict_spatial_volume(spatial_signals: np.ndarray, model: torch.nn.Module,
                           device: torch.device, m0_data: np.ndarray = None,
                           batch_size: int = 8, norm_stats: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run spatial U-Net inference on a full 3D volume (slice-by-slice).

    Args:
        spatial_signals: (Z, 2*n_plds, H, W) - preprocessed spatial stack
        model: SpatialASLNet model
        device: torch device
        m0_data: (H, W, Z) M0 calibration volume
        batch_size: Number of slices to process at once
        norm_stats: dict with y_mean_cbf, y_std_cbf, y_mean_att, y_std_att for denormalization

    Returns:
        cbf_volume: (H, W, Z) CBF maps
        att_volume: (H, W, Z) ATT maps
    """
    n_slices = spatial_signals.shape[0]
    _, h, w = spatial_signals.shape[1], spatial_signals.shape[2], spatial_signals.shape[3]

    # Pad spatial dimensions
    sample_padded, padding = pad_to_multiple(spatial_signals[0], multiple=16)
    padded_h, padded_w = sample_padded.shape[-2:]

    cbf_volume = np.zeros((n_slices, h, w), dtype=np.float32)
    att_volume = np.zeros((n_slices, h, w), dtype=np.float32)

    model.eval()

    for start_idx in range(0, n_slices, batch_size):
        end_idx = min(start_idx + batch_size, n_slices)
        batch_slices = spatial_signals[start_idx:end_idx]

        # Normalize by M0 if provided
        if m0_data is not None:
            m0_batch = m0_data[:, :, start_idx:end_idx].transpose(2, 0, 1)  # (batch, H, W)
            m0_safe = np.maximum(m0_batch, np.percentile(m0_batch, 5, axis=(1, 2), keepdims=True))
            batch_slices = batch_slices / m0_safe[:, np.newaxis, :, :]

        # CRITICAL: Per-pixel temporal normalization (must match training)
        # Z-score each pixel's temporal signal across channels
        # batch_slices shape: (batch, C, H, W) where C is temporal
        temporal_mean = np.mean(batch_slices, axis=1, keepdims=True)  # (batch, 1, H, W)
        temporal_std = np.std(batch_slices, axis=1, keepdims=True) + 1e-6  # (batch, 1, H, W)
        batch_slices = (batch_slices - temporal_mean) / temporal_std

        # Pad each slice
        padded_batch = np.zeros((end_idx - start_idx, batch_slices.shape[1], padded_h, padded_w), dtype=np.float32)
        for i in range(end_idx - start_idx):
            padded_batch[i], _ = pad_to_multiple(batch_slices[i], multiple=16)

        # Convert to tensor
        input_tensor = torch.from_numpy(padded_batch).float().to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                cbf_pred_norm, att_pred_norm, _, _ = model(input_tensor)

        # DENORMALIZE predictions - model outputs normalized z-scores
        if norm_stats is not None:
            cbf_pred = cbf_pred_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
            att_pred = att_pred_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
            # Apply physical constraints
            cbf_pred = torch.clamp(cbf_pred, min=0.0, max=200.0)
            att_pred = torch.clamp(att_pred, min=0.0, max=5000.0)
        else:
            cbf_pred = cbf_pred_norm
            att_pred = att_pred_norm

        # Convert to numpy
        cbf_batch = cbf_pred[:, 0].cpu().numpy()
        att_batch = att_pred[:, 0].cpu().numpy()
        
        # Unpad and store
        for i in range(end_idx - start_idx):
            cbf_volume[start_idx + i] = unpad(cbf_batch[i], padding)
            att_volume[start_idx + i] = unpad(att_batch[i], padding)
    
    # Transpose back to (H, W, Z) format
    return cbf_volume.transpose(1, 2, 0), att_volume.transpose(1, 2, 0)

def denormalize_predictions(cbf_norm, att_norm, cbf_log_var_norm, att_log_var_norm, norm_stats):
    """Applies de-normalization to all NN outputs, including uncertainty."""
    y_mean_cbf, y_std_cbf = norm_stats['y_mean_cbf'], norm_stats['y_std_cbf']
    y_mean_att, y_std_att = norm_stats['y_mean_att'], norm_stats['y_std_att']

    cbf_denorm = cbf_norm * y_std_cbf + y_mean_cbf
    att_denorm = att_norm * y_std_att + y_mean_att
    
    cbf_std_denorm = None
    if cbf_log_var_norm is not None:
        cbf_log_var_norm_clamped = np.clip(cbf_log_var_norm, -20, 20)
        cbf_std_denorm = np.exp(cbf_log_var_norm_clamped / 2.0) * y_std_cbf

    att_std_denorm = None
    if att_log_var_norm is not None:
        att_log_var_norm_clamped = np.clip(att_log_var_norm, -20, 20)
        att_std_denorm = np.exp(att_log_var_norm_clamped / 2.0) * y_std_att

    return cbf_denorm, att_denorm, cbf_std_denorm, att_std_denorm

def batch_predict_nn(signals_masked: np.ndarray, subject_plds: np.ndarray, models: List, config: Dict, norm_stats: Dict, device: torch.device, is_disentangled: bool, expected_scalars: int) -> Tuple:
    """
    Runs batched inference with dynamic feature processing based on active_features config.
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
    
    # Build processing config for dynamic feature selection
    # DEFENSIVE: Require active_features from config - no silent defaults
    active_features = config.get('active_features')
    if active_features is None:
        print("[WARNING] Config missing 'active_features'. Using detected scalar count for legacy compatibility.")
        active_features = ['mean', 'std', 'peak', 't1_artery']  # Legacy fallback
    
    processing_config = {
        'pld_values': model_plds_list,
        'active_features': active_features
    }
    
    # T1 Injection
    t1_val = config.get('T1_artery', 1850.0)
    t1_values = np.full((resampled_signals.shape[0], 1), t1_val, dtype=np.float32)

    # Use dynamic feature processing - pass raw resampled signals
    norm_input = process_signals_dynamic(resampled_signals, norm_stats, processing_config, t1_values=t1_values)
    
    # Feature Compatibility Check (Legacy Mode) - truncate if model expects fewer features
    shape_dim = num_model_plds * 2
    current_scalar_dim = norm_input.shape[1] - shape_dim
    
    if current_scalar_dim > expected_scalars:
        scalars_only = norm_input[:, shape_dim:]
        scalars_truncated = scalars_only[:, :expected_scalars]
        norm_input = np.concatenate([norm_input[:, :shape_dim], scalars_truncated], axis=1)
            
    input_tensor = torch.FloatTensor(norm_input).to(device)
    
    with torch.no_grad():
        cbf_means, att_means, cbf_log_vars, att_log_vars = [], [], [], []
        batch_size = 8192
        for i in range(0, len(input_tensor), batch_size):
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
    
    # Denormalize and Clamp
    cbf_denorm, att_denorm, cbf_std, att_std = denormalize_predictions(
        cbf_norm_ens.squeeze(), att_norm_ens.squeeze(),
        cbf_log_var_ens.squeeze(), att_log_var_ens.squeeze(), norm_stats
    )
    
    cbf_denorm = np.maximum(cbf_denorm, 0)
    att_denorm = np.maximum(att_denorm, 0)
    
    return cbf_denorm, att_denorm, cbf_std, att_std

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

def predict_subject(subject_dir: Path, models: List, config: Dict, norm_stats: Dict, device: torch.device, output_root: Path, is_disentangled: bool, expected_scalars: int):
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
    nn_results = batch_predict_nn(low_snr_masked, plds, models, config, norm_stats, device, is_disentangled, expected_scalars)
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

def load_artifacts(model_root: Path) -> tuple:
    print(f"--> Loading artifacts from: {model_root}")
    with open(model_root / 'research_config.json', 'r') as f: config = json.load(f)
    with open(model_root / 'norm_stats.json', 'r') as f: norm_stats = json.load(f)

    models, models_dir = [], model_root / 'trained_models'
    num_plds = len(config['pld_values'])
    
    is_disentangled = 'Disentangled' in config.get('model_class_name', '')
    model_class = DisentangledASLNet if is_disentangled else EnhancedASLNet
    
    # Input size is dynamically determined from checkpoint - scalars are auto-detected
    # Note: base_input_size is only used for model construction, not data processing

    # Automatically detect expected input size from the checkpoint
    sample_checkpoint = list(models_dir.glob('ensemble_model_*.pt'))[0]
    state_dict = torch.load(sample_checkpoint, map_location='cpu')
    
    expected_scalars = 0
    if 'encoder.pcasl_film.generator.0.weight' in state_dict:
        expected_scalars = state_dict['encoder.pcasl_film.generator.0.weight'].shape[1]
    else:
        # Fallback if specific layer not found (e.g. older architecture)
        expected_scalars = len(norm_stats['scalar_features_mean']) + 1
    
    print(f"  --> Detected model expectation: {expected_scalars} scalar features.")

    for model_path in models_dir.glob('ensemble_model_*.pt'):
        # Calculate input_size dynamically from detected scalars
        base_input_size = num_plds * 2 + expected_scalars
        model = model_class(mode='regression', input_size=base_input_size, num_scalar_features=expected_scalars, **config)
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        except RuntimeError as e:
            print(f"[WARN] Partial load for {model_path.name}: {e}")
        
        model.eval()
        models.append(model)
    
    if not models: raise FileNotFoundError("No models found.")
    return models, config, norm_stats, is_disentangled, expected_scalars

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on preprocessed in-vivo ASL data.")
    parser.add_argument("preprocessed_dir", type=str)
    parser.add_argument("model_artifacts_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    preprocessed_root, model_root, output_root = Path(args.preprocessed_dir), Path(args.model_artifacts_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        models, config, norm_stats, is_disentangled, expected_scalars = load_artifacts(model_root)
    except Exception as e:
        print(f"[FATAL ERROR] Could not load artifacts: {e}. Exiting.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device, dtype=torch.float16)
    print(f"Loaded {len(models)} models. Using device: {device} with float16 (T4 Optimized).")

    subject_dirs = sorted([d for d in preprocessed_root.iterdir() if d.is_dir()])
    for subject_dir in tqdm(subject_dirs, desc="Processing All Subjects"):
        predict_subject(subject_dir, models, config, norm_stats, device, output_root, is_disentangled, expected_scalars)

    print("\n--- All subjects predicted successfully! ---")
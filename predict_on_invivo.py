# adikondepudi-asl-multiverse/predict_on_invivo.py

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

from enhanced_asl_network import EnhancedASLNet
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep

# --- Feature Engineering & Normalization Helpers (must match training) ---
def engineer_signal_features(raw_signal: np.ndarray, num_plds: int) -> np.ndarray:
    if raw_signal.ndim == 1: raw_signal = raw_signal.reshape(1, -1)
    num_samples = raw_signal.shape[0]
    engineered_features = np.zeros((num_samples, 4))
    plds_indices = np.arange(num_plds)
    for i in range(num_samples):
        pcasl_curve, vsasl_curve = raw_signal[i, :num_plds], raw_signal[i, num_plds:]
        engineered_features[i, 0] = np.argmax(pcasl_curve)
        engineered_features[i, 1] = np.argmax(vsasl_curve)
        pcasl_sum = np.sum(pcasl_curve) + 1e-6
        vsasl_sum = np.sum(vsasl_curve) + 1e-6
        engineered_features[i, 2] = np.sum(pcasl_curve * plds_indices) / pcasl_sum
        engineered_features[i, 3] = np.sum(vsasl_curve * plds_indices) / vsasl_sum
    return engineered_features

def apply_normalization_vectorized(batch: np.ndarray, norm_stats: Dict, num_plds: int) -> np.ndarray:
    pcasl_raw, vsasl_raw = batch[:, :num_plds], batch[:, num_plds:num_plds*2]
    features = batch[:, num_plds*2:]
    pcasl_norm = (pcasl_raw - norm_stats['pcasl_mean']) / (np.array(norm_stats['pcasl_std']) + 1e-6)
    vsasl_norm = (vsasl_raw - norm_stats['vsasl_mean']) / (np.array(norm_stats['vsasl_std']) + 1e-6)
    return np.concatenate([pcasl_norm, vsasl_norm, features], axis=1)

def denormalize_predictions(cbf_norm: np.ndarray, att_norm: np.ndarray, norm_stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
    cbf_denorm = cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att_denorm = att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    return cbf_denorm, att_denorm

def resample_signal_for_model(signals_masked: np.ndarray, subject_plds: np.ndarray, config: Dict) -> Tuple[np.ndarray, int]:
    model_plds_list = config.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000])
    num_model_plds = len(model_plds_list)
    num_subject_plds = len(subject_plds)
    resampled_signals = np.zeros((signals_masked.shape[0], num_model_plds * 2))
    target_indices, source_indices = [], []
    for i, pld in enumerate(subject_plds):
        try:
            target_idx = model_plds_list.index(pld)
            target_indices.append(target_idx)
            source_indices.append(i)
        except ValueError:
            pass
    if source_indices:
        target_indices, source_indices = np.array(target_indices), np.array(source_indices)
        resampled_signals[:, target_indices] = signals_masked[:, source_indices]
        resampled_signals[:, target_indices + num_model_plds] = signals_masked[:, source_indices + num_subject_plds]
    return resampled_signals, num_model_plds

def load_artifacts(model_results_root: Path) -> Tuple[List[EnhancedASLNet], Dict, Dict]:
    """Robustly loads model ensemble, final config, and norm stats."""
    # Load base config
    with open(model_results_root / 'research_config.json', 'r') as f:
        config = json.load(f)
    
    # Check for final results and update config with Optuna's best params if they exist
    final_results_path = model_results_root / 'final_research_results.json'
    if final_results_path.exists():
        with open(final_results_path, 'r') as f:
            final_results = json.load(f)
        if 'optuna_best_params' in final_results and final_results['optuna_best_params']:
            print("  --> Optuna parameters found. Updating config for model loading.")
            best_params = final_results['optuna_best_params']
            config['hidden_sizes'] = [
                best_params.get('hidden_size_1'), best_params.get('hidden_size_2'), best_params.get('hidden_size_3')
            ]
            config['dropout_rate'] = best_params.get('dropout_rate')

    with open(model_results_root / 'norm_stats.json', 'r') as f:
        norm_stats = json.load(f)
        
    models = []
    models_dir = model_results_root / 'trained_models'
    num_plds = len(config['pld_values'])
    base_input_size = num_plds * 2 + 4
    
    for model_path in models_dir.glob('ensemble_model_*.pt'):
        # Instantiate model with the final, correct config
        model = EnhancedASLNet(input_size=base_input_size, **config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        models.append(model)
        
    if not models: raise FileNotFoundError("No models found in trained_models folder.")
    return models, config, norm_stats

def batch_predict_nn(signals_masked: np.ndarray, subject_plds: np.ndarray, models: List, config: Dict, norm_stats: Dict, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    resampled_signals, num_plds_for_model = resample_signal_for_model(signals_masked, subject_plds, config)
    eng_feats = engineer_signal_features(resampled_signals, num_plds_for_model)
    nn_input = np.concatenate([resampled_signals, eng_feats], axis=1)
    norm_input = apply_normalization_vectorized(nn_input, norm_stats, num_plds_for_model)
    input_tensor = torch.FloatTensor(norm_input).to(device)
    with torch.no_grad():
        cbf_preds = [model(input_tensor)[0].cpu().numpy() for model in models]
        att_preds = [model(input_tensor)[1].cpu().numpy() for model in models]
    cbf_norm, att_norm = np.mean(cbf_preds, axis=0), np.mean(att_preds, axis=0)
    return denormalize_predictions(cbf_norm.squeeze(), att_norm.squeeze(), norm_stats)

def fit_ls_robust(signals_masked: np.ndarray, plds: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    pldti = np.column_stack([plds, plds])
    ls_params = {k:v for k,v in config.items() if k in ['T1_artery','T_tau','T2_factor','alpha_BS1','alpha_PCASL','alpha_VSASL']}
    num_voxels = signals_masked.shape[0]
    cbf_results, att_results = np.full(num_voxels, np.nan), np.full(num_voxels, np.nan)
    signals_reshaped = signals_masked.reshape((num_voxels, len(plds), 2), order='F')
    init_guess = [50.0 / 6000.0, 1500.0]
    for i in tqdm(range(num_voxels), desc="  Fitting LS", leave=False, ncols=100):
        try:
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signals_reshaped[i], init_guess, **ls_params)
            cbf_results[i], att_results[i] = beta[0] * 6000.0, beta[1]
        except Exception:
            continue
    return cbf_results, att_results

def predict_subject(subject_dir: Path, models: List, config: Dict, norm_stats: Dict, device: torch.device, output_root: Path):
    subject_id = subject_dir.name
    subject_output_dir = output_root / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Predicting for Subject: {subject_id} ---")
    try:
        low_snr_signals = np.load(subject_dir / 'low_snr_signals.npy')
        high_snr_signals = np.load(subject_dir / 'high_snr_signals.npy')
        brain_mask_flat = np.load(subject_dir / 'brain_mask.npy').flatten()
        plds = np.load(subject_dir / 'plds.npy')
        dims = tuple(np.load(subject_dir / 'image_dims.npy'))
        affine = np.load(subject_dir / 'image_affine.npy')
        header = np.load(subject_dir / 'image_header.npy', allow_pickle=True).item()
        
        low_snr_masked = low_snr_signals[brain_mask_flat]
        high_snr_masked = high_snr_signals[brain_mask_flat]
        print(f"  --> Applied brain mask. Processing {np.sum(brain_mask_flat)} voxels.")

        def run_and_save(prediction_func, signals_data, output_filename, *args):
            start_time = time.time()
            cbf_masked, att_masked = prediction_func(signals_data, *args)
            duration = time.time() - start_time
            print(f"  --> Generated '{output_filename}' in {duration:.2f}s.")
            cbf_map, att_map = np.zeros_like(brain_mask_flat, dtype=np.float32), np.zeros_like(brain_mask_flat, dtype=np.float32)
            cbf_map[brain_mask_flat], att_map[brain_mask_flat] = cbf_masked, att_masked
            nib.save(nib.Nifti1Image(cbf_map.reshape(dims), affine, header), subject_output_dir / f'{output_filename}_cbf.nii.gz')
            nib.save(nib.Nifti1Image(att_map.reshape(dims), affine, header), subject_output_dir / f'{output_filename}_att.nii.gz')

        print("\n  [Primary Goal]: Testing NN on low-SNR (1-repeat) data.")
        run_and_save(batch_predict_nn, low_snr_masked, "nn_from_1_repeat", plds, models, config, norm_stats, device)
        print("  [Benchmark]: Running LS fit on high-SNR (4-repeat) data.")
        run_and_save(fit_ls_robust, high_snr_masked, "ls_from_4_repeats", plds, config)
        print("\n  [Additional Comparisons]:")
        run_and_save(batch_predict_nn, high_snr_masked, "nn_from_4_repeats", plds, models, config, norm_stats, device)
        run_and_save(fit_ls_robust, low_snr_masked, "ls_from_1_repeat", plds, config)

    except Exception as e:
        print(f"  [FATAL ERROR] predicting on subject {subject_id}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run NN and LS fitting on preprocessed in-vivo ASL data.")
    parser.add_argument("preprocessed_dir", type=str, help="Path to the directory with preprocessed NumPy arrays.")
    parser.add_argument("model_results_dir", type=str, help="Path to the results directory containing models, config, and norm_stats.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where final NIfTI maps will be saved.")
    args = parser.parse_args()

    preprocessed_root = Path(args.preprocessed_dir)
    model_results_root = Path(args.model_results_dir)
    output_root = Path(args.output_dir)

    print("--- Loading Model Artifacts ---")
    try:
        models, config, norm_stats = load_artifacts(model_results_root)
    except Exception as e:
        print(f"Error loading artifacts: {e}. Exiting.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device)
    print(f"Loaded {len(models)} models and config. Using device: {device}")
    
    subject_dirs = sorted([d for d in preprocessed_root.iterdir() if d.is_dir()])
    print(f"\nFound {len(subject_dirs)} preprocessed subjects to predict.")
    
    for sub_dir in subject_dirs:
        predict_subject(sub_dir, models, config, norm_stats, device, output_root)

    print("\n--- In-vivo prediction pipeline complete! ---")
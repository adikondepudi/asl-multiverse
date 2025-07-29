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

from enhanced_asl_network import EnhancedASLNet
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from utils import engineer_signal_features

def apply_normalization_vectorized(batch: np.ndarray, norm_stats: Dict, num_plds: int) -> np.ndarray:
    """Applies normalization to a batch of input signals for the NN."""
    pcasl_raw, vsasl_raw = batch[:, :num_plds], batch[:, num_plds:num_plds*2]
    features = batch[:, num_plds*2:]
    
    pcasl_norm = (pcasl_raw - norm_stats['pcasl_mean']) / (np.array(norm_stats['pcasl_std']) + 1e-6)
    vsasl_norm = (vsasl_raw - norm_stats['vsasl_mean']) / (np.array(norm_stats['vsasl_std']) + 1e-6)
    
    return np.concatenate([pcasl_norm, vsasl_norm, features], axis=1)

def denormalize_predictions(cbf_norm: np.ndarray, att_norm: np.ndarray, norm_stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Applies de-normalization to the NN's outputs."""
    cbf_denorm = cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att_denorm = att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    return cbf_denorm, att_denorm

# --- Main Prediction and Fitting Functions ---

def batch_predict_nn(signals_masked: np.ndarray, subject_plds: np.ndarray, models: List, config: Dict, norm_stats: Dict, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs batched inference using the trained NN ensemble on masked data.
    This function now pads/resamples the input signal to match the PLD structure
    the neural network was trained on.
    """
    # Resample subject signal to match model's expected PLDs for robustness
    model_plds_list = config['pld_values']
    num_model_plds = len(model_plds_list)
    num_subject_plds = len(subject_plds)
    
    # Create a zero-filled array with the shape the model expects
    resampled_signals = np.zeros((signals_masked.shape[0], num_model_plds * 2))

    # Create a mapping from subject PLD indices to model PLD indices
    target_indices, source_indices = [], []
    for i, pld in enumerate(subject_plds):
        try:
            target_idx = model_plds_list.index(int(pld)) # Ensure pld is int for list.index
            target_indices.append(target_idx)
            source_indices.append(i)
        except ValueError:
            # This subject PLD is not in the model's training PLDs; it will be ignored (left as zero).
            pass
            
    # Convert to numpy arrays for efficient indexing
    source_indices = np.array(source_indices)
    target_indices = np.array(target_indices)

    if source_indices.size > 0: # Only map data if there are common PLDs
        # Use index lists to place PCASL data correctly
        resampled_signals[:, target_indices] = signals_masked[:, source_indices]
        # Use index lists to place VSASL data correctly (offset by total PLD count)
        resampled_signals[:, target_indices + num_model_plds] = signals_masked[:, source_indices + num_subject_plds]
    
    # Continue with the original logic, but using the correctly shaped, resampled signal
    num_plds_train = num_model_plds # The number of PLDs is now fixed to the model's requirement
    eng_feats = engineer_signal_features(resampled_signals, num_plds_train)
    nn_input = np.concatenate([resampled_signals, eng_feats], axis=1)
    
    norm_input = apply_normalization_vectorized(nn_input, norm_stats, num_plds_train)
    input_tensor = torch.FloatTensor(norm_input).to(device)
    
    with torch.no_grad():
        cbf_preds_norm = [model(input_tensor)[0].cpu().numpy() for model in models]
        att_preds_norm = [model(input_tensor)[1].cpu().numpy() for model in models]
    
    cbf_norm_ensemble, att_norm_ensemble = np.mean(cbf_preds_norm, axis=0), np.mean(att_preds_norm, axis=0)
    return denormalize_predictions(cbf_norm_ensemble.squeeze(), att_norm_ensemble.squeeze(), norm_stats)

# Helper function for parallel LS fitting
def _fit_single_voxel_ls(signal_reshaped, pldti, ls_params, init_guess):
    """Helper function to fit a single voxel, designed for use with joblib."""
    try:
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signal_reshaped, init_guess, **ls_params)
        return beta[0] * 6000.0, beta[1]
    except Exception:
        return np.nan, np.nan

# Rewritten to use joblib for parallel processing 
def fit_ls_robust(signals_masked: np.ndarray, plds: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Runs a robust, parallelized LS fit for each voxel in the masked data."""
    pldti = np.column_stack([plds, plds])
    ls_params = {k:v for k,v in config.items() if k in ['T1_artery','T_tau','T2_factor','alpha_BS1','alpha_PCASL','alpha_VSASL']}
    num_voxels = signals_masked.shape[0]
    signals_reshaped = signals_masked.reshape((num_voxels, len(plds), 2), order='F')
    init_guess = [50.0 / 6000.0, 1500.0]
    
    num_cores = multiprocessing.cpu_count()
    
    results = Parallel(n_jobs=num_cores)(
        delayed(_fit_single_voxel_ls)(signals_reshaped[i], pldti, ls_params, init_guess)
        for i in tqdm(range(num_voxels), desc="  Fitting LS (Parallel)", leave=False, ncols=100)
    )
    
    results_arr = np.array(results)
    return results_arr[:, 0], results_arr[:, 1]

def predict_subject(subject_dir: Path, models: List, config: Dict, norm_stats: Dict, device: torch.device, output_root: Path):
    """Main processing function for a single subject."""
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
            cbf_map = np.zeros(brain_mask_flat.shape, dtype=np.float32)
            att_map = np.zeros(brain_mask_flat.shape, dtype=np.float32)
            cbf_map[brain_mask_flat] = cbf_masked
            att_map[brain_mask_flat] = att_masked
            cbf_map_3d = cbf_map.reshape(dims)
            att_map_3d = att_map.reshape(dims)
            nib.save(nib.Nifti1Image(cbf_map_3d, affine, header), subject_output_dir / f'{output_filename}_cbf.nii.gz')
            nib.save(nib.Nifti1Image(att_map_3d, affine, header), subject_output_dir / f'{output_filename}_att.nii.gz')

        print("\n  [Primary Goal]: Testing NN on low-SNR data.")
        run_and_save(batch_predict_nn, low_snr_masked, "nn_from_1_repeat", plds, models, config, norm_stats, device)
        print("\n  [Benchmark]: Running conventional LS fit on high-SNR data.")
        run_and_save(fit_ls_robust, high_snr_masked, "ls_from_4_repeats", plds, config)
        print("\n  [For Completeness]: Running other comparisons.")
        run_and_save(batch_predict_nn, high_snr_masked, "nn_from_4_repeats", plds, models, config, norm_stats, device)
        run_and_save(fit_ls_robust, low_snr_masked, "ls_from_1_repeat", plds, config)

    except Exception as e:
        print(f"  [FATAL ERROR] predicting on subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run NN and LS fitting on preprocessed in-vivo ASL data.")
    parser.add_argument("preprocessed_dir", type=str)
    parser.add_argument("model_results_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    preprocessed_root = Path(args.preprocessed_dir)
    model_results_root = Path(args.model_results_dir)
    output_root = Path(args.output_dir)

    print("--- Loading Model Artifacts ---")
    try:
        # Robustly load config for potentially Optuna-tuned models
        with open(model_results_root / 'research_config.json', 'r') as f:
            config = json.load(f)
        
        final_results_path = model_results_root / 'final_research_results.json'
        if final_results_path.exists():
            with open(final_results_path, 'r') as f:
                final_results = json.load(f)
            if 'optuna_best_params' in final_results and final_results['optuna_best_params']:
                print("  --> Optuna parameters found. Updating config for model instantiation.")
                best_params = final_results['optuna_best_params']
                config['hidden_sizes'] = [
                    best_params.get('hidden_size_1'), best_params.get('hidden_size_2'), best_params.get('hidden_size_3')
                ]
                config['dropout_rate'] = best_params.get('dropout_rate')

        with open(model_results_root / 'norm_stats.json', 'r') as f:
            norm_stats = json.load(f)
        
        models = []
        models_dir = model_results_root / 'trained_models'
        model_files = list(models_dir.glob('ensemble_model_*.pt'))
        
        num_plds = len(config['pld_values'])
        base_input_size = num_plds * 2 + 4

        for model_path in model_files:
            model = EnhancedASLNet(input_size=base_input_size, **config)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            models.append(model)
        
        if not models: raise FileNotFoundError("No models found in trained_models folder.")
        
    except Exception as e:
        print(f"Error loading artifacts: {e}. Make sure paths are correct and JSON files are valid.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device)
    print(f"Loaded {len(models)} models and config. Using device: {device}")
    
    subject_dirs = sorted([d for d in preprocessed_root.iterdir() if d.is_dir()])
    print(f"\nFound {len(subject_dirs)} preprocessed subjects to predict.")
    
    for sub_dir in subject_dirs:
        predict_subject(sub_dir, models, config, norm_stats, device, output_root)

    print("\n--- In-vivo prediction pipeline complete! ---")
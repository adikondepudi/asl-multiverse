# adikondepudi-asl-multiverse/run_all_evaluations.py

import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import sys
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
import torch
from typing import List, Dict, Tuple
import json
import re

# --- Import necessary functions from other project files ---
from enhanced_asl_network import EnhancedASLNet
from predict_on_invivo import batch_predict_nn # Re-use the prediction function
from prepare_invivo_data import find_and_sort_files_by_pld # Re-use the file finder

def load_nifti_data(file_path: Path) -> np.ndarray:
    """Loads NIfTI data and cleans non-finite values."""
    img = nib.load(file_path)
    return np.nan_to_num(img.get_fdata(dtype=np.float64))

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
            # Add any other optimized params that affect model architecture here

    # Load normalization stats
    with open(model_results_root / 'norm_stats.json', 'r') as f:
        norm_stats = json.load(f)
        
    # Load all models in the ensemble
    models = []
    models_dir = model_results_root / 'trained_models'
    num_plds = len(config['pld_values'])
    base_input_size = num_plds * 2 + 4 # 2 modalities, 4 engineered features
    
    for model_path in models_dir.glob('ensemble_model_*.pt'):
        # Instantiate model with the final, correct config
        model = EnhancedASLNet(input_size=base_input_size, **config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        models.append(model)
        
    if not models: raise FileNotFoundError("No models found in trained_models folder.")
    return models, config, norm_stats

def calculate_spatial_cov(cbf_map_data: np.ndarray, tissue_mask: np.ndarray) -> float:
    """Calculates the spatial coefficient of variation within a given tissue mask."""
    if np.sum(tissue_mask) < 10: return np.nan
    masked_data = cbf_map_data[tissue_mask]
    mean_val, std_val = np.nanmean(masked_data), np.nanstd(masked_data)
    return (std_val / mean_val) * 100 if mean_val > 1e-6 else np.nan

def calculate_gm_wm_ratio(cbf_map_data: np.ndarray, gm_mask: np.ndarray, wm_mask: np.ndarray) -> float:
    """Calculates the gray matter to white matter CBF ratio."""
    if np.sum(gm_mask) < 10 or np.sum(wm_mask) < 10: return np.nan
    mean_gm, mean_wm = np.nanmean(cbf_map_data[gm_mask]), np.nanmean(cbf_map_data[wm_mask])
    return mean_gm / mean_wm if mean_wm > 1e-6 else np.nan

def calculate_fit_success_rate(cbf_map_data: np.ndarray, brain_mask: np.ndarray) -> float:
    """Calculates the percentage of voxels within the brain mask that have a valid fit."""
    total_voxels = np.sum(brain_mask)
    if total_voxels == 0: return 0.0
    valid_fits = np.sum(~np.isnan(cbf_map_data[brain_mask]))
    return (valid_fits / total_voxels) * 100

def calculate_nn_test_retest_cov(
    raw_subject_dir: Path, brain_mask: np.ndarray, gm_mask: np.ndarray,
    models: List, config: Dict, norm_stats: Dict, device: torch.device
) -> float:
    """
    Calculates the test-retest CoV for the NN by predicting on each of the 4 repeats
    and measuring the voxel-wise variation.
    """
    pcasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_PCASL_*.nii*'])
    vsasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_VSASL_*.nii*'])
    plds = np.array([int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files])

    cbf_maps_from_repeats = []
    for i in range(4): # Loop over the 4 repeats
        pcasl_repeat_i = np.stack([load_nifti_data(f)[..., i] for f in pcasl_files], axis=-1)
        vsasl_repeat_i = np.stack([load_nifti_data(f)[..., i] for f in vsasl_files], axis=-1)
        
        signal_i_flat = np.concatenate([pcasl_repeat_i.reshape(-1, len(plds)), 
                                        vsasl_repeat_i.reshape(-1, len(plds))], axis=1)
        
        cbf_masked, _ = batch_predict_nn(signal_i_flat[brain_mask.flatten()], plds, models, config, norm_stats, device)
        
        cbf_map = np.full(brain_mask.shape, np.nan, dtype=np.float32)
        cbf_map[brain_mask] = cbf_masked
        cbf_maps_from_repeats.append(cbf_map)
        
    stacked_maps = np.stack(cbf_maps_from_repeats, axis=-1)
    mean_cbf = np.nanmean(stacked_maps, axis=-1)
    std_cbf = np.nanstd(stacked_maps, axis=-1)
    
    cov_map = np.divide(std_cbf, mean_cbf, out=np.full_like(mean_cbf, np.nan), where=mean_cbf > 1e-6)
    
    return np.nanmean(cov_map[gm_mask]) * 100 # Return as percentage

def evaluate_single_subject(
    maps_dir: Path, preprocessed_dir: Path, raw_validated_dir: Path,
    models: List, config: Dict, norm_stats: Dict, device: torch.device
) -> dict:
    subject_id = maps_dir.name
    results = {"subject_id": subject_id}
    try:
        # Load preprocessed masks
        brain_mask = np.load(preprocessed_dir / subject_id / 'brain_mask.npy')
        gm_mask = np.load(preprocessed_dir / subject_id / 'gm_mask.npy')
        wm_mask = np.load(preprocessed_dir / subject_id / 'wm_mask.npy')

        # Load final CBF maps
        map_files = {
            "nn_1r": maps_dir / 'nn_from_1_repeat_cbf.nii.gz',
            "ls_1r": maps_dir / 'ls_from_1_repeat_cbf.nii.gz',
            "ls_4r": maps_dir / 'ls_from_4_repeats_cbf.nii.gz'
        }
        maps_data = {key: load_nifti_data(path) for key, path in map_files.items()}

        # --- Calculate Metrics ---
        # Spatial CoV in GM (your original metric)
        results["nn_1r_spatial_cov_gm"] = calculate_spatial_cov(maps_data["nn_1r"], gm_mask)
        results["ls_1r_spatial_cov_gm"] = calculate_spatial_cov(maps_data["ls_1r"], gm_mask)
        results["ls_4r_spatial_cov_gm"] = calculate_spatial_cov(maps_data["ls_4r"], gm_mask)

        # GM/WM Ratio
        results["nn_1r_gm_wm_ratio"] = calculate_gm_wm_ratio(maps_data["nn_1r"], gm_mask, wm_mask)
        results["ls_1r_gm_wm_ratio"] = calculate_gm_wm_ratio(maps_data["ls_1r"], gm_mask, wm_mask)
        results["ls_4r_gm_wm_ratio"] = calculate_gm_wm_ratio(maps_data["ls_4r"], gm_mask, wm_mask)

        # Fit Success Rate
        results["nn_1r_fit_success_rate"] = calculate_fit_success_rate(maps_data["nn_1r"], brain_mask)
        results["ls_1r_fit_success_rate"] = calculate_fit_success_rate(maps_data["ls_1r"], brain_mask)
        results["ls_4r_fit_success_rate"] = calculate_fit_success_rate(maps_data["ls_4r"], brain_mask)

        # NEW: Test-Retest CoV for NN
        results["nn_test_retest_cov_gm"] = calculate_nn_test_retest_cov(
            raw_validated_dir / subject_id, brain_mask, gm_mask, models, config, norm_stats, device
        )
        
        # Correlation vs Benchmark (LS 4-repeat)
        corr, _ = pearsonr(maps_data["nn_1r"][brain_mask], maps_data["ls_4r"][brain_mask])
        results["correlation_vs_benchmark"] = corr

    except Exception as e:
        results["error"] = str(e)
    return results

def main():
    parser = argparse.ArgumentParser(description="Run a comprehensive quantitative evaluation on the final generated maps.")
    parser.add_argument("final_maps_dir", type=str)
    parser.add_argument("preprocessed_dir", type=str)
    parser.add_argument("raw_validated_dir", type=str)
    parser.add_argument("model_results_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    # Load model artifacts once
    print("--- Loading Model Artifacts for Evaluation ---")
    try:
        models, config, norm_stats = load_artifacts(Path(args.model_results_dir))
    except Exception as e:
        print(f"Error loading artifacts: {e}. Exiting.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device)
    
    subject_dirs = sorted([d for d in Path(args.final_maps_dir).iterdir() if d.is_dir()])
    all_results = [
        evaluate_single_subject(
            s_dir, Path(args.preprocessed_dir), Path(args.raw_validated_dir),
            models, config, norm_stats, device
        ) for s_dir in tqdm(subject_dirs, desc="Evaluating all subjects")
    ]
    
    df = pd.DataFrame(all_results)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "comprehensive_invivo_evaluation_summary.csv"
    df.to_csv(summary_path, index=False, float_format='%.4f')

    print("\n" + "="*80)
    print(" " * 20 + "COMPREHENSIVE IN-VIVO EVALUATION SUMMARY")
    print("="*80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df.to_string(float_format="%.2f"))
    print(f"\nSummary report saved to: {summary_path}")
    print("="*80)

if __name__ == '__main__':
    main()
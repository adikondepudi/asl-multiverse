# adikondepudi-asl-multiverse/run_all_evaluations.py

import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import sys
import pandas as pd
from tqdm import tqdm
import torch
from typing import List, Dict, Tuple
import json
import re

# --- Import necessary functions from other project files ---
from enhanced_asl_network import EnhancedASLNet
from predict_on_invivo import batch_predict_nn, fit_ls_robust
from prepare_invivo_data import find_and_sort_files_by_pld

def load_nifti_data(file_path: Path) -> np.ndarray:
    """Loads NIfTI data and cleans non-finite values."""
    img = nib.load(file_path)
    return np.nan_to_num(img.get_fdata(dtype=np.float64))

def load_artifacts(model_results_root: Path) -> Tuple[List[EnhancedASLNet], Dict, Dict]:
    """Robustly loads model ensemble, final config, and norm stats."""
    with open(model_results_root / 'research_config.json', 'r') as f: config = json.load(f)
    final_results_path = model_results_root / 'final_research_results.json'
    if final_results_path.exists():
        with open(final_results_path, 'r') as f: final_results = json.load(f)
        if 'optuna_best_params' in final_results and final_results['optuna_best_params']:
            best_params = final_results['optuna_best_params']
            config['hidden_sizes'] = [best_params.get('hidden_size_1'), best_params.get('hidden_size_2'), best_params.get('hidden_size_3')]
            config['dropout_rate'] = best_params.get('dropout_rate')
    with open(model_results_root / 'norm_stats.json', 'r') as f: norm_stats = json.load(f)
    models, models_dir = [], model_results_root / 'trained_models'
    num_plds = len(config['pld_values'])
    base_input_size = num_plds * 2 + 4
    for model_path in models_dir.glob('ensemble_model_*.pt'):
        model = EnhancedASLNet(input_size=base_input_size, **config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        models.append(model)
    if not models: raise FileNotFoundError("No models found.")
    return models, config, norm_stats

def calculate_test_retest_cov(
    raw_subject_dir: Path, brain_mask: np.ndarray, gm_mask: np.ndarray, config: Dict,
    method: str, models: List = None, norm_stats: Dict = None, device: torch.device = None
) -> Tuple[float, float]:
    """
    Calculates the test-retest CoV for a given method (NN or LS) by processing each of the 4 repeats.
    """
    pcasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_PCASL_*.nii*'])
    vsasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_VSASL_*.nii*'])
    plds = np.array([int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files])

    cbf_maps_from_repeats, att_maps_from_repeats = [], []
    for i in range(4): # Loop over the 4 repeats
        pcasl_repeat_i = np.stack([load_nifti_data(f)[..., i] for f in pcasl_files], axis=-1)
        vsasl_repeat_i = np.stack([load_nifti_data(f)[..., i] for f in vsasl_files], axis=-1)
        signal_i_flat = np.concatenate([pcasl_repeat_i.reshape(-1, len(plds)), vsasl_repeat_i.reshape(-1, len(plds))], axis=1)
        
        cbf_masked, att_masked = np.nan, np.nan
        if method == 'nn':
            cbf_masked, att_masked = batch_predict_nn(signal_i_flat[brain_mask.flatten()], plds, models, config, norm_stats, device)
        elif method == 'ls':
            cbf_masked, att_masked = fit_ls_robust(signal_i_flat[brain_mask.flatten()], plds, config)
        
        cbf_map = np.full(brain_mask.shape, np.nan, dtype=np.float32)
        att_map = np.full(brain_mask.shape, np.nan, dtype=np.float32)
        cbf_map[brain_mask] = cbf_masked
        att_map[brain_mask] = att_masked
        cbf_maps_from_repeats.append(cbf_map)
        att_maps_from_repeats.append(att_map)
        
    stacked_cbf, stacked_att = np.stack(cbf_maps_from_repeats, axis=-1), np.stack(att_maps_from_repeats, axis=-1)
    
    mean_cbf, std_cbf = np.nanmean(stacked_cbf, axis=-1), np.nanstd(stacked_cbf, axis=-1)
    cov_map_cbf = np.divide(std_cbf, mean_cbf, out=np.full_like(mean_cbf, np.nan), where=mean_cbf > 1e-6)
    
    mean_att, std_att = np.nanmean(stacked_att, axis=-1), np.nanstd(stacked_att, axis=-1)
    cov_map_att = np.divide(std_att, mean_att, out=np.full_like(mean_att, np.nan), where=mean_att > 1e-6)
    
    return np.nanmean(cov_map_cbf[gm_mask]), np.nanmean(cov_map_att[gm_mask])

def evaluate_single_subject(
    maps_dir: Path, preprocessed_dir: Path, raw_validated_dir: Path,
    models: List, config: Dict, norm_stats: Dict, device: torch.device
) -> dict:
    subject_id = maps_dir.name
    results = {"subject_id": subject_id}
    try:
        brain_mask = np.load(preprocessed_dir / subject_id / 'brain_mask.npy')
        gm_mask = np.load(preprocessed_dir / subject_id / 'gm_mask.npy')

        map_files = {
            "ls_4r_cbf": maps_dir / 'ls_from_4_repeats_cbf.nii.gz', "ls_4r_att": maps_dir / 'ls_from_4_repeats_att.nii.gz',
            "ls_1r_cbf": maps_dir / 'ls_from_1_repeat_cbf.nii.gz', "ls_1r_att": maps_dir / 'ls_from_1_repeat_att.nii.gz',
            "nn_4r_cbf": maps_dir / 'nn_from_4_repeats_cbf.nii.gz', "nn_4r_att": maps_dir / 'nn_from_4_repeats_att.nii.gz',
            "nn_1r_cbf": maps_dir / 'nn_from_1_repeat_cbf.nii.gz', "nn_1r_att": maps_dir / 'nn_from_1_repeat_att.nii.gz'
        }
        maps_data = {key: load_nifti_data(path) for key, path in map_files.items()}

        cbf_thresh_mask_ls = (maps_data["ls_1r_cbf"] > 0) & (maps_data["ls_1r_cbf"] < 100)
        cbf_thresh_mask_nn = (maps_data["nn_1r_cbf"] > 0) & (maps_data["nn_1r_cbf"] < 100)
        common_gm_mask_cbf = gm_mask & cbf_thresh_mask_ls & cbf_thresh_mask_nn

        att_thresh_mask_ls = (maps_data["ls_1r_att"] > 0) & (maps_data["ls_1r_att"] < 4000)
        att_thresh_mask_nn = (maps_data["nn_1r_att"] > 0) & (maps_data["nn_1r_att"] < 4000)
        common_gm_mask_att = gm_mask & att_thresh_mask_ls & att_thresh_mask_nn

        # --- FIXED: Use underscores in key names consistently ---
        for key in ["ls_4r", "ls_1r", "nn_4r", "nn_1r"]:
            results[f"GM_CBF_{key}"] = np.nanmean(maps_data[f"{key}_cbf"][common_gm_mask_cbf])
            results[f"GM_ATT_{key}"] = np.nanmean(maps_data[f"{key}_att"][common_gm_mask_att])
        
        results["CBF_GM_CoV_ls_1_repeat"] = np.nanstd(maps_data["ls_1r_cbf"][common_gm_mask_cbf]) / results["GM_CBF_ls_1r"]
        results["CBF_GM_CoV_nn_1_repeat"] = np.nanstd(maps_data["nn_1r_cbf"][common_gm_mask_cbf]) / results["GM_CBF_nn_1r"]
        results["ATT_GM_CoV_ls_1_repeat"] = np.nanstd(maps_data["ls_1r_att"][common_gm_mask_att]) / results["GM_ATT_ls_1r"]
        results["ATT_GM_CoV_nn_1_repeat"] = np.nanstd(maps_data["nn_1r_att"][common_gm_mask_att]) / results["GM_ATT_nn_1r"]
        
        tqdm.write(f"  Calculating Test-Retest CoV for {subject_id}...")
        results["GM_CBF_test_retest_CoV_nn_1_repeat"], results["GM_ATT_test_retest_CoV_nn_1_repeat"] = calculate_test_retest_cov(
            raw_validated_dir / subject_id, brain_mask, gm_mask, config, 'nn', models, norm_stats, device
        )
        results["GM_CBF_test_retest_CoV_ls_1_repeat"], results["GM_ATT_test_retest_CoV_ls_1_repeat"] = calculate_test_retest_cov(
            raw_validated_dir / subject_id, brain_mask, gm_mask, config, 'ls'
        )
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

    # --- FIXED: Use underscores in column lists and select directly ---
    cbf_cols = [
        "subject_id", "GM_CBF_ls_4r", "GM_CBF_ls_1r", "GM_CBF_nn_4r", "GM_CBF_nn_1r",
        "GM_CBF_test_retest_CoV_ls_1_repeat", "GM_CBF_test_retest_CoV_nn_1_repeat",
        "CBF_GM_CoV_ls_1_repeat", "CBF_GM_CoV_nn_1_repeat"
    ]
    att_cols = [
        "subject_id", "GM_ATT_ls_4r", "GM_ATT_ls_1r", "GM_ATT_nn_4r", "GM_ATT_nn_1r",
        "GM_ATT_test_retest_CoV_ls_1_repeat", "GM_ATT_test_retest_CoV_nn_1_repeat",
        "ATT_GM_CoV_ls_1_repeat", "ATT_GM_CoV_nn_1_repeat"
    ]

    df_cbf = df[[col for col in cbf_cols if col in df.columns]]
    df_att = df[[col for col in att_cols if col in df.columns]]
    
    cbf_summary_path = output_path / "cbf_evaluation_summary.csv"
    att_summary_path = output_path / "att_evaluation_summary.csv"
    df_cbf.to_csv(cbf_summary_path, index=False, float_format='%.4f')
    df_att.to_csv(att_summary_path, index=False, float_format='%.4f')

    print("\n" + "="*80)
    print(" " * 28 + "CBF EVALUATION SUMMARY")
    print("="*80)
    pd.set_option('display.max_rows', None); pd.set_option('display.max_columns', None); pd.set_option('display.width', 200)
    print(df_cbf.to_string(index=False, float_format="%.2f"))
    print(f"\nCBF summary saved to: {cbf_summary_path}")
    
    print("\n" + "="*80)
    print(" " * 28 + "ATT EVALUATION SUMMARY")
    print("="*80)
    print(df_att.to_string(index=False, float_format="%.2f"))
    print(f"\nATT summary saved to: {att_summary_path}")
    print("="*80)

if __name__ == '__main__':
    main()
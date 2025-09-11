# run_all_evaluations.py
#
# A scientifically rigorous script for the quantitative evaluation of in-vivo results.
# This script abandons flawed "accuracy" comparisons in the absence of ground truth,
# and instead focuses on three objective, defensible metrics:
#   1. ROBUSTNESS: How well does each method handle noisy, single-repeat data?
#      - Metrics: Fit failure rate, physiologically implausible voxels.
#   2. PRECISION: How repeatable is each method?
#      - Metric: Voxelwise test-retest Coefficient of Variation (CoV) across 4 repeats.
#   3. CONCORDANCE: How well do the methods agree under high-SNR conditions?
#      - Metrics: Pearson correlation and Bland-Altman analysis.

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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- Import necessary functions from other project files ---
from enhanced_asl_network import EnhancedASLNet
from predict_on_invivo import batch_predict_nn, fit_ls_robust
from prepare_invivo_data import find_and_sort_files_by_pld

# --- Helper Functions ---

def load_nifti_data(file_path: Path) -> np.ndarray:
    """Loads NIfTI data and cleans non-finite values."""
    if not file_path.exists():
        print(f"Warning: File not found {file_path}", file=sys.stderr)
        return None
    img = nib.load(file_path)
    return np.nan_to_num(img.get_fdata(dtype=np.float64))

def load_artifacts(model_results_root: Path) -> Tuple[List[EnhancedASLNet], Dict, Dict]:
    """Robustly loads model ensemble, final config, and norm stats."""
    # This function remains the same as your original
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

# --- Core Evaluation Functions ---

def evaluate_robustness(ls_1r_map: np.ndarray, nn_1r_map: np.ndarray, brain_mask: np.ndarray, param: str) -> dict:
    """Analyzes fit failures and physiological plausibility on single-repeat data."""
    if ls_1r_map is None or nn_1r_map is None or brain_mask is None:
        return {}
    
    total_voxels = np.sum(brain_mask)
    if total_voxels == 0: return {}
    
    # Analyze LS fit failures (NN never fails)
    ls_masked = np.where(brain_mask, ls_1r_map, np.nan)
    # The original LS fit function returns NaN on failure. We count these.
    # Note: `fit_ls_robust` should already produce NaNs, but we check again.
    failed_fits = np.isnan(ls_masked).sum()
    
    # Analyze physiological plausibility
    plausibility_range = (0, 200) if param == 'cbf' else (0, 6000)
    
    ls_implausible = np.sum((ls_masked < plausibility_range[0]) | (ls_masked > plausibility_range[1]))
    
    nn_masked = nn_1r_map[brain_mask]
    nn_implausible = np.sum((nn_masked < plausibility_range[0]) | (nn_masked > plausibility_range[1]))
    
    return {
        f'{param}_ls_fit_failure_perc': (failed_fits / total_voxels) * 100,
        f'{param}_ls_implausible_perc': (ls_implausible / total_voxels) * 100,
        f'{param}_nn_implausible_perc': (nn_implausible / total_voxels) * 100
    }

def calculate_test_retest_cov(raw_subject_dir: Path, brain_mask: np.ndarray, gm_mask: np.ndarray, config: Dict,
                                method: str, models: List = None, norm_stats: Dict = None, device: torch.device = None) -> Tuple[float, float]:
    """Calculates the test-retest CoV by processing each of the 4 repeats."""
    # This function is correct and scientifically sound, so we keep it.
    pcasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_PCASL_*.nii*'])
    vsasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_VSASL_*.nii*'])
    if not pcasl_files or not vsasl_files: return np.nan, np.nan
    plds = np.array([int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files])

    cbf_maps_from_repeats, att_maps_from_repeats = [], []
    for i in range(4): # Loop over the 4 repeats
        pcasl_repeat_i = np.stack([load_nifti_data(f)[..., i] for f in pcasl_files], axis=-1)
        vsasl_repeat_i = np.stack([load_nifti_data(f)[..., i] for f in vsasl_files], axis=-1)
        signal_i_flat = np.concatenate([pcasl_repeat_i.reshape(-1, len(plds)), vsasl_repeat_i.reshape(-1, len(plds))], axis=1)
        
        cbf_masked, att_masked = np.array([np.nan]), np.array([np.nan])
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

def evaluate_concordance(ls_4r_map: np.ndarray, nn_4r_map: np.ndarray, gm_mask: np.ndarray, param: str, subject_id: str, output_dir: Path) -> dict:
    """Performs correlation and Bland-Altman analysis on high-SNR maps."""
    if ls_4r_map is None or nn_4r_map is None or gm_mask is None:
        return {}
    
    # Apply GM mask and flatten, removing voxels that are NaN in either map
    ls_vals = ls_4r_map[gm_mask]
    nn_vals = nn_4r_map[gm_mask]
    valid_mask = ~np.isnan(ls_vals) & ~np.isnan(nn_vals)
    ls_vals, nn_vals = ls_vals[valid_mask], nn_vals[valid_mask]
    
    if len(ls_vals) < 2: return {}

    # Pearson Correlation
    corr, _ = pearsonr(ls_vals, nn_vals)
    
    # Bland-Altman Analysis
    diff = nn_vals - ls_vals
    avg = (nn_vals + ls_vals) / 2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff
    
    # Generate and save plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(x=ls_vals, y=nn_vals, ax=axes[0], scatter_kws={'alpha':0.2, 's':5}, line_kws={'color':'red'})
    axes[0].set_title(f'{param.upper()} Concordance (r={corr:.3f})')
    axes[0].set_xlabel(f'LS 4-Repeat {param.upper()}')
    axes[0].set_ylabel(f'NN 4-Repeat {param.upper()}')
    
    sns.scatterplot(x=avg, y=diff, ax=axes[1], alpha=0.2, s=5)
    axes[1].axhline(mean_diff, color='red', linestyle='-', label=f'Bias: {mean_diff:.2f}')
    axes[1].axhline(loa_upper, color='red', linestyle='--', label=f'95% LoA: {loa_upper:.2f}')
    axes[1].axhline(loa_lower, color='red', linestyle='--')
    axes[1].set_title('Bland-Altman Plot')
    axes[1].set_xlabel(f'Average of Methods ({param.upper()})')
    axes[1].set_ylabel(f'Difference (NN - LS) ({param.upper()})')
    axes[1].legend()
    
    plt.suptitle(f'Concordance Analysis for Subject {subject_id}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = output_dir / f'plots/{subject_id}_{param}_concordance.png'
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path)
    plt.close(fig)
    
    return {
        f'{param}_concordance_pearson_r': corr,
        f'{param}_concordance_bias': mean_diff,
        f'{param}_concordance_loa_upper': loa_upper,
        f'{param}_concordance_loa_lower': loa_lower
    }

def main():
    parser = argparse.ArgumentParser(description="Run a comprehensive quantitative evaluation on the final generated maps.")
    parser.add_argument("final_maps_dir", type=str)
    parser.add_argument("preprocessed_dir", type=str)
    parser.add_argument("raw_validated_dir", type=str)
    parser.add_argument("model_results_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("--- Loading Model Artifacts for Evaluation ---")
    models, config, norm_stats = load_artifacts(Path(args.model_results_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device)
    
    subject_dirs = sorted([d for d in Path(args.final_maps_dir).iterdir() if d.is_dir()])
    all_subject_results = []
    
    for subject_dir in tqdm(subject_dirs, desc="Evaluating all subjects"):
        subject_id = subject_dir.name
        results = {"subject_id": subject_id}
        
        try:
            # Load necessary data
            brain_mask = np.load(Path(args.preprocessed_dir) / subject_id / 'brain_mask.npy')
            gm_mask = np.load(Path(args.preprocessed_dir) / subject_id / 'gm_mask.npy')
            raw_subject_dir = Path(args.raw_validated_dir) / subject_id

            map_files = {
                "ls_4r_cbf": subject_dir / 'ls_from_4_repeats_cbf.nii.gz', "ls_4r_att": subject_dir / 'ls_from_4_repeats_att.nii.gz',
                "ls_1r_cbf": subject_dir / 'ls_from_1_repeat_cbf.nii.gz', "ls_1r_att": subject_dir / 'ls_from_1_repeat_att.nii.gz',
                "nn_4r_cbf": subject_dir / 'nn_from_4_repeats_cbf.nii.gz', "nn_4r_att": subject_dir / 'nn_from_4_repeats_att.nii.gz',
                "nn_1r_cbf": subject_dir / 'nn_from_1_repeat_cbf.nii.gz', "nn_1r_att": subject_dir / 'nn_from_1_repeat_att.nii.gz'
            }
            maps = {key: load_nifti_data(path) for key, path in map_files.items()}
            
            # --- 1. Robustness Evaluation ---
            results.update(evaluate_robustness(maps["ls_1r_cbf"], maps["nn_1r_cbf"], brain_mask, 'cbf'))
            results.update(evaluate_robustness(maps["ls_1r_att"], maps["nn_1r_att"], brain_mask, 'att'))

            # --- 2. Precision Evaluation ---
            tqdm.write(f"  Calculating Test-Retest Precision CoV for {subject_id}...")
            nn_cbf_cov, nn_att_cov = calculate_test_retest_cov(raw_subject_dir, brain_mask, gm_mask, config, 'nn', models, norm_stats, device)
            ls_cbf_cov, ls_att_cov = calculate_test_retest_cov(raw_subject_dir, brain_mask, gm_mask, config, 'ls')
            results.update({'cbf_precision_test_retest_cov_nn': nn_cbf_cov, 'att_precision_test_retest_cov_nn': nn_att_cov,
                            'cbf_precision_test_retest_cov_ls': ls_cbf_cov, 'att_precision_test_retest_cov_ls': ls_att_cov})
            
            # --- 3. Concordance Evaluation ---
            results.update(evaluate_concordance(maps["ls_4r_cbf"], maps["nn_4r_cbf"], gm_mask, 'cbf', subject_id, output_path))
            results.update(evaluate_concordance(maps["ls_4r_att"], maps["nn_4r_att"], gm_mask, 'att', subject_id, output_path))
            
            all_subject_results.append(results)

        except Exception as e:
            tqdm.write(f"  [ERROR] Failed processing subject {subject_id}: {e}")
            all_subject_results.append({"subject_id": subject_id, "error": str(e)})

    # --- Final Reporting ---
    df = pd.DataFrame(all_subject_results).set_index('subject_id')
    pd.set_option('display.max_rows', None); pd.set_option('display.max_columns', None); pd.set_option('display.width', 200)

    # Robustness Summary
    robustness_cols = [c for c in df.columns if 'failure' in c or 'implausible' in c]
    df_robustness = df[robustness_cols]
    print("\n" + "="*80)
    print(" " * 28 + "ROBUSTNESS SUMMARY (1-Repeat Data)")
    print("="*80)
    print(df_robustness.mean().to_frame(name='Mean Percentage (%)').to_string(float_format="%.2f"))
    df_robustness.to_csv(output_path / "robustness_summary.csv")

    # Precision Summary
    precision_cols = [c for c in df.columns if 'precision' in c]
    df_precision = df[precision_cols]
    print("\n" + "="*80)
    print(" " * 28 + "PRECISION SUMMARY (Test-Retest CoV)")
    print("="*80)
    print(df_precision.mean().to_frame(name='Mean CoV').to_string(float_format="%.4f"))
    df_precision.to_csv(output_path / "precision_summary.csv")

    # Concordance Summary
    concordance_cols = [c for c in df.columns if 'concordance' in c]
    df_concordance = df[concordance_cols]
    print("\n" + "="*80)
    print(" " * 28 + "CONCORDANCE SUMMARY (High-SNR 4-Repeat Data)")
    print("="*80)
    print(df_concordance.mean().to_frame(name='Mean Value').to_string(float_format="%.3f"))
    df_concordance.to_csv(output_path / "concordance_summary.csv")
    
    print(f"\nAll summary tables saved to: {output_path.resolve()}")
    print(f"Concordance plots saved to: {output_path.resolve()}/plots/")
    print("\n--- In-vivo evaluation pipeline complete! ---")

if __name__ == '__main__':
    main()
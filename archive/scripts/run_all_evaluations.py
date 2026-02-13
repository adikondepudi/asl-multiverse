# run_all_evaluations.py
# FINAL CORRECTED VERSION

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
import warnings

# --- Import necessary functions from other project files ---
from enhanced_asl_network import DisentangledASLNet
from predict_on_invivo import batch_predict_nn, fit_ls_robust
from prepare_invivo_data import find_and_sort_files_by_pld
from feature_registry import FeatureRegistry, validate_norm_stats

# --- Helper Functions ---

def load_nifti_data(file_path: Path) -> np.ndarray:
    """Loads NIfTI data and cleans non-finite values."""
    if not file_path.exists():
        print(f"Warning: File not found {file_path}", file=sys.stderr)
        return None
    try:
        img = nib.load(file_path)
        return np.nan_to_num(img.get_fdata(dtype=np.float64))
    except Exception as e:
        print(f"Warning: Could not load {file_path}. Error: {e}", file=sys.stderr)
        return None

def load_artifacts_for_eval(model_root: Path) -> tuple:
    """
    Loads artifacts and determines model type for evaluation scripts.
    Includes robust feature dimension detection.
    """
    print(f"--> Loading artifacts from: {model_root}")
    with open(model_root / 'research_config.json', 'r') as f: config = json.load(f)
    with open(model_root / 'norm_stats.json', 'r') as f: norm_stats = json.load(f)
    
    models, models_dir = [], model_root / 'trained_models'
    num_plds = len(config['pld_values'])
    
    is_disentangled = 'Disentangled' in config.get('model_class_name', '')
    model_class = DisentangledASLNet if is_disentangled else EnhancedASLNet
    
    # Input size is dynamically determined from checkpoint - scalars are auto-detected
    
    # --- ROBUST LOADING LOGIC (Copied from predict_on_invivo.py) ---
    sample_checkpoint = list(models_dir.glob('ensemble_model_*.pt'))[0]
    state_dict = torch.load(sample_checkpoint, map_location='cpu')
    
    expected_scalars = 0
    if 'encoder.pcasl_film.generator.0.weight' in state_dict:
        expected_scalars = state_dict['encoder.pcasl_film.generator.0.weight'].shape[1]
    else:
        # Fallback
        expected_scalars = len(norm_stats['scalar_features_mean']) + 1
        
    print(f"  --> Detected model expectation: {expected_scalars} scalar features.")

    for model_path in models_dir.glob('ensemble_model_*.pt'):
        # Calculate input_size dynamically from detected scalars
        base_input_size = num_plds * 2 + expected_scalars
        model = model_class(mode='regression', input_size=base_input_size, num_scalar_features=expected_scalars, **config)
        # strict=False is important for potential backward compatibility experiments
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        except RuntimeError as e:
             print(f"[WARN] Partial load for {model_path.name}: {e}")
             
        model.eval()
        models.append(model)
        
    if not models: raise FileNotFoundError(f"No models found in {models_dir}")
    return models, config, norm_stats, is_disentangled, expected_scalars

def analyze_computational_performance(final_maps_dir: Path) -> pd.DataFrame:
    """Analyzes the timing information from prediction runs."""
    timings = []
    for subject_dir in final_maps_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        timing_file = subject_dir / "timings.json"
        if timing_file.exists():
            with open(timing_file, 'r') as f:
                data = json.load(f)
            data['subject_id'] = subject_dir.name
            timings.append(data)
    if not timings:
        print("Warning: No timing.json files found to analyze computational performance.", file=sys.stderr)
        return pd.DataFrame()
    df = pd.DataFrame(timings).set_index('subject_id')
    if 'nn_total_s' in df.columns and 'ls_total_s' in df.columns:
        df['speedup_factor'] = df['ls_total_s'] / df['nn_total_s'].replace(0, 1e-6)
    return df

# --- Core Evaluation Functions ---

def evaluate_robustness(ls_1r_map: np.ndarray, nn_1r_map: np.ndarray, brain_mask: np.ndarray, param: str) -> dict:
    """Analyzes fit failures and physiological plausibility on single-repeat data."""
    if ls_1r_map is None or nn_1r_map is None or brain_mask is None: return {}
    
    total_voxels = np.sum(brain_mask)
    if total_voxels == 0: return {}
    
    ls_masked = ls_1r_map[brain_mask]
    failed_fits = np.isnan(ls_masked).sum()
    
    plausibility_range = (0, 200) if param == 'cbf' else (0, 6000)
    ls_implausible = np.sum((ls_masked < plausibility_range[0]) | (ls_masked > plausibility_range[1]))
    
    nn_masked = nn_1r_map[brain_mask]
    nn_implausible = np.sum((nn_masked < plausibility_range[0]) | (nn_masked > plausibility_range[1]))
    
    return {
        f'{param}_ls_fit_failure_perc': (failed_fits / total_voxels) * 100,
        f'{param}_ls_implausible_perc': (ls_implausible / total_voxels) * 100,
        f'{param}_nn_implausible_perc': (nn_implausible / total_voxels) * 100,
    }

def calculate_test_retest_cov(raw_subject_dir: Path, brain_mask: np.ndarray, gm_mask: np.ndarray, config: Dict,
                                method: str, is_disentangled: bool, models: List = None, norm_stats: Dict = None, device: torch.device = None, expected_scalars: int = 0) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Calculates the test-retest CoV by processing each available repeat dynamically."""
    pcasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_PCASL_*.nii*'])
    vsasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_VSASL_*.nii*'])
    if not pcasl_files or not vsasl_files: return np.nan, np.nan, None, None
    plds = np.array([int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files])

    # --- ROBUSTNESS FIX: Pre-load data and determine number of repeats dynamically ---
    pcasl_data_all_repeats = [load_nifti_data(f) for f in pcasl_files]
    vsasl_data_all_repeats = [load_nifti_data(f) for f in vsasl_files]

    if any(d is None for d in pcasl_data_all_repeats) or any(d is None for d in vsasl_data_all_repeats):
        tqdm.write(f"  [Warning] Could not load all NIfTI files for test-retest. Skipping.")
        return np.nan, np.nan, None, None

    try:
        num_repeats = min(d.shape[-1] for d in pcasl_data_all_repeats + vsasl_data_all_repeats)
        if num_repeats < 2:
            tqdm.write(f"  [Warning] Subject has < 2 repeats ({num_repeats}). Cannot calculate test-retest. Skipping.")
            return np.nan, np.nan, None, None
    except (IndexError, ValueError):
        tqdm.write(f"  [Warning] Could not determine number of repeats. Skipping test-retest.")
        return np.nan, np.nan, None, None
    # --- END FIX ---

    cbf_maps, att_maps = [], []
    for i in range(num_repeats): # Loop over dynamically found number of repeats
        # Use the pre-loaded data
        pcasl_repeat_i = np.stack([d[..., i] for d in pcasl_data_all_repeats], axis=-1)
        vsasl_repeat_i = np.stack([d[..., i] for d in vsasl_data_all_repeats], axis=-1)
        signal_i_flat = np.concatenate([pcasl_repeat_i.reshape(-1, len(plds)), vsasl_repeat_i.reshape(-1, len(plds))], axis=1)
        
        cbf_masked, att_masked = np.array([np.nan]), np.array([np.nan])
        if method == 'nn':
            results = batch_predict_nn(signal_i_flat[brain_mask.flatten()], plds, models, config, norm_stats, device, is_disentangled, expected_scalars)
            cbf_masked, att_masked = results[0], results[1]
        elif method == 'ls':
            cbf_masked, att_masked = fit_ls_robust(signal_i_flat[brain_mask.flatten()], plds, config)
        
        cbf_map, att_map = np.full(brain_mask.shape, np.nan), np.full(brain_mask.shape, np.nan)
        cbf_map[brain_mask], att_map[brain_mask] = cbf_masked, att_masked
        cbf_maps.append(cbf_map); att_maps.append(att_map)
        
    if len(cbf_maps) < 2:
        return np.nan, np.nan, None, None
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stacked_cbf, stacked_att = np.stack(cbf_maps, axis=-1), np.stack(att_maps, axis=-1)
        
        mean_cbf, std_cbf = np.nanmean(stacked_cbf, axis=-1), np.nanstd(stacked_cbf, axis=-1)
        cov_map_cbf = np.divide(std_cbf, mean_cbf, out=np.full_like(mean_cbf, np.nan), where=mean_cbf > 1e-6)
        
        mean_att, std_att = np.nanmean(stacked_att, axis=-1), np.nanstd(stacked_att, axis=-1)
        cov_map_att = np.divide(std_att, mean_att, out=np.full_like(mean_att, np.nan), where=mean_att > 1e-6)
        
        return np.nanmean(cov_map_cbf[gm_mask]), np.nanmean(cov_map_att[gm_mask]), std_cbf, std_att

def evaluate_concordance(ls_map: np.ndarray, nn_map: np.ndarray, gm_mask: np.ndarray, param: str, subject_id: str, output_dir: Path) -> dict:
    if ls_map is None or nn_map is None or gm_mask is None: return {}
    ls_vals, nn_vals = ls_map[gm_mask], nn_map[gm_mask]
    valid_mask = ~np.isnan(ls_vals) & ~np.isnan(nn_vals)
    ls_vals, nn_vals = ls_vals[valid_mask], nn_vals[valid_mask]
    if len(ls_vals) < 2: return {}

    corr, _ = pearsonr(ls_vals, nn_vals)
    diff, avg = nn_vals - ls_vals, (nn_vals + ls_vals) / 2
    mean_diff, std_diff = np.mean(diff), np.std(diff)
    loa_lower, loa_upper = mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(x=ls_vals, y=nn_vals, ax=axes[0], scatter_kws={'alpha':0.2, 's':5}, line_kws={'color':'red'})
    axes[0].set_title(f'{param.upper()} Concordance (r={corr:.3f})'); axes[0].set_xlabel(f'LS {param.upper()}'); axes[0].set_ylabel(f'NN {param.upper()}')
    sns.scatterplot(x=avg, y=diff, ax=axes[1], alpha=0.2, s=5)
    axes[1].axhline(mean_diff, color='red', linestyle='-', label=f'Bias: {mean_diff:.2f}')
    axes[1].axhline(loa_upper, color='red', linestyle='--', label=f'95% LoA'); axes[1].axhline(loa_lower, color='red', linestyle='--')
    axes[1].set_title('Bland-Altman Plot'); axes[1].set_xlabel(f'Average of Methods'); axes[1].set_ylabel(f'Difference (NN - LS)'); axes[1].legend()
    plt.suptitle(f'Concordance Analysis for Subject {subject_id}'); plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = output_dir / f'plots/{subject_id}_{param}_concordance.png'; plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path); plt.close(fig)
    
    return {f'{param}_concordance_pearson_r': corr, f'{param}_concordance_bias': mean_diff}

def evaluate_uncertainty_calibration(predicted_std, measured_std, gm_mask, param, subject_id, output_dir):
    if predicted_std is None or measured_std is None or gm_mask is None: return {}
    pred_vals, meas_vals = predicted_std[gm_mask], measured_std[gm_mask]
    valid_mask = ~np.isnan(pred_vals) & ~np.isnan(meas_vals) & (meas_vals > 0)
    pred_vals, meas_vals = pred_vals[valid_mask], meas_vals[valid_mask]
    if len(pred_vals) < 10: return {}
    corr, _ = pearsonr(pred_vals, meas_vals)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.regplot(x=meas_vals, y=pred_vals, ax=ax, scatter_kws={'alpha':0.1, 's':5}, line_kws={'color':'red'})
    ax.set_title(f'{param.upper()} Uncertainty Calibration (r={corr:.3f})'); ax.set_xlabel('Measured Test-Retest Std Dev'); ax.set_ylabel('NN Predicted Aleatoric Std Dev')
    plot_path = output_dir / f'plots/{subject_id}_{param}_uncertainty_calibration.png'; plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path); plt.close(fig)
    return {f'{param}_uncertainty_pearson_r': corr}

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive quantitative evaluation on final generated maps.")
    parser.add_argument("final_maps_dir", type=str); parser.add_argument("preprocessed_dir", type=str)
    parser.add_argument("raw_validated_dir", type=str); parser.add_argument("model_results_dir", type=str)
    parser.add_argument("output_dir", type=str); args = parser.parse_args()
    
    output_path = Path(args.output_dir); output_path.mkdir(parents=True, exist_ok=True)

    try: models, config, norm_stats, is_disentangled, expected_scalars = load_artifacts_for_eval(Path(args.model_results_dir))
    except Exception as e: print(f"[FATAL] Could not load artifacts: {e}"); sys.exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device, dtype=torch.bfloat16)
    
    all_subject_results = []
    for subject_dir in tqdm(sorted([d for d in Path(args.final_maps_dir).iterdir() if d.is_dir()]), desc="Evaluating all subjects"):
        subject_id = subject_dir.name
        results = {"subject_id": subject_id}
        
        try:
            brain_mask = np.load(Path(args.preprocessed_dir) / subject_id / 'brain_mask.npy')
            gm_mask = np.load(Path(args.preprocessed_dir) / subject_id / 'gm_mask.npy')
            raw_subject_dir = Path(args.raw_validated_dir) / subject_id

            maps = {name: load_nifti_data(subject_dir / f'{name}.nii.gz') for name in ["ls_1r_cbf", "ls_1r_att", "nn_1r_cbf", "nn_1r_att", "nn_1r_cbf_uncertainty", "nn_1r_att_uncertainty"]}
            
            results.update(evaluate_robustness(maps["ls_1r_cbf"], maps["nn_1r_cbf"], brain_mask, 'cbf'))
            results.update(evaluate_robustness(maps["ls_1r_att"], maps["nn_1r_att"], brain_mask, 'att'))
            results.update(evaluate_concordance(maps["ls_1r_cbf"], maps["nn_1r_cbf"], gm_mask, 'cbf', subject_id, output_path))
            results.update(evaluate_concordance(maps["ls_1r_att"], maps["nn_1r_att"], gm_mask, 'att', subject_id, output_path))
            
            tqdm.write(f"  Calculating Test-Retest Metrics for {subject_id}...")
            # For NN, pass the full config. For LS, create a smaller dict of only physical params.
            nn_cbf_cov, nn_att_cov, _, _ = calculate_test_retest_cov(raw_subject_dir, brain_mask, gm_mask, config, 'nn', is_disentangled, models, norm_stats, device, expected_scalars)
            ls_params = {k:v for k,v in config.items() if k in ['T1_artery','T_tau','T2_factor','alpha_BS1','alpha_PCASL','alpha_VSASL']}
            ls_cbf_cov, ls_att_cov, meas_std_cbf, meas_std_att = calculate_test_retest_cov(raw_subject_dir, brain_mask, gm_mask, ls_params, 'ls', is_disentangled)
            results.update({'cbf_precision_cov_nn': nn_cbf_cov, 'att_precision_cov_nn': nn_att_cov, 'cbf_precision_cov_ls': ls_cbf_cov, 'att_precision_cov_ls': ls_att_cov})
            
            results.update(evaluate_uncertainty_calibration(maps["nn_1r_cbf_uncertainty"], meas_std_cbf, gm_mask, 'cbf', subject_id, output_path))
            results.update(evaluate_uncertainty_calibration(maps["nn_1r_att_uncertainty"], meas_std_att, gm_mask, 'att', subject_id, output_path))
            
            all_subject_results.append(results)

        except Exception as e:
            tqdm.write(f"  [ERROR] Failed processing subject {subject_id}: {e}"); import traceback; traceback.print_exc()
            all_subject_results.append({"subject_id": subject_id, "error": str(e)})

    df = pd.DataFrame(all_subject_results).set_index('subject_id')
    pd.set_option('display.max_rows', None); pd.set_option('display.max_columns', None); pd.set_option('display.width', 200)

    for metric_name, cols_keyword in [("ROBUSTNESS", ["failure", "implausible"]), ("PRECISION", ["precision"]), ("CONCORDANCE", ["concordance"]), ("UNCERTAINTY", ["uncertainty"])]:
        cols = [c for c in df.columns if any(kw in c for kw in cols_keyword)]
        if not cols: continue
        df_metric = df[cols]
        print(f"\n" + "="*80); print(f" " * (30 - len(metric_name)//2) + f"{metric_name} SUMMARY"); print("="*80)
        print(df_metric.mean().to_frame(name='Mean Value').to_string(float_format="%.3f"))
        df_metric.to_csv(output_path / f"summary_{metric_name.lower()}.csv")

    df_timings = analyze_computational_performance(Path(args.final_maps_dir))
    if not df_timings.empty:
        print("\n" + "="*80); print(" " * 22 + "COMPUTATIONAL PERFORMANCE SUMMARY"); print("="*80)
        print(df_timings.mean().to_frame(name='Mean Value (s)').to_string(float_format="%.2f"))
        df_timings.to_csv(output_path / "summary_computational_performance.csv")
    else:
        print("\nSkipping computational performance summary as no timing data was found.")

    print(f"\nAll summary tables saved to: {output_path.resolve()}"); print(f"All plots saved to: {output_path.resolve()}/plots/")
    print("\n--- In-vivo evaluation pipeline complete! ---")

if __name__ == '__main__':
    main()
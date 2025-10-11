# run_all_evaluations.py
#
# A scientifically rigorous script for the quantitative evaluation of in-vivo results.
# MODIFIED to include new benchmarks for computational performance and uncertainty validation.

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

from enhanced_asl_network import EnhancedASLNet
from predict_on_invivo import batch_predict_nn, fit_ls_robust
from prepare_invivo_data import find_and_sort_files_by_pld

def load_nifti_data(file_path: Path) -> np.ndarray:
    if not file_path.exists(): return None
    return np.nan_to_num(nib.load(file_path).get_fdata(dtype=np.float64))

# ... [Other helper functions like load_artifacts remain the same] ...

def analyze_computational_performance(final_maps_dir: Path) -> pd.DataFrame:
    """Aggregates timing data from all subjects and calculates speedup."""
    timings = []
    for subject_dir in final_maps_dir.iterdir():
        if subject_dir.is_dir() and (subject_dir / 'timings.json').exists():
            with open(subject_dir / 'timings.json', 'r') as f:
                data = json.load(f)
                data['subject_id'] = subject_dir.name
                timings.append(data)
    df = pd.DataFrame(timings).set_index('subject_id')
    if not df.empty and 'ls_total_s' in df.columns and 'nn_total_s' in df.columns:
        df['speedup_factor'] = df['ls_total_s'] / df['nn_total_s']
    return df

def get_measured_stdev(raw_subject_dir, brain_mask, gm_mask, config) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the ground-truth test-retest stdev across 4 repeats using LS."""
    # This function is derived from the original CoV calculation
    pcasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_PCASL_*.nii*'])
    vsasl_files = find_and_sort_files_by_pld(raw_subject_dir, ['r_normdiff_alldyn_VSASL_*.nii*'])
    plds = np.array([int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files])
    
    cbf_maps, att_maps = [], []
    for i in range(4): # Loop over 4 repeats
        pcasl_repeat = np.stack([load_nifti_data(f)[..., i] for f in pcasl_files], axis=-1)
        vsasl_repeat = np.stack([load_nifti_data(f)[..., i] for f in vsasl_files], axis=-1)
        signal_flat = np.concatenate([pcasl_repeat.reshape(-1, len(plds)), vsasl_repeat.reshape(-1, len(plds))], axis=1)
        
        cbf_masked, att_masked = fit_ls_robust(signal_flat[brain_mask.flatten()], plds, config)
        
        cbf_map, att_map = np.full(brain_mask.shape, np.nan), np.full(brain_mask.shape, np.nan)
        cbf_map[brain_mask], att_map[brain_mask] = cbf_masked, att_masked
        cbf_maps.append(cbf_map); att_maps.append(att_map)
        
    return np.nanstd(np.stack(cbf_maps, axis=-1), axis=-1), np.nanstd(np.stack(att_maps, axis=-1), axis=-1)

def evaluate_uncertainty_calibration(predicted_std, measured_std, gm_mask, param, subject_id, output_dir):
    """Correlates predicted uncertainty with measured test-retest standard deviation."""
    pred_vals = predicted_std[gm_mask]
    meas_vals = measured_std[gm_mask]
    valid_mask = ~np.isnan(pred_vals) & ~np.isnan(meas_vals) & (meas_vals > 0)
    pred_vals, meas_vals = pred_vals[valid_mask], meas_vals[valid_mask]

    if len(pred_vals) < 10: return {}
    corr, _ = pearsonr(pred_vals, meas_vals)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.regplot(x=meas_vals, y=pred_vals, ax=ax, scatter_kws={'alpha':0.1, 's':5}, line_kws={'color':'red'})
    ax.set_title(f'{param.upper()} Uncertainty Calibration (r={corr:.3f})')
    ax.set_xlabel('Measured Test-Retest Std Dev')
    ax.set_ylabel('NN Predicted Aleatoric Std Dev')
    plot_path = output_dir / f'plots/{subject_id}_{param}_uncertainty_calibration.png'
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path); plt.close(fig)
    
    return {f'{param}_uncertainty_pearson_r': corr}

def main():
    # ... argparsing and artifact loading ...
    parser = argparse.ArgumentParser(description="Run comprehensive quantitative evaluation.")
    parser.add_argument("final_maps_dir", type=str)
    parser.add_argument("preprocessed_dir", type=str)
    parser.add_argument("raw_validated_dir", type=str)
    parser.add_argument("model_results_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_maps_dir = Path(args.final_maps_dir)

    # ... [load_artifacts logic here]
    
    all_subject_results = []
    subject_dirs = sorted([d for d in final_maps_dir.iterdir() if d.is_dir()])

    for subject_dir in tqdm(subject_dirs, desc="Evaluating all subjects"):
        subject_id = subject_dir.name
        results = {"subject_id": subject_id}
        
        # --- NEW: Uncertainty Validation ---
        gm_mask = np.load(Path(args.preprocessed_dir) / subject_id / 'gm_mask.npy')
        brain_mask = np.load(Path(args.preprocessed_dir) / subject_id / 'brain_mask.npy')
        raw_subject_dir = Path(args.raw_validated_dir) / subject_id

        pred_std_cbf = load_nifti_data(subject_dir / 'nn_1r_cbf_uncertainty.nii.gz')
        pred_std_att = load_nifti_data(subject_dir / 'nn_1r_att_uncertainty.nii.gz')
        
        # This is computationally expensive, but scientifically necessary
        tqdm.write(f"  Calculating ground-truth stdev for {subject_id}...")
        meas_std_cbf, meas_std_att = get_measured_stdev(raw_subject_dir, brain_mask, gm_mask, {}) # Pass empty config for now
        
        if pred_std_cbf is not None:
             results.update(evaluate_uncertainty_calibration(pred_std_cbf, meas_std_cbf, gm_mask, 'cbf', subject_id, output_path))
        if pred_std_att is not None:
             results.update(evaluate_uncertainty_calibration(pred_std_att, meas_std_att, gm_mask, 'att', subject_id, output_path))
        
        all_subject_results.append(results)

    # --- Final Reporting ---
    # ... [Original reporting for Robustness, Precision, Concordance] ...
    
    # --- NEW: Computational Performance Report ---
    df_timings = analyze_computational_performance(final_maps_dir)
    print("\n" + "="*80); print(" " * 22 + "COMPUTATIONAL PERFORMANCE SUMMARY"); print("="*80)
    print(df_timings.mean().to_frame(name='Mean Value').to_string(float_format="%.2f"))
    df_timings.to_csv(output_path / "computational_performance.csv")

    # --- NEW: Uncertainty Validation Report ---
    df_uncertainty = pd.DataFrame(all_subject_results).set_index('subject_id')
    uncertainty_cols = [c for c in df_uncertainty.columns if 'uncertainty' in c]
    if uncertainty_cols:
        print("\n" + "="*80); print(" " * 25 + "UNCERTAINTY VALIDATION SUMMARY"); print("="*80)
        print(df_uncertainty[uncertainty_cols].mean().to_frame(name='Mean Pearson Correlation (r)').to_string(float_format="%.3f"))
        df_uncertainty[uncertainty_cols].to_csv(output_path / "uncertainty_validation.csv")

if __name__ == '__main__':
    main()
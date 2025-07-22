# run_all_evaluations.py (Final Version)
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import sys
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def analyze_correlation_and_bias(map1_data, map2_data, mask):
    """Calculates Pearson correlation and data for Bland-Altman plot."""
    v_map1, v_map2 = map1_data[mask], map2_data[mask]
    
    # Check if arrays contain valid numbers before calculating
    if np.isnan(v_map1).all() or np.isnan(v_map2).all():
        return {"correlation": np.nan, "mean_difference": np.nan, "upper_loa": np.nan, "lower_loa": np.nan, "plot_data": (None, None)}

    correlation, _ = pearsonr(v_map1, v_map2)
    difference = v_map1 - v_map2
    average = (v_map1 + v_map2) / 2
    mean_diff, std_diff = np.mean(difference), np.std(difference)
    return {
        "correlation": correlation,
        "mean_difference": mean_diff,
        "upper_loa": mean_diff + 1.96 * std_diff,
        "lower_loa": mean_diff - 1.96 * std_diff,
        "plot_data": (average, difference)
    }

def analyze_tissue_properties(cbf_map_data, m0_data, brain_mask):
    """Calculates GM/WM ratio and CoV within GM."""
    m0_masked = m0_data[brain_mask]
    if len(m0_masked) == 0: return {"gm_wm_ratio": np.nan, "cov_in_gm": np.nan, "fit_success_rate": 0}

    # --- NEW: Calculate Fit Success Rate ---
    cbf_masked = cbf_map_data[brain_mask]
    valid_fits = np.sum(~np.isnan(cbf_masked))
    total_voxels = len(cbf_masked)
    fit_success_rate = (valid_fits / total_voxels) * 100 if total_voxels > 0 else 0

    # Only proceed with tissue analysis if there are enough valid fits
    if fit_success_rate < 1.0: # If less than 1% of voxels were fit, metrics are meaningless
        return {"gm_wm_ratio": np.nan, "cov_in_gm": np.nan, "fit_success_rate": fit_success_rate}

    gm_threshold = np.percentile(m0_masked, 60)
    gm_mask = (m0_data > gm_threshold) & brain_mask
    wm_threshold_lower = np.percentile(m0_masked, 20)
    wm_mask = (m0_data > wm_threshold_lower) & (m0_data <= gm_threshold) & brain_mask

    if np.sum(gm_mask) > 10 and np.sum(wm_mask) > 10:
        mean_gm_cbf = np.nanmean(cbf_map_data[gm_mask])
        mean_wm_cbf = np.nanmean(cbf_map_data[wm_mask])
        gm_wm_ratio = mean_gm_cbf / mean_wm_cbf if mean_wm_cbf > 1e-6 else np.nan
        std_gm_cbf = np.nanstd(cbf_map_data[gm_mask])
        cov_gm = std_gm_cbf / mean_gm_cbf if mean_gm_cbf > 1e-6 else np.nan
    else:
        gm_wm_ratio, cov_gm = np.nan, np.nan
        
    return {"gm_wm_ratio": gm_wm_ratio, "cov_in_gm": cov_gm, "fit_success_rate": fit_success_rate}

def create_bland_altman_plot(analysis_results, title, output_path):
    """Generates and saves a Bland-Altman plot."""
    avg, diff = analysis_results['plot_data']
    if avg is None: return # Skip plot if data is invalid
    
    mean_diff, upper_loa, lower_loa = analysis_results['mean_difference'], analysis_results['upper_loa'], analysis_results['lower_loa']
    plt.figure(figsize=(10, 7))
    plt.scatter(avg, diff, alpha=0.2, s=5, c='blue')
    plt.axhline(mean_diff, color='red', linestyle='--', label=f"Mean Bias: {mean_diff:.2f}")
    plt.axhline(upper_loa, color='gray', linestyle=':', label=f"Upper LoA (+1.96 SD): {upper_loa:.2f}")
    plt.axhline(lower_loa, color='gray', linestyle=':', label=f"Lower LoA (-1.96 SD): {lower_loa:.2f}")
    plt.title(title, fontsize=16)
    plt.xlabel("Average of Methods (CBF mL/100g/min)", fontsize=12)
    plt.ylabel("Difference (NN 1-Repeat - LS 4-Repeat)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def evaluate_single_subject(subject_maps_dir: Path, raw_data_root: Path) -> dict:
    subject_id = subject_maps_dir.name
    subject_raw_dir = raw_data_root / subject_id
    try:
        nn_1r_cbf = nib.load(subject_maps_dir / 'nn_from_1_repeat_cbf.nii.gz').get_fdata()
        ls_4r_cbf = nib.load(subject_maps_dir / 'ls_from_4_repeats_cbf.nii.gz').get_fdata()
        ls_1r_cbf = nib.load(subject_maps_dir / 'ls_from_1_repeat_cbf.nii.gz').get_fdata()
        
        m0_file = next(subject_raw_dir.rglob('r_M0.nii*'), None) or next(subject_raw_dir.rglob('M0.nii*'), None)
        if not m0_file: return {"subject_id": subject_id, "error": "M0 scan not found"}
        m0_data = nib.load(m0_file).get_fdata()
        brain_mask = nib.load(subject_maps_dir / 'nn_from_1_repeat_cbf.nii.gz').get_fdata() != 0

        core_analysis = analyze_correlation_and_bias(nn_1r_cbf, ls_4r_cbf, brain_mask)
        nn_1r_tissue = analyze_tissue_properties(nn_1r_cbf, m0_data, brain_mask)
        ls_4r_tissue = analyze_tissue_properties(ls_4r_cbf, m0_data, brain_mask)
        ls_1r_tissue = analyze_tissue_properties(ls_1r_cbf, m0_data, brain_mask)

        plot_path = subject_maps_dir / f"{subject_id}_bland_altman.png"
        create_bland_altman_plot(core_analysis, f"Bland-Altman: NN (1-Repeat) vs LS (4-Repeat) for {subject_id}", plot_path)
        
        return {
            "subject_id": subject_id,
            "correlation_vs_benchmark": core_analysis['correlation'],
            "mean_bias_vs_benchmark": core_analysis['mean_difference'],
            "nn_1r_gm_wm_ratio": nn_1r_tissue['gm_wm_ratio'],
            "ls_4r_gm_wm_ratio": ls_4r_tissue['gm_wm_ratio'],
            "ls_1r_gm_wm_ratio": ls_1r_tissue['gm_wm_ratio'],
            "nn_1r_cov_in_gm": nn_1r_tissue['cov_in_gm'],
            "ls_4r_cov_in_gm": ls_4r_tissue['cov_in_gm'],
            "ls_1r_cov_in_gm": ls_1r_tissue['cov_in_gm'],
            "nn_1r_fit_success_rate": nn_1r_tissue['fit_success_rate'],
            "ls_4r_fit_success_rate": ls_4r_tissue['fit_success_rate'],
            "ls_1r_fit_success_rate": ls_1r_tissue['fit_success_rate'],
            "error": None
        }
    except Exception as e:
        return {"subject_id": subject_id, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Run quantitative evaluation on all subjects in the final maps folder.")
    parser.add_argument("final_maps_dir", type=str)
    parser.add_argument("raw_validated_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    final_maps_root, raw_data_root, output_root = Path(args.final_maps_dir), Path(args.raw_validated_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    subject_dirs = sorted([d for d in final_maps_root.iterdir() if d.is_dir()])
    if not subject_dirs:
        print(f"[ERROR] No subject folders found in {final_maps_root}.")
        sys.exit(1)

    all_results = [evaluate_single_subject(sub_dir, raw_data_root) for sub_dir in tqdm(subject_dirs, desc="Evaluating all subjects")]
    df = pd.DataFrame(all_results)
    
    print("\n\n" + "="*120)
    print(" " * 45 + "QUANTITATIVE EVALUATION SUMMARY")
    print("="*120)
    
    failed_subjects = df[df['error'].notna()]
    if not failed_subjects.empty:
        print("\n--- Subjects with Errors ---")
        for _, row in failed_subjects.iterrows(): print(f"  - {row['subject_id']}: {row['error']}")
    
    df_success = df[df['error'].isna()].drop(columns=['error'])
    if not df_success.empty:
        cols = [
            "subject_id", "correlation_vs_benchmark", "mean_bias_vs_benchmark",
            "nn_1r_gm_wm_ratio", "ls_4r_gm_wm_ratio", "ls_1r_gm_wm_ratio",
            "nn_1r_cov_in_gm", "ls_4r_cov_in_gm", "ls_1r_cov_in_gm",
            "nn_1r_fit_success_rate", "ls_4r_fit_success_rate", "ls_1r_fit_success_rate"
        ]
        df_display = df_success[[c for c in cols if c in df_success.columns]].copy()
        
        for col in [c for c in df_display.columns if "cov_in_gm" in c]: df_display[col] *= 100
        
        print("\n--- Performance Metrics Summary ---")
        pd.set_option('display.max_rows', None); pd.set_option('display.max_columns', None); pd.set_option('display.width', 200)
        print(df_display.to_string(index=False, float_format="%.2f"))

        summary_path = output_root / "invivo_evaluation_summary.csv"
        df_display.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"\nSummary report saved to: {summary_path}")
        print("Individual Bland-Altman plots saved in each subject's 'final_maps' subfolder.")
    print("="*120)

if __name__ == '__main__':
    main()
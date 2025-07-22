# adikondepudi-asl-multiverse/generate_in_vivo_publication_figures.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from pathlib import Path
import argparse
from scipy.ndimage import zoom

def plot_quantitative_summary(df: pd.DataFrame, output_dir: Path):
    """Generates and saves bar charts for key performance metrics."""
    print("--> Generating quantitative summary plot...")
    
    # Select and rename columns for plotting
    df_plot = df.rename(columns={
        'nn_1r_spatial_cov_gm': 'NN (1-Repeat) Spatial CoV',
        'ls_4r_spatial_cov_gm': 'LS (4-Repeat) Spatial CoV',
        'nn_test_retest_cov_gm': 'NN (1-Repeat) Test-Retest CoV',
        'nn_1r_gm_wm_ratio': 'NN (1-Repeat) GM/WM Ratio',
        'ls_4r_gm_wm_ratio': 'LS (4-Repeat) GM/WM Ratio'
    })

    # Melt the dataframe to make it suitable for seaborn's catplot
    metrics_to_plot = [
        'NN (1-Repeat) Spatial CoV', 'LS (4-Repeat) Spatial CoV', 
        'NN (1-Repeat) Test-Retest CoV',
        'NN (1-Repeat) GM/WM Ratio', 'LS (4-Repeat) GM/WM Ratio'
    ]
    df_melt = df_plot.melt(id_vars=['subject_id'], value_vars=metrics_to_plot, var_name='Metric', value_name='Value')

    # Create separate plots for CoV and Ratios as they have different scales
    sns.set_theme(style="whitegrid", context="talk")

    # Plot for CoV
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.barplot(data=df_melt[df_melt['Metric'].str.contains('CoV')], x='Metric', y='Value', ax=ax1, palette="viridis")
    ax1.set_title('Noise & Reproducibility Comparison (Lower is Better)', fontsize=18, weight='bold')
    ax1.set_ylabel('Coefficient of Variation (%)', fontsize=14)
    ax1.set_xlabel('')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig1.savefig(output_dir / "figure_1_quantitative_cov.png", dpi=300)
    plt.close(fig1)

    # Plot for GM/WM Ratio
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.barplot(data=df_melt[df_melt['Metric'].str.contains('Ratio')], x='Metric', y='Value', ax=ax2, palette="plasma")
    ax2.set_title('Physiological Plausibility (GM/WM CBF Ratio)', fontsize=18, weight='bold')
    ax2.set_ylabel('GM/WM CBF Ratio', fontsize=14)
    ax2.set_xlabel('')
    ax2.axhline(y=1.5, color='gray', linestyle='--', label='Typical Physiological Range (~1.5-2.5)')
    ax2.axhline(y=2.5, color='gray', linestyle='--')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig2.savefig(output_dir / "figure_2_quantitative_ratio.png", dpi=300)
    plt.close(fig2)

    print(f"    Saved quantitative plots to {output_dir}")

def plot_visual_comparison(maps_dir: Path, output_dir: Path, subject_id: str):
    """Generates a visual comparison of CBF maps for a single subject, with interpolation."""
    print(f"--> Generating visual comparison for subject {subject_id}...")
    
    subject_maps_dir = maps_dir / subject_id
    if not subject_maps_dir.exists():
        print(f"    [ERROR] Maps directory not found for subject {subject_id}. Skipping visual plot.")
        return

    maps_to_load = {
        'LS (4-Repeat)': subject_maps_dir / 'ls_from_4_repeats_cbf.nii.gz',
        'LS (1-Repeat)': subject_maps_dir / 'ls_from_1_repeat_cbf.nii.gz',
        'NN (1-Repeat)': subject_maps_dir / 'nn_from_1_repeat_cbf.nii.gz'
    }

    loaded_maps = {}
    for name, path in maps_to_load.items():
        if path.exists():
            loaded_maps[name] = nib.load(path).get_fdata()
        else:
            print(f"    [WARNING] Map file not found: {path}. Skipping this map.")

    if not loaded_maps:
        return

    # Determine a consistent color scale (vmax) from the most reliable map
    vmax = np.percentile(loaded_maps.get('LS (4-Repeat)', np.zeros((1,1))), 98)

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, len(loaded_maps), figsize=(len(loaded_maps) * 5, 5))
    fig.suptitle(f'Visual CBF Map Comparison for Subject: {subject_id}', fontsize=20, weight='bold')

    # Find a good middle slice with brain tissue
    brain_mask = loaded_maps.get('LS (4-Repeat)', np.zeros((1,1))) > 0
    slice_idx = int(np.where(brain_mask.any(axis=(0, 1)))[0].mean())

    for i, (name, data) in enumerate(loaded_maps.items()):
        # Interpolate from 64x64 to 128x128
        slice_data = data[:, :, slice_idx]
        interpolated_slice = zoom(slice_data, zoom=2, order=1) # order=1 is bilinear

        ax = axes[i]
        im = ax.imshow(interpolated_slice.T, cmap='viridis', vmin=0, vmax=vmax, origin='lower')
        ax.set_title(name, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label='CBF (mL/100g/min)')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / f"figure_3_visual_comparison_{subject_id}.png", dpi=300)
    plt.close(fig)
    print(f"    Saved visual comparison plot to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready figures from final evaluation results.")
    parser.add_argument("summary_csv", type=str, help="Path to the comprehensive_invivo_evaluation_summary.csv file.")
    parser.add_argument("maps_dir", type=str, help="Path to the parent directory containing the final NIfTI maps for each subject.")
    parser.add_argument("output_dir", type=str, help="Directory where the generated figures will be saved.")
    parser.add_argument("--subject_id", type=str, default="20231004_MR1_A151", help="A representative subject ID for the visual comparison plot.")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.summary_csv)
    
    plot_quantitative_summary(df, output_path)
    plot_visual_comparison(Path(args.maps_dir), output_path, args.subject_id)

    print("\n--- Figure generation complete! ---")

if __name__ == '__main__':
    main()
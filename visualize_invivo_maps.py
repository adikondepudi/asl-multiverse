# visualize_invivo_maps.py
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def plot_map_slices(map_data: np.ndarray, bg_data: np.ndarray, title: str, output_path: Path,
                      cmap: str = 'jet', vmin: float = 0, vmax: float = None):
    """
    Generates and saves a mosaic of axial slices for a given map overlaid on a background image.
    """
    if vmax is None:
        vmax = np.percentile(map_data[map_data > 0], 98) if np.any(map_data > 0) else 1.0

    num_slices = map_data.shape[2]
    # Create a grid of subplots (e.g., 4x5 for 20 slices)
    grid_size = int(np.ceil(np.sqrt(num_slices)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2),
                             facecolor='black')
    axes = axes.ravel()

    for i in range(num_slices):
        ax = axes[i]
        # Rotate for standard radiological view
        bg_slice = np.rot90(bg_data[:, :, i])
        map_slice = np.rot90(map_data[:, :, i])

        ax.imshow(bg_slice, cmap='gray')
        # Use a mask to only show the overlay where the signal is meaningful
        masked_overlay = np.ma.masked_where(map_slice <= vmin, map_slice)
        im = ax.imshow(masked_overlay, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax)
        
        ax.axis('off')
        ax.set_title(f'Slice {i}', color='white', fontsize=8)

    # Hide unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=16, color='white', y=0.98)
    # Add a single colorbar for the whole figure
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, pad=0.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    fig.savefig(output_path, dpi=150, facecolor='black')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate visual plots of the final NIfTI maps.")
    parser.add_argument("invivo_maps_dir", type=str, help="Directory with the final map outputs from predict_on_invivo.py.")
    parser.add_argument("preprocessed_dir", type=str, help="Directory with preprocessed data (for M0/brain masks).")
    parser.add_argument("output_viz_dir", type=str, help="Directory to save the output PNG images.")
    args = parser.parse_args()

    maps_root = Path(args.invivo_maps_dir)
    preprocessed_root = Path(args.preprocessed_dir)
    output_root = Path(args.output_viz_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([d for d in maps_root.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} subjects with maps to visualize.")

    for subject_dir in tqdm(subject_dirs, desc="Visualizing Subjects"):
        subject_id = subject_dir.name
        subject_viz_dir = output_root / subject_id
        subject_viz_dir.mkdir(exist_ok=True)

        try:
            # Load the brain mask to use as a background
            brain_mask = np.load(preprocessed_root / subject_id / 'brain_mask.npy')
            
            # --- Visualize NN (1-repeat) vs LS (4-repeat) ---
            # CBF comparison
            nn_cbf = nib.load(subject_dir / "nn_from_1_repeat_cbf.nii.gz").get_fdata()
            ls_cbf = nib.load(subject_dir / "ls_from_4_repeats_cbf.nii.gz").get_fdata()
            plot_map_slices(nn_cbf, brain_mask, f"{subject_id}: NN CBF (from 1 repeat)", subject_viz_dir / "nn_cbf_1r.png", vmin=0, vmax=100)
            plot_map_slices(ls_cbf, brain_mask, f"{subject_id}: LS CBF (from 4 repeats)", subject_viz_dir / "ls_cbf_4r.png", vmin=0, vmax=100)

            # ATT comparison
            nn_att = nib.load(subject_dir / "nn_from_1_repeat_att.nii.gz").get_fdata()
            ls_att = nib.load(subject_dir / "ls_from_4_repeats_att.nii.gz").get_fdata()
            plot_map_slices(nn_att, brain_mask, f"{subject_id}: NN ATT (from 1 repeat)", subject_viz_dir / "nn_att_1r.png", vmin=500, vmax=3500)
            plot_map_slices(ls_att, brain_mask, f"{subject_id}: LS ATT (from 4 repeats)", subject_viz_dir / "ls_att_4r.png", vmin=500, vmax=3500)
        
        except FileNotFoundError as e:
            tqdm.write(f"  [WARN] Skipping {subject_id}: Missing file - {e.filename}")
        except Exception as e:
            tqdm.write(f"  [ERROR] Failed to visualize {subject_id}: {e}")

    print("\n--- Visualization complete! Check the output directory for PNG files. ---")

if __name__ == '__main__':
    main()
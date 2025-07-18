# process_invivo_data.py
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import re
from tqdm import tqdm
from typing import List

def find_and_sort_files_robustly(subject_dir: Path, patterns: list) -> List[Path]:
    """
    Finds files matching a list of patterns and sorts them by the numeric
    Post-Labeling Delay (PLD) value found in their filenames.
    """
    def get_pld_from_path(path: Path) -> int:
        match = re.search(r'_(\d+)', path.name)
        return int(match.groups()[0]) if match else -1

    for pattern in patterns:
        files = list(subject_dir.glob(pattern))
        if files:
            print(f"  --> Found files for '{subject_dir.name}' using pattern: '{pattern}'")
            return sorted(files, key=get_pld_from_path)
    return []

def load_and_scale_nifti(file_path: Path) -> np.ndarray:
    """
    Loads NIfTI data, applies scaling factors, and cleans non-finite values.

    This function performs two critical data cleaning steps:
    1.  Applies `scl_slope` and `scl_inter` to convert raw integer data to
        its true physical floating-point values.
    2.  Replaces any NaN or infinity values in the data with 0.0,
        ensuring the data is numerically stable for downstream processing.
    """
    img = nib.load(file_path)
    data = img.get_fdata(dtype=np.float64)
    
    scl_slope = img.header.get('scl_slope', 0)
    scl_inter = img.header.get('scl_inter', 0)

    if scl_slope != 0:
        data = data * scl_slope + scl_inter
        
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

def preprocess_subject(subject_dir: Path, output_root: Path):
    """
    Processes a single subject's data from NIfTI to NumPy arrays, 
    now including a robustly generated brain mask.
    """
    subject_id = subject_dir.name
    subject_output_dir = output_root / subject_id
    
    try:
        pcasl_patterns = ['rPCASL_*_aslrawimages.nii*', 'r_normdiff_alldyn_PCASL_*.nii*', 'r_PCASL_*.nii*']
        vsasl_patterns = ['rVSASL_*_aslrawimages.nii*', 'r_normdiff_alldyn_VSASL_*.nii*', 'r_VSASL_*.nii*']
        
        pcasl_files = find_and_sort_files_robustly(subject_dir, pcasl_patterns)
        vsasl_files = find_and_sort_files_robustly(subject_dir, vsasl_patterns)

        if not pcasl_files or not vsasl_files or len(pcasl_files) != len(vsasl_files):
            print(f"Warning: Inconsistent or missing files for {subject_id}. Skipping.")
            return

        subject_output_dir.mkdir(parents=True, exist_ok=True)

        pcasl_data_list = [load_and_scale_nifti(f) for f in pcasl_files]
        pcasl_full_data = np.stack(pcasl_data_list, axis=-2)
        
        vsasl_data_list = [load_and_scale_nifti(f) for f in vsasl_files]
        vsasl_full_data = np.stack(vsasl_data_list, axis=-2)
        
        first_img = nib.load(pcasl_files[0])
        affine, header = first_img.affine, first_img.header
        x_dim, y_dim, z_dim = pcasl_full_data.shape[:3]
        num_repeats = pcasl_full_data.shape[-1] if pcasl_full_data.ndim == 5 else 1
        
        # Prepare low-SNR (single repeat) and high-SNR (averaged) data
        pcasl_low_snr = np.squeeze(pcasl_full_data[..., 0])
        vsasl_low_snr = np.squeeze(vsasl_full_data[..., 0])
        pcasl_high_snr = np.mean(pcasl_full_data[..., :min(4, num_repeats)], axis=-1)
        vsasl_high_snr = np.mean(vsasl_full_data[..., :min(4, num_repeats)], axis=-1)

        # --- Generate and save a brain mask ---
        mean_signal_vol = np.mean(pcasl_high_snr, axis=-1)
        threshold = np.percentile(mean_signal_vol[mean_signal_vol > 0], 98) * 0.15
        brain_mask = mean_signal_vol > threshold
        np.save(subject_output_dir / 'brain_mask.npy', brain_mask)

        # Flatten the 4D/5D data into a 2D (voxel, time) array using C-style ordering
        low_snr_signals = np.concatenate([
            pcasl_low_snr.reshape(-1, len(pcasl_files)),
            vsasl_low_snr.reshape(-1, len(vsasl_files))
        ], axis=1)
        high_snr_signals = np.concatenate([
            pcasl_high_snr.reshape(-1, len(pcasl_files)),
            vsasl_high_snr.reshape(-1, len(vsasl_files))
        ], axis=1)
        
        subject_plds = [int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files]

        # Save all processed artifacts
        np.save(subject_output_dir / 'plds.npy', np.array(subject_plds))
        np.save(subject_output_dir / 'low_snr_signals.npy', low_snr_signals)
        np.save(subject_output_dir / 'high_snr_signals.npy', high_snr_signals)
        np.save(subject_output_dir / 'image_affine.npy', affine)
        np.save(subject_output_dir / 'image_header.npy', header)
        np.save(subject_output_dir / 'image_dims.npy', np.array([x_dim, y_dim, z_dim]))

    except Exception as e:
        print(f"FATAL ERROR processing subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_invivo_data.py <path_to_raw_data_folder> <output_preprocessed_folder>")
        sys.exit(1)
    
    root_data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    subject_dirs = [d for d in root_data_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(subject_dirs)} potential subject folders in {root_data_dir}.")
    print("Starting preprocessing...")
    
    for sub_dir in tqdm(subject_dirs, desc="Processing Subjects"):
        preprocess_subject(sub_dir, output_dir)
        
    print("\n--- In-vivo data preprocessing complete! ---")
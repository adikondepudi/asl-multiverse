# process_invivo_data.py
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import re
from tqdm import tqdm
from typing import List

def find_and_sort_files_robustly(subject_dir: Path, patterns: list) -> List[Path]:
    """Finds files matching a list of patterns and sorts them by PLD."""
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

    This function now performs two critical data cleaning steps:
    1.  Applies `scl_slope` and `scl_inter` to convert raw integer data to
        its true physical floating-point values.
    2.  (NEW) Replaces any NaN or infinity values in the data with 0.0,
        ensuring the data is numerically stable for downstream processing.
    """
    img = nib.load(file_path)
    data = img.get_fdata(dtype=np.float64)
    
    scl_slope = img.header.get('scl_slope', 0)
    scl_inter = img.header.get('scl_inter', 0)

    if scl_slope != 0:
        data = data * scl_slope + scl_inter
        
    # === ELEGANT FIX: Clean non-finite values ===
    # This is the single most important change to fix the "black image" issue.
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # === END OF FIX ===
        
    return data

def preprocess_subject(subject_dir: Path, output_root: Path):
    """
    Processes a single subject's data from raw NIfTI files to NumPy arrays.
    """
    subject_id = subject_dir.name
    subject_output_dir = output_root / subject_id
    
    try:
        pcasl_patterns = ['rPCASL_*_aslrawimages.nii*', 'r_normdiff_alldyn_PCASL_*.nii*', 'r_PCASL_*.nii*']
        vsasl_patterns = ['rVSASL_*_aslrawimages.nii*', 'r_normdiff_alldyn_VSASL_*.nii*', 'r_VSASL_*.nii*']
        
        pcasl_files = find_and_sort_files_robustly(subject_dir, pcasl_patterns)
        vsasl_files = find_and_sort_files_robustly(subject_dir, vsasl_patterns)

        if not pcasl_files or not vsasl_files:
            print(f"Warning: Could not find EITHER PCASL or VSASL files for {subject_id}. Skipping.")
            return
            
        if len(pcasl_files) != len(vsasl_files):
            print(f"Warning: Mismatched PLD counts for {subject_id} "
                  f"({len(pcasl_files)} PCASL, {len(vsasl_files)} VSASL). Skipping.")
            return

        subject_output_dir.mkdir(parents=True, exist_ok=True)

        pcasl_data_list = [load_and_scale_nifti(f) for f in pcasl_files]
        pcasl_full_data = np.stack(pcasl_data_list, axis=-2)
        
        vsasl_data_list = [load_and_scale_nifti(f) for f in vsasl_files]
        vsasl_full_data = np.stack(vsasl_data_list, axis=-2)
        
        first_img = nib.load(pcasl_files[0])
        affine = first_img.affine
        header = first_img.header
        # Robustly determine dimensions, handling both 3D and 4D cases per file
        base_shape = pcasl_full_data.shape[:3]
        num_repeats = pcasl_full_data.shape[-1] if pcasl_full_data.ndim == 5 else 1
        x_dim, y_dim, z_dim = base_shape
        
        num_repeats_for_avg = min(4, num_repeats)
        # Squeeze to handle cases where there is no repeat dimension
        pcasl_low_snr = np.squeeze(pcasl_full_data[..., 0])
        vsasl_low_snr = np.squeeze(vsasl_full_data[..., 0])
        pcasl_high_snr = np.mean(pcasl_full_data[..., :num_repeats_for_avg], axis=-1)
        vsasl_high_snr = np.mean(vsasl_full_data[..., :num_repeats_for_avg], axis=-1)

        low_snr_signals, high_snr_signals = [], []
        for z in range(z_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    low_snr_signals.append(np.concatenate([pcasl_low_snr[x, y, z, :], vsasl_low_snr[x, y, z, :]]))
                    high_snr_signals.append(np.concatenate([pcasl_high_snr[x, y, z, :], vsasl_high_snr[x, y, z, :]]))
        
        subject_plds = [int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files]

        np.save(subject_output_dir / 'plds.npy', np.array(subject_plds))
        np.save(subject_output_dir / 'low_snr_signals.npy', np.array(low_snr_signals))
        np.save(subject_output_dir / 'high_snr_signals.npy', np.array(high_snr_signals))
        np.save(subject_output_dir / 'image_affine.npy', affine)
        np.save(subject_output_dir / 'image_header.npy', header)
        np.save(subject_output_dir / 'image_dims.npy', np.array([x_dim, y_dim, z_dim]))

    except Exception as e:
        print(f"FATAL ERROR processing subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_invivo_data.py <path_to_multiverse_folder> <output_preprocessed_folder>")
        sys.exit(1)
    
    root_data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    subject_dirs = [d for d in root_data_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(subject_dirs)} potential subject folders in {root_data_dir}.")
    print("Starting preprocessing...")
    
    for sub_dir in tqdm(subject_dirs, desc="Processing Subjects"):
        preprocess_subject(sub_dir, output_dir)
        
    print("\n--- In-vivo data preprocessing complete! ---")
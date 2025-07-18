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
    Loads NIfTI data, applies scaling factors from the header, and cleans
    non-finite values to ensure numerical stability.
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
    using the M0 scan for robust brain masking.
    """
    subject_id = subject_dir.name
    subject_output_dir = output_root / subject_id
    
    try:
        pcasl_patterns = ['rPCASL_*_aslrawimages.nii*', 'r_normdiff_alldyn_PCASL_*.nii*', 'r_PCASL_*.nii*']
        vsasl_patterns = ['rVSASL_*_aslrawimages.nii*', 'r_normdiff_alldyn_VSASL_*.nii*', 'r_VSASL_*.nii*']
        m0_patterns = ['r_M0.nii*', 'M0.nii*'] # Patterns to find the M0 scan
        
        pcasl_files = find_and_sort_files_robustly(subject_dir, pcasl_patterns)
        vsasl_files = find_and_sort_files_robustly(subject_dir, vsasl_patterns)
        m0_file_list = find_and_sort_files_robustly(subject_dir, m0_patterns)

        if not pcasl_files or not vsasl_files or len(pcasl_files) != len(vsasl_files):
            print(f"Warning: Inconsistent or missing ASL files for {subject_id}. Skipping.")
            return

        subject_output_dir.mkdir(parents=True, exist_ok=True)

        # === THE DEFINITIVE FIX: Use M0 Scan for Brain Masking ===
        brain_mask: np.ndarray
        if m0_file_list:
            print(f"  --> Found M0 scan for robust masking.")
            m0_img_scaled = load_and_scale_nifti(m0_file_list[0])
            # A simple threshold on the high-SNR M0 image is very effective.
            threshold = np.percentile(m0_img_scaled[m0_img_scaled > 0], 50) * 0.5
            brain_mask = m0_img_scaled > threshold
        else:
            # Fallback if no M0 is found (less reliable, but won't crash)
            print(f"Warning: No M0 scan found for {subject_id}. Creating fallback mask from ASL data.")
            pcasl_data_list_for_mask = [load_and_scale_nifti(f) for f in pcasl_files]
            pcasl_full_data_for_mask = np.stack(pcasl_data_list_for_mask, axis=-2)
            mean_abs_signal = np.mean(np.abs(pcasl_full_data_for_mask), axis=(-1, -2))
            threshold = np.percentile(mean_abs_signal[mean_abs_signal > 0], 95) * 0.2
            brain_mask = mean_abs_signal > threshold
        
        np.save(subject_output_dir / 'brain_mask.npy', brain_mask)
        # === END OF FIX ===
        
        pcasl_data_list = [load_and_scale_nifti(f) for f in pcasl_files]
        pcasl_full_data = np.stack(pcasl_data_list, axis=-2)
        vsasl_data_list = [load_and_scale_nifti(f) for f in vsasl_files]
        vsasl_full_data = np.stack(vsasl_data_list, axis=-2)

        first_img = nib.load(pcasl_files[0])
        affine, header = first_img.affine, first_img.header
        x_dim, y_dim, z_dim = pcasl_full_data.shape[:3]
        num_repeats = pcasl_full_data.shape[-1] if pcasl_full_data.ndim == 5 else 1

        pcasl_low_snr = np.squeeze(pcasl_full_data[..., 0])
        vsasl_low_snr = np.squeeze(vsasl_full_data[..., 0])
        pcasl_high_snr = np.mean(pcasl_full_data[..., :min(4, num_repeats)], axis=-1)
        vsasl_high_snr = np.mean(vsasl_full_data[..., :min(4, num_repeats)], axis=-1)

        low_snr_signals = np.concatenate([pcasl_low_snr.reshape(-1, len(pcasl_files)), vsasl_low_snr.reshape(-1, len(vsasl_files))], axis=1)
        high_snr_signals = np.concatenate([pcasl_high_snr.reshape(-1, len(pcasl_files)), vsasl_high_snr.reshape(-1, len(vsasl_files))], axis=1)
        
        subject_plds = [int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files]

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
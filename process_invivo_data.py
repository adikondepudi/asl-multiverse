# adikondepudi-asl-multiverse/process_invivo_data.py

import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import re
from tqdm import tqdm
from typing import List
import argparse

def find_and_sort_files_by_pld(subject_dir: Path, patterns: list) -> List[Path]:
    """
    Finds files matching a list of patterns and sorts them by the numeric
    Post-Labeling Delay (PLD) value found in their filenames.
    """
    def get_pld_from_path(path: Path) -> int:
        match = re.search(r'_(\d+)', path.name)
        return int(match.groups()[0]) if match else -1

    for pattern in patterns:
        files = list(subject_dir.rglob(pattern))
        if files:
            return sorted(files, key=get_pld_from_path)
    return []

def load_nifti_data(file_path: Path) -> np.ndarray:
    """Loads NIfTI data and cleans non-finite values."""
    img = nib.load(file_path)
    data = img.get_fdata(dtype=np.float64)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

def preprocess_subject(subject_dir: Path, output_root: Path):
    """
    Processes a single subject's 'normdiff' data and gold-standard masks
    into NumPy arrays suitable for NN and LS methods.
    """
    subject_id = subject_dir.name
    subject_output_dir = output_root / subject_id
    print(f"\n--- Processing Subject: {subject_id} ---")

    try:
        pcasl_files = find_and_sort_files_by_pld(subject_dir, ['r_normdiff_alldyn_PCASL_*.nii*'])
        vsasl_files = find_and_sort_files_by_pld(subject_dir, ['r_normdiff_alldyn_VSASL_*.nii*'])

        if not pcasl_files or not vsasl_files:
            print(f"  [ERROR] Missing ASL files for {subject_id}. Skipping.")
            return

        subject_output_dir.mkdir(parents=True, exist_ok=True)

        # --- NEW: Load Gold-Standard Masks from provided files ---
        print("  --> Loading gold-standard masks.")
        gm_prob_map_file = next(subject_dir.rglob('*GM_axial_philiport.nii*'))
        wm_prob_map_file = next(subject_dir.rglob('*WM_axial_philiport.nii*'))
        wb_mask_file = next(subject_dir.rglob('*M0_WBmask_by_mprage_tight.nii*'))

        gm_prob_map = load_nifti_data(gm_prob_map_file)
        wm_prob_map = load_nifti_data(wm_prob_map_file)
        wb_mask_data = load_nifti_data(wb_mask_file)

        gm_mask = gm_prob_map > 0.75  # Threshold as per PI's instruction
        wm_mask = wm_prob_map > 0.75  # Assuming same threshold for WM
        brain_mask = wb_mask_data > 0.5 # Simple threshold on the whole brain mask

        if np.sum(brain_mask) == 0:
            print(f"  [ERROR] Brain mask for {subject_id} is empty. Skipping.")
            return
            
        np.save(subject_output_dir / 'brain_mask.npy', brain_mask)
        np.save(subject_output_dir / 'gm_mask.npy', gm_mask)
        np.save(subject_output_dir / 'wm_mask.npy', wm_mask)
        print(f"  --> Masks created: {np.sum(brain_mask)} brain, {np.sum(gm_mask)} GM, {np.sum(wm_mask)} WM voxels.")

        # --- Load, Process, and Stack Signal Data ---
        pcasl_1_repeat, pcasl_4_repeat_avg = [], []
        vsasl_1_repeat, vsasl_4_repeat_avg = [], []

        for f in pcasl_files:
            data = load_nifti_data(f)
            pcasl_1_repeat.append(data[..., 0])
            pcasl_4_repeat_avg.append(np.mean(data, axis=-1))

        for f in vsasl_files:
            data = load_nifti_data(f)
            vsasl_1_repeat.append(data[..., 0])
            vsasl_4_repeat_avg.append(np.mean(data, axis=-1))

        pcasl_low_snr_4d = np.stack(pcasl_1_repeat, axis=-1)
        pcasl_high_snr_4d = np.stack(pcasl_4_repeat_avg, axis=-1)
        vsasl_low_snr_4d = np.stack(vsasl_1_repeat, axis=-1)
        vsasl_high_snr_4d = np.stack(vsasl_4_repeat_avg, axis=-1)
        
        pcasl_low_flat = pcasl_low_snr_4d.reshape(-1, len(pcasl_files))
        vsasl_low_flat = vsasl_low_snr_4d.reshape(-1, len(vsasl_files))
        pcasl_high_flat = pcasl_high_snr_4d.reshape(-1, len(pcasl_files))
        vsasl_high_flat = vsasl_high_snr_4d.reshape(-1, len(vsasl_files))
        
        low_snr_signals = np.concatenate([pcasl_low_flat, vsasl_low_flat], axis=1)
        high_snr_signals = np.concatenate([pcasl_high_flat, vsasl_high_flat], axis=1)

        first_img = nib.load(pcasl_files[0])
        subject_plds = [int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files]

        np.save(subject_output_dir / 'low_snr_signals.npy', low_snr_signals)
        np.save(subject_output_dir / 'high_snr_signals.npy', high_snr_signals)
        np.save(subject_output_dir / 'plds.npy', np.array(subject_plds))
        np.save(subject_output_dir / 'image_affine.npy', first_img.affine)
        np.save(subject_output_dir / 'image_header.npy', first_img.header)
        np.save(subject_output_dir / 'image_dims.npy', np.array(first_img.shape[:3]))
        print(f"  --> Saved processed NumPy arrays to {subject_output_dir}")

    except Exception as e:
        print(f"  [FATAL ERROR] processing subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processes validated raw in-vivo data into structured NumPy arrays for analysis.")
    parser.add_argument("validated_data_dir", type=str, help="Path to the directory containing validated subject folders.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where preprocessed NumPy arrays will be saved (e.g., 'data/preprocessed_invivo').")
    args = parser.parse_args()

    root_data_dir = Path(args.validated_data_dir)
    output_dir = Path(args.output_dir)
    
    subject_dirs = sorted([d for d in root_data_dir.iterdir() if d.is_dir()])
    if not subject_dirs:
         print(f"[ERROR] No subject folders found in '{root_data_dir}'.")
         sys.exit(1)

    print(f"Found {len(subject_dirs)} valid subject folders to process.")
    for sub_dir in subject_dirs:
        preprocess_subject(sub_dir, output_dir)
        
    print("\n--- In-vivo data preprocessing complete! ---")
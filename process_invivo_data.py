# process_invivo_data.py
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import re
from tqdm import tqdm
from typing import List, Dict

def find_and_sort_files_by_pld(subject_dir: Path, patterns: list) -> List[Path]:
    def get_pld_from_path(path: Path) -> int:
        match = re.search(r'_(\d+)', path.name)
        return int(match.groups()[0]) if match else -1

    for pattern in patterns:
        files = list(subject_dir.rglob(pattern))
        if files:
            return sorted(files, key=get_pld_from_path)
    return []

def load_nifti_data(file_path: Path) -> np.ndarray:
    img = nib.load(file_path)
    data = img.get_fdata(dtype=np.float64)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

def preprocess_subject(subject_dir: Path, output_root: Path):
    subject_id = subject_dir.name
    subject_output_dir = output_root / subject_id
    tqdm.write(f"\n--- Processing Subject: {subject_id} ---")

    try:
        pcasl_patterns = ['r_normdiff_alldyn_PCASL_*.nii*']
        vsasl_patterns = ['r_normdiff_alldyn_VSASL_*.nii*']
        m0_patterns = ['r_M0.nii*', 'M0.nii*']
        # --- ADDED: Pattern for the high-res gray matter probability map ---
        gm_prob_map_patterns = ['cor2M0_mprage_GM_axial_philiport.nii*']

        pcasl_files = find_and_sort_files_by_pld(subject_dir, pcasl_patterns)
        vsasl_files = find_and_sort_files_by_pld(subject_dir, vsasl_patterns)
        m0_file_list = find_and_sort_files_by_pld(subject_dir, m0_patterns)
        # --- ADDED: Find the GM probability map file ---
        gm_prob_map_file_list = find_and_sort_files_by_pld(subject_dir, gm_prob_map_patterns)

        if not pcasl_files or not vsasl_files or len(pcasl_files) != len(vsasl_files):
            tqdm.write(f"  [WARNING] Inconsistent or missing ASL files for {subject_id}. Skipping.")
            return
        
        # --- ADDED: A failsafe check; this should always pass now due to the updated validation script ---
        if not gm_prob_map_file_list:
            tqdm.write(f"  [ERROR] Gray Matter probability map not found for {subject_id} during processing. Skipping.")
            return

        subject_output_dir.mkdir(parents=True, exist_ok=True)

        if m0_file_list:
            tqdm.write("  --> Found M0 scan, creating robust brain mask.")
            m0_data = load_nifti_data(m0_file_list[0])
            threshold = np.percentile(m0_data[m0_data > 0], 50) * 0.5
            brain_mask = m0_data > threshold
        else:
            tqdm.write("  [WARNING] No M0 scan found. Masking quality may be reduced.")
            first_pcasl_data = load_nifti_data(pcasl_files[0])
            mean_signal = np.mean(np.abs(first_pcasl_data), axis=-1)
            threshold = np.percentile(mean_signal[mean_signal > 0], 95) * 0.2
            brain_mask = mean_signal > threshold
        
        if np.sum(brain_mask) == 0:
            tqdm.write(f"  [ERROR] Brain mask for {subject_id} is empty. Skipping.")
            return
            
        np.save(subject_output_dir / 'brain_mask.npy', brain_mask)
        tqdm.write(f"  --> Brain mask created with {np.sum(brain_mask)} voxels.")
        
        # --- ADDED: Generate and save the final Gray Matter mask ---
        tqdm.write("  --> Found GM probability map, creating final GM mask.")
        gm_prob_data = load_nifti_data(gm_prob_map_file_list[0])
        # Threshold at 0.75 as specified and combine with the main brain mask
        gm_mask = (gm_prob_data > 0.75) & brain_mask
        np.save(subject_output_dir / 'gm_mask.npy', gm_mask)
        tqdm.write(f"  --> Gray Matter mask created with {np.sum(gm_mask)} voxels.")
        # --- END OF ADDED CODE ---

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
        vsasl_high_flat = vsasl_high_snr_4d.reshape(-1, len(pcasl_files))
        
        low_snr_signals = np.concatenate([pcasl_low_flat, vsasl_low_flat], axis=1)
        high_snr_signals = np.concatenate([pcasl_high_flat, vsasl_high_flat], axis=1)

        # Create Z-coordinate map (Enhancement C)
        # Shape matches flattened brain mask. Contains slice index (0..Z-1)
        zs = np.indices(brain_mask.shape)[2]
        z_coords_masked = zs[brain_mask].astype(np.float32).reshape(-1, 1)
        np.save(subject_output_dir / 'z_coords.npy', z_coords_masked)

        first_img = nib.load(pcasl_files[0])
        subject_plds = [int(re.search(r'_(\d+)', p.name).group(1)) for p in pcasl_files]

        np.save(subject_output_dir / 'low_snr_signals.npy', low_snr_signals)
        np.save(subject_output_dir / 'high_snr_signals.npy', high_snr_signals)
        np.save(subject_output_dir / 'plds.npy', np.array(subject_plds))
        np.save(subject_output_dir / 'image_affine.npy', first_img.affine)
        np.save(subject_output_dir / 'image_header.npy', first_img.header)
        np.save(subject_output_dir / 'image_dims.npy', np.array(first_img.shape[:3]))
        tqdm.write(f"  --> Saved processed NumPy arrays to {subject_output_dir}")

    except Exception as e:
        tqdm.write(f"  [FATAL ERROR] processing subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_invivo_data.py <path_to_validated_data_folder> <output_preprocessed_folder>")
        sys.exit(1)

    root_data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    subject_dirs = sorted([d for d in root_data_dir.iterdir() if d.is_dir()])
    
    if not subject_dirs:
         print(f"[ERROR] No subject folders found in '{root_data_dir}'. Did the prepare script run correctly?")
         sys.exit(1)

    print(f"Found {len(subject_dirs)} valid subject folders to process.")
    
    for sub_dir in tqdm(subject_dirs, desc="Processing all subjects"):
        preprocess_subject(sub_dir, output_dir)
        
    print("\n--- In-vivo data preprocessing complete! ---")
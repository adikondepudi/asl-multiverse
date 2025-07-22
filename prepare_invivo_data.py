# prepare_invivo_data.py
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import re
from tqdm import tqdm
import shutil
import argparse
from typing import List

def find_and_sort_files_by_pld(subject_dir: Path, patterns: list) -> List[Path]:
    """
    Finds files matching a list of glob patterns and sorts them by the numeric
    PLD value found in their filenames.
    """
    def get_pld_from_path(path: Path) -> int:
        match = re.search(r'_(\d+)', path.name)
        return int(match.groups()[0]) if match else -1

    for pattern in patterns:
        files = list(subject_dir.rglob(pattern))
        if files:
            return sorted(files, key=get_pld_from_path)
    return []

def main(source_dir: Path, dest_dir: Path):
    """
    Scans a source directory of subject data, validates each subject, and copies
    only the valid ones to a clean destination directory.
    """
    if not source_dir.is_dir():
        print(f"[ERROR] Source directory not found: {source_dir}")
        sys.exit(1)

    # Create the destination directory, clearing it first if it exists
    if dest_dir.exists():
        print(f"[WARNING] Destination directory {dest_dir} already exists. It will be removed and recreated.")
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    print(f"Created clean destination directory: {dest_dir}")

    subject_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} total subjects to validate in {source_dir}.\n")
    
    passed_count = 0
    failed_count = 0

    for subject_dir in tqdm(subject_dirs, desc="Validating Subjects", unit="subject"):
        subject_id = subject_dir.name
        is_valid = True
        fail_reason = ""

        # --- CHECK 1: Ensure all required normdiff files exist and are consistent ---
        pcasl_files = find_and_sort_files_by_pld(subject_dir, ['r_normdiff_alldyn_PCASL_*.nii*'])
        vsasl_files = find_and_sort_files_by_pld(subject_dir, ['r_normdiff_alldyn_VSASL_*.nii*'])

        if not pcasl_files or not vsasl_files:
            is_valid = False
            fail_reason = "Missing required 'normdiff' PCASL or VSASL files."
        elif len(pcasl_files) != len(vsasl_files):
            is_valid = False
            fail_reason = f"Mismatched file counts: {len(pcasl_files)} PCASL vs {len(vsasl_files)} VSASL."
        
        if not is_valid:
            tqdm.write(f"  - [FAIL] {subject_id}: {fail_reason}")
            failed_count += 1
            continue

        # --- CHECK 2: Validate the M0 scan (if it exists) ---
        m0_files = find_and_sort_files_by_pld(subject_dir, ['r_M0.nii*', 'M0.nii*'])
        if m0_files:
            try:
                m0_img = nib.load(m0_files[0])
                m0_data = m0_img.get_fdata(dtype=np.float32)
                # A standard deviation near zero means the image is blank or constant-valued
                if np.std(m0_data) < 1e-6:
                    is_valid = False
                    fail_reason = "M0 scan exists but appears to be empty or zero-filled (std dev is ~0)."
            except Exception as e:
                is_valid = False
                fail_reason = f"Could not load or read M0 scan. Error: {e}"
        else:
            tqdm.write(f"  - [WARN] {subject_id}: No M0 scan found. Will rely on ASL data for masking.")

        if not is_valid:
            tqdm.write(f"  - [FAIL] {subject_id}: {fail_reason}")
            failed_count += 1
            continue

        # --- If all checks pass, copy the entire subject folder ---
        try:
            shutil.copytree(subject_dir, dest_dir / subject_id)
            tqdm.write(f"  - [PASS] {subject_id}: All checks passed. Copied to destination.")
            passed_count += 1
        except Exception as e:
            tqdm.write(f"  - [FAIL] {subject_id}: Could not copy directory. Error: {e}")
            failed_count += 1

    print("\n" + "="*50)
    print("      PRE-VALIDATION SUMMARY")
    print("="*50)
    print(f"✅ Copied {passed_count} valid subjects to: {dest_dir.resolve()}")
    print(f"❌ Skipped {failed_count} invalid subjects.")
    print("You can now safely use the destination folder as input for 'process_invivo_data.py'.")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Scans a source data directory, validates each subject, and copies only the valid subjects to a new clean directory for processing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "source_dir",
        type=str,
        help="The path to the source directory containing all raw subject folders (e.g., the original 18)."
    )
    parser.add_argument(
        "destination_dir",
        type=str,
        help="The path to the new, clean output directory where valid subjects will be copied (e.g., 'raw_invivo_data')."
    )
    args = parser.parse_args()
    main(Path(args.source_dir), Path(args.destination_dir))
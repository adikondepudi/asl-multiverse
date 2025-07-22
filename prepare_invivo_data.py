# adikondepudi-asl-multiverse/prepare_invivo_data.py

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
    Scans a source directory of subject data, validates each subject based on
    date and file integrity (including new masks), and copies only the valid
    ones to a clean destination directory.
    """
    if not source_dir.is_dir():
        print(f"[ERROR] Source directory not found: {source_dir}")
        sys.exit(1)

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

        # --- CHECK 1: Filter by Date (as per PI's instruction) ---
        match = re.search(r'^(\d{8})', subject_id)
        if match:
            subject_date = int(match.group(1))
            if subject_date < 20231001:
                is_valid = False
                fail_reason = "Data is from before October 2023."
        
        if not is_valid:
            tqdm.write(f"  - [SKIP] {subject_id}: {fail_reason}")
            failed_count += 1
            continue
            
        # --- CHECK 2: Ensure all required files exist, including new masks ---
        pcasl_files = find_and_sort_files_by_pld(subject_dir, ['r_normdiff_alldyn_PCASL_*.nii*'])
        vsasl_files = find_and_sort_files_by_pld(subject_dir, ['r_normdiff_alldyn_VSASL_*.nii*'])
        gm_mask_file = next(subject_dir.rglob('*GM_axial_philiport.nii*'), None)
        wb_mask_file = next(subject_dir.rglob('*M0_WBmask_by_mprage_tight.nii*'), None)

        if not pcasl_files or not vsasl_files:
            is_valid = False
            fail_reason = "Missing required 'normdiff' PCASL or VSASL files."
        elif len(pcasl_files) != len(vsasl_files):
            is_valid = False
            fail_reason = f"Mismatched file counts: {len(pcasl_files)} PCASL vs {len(vsasl_files)} VSASL."
        elif not gm_mask_file or not wb_mask_file:
            is_valid = False
            fail_reason = "Missing required gold-standard GM or WB mask files provided by PI."

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
    print("      DATA PRE-VALIDATION SUMMARY")
    print("="*50)
    print(f"✅ Copied {passed_count} valid subjects to: {dest_dir.resolve()}")
    print(f"❌ Skipped {failed_count} invalid or outdated subjects.")
    print("You can now safely use the destination folder as input for 'process_invivo_data.py'.")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Validates raw subject data folders based on date and file integrity (including new masks) and copies valid subjects to a clean directory for processing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("source_dir", type=str, help="The path to the source directory containing all raw subject folders.")
    parser.add_argument("destination_dir", type=str, help="The path to the new, clean output directory where valid subjects will be copied (e.g., 'data/raw_invivo_validated').")
    args = parser.parse_args()
    main(Path(args.source_dir), Path(args.destination_dir))
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
    def get_pld_from_path(path: Path) -> int:
        match = re.search(r'_(\d+)', path.name)
        return int(match.groups()[0]) if match else -1

    for pattern in patterns:
        files = list(subject_dir.rglob(pattern))
        if files:
            return sorted(files, key=get_pld_from_path)
    return []

def main(source_dir: Path, dest_dir: Path):
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

        pcasl_files = find_and_sort_files_by_pld(subject_dir, ['r_normdiff_alldyn_PCASL_*.nii*'])
        vsasl_files = find_and_sort_files_by_pld(subject_dir, ['r_normdiff_alldyn_VSASL_*.nii*'])
        # --- NEW: Also check for the essential GM probability map file ---
        gm_prob_map_files = find_and_sort_files_by_pld(subject_dir, ['cor2M0_mprage_GM_axial_philiport.nii*'])

        if not pcasl_files or not vsasl_files:
            is_valid = False
            fail_reason = "Missing required 'normdiff' PCASL or VSASL files."
        elif len(pcasl_files) != len(vsasl_files):
            is_valid = False
            fail_reason = f"Mismatched file counts: {len(pcasl_files)} PCASL vs {len(vsasl_files)} VSASL."
        # --- NEW: Add the check for the GM map ---
        elif not gm_prob_map_files:
            is_valid = False
            fail_reason = "Missing the required Gray Matter probability map ('cor2M0_mprage_GM_axial_philiport.nii.gz')."

        if not is_valid:
            tqdm.write(f"  - [FAIL] {subject_id}: {fail_reason}")
            failed_count += 1
            continue
        
        # --- M0 Scan check remains a warning, not a failure condition ---
        m0_files = find_and_sort_files_by_pld(subject_dir, ['r_M0.nii*', 'M0.nii*'])
        if not m0_files:
            tqdm.write(f"  - [WARN] {subject_id}: No M0 scan found. Will rely on ASL data for masking.")

        # --- If all critical checks pass, copy the folder ---
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
    parser.add_argument("source_dir", type=str, help="The path to the source directory containing all raw subject folders.")
    parser.add_argument("destination_dir", type=str, help="The path to the new, clean output directory where valid subjects will be copied.")
    args = parser.parse_args()
    main(Path(args.source_dir), Path(args.destination_dir))
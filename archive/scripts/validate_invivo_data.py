# validate_invivo_data.py
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import re
from tqdm import tqdm
from typing import List, Dict, Any

def find_files(subject_dir: Path, patterns: list) -> List[Path]:
    """Finds and sorts files by PLD based on a list of glob patterns."""
    def get_pld(path: Path) -> int:
        match = re.search(r'_(\d+)', path.name)
        return int(match.groups()[0]) if match else -1

    for pattern in patterns:
        files = sorted(list(subject_dir.glob(pattern)), key=get_pld)
        if files:
            return files
    return []

def load_and_analyze_nifti(file_path: Path) -> Dict[str, Any]:
    """
    Loads a NIfTI file, scales it, and returns its data along with a
    report on its numerical and structural properties.
    """
    report = {}
    try:
        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float64)
        
        report['shape'] = data.shape
        report['affine'] = img.affine
        
        scl_slope = img.header.get('scl_slope', 0)
        scl_inter = img.header.get('scl_inter', 0)

        if scl_slope != 0:
            data = data * scl_slope + scl_inter
        
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        report['nan_count'] = nan_count
        report['inf_count'] = inf_count
        
        # Clean data for further analysis
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        report['min_value'] = data.min()
        report['max_value'] = data.max()
        report['mean_value'] = data.mean()
        report['std_dev'] = data.std()
        report['data'] = data
        report['error'] = None

    except Exception as e:
        report['error'] = str(e)

    return report

def validate_subject(subject_dir: Path) -> Dict[str, Any]:
    """
    Performs a comprehensive validation of a single subject's data.
    """
    report = {'subject_id': subject_dir.name, 'checks': [], 'is_valid': True}

    # --- 1. File Discovery ---
    pcasl_patterns = ['rPCASL_*_aslrawimages.nii*', 'r_normdiff_alldyn_PCASL_*.nii*', 'r_PCASL_*.nii*']
    vsasl_patterns = ['rVSASL_*_aslrawimages.nii*', 'r_normdiff_alldyn_VSASL_*.nii*', 'r_VSASL_*.nii*']
    m0_patterns = ['r_M0.nii*', 'M0.nii*']
    
    pcasl_files = find_files(subject_dir, pcasl_patterns)
    vsasl_files = find_files(subject_dir, vsasl_patterns)
    m0_files = find_files(subject_dir, m0_patterns)
    
    # --- 2. Consistency Checks ---
    if not pcasl_files or not vsasl_files:
        report['is_valid'] = False
        report['checks'].append({'check': 'File Existence', 'status': 'FAIL', 'message': 'Missing PCASL or VSASL files.'})
        return report

    if len(pcasl_files) != len(vsasl_files):
        report['is_valid'] = False
        report['checks'].append({'check': 'File Count Consistency', 'status': 'FAIL', 
                                'message': f'Mismatched PLD counts: {len(pcasl_files)} PCASL vs {len(vsasl_files)} VSASL.'})
        return report
    
    report['checks'].append({'check': 'File Discovery', 'status': 'PASS', 
                            'message': f'Found {len(pcasl_files)} PCASL/VSASL pairs.'})

    # --- 3. Deep File Inspection ---
    all_files = pcasl_files + vsasl_files + m0_files
    file_reports = {f.name: load_and_analyze_nifti(f) for f in all_files}
    
    base_shape = file_reports[pcasl_files[0].name]['shape'][:3]
    base_affine = file_reports[pcasl_files[0].name]['affine']
    total_nan, total_inf = 0, 0

    for name, file_report in file_reports.items():
        if file_report['error']:
            report['is_valid'] = False
            report['checks'].append({'check': f'File Load: {name}', 'status': 'FAIL', 'message': file_report['error']})
            return report
        if file_report['shape'][:3] != base_shape:
            report['is_valid'] = False
            report['checks'].append({'check': 'Dimension Consistency', 'status': 'FAIL', 'message': f'File {name} has shape {file_report["shape"]} vs base {base_shape}.'})
        if not np.allclose(file_report['affine'], base_affine):
            report['checks'].append({'check': 'Affine Consistency', 'status': 'WARN', 'message': f'File {name} has a different affine matrix.'})
        total_nan += file_report['nan_count']
        total_inf += file_report['inf_count']

    report['checks'].append({'check': 'Structural Consistency', 'status': 'PASS', 'message': 'All files share same dimensions and affine.'})
    report['checks'].append({'check': 'Numerical Stability', 'status': 'PASS' if total_nan + total_inf == 0 else 'WARN', 
                             'message': f'Found {total_nan} NaNs and {total_inf} Infs across all files.'})

    # --- 4. Maskability Check ---
    if m0_files:
        m0_report = file_reports[m0_files[0].name]
        if m0_report['std_dev'] < 1e-6:
             report['is_valid'] = False
             report['checks'].append({'check': 'M0 Signal Presence', 'status': 'FAIL', 'message': 'M0 scan appears to be empty or zero-filled.'})
        else:
             report['checks'].append({'check': 'M0 Signal Presence', 'status': 'PASS', 'message': 'M0 scan contains a valid signal.'})
    else:
        report['checks'].append({'check': 'M0 Signal Presence', 'status': 'WARN', 'message': 'No M0 scan found. Masking will rely on less stable ASL data.'})

    return report

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python validate_invivo_data.py <path_to_raw_data_folder>")
        sys.exit(1)
    
    root_data_dir = Path(sys.argv[1])
    if not root_data_dir.is_dir():
        print(f"Error: Directory not found at '{root_data_dir}'")
        sys.exit(1)
        
    subject_dirs = sorted([d for d in root_data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} potential subject folders. Starting validation...\n")
    
    all_reports = [validate_subject(sub_dir) for sub_dir in tqdm(subject_dirs, desc="Validating Subjects")]
    
    # --- Generate Final Summary Report ---
    ready_subjects = []
    failed_subjects = []

    for report in all_reports:
        if report['is_valid']:
            ready_subjects.append(report['subject_id'])
        else:
            failed_subjects.append(report)
            
    print("\n" + "="*80)
    print(" " * 25 + "DATA VALIDATION SUMMARY REPORT")
    print("="*80)
    
    print(f"\n✅ PASSED ({len(ready_subjects)}/{len(all_reports)} subjects): These subjects appear clean and ready for preprocessing.")
    print(", ".join(ready_subjects))
    
    if failed_subjects:
        print(f"\n❌ FAILED ({len(failed_subjects)}/{len(all_reports)} subjects): These subjects have critical errors and will be skipped.")
        for report in failed_subjects:
            print(f"\n--- Subject: {report['subject_id']} ---")
            for check in report['checks']:
                if check['status'] != 'PASS':
                    print(f"  - [{check['status']}] {check['check']}: {check['message']}")
    
    print("\n" + "="*80)
    print("Validation complete.")

if __name__ == '__main__':
    main()
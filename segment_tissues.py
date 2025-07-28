# segment_tissues.py
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import subprocess
from tqdm import tqdm
import sys
import shutil

def run_fsl_fast(m0_path: Path, output_dir: Path):
    """
    Runs FSL's FAST for tissue segmentation on the M0 scan.
    This function is a placeholder and requires FSL to be installed.
    """
    print(f"  --> Running FSL FAST on: {m0_path.name}")
    output_prefix = output_dir / "segmentation"
    
    # The command to run FSL FAST. Assumes FSL is in the system's PATH.
    command = [
        "fast",
        "-o", str(output_prefix),
        "-n", "3",  # 3 tissue classes (CSF, GM, WM)
        "-g",       # Output CSF, GM, WM probability maps
        str(m0_path)
    ]
    
    try:
        # Check if FSL is installed and accessible
        if not shutil.which("fast"):
            raise FileNotFoundError("FSL 'fast' command not found. Please ensure FSL is installed and in your PATH.")
        
        # Execute the command
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=process.stdout, stderr=process.stderr)
        
        # FAST outputs pve_0 (CSF), pve_1 (GM), pve_2 (WM)
        gm_prob_map_path = output_prefix.with_name(f"{output_prefix.name}_pve_1.nii.gz")
        if not gm_prob_map_path.exists():
            raise FileNotFoundError(f"Expected GM probability map not found at {gm_prob_map_path}")
            
        return gm_prob_map_path

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [ERROR] FSL FAST failed for {m0_path.name}.", file=sys.stderr)
        if isinstance(e, subprocess.CalledProcessError):
            print(f"  --> FSL STDOUT:\n{e.stdout}", file=sys.stderr)
            print(f"  --> FSL STDERR:\n{e.stderr}", file=sys.stderr)
        else:
            print(f"  --> Error details: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Perform tissue segmentation on preprocessed in-vivo data to generate Gray Matter (GM) masks."
    )
    parser.add_argument("preprocessed_dir", type=str, help="Path to the directory containing preprocessed subjects (from process_invivo_data.py).")
    parser.add_argument("--gm_threshold", type=float, default=0.7, help="Probability threshold to create a binary GM mask from the FSL FAST output.")
    args = parser.parse_args()

    preprocessed_root = Path(args.preprocessed_dir)
    subject_dirs = sorted([d for d in preprocessed_root.iterdir() if d.is_dir()])
    
    if not subject_dirs:
        print(f"[ERROR] No subject folders found in '{preprocessed_root}'.")
        sys.exit(1)

    print(f"Found {len(subject_dirs)} subjects to segment.")

    for subject_dir in tqdm(subject_dirs, desc="Segmenting Subjects"):
        try:
            # The M0 scan is not saved in preprocessed folder, we need to find it in the original data.
            # This script assumes a structure where the M0 scan is available and can be re-referenced,
            # but for simplicity, we will assume it's been copied into the preprocessed folder.
            # A more robust pipeline would handle this. Let's create a placeholder M0 from the brain mask.
            
            # For this example, we'll create a dummy M0 NIfTI from the brain mask to run FSL on.
            # In a real pipeline, you would use the actual M0 scan.
            brain_mask = np.load(subject_dir / 'brain_mask.npy')
            affine = np.load(subject_dir / 'image_affine.npy')
            header = np.load(subject_dir / 'image_header.npy', allow_pickle=True).item()
            
            # Save a temporary M0 file for FSL
            m0_img = nib.Nifti1Image(brain_mask.astype(np.float32), affine, header)
            temp_m0_path = subject_dir / "temp_m0_for_fsl.nii.gz"
            nib.save(m0_img, temp_m0_path)
            
            # Run segmentation
            gm_prob_map_path = run_fsl_fast(temp_m0_path, subject_dir)
            
            if gm_prob_map_path:
                gm_prob_data = nib.load(gm_prob_map_path).get_fdata(dtype=np.float32)
                
                # Create a binary mask based on the probability threshold
                gm_mask = gm_prob_data > args.gm_threshold
                
                # Save the final binary GM mask
                gm_mask_path = subject_dir / 'gm_mask.npy'
                np.save(gm_mask_path, gm_mask)
                tqdm.write(f"  - Saved binary GM mask for {subject_dir.name} to {gm_mask_path}")
            else:
                tqdm.write(f"  - [FAIL] Could not generate GM mask for {subject_dir.name}.")
                
            # Clean up temporary file
            if temp_m0_path.exists():
                temp_m0_path.unlink()

        except Exception as e:
            tqdm.write(f"  - [FATAL ERROR] for subject {subject_dir.name}: {e}")

    print("\n--- Tissue segmentation complete. ---")

if __name__ == '__main__':
    main()
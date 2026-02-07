#!/usr/bin/env python3
"""
BASIL/Bayesian baseline for ASL parameter estimation.

Wraps FSL's oxford_asl / basil CLI tools to provide Bayesian inference
as a third baseline for comparison (NN vs NLLS vs BASIL).

Reference: Woods et al. (2024) MRM Consensus recommends Bayesian inference
for multi-timepoint ASL data.

Usage:
    from basil_baseline import check_basil_available, run_basil_fitting

    if check_basil_available():
        cbf_map, att_map = run_basil_fitting(signals, brain_mask, plds, output_dir)
"""

import numpy as np
import nibabel as nib
import subprocess
import shutil
import tempfile
import warnings
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


def check_basil_available() -> bool:
    """
    Check if FSL's oxford_asl (BASIL) is available on PATH.

    Returns:
        True if oxford_asl is found, False otherwise.
    """
    return shutil.which("oxford_asl") is not None


def _save_temp_nifti(data: np.ndarray, filepath: Path,
                     affine: Optional[np.ndarray] = None):
    """
    Save a numpy array as a NIfTI file.

    Args:
        data: Array to save.
        filepath: Output path.
        affine: Affine transform. Uses identity if None.
    """
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, str(filepath))


def _build_oxford_asl_cmd(input_nifti: Path, output_dir: Path,
                          mask_nifti: Path, plds: np.ndarray,
                          is_pcasl: bool = True,
                          bolus_duration: float = 1.8,
                          t1b: float = 1.65,
                          spatial: bool = True,
                          mc: bool = False) -> list:
    """
    Build the oxford_asl CLI command.

    Args:
        input_nifti: Path to 4D input NIfTI (difference images).
        output_dir: Output directory for BASIL results.
        mask_nifti: Path to brain mask NIfTI.
        plds: PLD values in seconds.
        is_pcasl: If True, use --casl flag (PCASL). If False, pulsed/VSASL.
        bolus_duration: Labeling duration in seconds.
        t1b: Arterial blood T1 in seconds (1.65s at 3T per consensus).
        spatial: Enable spatial regularization (--spatial).
        mc: Enable motion correction (--mc).

    Returns:
        List of command arguments.
    """
    # Format PLDs as comma-separated string
    pld_str = ",".join(f"{p:.4f}" for p in plds)

    cmd = [
        "oxford_asl",
        "-i", str(input_nifti),
        "-o", str(output_dir),
        "--iaf=diff",       # Input is already difference images
        "--ibf=tis",        # Input grouped by TIs/PLDs
        f"--plds={pld_str}",
        f"--bolus={bolus_duration}",
        f"--t1b={t1b}",
        "-m", str(mask_nifti),
    ]

    if is_pcasl:
        cmd.append("--casl")

    if spatial:
        cmd.append("--spatial")

    if mc:
        cmd.append("--mc")

    return cmd


def _find_basil_output(basil_output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate CBF and ATT output files in BASIL's output directory structure.

    BASIL/oxford_asl creates a nested directory structure. The perfusion
    (CBF) and arrival time (ATT) maps can be found in several locations
    depending on the version and options used.

    Args:
        basil_output_dir: The -o directory passed to oxford_asl.

    Returns:
        Tuple of (cbf_path, att_path), either may be None if not found.
    """
    # oxford_asl output structure (common locations):
    #   <output_dir>/native_space/perfusion.nii.gz          (CBF, relative)
    #   <output_dir>/native_space/perfusion_calib.nii.gz    (CBF, calibrated)
    #   <output_dir>/native_space/arrival.nii.gz            (ATT)
    #   <output_dir>/basil/step2/mean_ftiss.nii.gz          (CBF from BASIL step)
    #   <output_dir>/basil/step2/mean_delttiss.nii.gz       (ATT from BASIL step)

    cbf_path = None
    att_path = None

    # Priority order for CBF
    cbf_candidates = [
        basil_output_dir / "native_space" / "perfusion.nii.gz",
        basil_output_dir / "native_space" / "perfusion_calib.nii.gz",
        basil_output_dir / "basil" / "step2" / "mean_ftiss.nii.gz",
        basil_output_dir / "basil" / "step1" / "mean_ftiss.nii.gz",
    ]

    for candidate in cbf_candidates:
        if candidate.exists():
            cbf_path = candidate
            break

    # Priority order for ATT
    att_candidates = [
        basil_output_dir / "native_space" / "arrival.nii.gz",
        basil_output_dir / "basil" / "step2" / "mean_delttiss.nii.gz",
        basil_output_dir / "basil" / "step1" / "mean_delttiss.nii.gz",
    ]

    for candidate in att_candidates:
        if candidate.exists():
            att_path = candidate
            break

    return cbf_path, att_path


def run_basil_fitting(signals: np.ndarray, brain_mask: np.ndarray,
                      plds: np.ndarray, output_dir: Path,
                      params: Optional[Dict] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run BASIL (Bayesian) fitting on ASL signals.

    Takes the same combined PCASL+VSASL input format as run_ls_fitting.
    Since BASIL expects single-modality input, PCASL and VSASL are
    processed separately, then the results are averaged.

    Args:
        signals: (H, W, Z, 2*n_plds) - first n_plds are PCASL, last n_plds are VSASL.
        brain_mask: (H, W, Z) binary mask.
        plds: PLD values in milliseconds.
        output_dir: Directory to save BASIL outputs (for debugging/inspection).
        params: Optional dict with override parameters:
            - 'bolus_duration': Label duration in seconds (default 1.8).
            - 't1b': Blood T1 in seconds (default 1.65 at 3T).
            - 'spatial': Enable spatial regularization (default True).
            - 'mc': Enable motion correction (default False).
            - 'alpha_BS1': Background suppression per-pulse efficiency (default None, no correction).
            - 'cleanup_temp': Whether to remove temp files (default True).

    Returns:
        cbf_map: (H, W, Z) CBF in ml/100g/min.
        att_map: (H, W, Z) ATT in milliseconds.

    Raises:
        RuntimeError: If BASIL is not installed or fails to run.
    """
    if not check_basil_available():
        raise RuntimeError(
            "FSL oxford_asl (BASIL) is not available. "
            "Install FSL (https://fsl.fmrib.ox.ac.uk/fsl/) and ensure "
            "oxford_asl is on your PATH."
        )

    # Parse parameters
    if params is None:
        params = {}

    bolus_duration = params.get('bolus_duration', 1.8)
    t1b = params.get('t1b', 1.65)
    spatial = params.get('spatial', True)
    mc = params.get('mc', False)
    cleanup_temp = params.get('cleanup_temp', True)

    h, w, z = brain_mask.shape
    n_plds = len(plds)

    # Convert PLDs from ms to seconds for BASIL
    plds_sec = plds / 1000.0

    # Split PCASL and VSASL signals
    pcasl_signals = signals[..., :n_plds]   # (H, W, Z, n_plds)
    vsasl_signals = signals[..., n_plds:]   # (H, W, Z, n_plds)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use identity affine (signals are in native voxel space)
    affine = np.eye(4)

    # Initialize output maps
    cbf_map = np.full((h, w, z), np.nan, dtype=np.float32)
    att_map = np.full((h, w, z), np.nan, dtype=np.float32)

    # Track which modalities succeeded
    modality_results = {}

    start_time = time.time()

    for modality_name, modality_signals, is_pcasl in [
        ("pcasl", pcasl_signals, True),
        ("vsasl", vsasl_signals, False),
    ]:
        print(f"  Running BASIL for {modality_name.upper()}...")

        # Create temporary directory for this modality
        temp_dir = tempfile.mkdtemp(prefix=f"basil_{modality_name}_")
        temp_dir = Path(temp_dir)

        try:
            # Save input as 4D NIfTI: (H, W, Z, n_plds)
            input_nifti = temp_dir / "input.nii.gz"
            _save_temp_nifti(modality_signals, input_nifti, affine)

            # Save brain mask
            mask_nifti = temp_dir / "mask.nii.gz"
            _save_temp_nifti(brain_mask.astype(np.float32), mask_nifti, affine)

            # BASIL output directory
            basil_out = temp_dir / "basil_output"

            # Build and run command
            cmd = _build_oxford_asl_cmd(
                input_nifti=input_nifti,
                output_dir=basil_out,
                mask_nifti=mask_nifti,
                plds=plds_sec,
                is_pcasl=is_pcasl,
                bolus_duration=bolus_duration,
                t1b=t1b,
                spatial=spatial,
                mc=mc,
            )

            print(f"    Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                warnings.warn(
                    f"BASIL {modality_name.upper()} failed (exit code {result.returncode}).\n"
                    f"stderr: {result.stderr[:500]}"
                )
                continue

            # Find and load output maps
            cbf_path, att_path = _find_basil_output(basil_out)

            if cbf_path is None:
                warnings.warn(
                    f"BASIL {modality_name.upper()}: CBF output not found in {basil_out}. "
                    f"Directory contents: {list(basil_out.rglob('*.nii.gz'))}"
                )
                continue

            # Load results
            mod_cbf = nib.load(str(cbf_path)).get_fdata()

            mod_att = None
            if att_path is not None:
                mod_att = nib.load(str(att_path)).get_fdata()

            modality_results[modality_name] = {
                'cbf': mod_cbf,
                'att': mod_att,
            }

            print(f"    {modality_name.upper()} BASIL completed successfully.")

            # Copy outputs to persistent output directory for inspection
            modality_out = output_dir / f"basil_{modality_name}"
            if basil_out.exists():
                shutil.copytree(basil_out, modality_out, dirs_exist_ok=True)

        except subprocess.TimeoutExpired:
            warnings.warn(f"BASIL {modality_name.upper()} timed out after 600s.")
            continue

        except Exception as e:
            warnings.warn(f"BASIL {modality_name.upper()} error: {e}")
            continue

        finally:
            # Clean up temp directory
            if cleanup_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - start_time

    # Combine modality results
    if not modality_results:
        warnings.warn("BASIL failed for all modalities. Returning NaN maps.")
        return cbf_map, att_map

    # Average CBF and ATT across available modalities
    cbf_maps = []
    att_maps = []

    for mod_name, mod_data in modality_results.items():
        if mod_data['cbf'] is not None:
            mod_cbf = mod_data['cbf']
            # BASIL outputs CBF in ml/100g/min already (when uncalibrated,
            # it outputs in arbitrary units proportional to perfusion).
            # The BASIL perfusion output (mean_ftiss) is in units of
            # perfusion rate. For uncalibrated data, the scale depends
            # on M0. We keep it as-is since our input signals are
            # already normalized difference images.
            #
            # Convert BASIL's native perfusion units to ml/100g/min:
            # BASIL ftiss is in s^-1, need to multiply by 6000 to get
            # ml/100g/min (same conversion as LS fitting).
            cbf_converted = mod_cbf * 6000.0
            cbf_maps.append(cbf_converted)

        if mod_data['att'] is not None:
            mod_att = mod_data['att']
            # BASIL outputs ATT in seconds; convert to ms
            att_converted = mod_att * 1000.0
            att_maps.append(att_converted)

    if cbf_maps:
        # Average across modalities, ignoring NaN
        cbf_stack = np.stack(cbf_maps, axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cbf_map = np.nanmean(cbf_stack, axis=-1).astype(np.float32)
        # Apply brain mask
        cbf_map[~brain_mask] = np.nan

    if att_maps:
        att_stack = np.stack(att_maps, axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            att_map = np.nanmean(att_stack, axis=-1).astype(np.float32)
        att_map[~brain_mask] = np.nan

    n_voxels = brain_mask.sum()
    print(f"  BASIL fitting completed in {elapsed:.1f}s "
          f"({elapsed/max(n_voxels,1)*1000:.2f}ms/voxel)")
    print(f"  Modalities completed: {list(modality_results.keys())}")

    return cbf_map, att_map


def run_basil_fitting_safe(signals: np.ndarray, brain_mask: np.ndarray,
                           plds: np.ndarray, output_dir: Path,
                           params: Optional[Dict] = None
                           ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Safe wrapper around run_basil_fitting that returns None if BASIL
    is unavailable or fails, instead of raising an exception.

    Args:
        Same as run_basil_fitting.

    Returns:
        Tuple of (cbf_map, att_map) if successful, None otherwise.
    """
    if not check_basil_available():
        warnings.warn(
            "FSL oxford_asl (BASIL) is not installed. Skipping BASIL fitting. "
            "Install FSL to enable Bayesian baseline comparison."
        )
        return None

    try:
        return run_basil_fitting(signals, brain_mask, plds, output_dir, params)
    except Exception as e:
        warnings.warn(f"BASIL fitting failed: {e}")
        return None

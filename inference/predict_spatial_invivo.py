# predict_spatial_invivo.py
"""
Spatial inference on in vivo ASL data using SpatialASLNet (U-Net).

CRITICAL: Uses global_scale normalization to match training.
DO NOT use per-pixel z-score normalization - it destroys CBF information!

Usage:
    python predict_spatial_invivo.py <invivo_data_dir> <model_dir> <output_dir>

Example:
    python predict_spatial_invivo.py Multiverse production_model_v1 invivo_results
"""

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import argparse
import re
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import yaml
from spatial_asl_network import SpatialASLNet
from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet


def find_and_sort_files_by_pld(subject_dir: Path, pattern: str) -> List[Path]:
    """Find files matching pattern and sort by PLD value."""
    def get_pld(path: Path) -> int:
        match = re.search(r'_(\d+)', path.name)
        return int(match.group(1)) if match else -1

    files = list(subject_dir.glob(pattern))
    return sorted(files, key=get_pld)


def load_nifti(file_path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI file and return data with NaNs replaced."""
    img = nib.load(file_path)
    data = img.get_fdata(dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, img


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad spatial dimensions to be divisible by multiple."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(tensor: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    """Remove padding from tensor."""
    pad_top, pad_bottom, pad_left, pad_right = padding
    _, _, h, w = tensor.shape
    return tensor[:, :,
                  pad_top:h-pad_bottom if pad_bottom else h,
                  pad_left:w-pad_right if pad_right else w]


def load_spatial_model(model_dir: Path, device: torch.device) -> Tuple[List[torch.nn.Module], Dict, Dict]:
    """Load SpatialASLNet or AmplitudeAwareSpatialASLNet ensemble from model directory."""
    print(f"Loading model from: {model_dir}")

    # Load config and norm stats
    with open(model_dir / 'research_config.json', 'r') as f:
        config = json.load(f)
    with open(model_dir / 'norm_stats.json', 'r') as f:
        norm_stats = json.load(f)

    # Load training config to determine model class
    config_yaml_path = model_dir / 'config.yaml'
    training_config = {}
    if config_yaml_path.exists():
        with open(config_yaml_path, 'r') as f:
            full_config = yaml.safe_load(f)
            training_config = full_config.get('training', {})

    model_class_name = training_config.get('model_class_name', 'SpatialASLNet')

    # Determine number of PLDs from config
    n_plds = len(config['pld_values'])
    print(f"  Model class: {model_class_name}")
    print(f"  Model expects {n_plds} PLDs: {config['pld_values']}")

    # Load ensemble models
    models = []
    models_dir = model_dir / 'trained_models'

    for model_path in sorted(models_dir.glob('ensemble_model_*.pt')):
        # Load state dict first to check for FiLM keys
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in state_dict:
            sd = state_dict['model_state_dict']
        else:
            sd = state_dict

        # Determine model architecture from checkpoint
        if model_class_name == 'AmplitudeAwareSpatialASLNet':
            has_film_keys = any('film' in k for k in sd.keys())
            features = training_config.get('hidden_sizes', [32, 64, 128, 256])

            if has_film_keys:
                # Full architecture (how models were actually trained due to bug)
                model = AmplitudeAwareSpatialASLNet(
                    n_plds=n_plds,
                    features=features,
                    use_film_at_bottleneck=True,
                    use_film_at_decoder=True,
                    use_amplitude_output_modulation=True,
                )
            else:
                model = AmplitudeAwareSpatialASLNet(
                    n_plds=n_plds,
                    features=features,
                    use_film_at_bottleneck=training_config.get('use_film_at_bottleneck', True),
                    use_film_at_decoder=training_config.get('use_film_at_decoder', True),
                    use_amplitude_output_modulation=training_config.get('use_amplitude_output_modulation', True),
                )
        else:
            model = SpatialASLNet(n_plds=n_plds)

        # Load weights
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        models.append(model)
        print(f"  Loaded: {model_path.name}")

    print(f"  Total ensemble members: {len(models)}")
    return models, config, norm_stats


def preprocess_subject(subject_dir: Path, model_plds: List[int], global_scale: float = 10.0) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image, np.ndarray]:
    """
    Preprocess a single subject's data for spatial model inference.

    CRITICAL: Uses global_scale normalization (NOT z-score) to preserve CBF amplitude!

    Returns:
        spatial_stack: (Z, 2*n_plds, H, W) tensor ready for model
        brain_mask: (H, W, Z) brain mask
        reference_img: NIfTI image for saving outputs
        subject_plds: PLDs found in subject data
    """
    # Find PCASL and VSASL files
    pcasl_files = find_and_sort_files_by_pld(subject_dir, 'r_normdiff_alldyn_PCASL_*.nii*')
    vsasl_files = find_and_sort_files_by_pld(subject_dir, 'r_normdiff_alldyn_VSASL_*.nii*')

    if not pcasl_files or not vsasl_files:
        raise ValueError(f"Missing PCASL or VSASL files in {subject_dir}")

    # Extract PLDs from filenames for both modalities
    pcasl_plds = [int(re.search(r'_(\d+)', f.name).group(1)) for f in pcasl_files]
    vsasl_plds = [int(re.search(r'_(\d+)', f.name).group(1)) for f in vsasl_files]

    # Use only PLDs that exist in BOTH PCASL and VSASL
    common_plds = sorted(set(pcasl_plds) & set(vsasl_plds))
    if len(common_plds) < len(pcasl_plds) or len(common_plds) < len(vsasl_plds):
        print(f"  WARNING: PCASL has {pcasl_plds}, VSASL has {vsasl_plds}")
        print(f"  Using common PLDs: {common_plds}")
        # Filter to only common PLDs
        pcasl_files = [f for f in pcasl_files if int(re.search(r'_(\d+)', f.name).group(1)) in common_plds]
        vsasl_files = [f for f in vsasl_files if int(re.search(r'_(\d+)', f.name).group(1)) in common_plds]

    subject_plds = common_plds
    print(f"  Found PLDs: {subject_plds}")

    # Load reference image for affine/header
    _, ref_img = load_nifti(pcasl_files[0])

    # Load M0 for brain mask
    m0_files = list(subject_dir.glob('r_M0.nii*'))
    if m0_files:
        m0_data, _ = load_nifti(m0_files[0])
        threshold = np.percentile(m0_data[m0_data > 0], 50) * 0.3
        brain_mask = m0_data > threshold
    else:
        # Fallback: use signal intensity
        first_pcasl, _ = load_nifti(pcasl_files[0])
        mean_signal = np.mean(np.abs(first_pcasl), axis=-1)
        threshold = np.percentile(mean_signal[mean_signal > 0], 90) * 0.1
        brain_mask = mean_signal > threshold

    # Load and stack ASL data
    # Data is already M0-normalized (normdiff), average across repeats
    pcasl_volumes = []
    for f in pcasl_files:
        data, _ = load_nifti(f)
        if data.ndim == 4:  # Has repeats dimension
            data = np.mean(data, axis=-1)  # Average repeats
        pcasl_volumes.append(data)

    vsasl_volumes = []
    for f in vsasl_files:
        data, _ = load_nifti(f)
        if data.ndim == 4:
            data = np.mean(data, axis=-1)
        vsasl_volumes.append(data)

    # Stack: (H, W, Z, n_plds)
    pcasl_stack = np.stack(pcasl_volumes, axis=-1)
    vsasl_stack = np.stack(vsasl_volumes, axis=-1)

    # Handle PLD mismatch - insert zeros for missing PLDs
    n_model_plds = len(model_plds)
    n_subject_plds = len(subject_plds)

    if set(subject_plds) != set(model_plds):
        print(f"  WARNING: PLD mismatch - model expects {model_plds}, subject has {subject_plds}")

        # Create full-size arrays with zeros
        h, w, z = pcasl_stack.shape[:3]
        pcasl_full = np.zeros((h, w, z, n_model_plds), dtype=np.float32)
        vsasl_full = np.zeros((h, w, z, n_model_plds), dtype=np.float32)

        # Map subject PLDs to model PLDs
        for i, pld in enumerate(subject_plds):
            if pld in model_plds:
                idx = model_plds.index(pld)
                pcasl_full[..., idx] = pcasl_stack[..., i]
                vsasl_full[..., idx] = vsasl_stack[..., i]
            else:
                print(f"    Dropping PLD {pld} (not in model)")

        pcasl_stack = pcasl_full
        vsasl_stack = vsasl_full

    # Concatenate PCASL + VSASL: (H, W, Z, 2*n_plds)
    combined = np.concatenate([pcasl_stack, vsasl_stack], axis=-1)

    # Transpose to (Z, 2*n_plds, H, W) for PyTorch
    spatial_stack = np.transpose(combined, (2, 3, 0, 1))

    # CRITICAL: Apply BOTH M0 scaling and global_scale to match training!
    #
    # Training pipeline (SpatialDataset):
    #   1. Generate signals (~0.01)
    #   2. M0_SCALE_FACTOR = 100 (signals *= 100)  <- THIS WAS MISSING!
    #   3. global_scale_factor = 10 (signals *= 10)
    #   Final training range: ~1-50
    #
    # In-vivo data is already M0-normalized (normdiff ~0.001-0.01)
    # We need to apply BOTH scaling factors to match training:
    M0_SCALE_FACTOR = 100.0  # Must match SpatialDataset.M0_SCALE_FACTOR
    spatial_stack = spatial_stack * M0_SCALE_FACTOR * global_scale

    print(f"  Preprocessed shape: {spatial_stack.shape}")
    print(f"  Signal range after scaling: [{spatial_stack.min():.4f}, {spatial_stack.max():.4f}]")

    return spatial_stack.astype(np.float32), brain_mask, ref_img, np.array(subject_plds)


def predict_volume(spatial_stack: np.ndarray, models: List[torch.nn.Module],
                   norm_stats: Dict, device: torch.device,
                   batch_size: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run ensemble inference on full volume.

    Args:
        spatial_stack: (Z, 2*n_plds, H, W) preprocessed input
        models: List of SpatialASLNet models
        norm_stats: Normalization statistics for denormalization
        device: torch device
        batch_size: Slices per batch

    Returns:
        cbf_volume: (H, W, Z) CBF in ml/100g/min
        att_volume: (H, W, Z) ATT in ms
        cbf_std: (H, W, Z) CBF uncertainty (ensemble std)
        att_std: (H, W, Z) ATT uncertainty (ensemble std)
    """
    n_slices, n_channels, h, w = spatial_stack.shape

    # Determine padding
    sample = torch.from_numpy(spatial_stack[0:1]).float()
    _, padding = pad_to_multiple(sample, 16)

    # Storage for all ensemble predictions
    all_cbf = []
    all_att = []

    for model in models:
        model.eval()
        cbf_slices = []
        att_slices = []

        for start_idx in range(0, n_slices, batch_size):
            end_idx = min(start_idx + batch_size, n_slices)
            batch = spatial_stack[start_idx:end_idx]

            # Convert to tensor and pad
            batch_tensor = torch.from_numpy(batch).float().to(device)
            batch_padded, _ = pad_to_multiple(batch_tensor, 16)

            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu',
                                       dtype=torch.float16 if device.type == 'cuda' else torch.float32):
                    cbf_norm, att_norm, _, _ = model(batch_padded)

            # Unpad
            cbf_norm = unpad(cbf_norm, padding)
            att_norm = unpad(att_norm, padding)

            cbf_slices.append(cbf_norm.cpu().numpy())
            att_slices.append(att_norm.cpu().numpy())

        # Stack slices: (Z, 1, H, W)
        cbf_vol = np.concatenate(cbf_slices, axis=0)
        att_vol = np.concatenate(att_slices, axis=0)

        all_cbf.append(cbf_vol)
        all_att.append(att_vol)

    # Ensemble: average predictions
    cbf_ensemble = np.mean(all_cbf, axis=0)  # (Z, 1, H, W)
    att_ensemble = np.mean(all_att, axis=0)

    # Uncertainty: standard deviation across ensemble
    cbf_std_ensemble = np.std(all_cbf, axis=0)
    att_std_ensemble = np.std(all_att, axis=0)

    # Denormalize: model outputs normalized z-scores
    y_mean_cbf = norm_stats['y_mean_cbf']
    y_std_cbf = norm_stats['y_std_cbf']
    y_mean_att = norm_stats['y_mean_att']
    y_std_att = norm_stats['y_std_att']

    cbf_denorm = cbf_ensemble * y_std_cbf + y_mean_cbf
    att_denorm = att_ensemble * y_std_att + y_mean_att

    # Denormalize uncertainty
    cbf_std_denorm = cbf_std_ensemble * y_std_cbf
    att_std_denorm = att_std_ensemble * y_std_att

    # Apply physical constraints
    cbf_denorm = np.clip(cbf_denorm, 0, 200)
    att_denorm = np.clip(att_denorm, 0, 5000)

    # Transpose to (H, W, Z) for NIfTI
    cbf_volume = np.transpose(cbf_denorm[:, 0, :, :], (1, 2, 0))
    att_volume = np.transpose(att_denorm[:, 0, :, :], (1, 2, 0))
    cbf_std_vol = np.transpose(cbf_std_denorm[:, 0, :, :], (1, 2, 0))
    att_std_vol = np.transpose(att_std_denorm[:, 0, :, :], (1, 2, 0))

    return cbf_volume, att_volume, cbf_std_vol, att_std_vol


def save_nifti(data: np.ndarray, reference: nib.Nifti1Image, output_path: Path):
    """Save array as NIfTI using reference image's affine/header."""
    img = nib.Nifti1Image(data.astype(np.float32), reference.affine, reference.header)
    nib.save(img, output_path)


def process_subject(subject_dir: Path, models: List[torch.nn.Module],
                    config: Dict, norm_stats: Dict, device: torch.device,
                    output_dir: Path):
    """Process a single subject."""
    subject_id = subject_dir.name
    subject_output = output_dir / subject_id
    subject_output.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {subject_id}")

    # Get global scale from config
    global_scale = config.get('global_scale_factor', 10.0)
    model_plds = config['pld_values']

    # Preprocess
    spatial_stack, brain_mask, ref_img, subject_plds = preprocess_subject(
        subject_dir, model_plds, global_scale
    )

    # Run inference
    cbf, att, cbf_std, att_std = predict_volume(
        spatial_stack, models, norm_stats, device
    )

    # Apply brain mask
    cbf_masked = cbf * brain_mask
    att_masked = att * brain_mask

    # Save outputs
    save_nifti(cbf_masked, ref_img, subject_output / 'nn_cbf.nii.gz')
    save_nifti(att_masked, ref_img, subject_output / 'nn_att.nii.gz')
    save_nifti(cbf_std, ref_img, subject_output / 'nn_cbf_uncertainty.nii.gz')
    save_nifti(att_std, ref_img, subject_output / 'nn_att_uncertainty.nii.gz')
    save_nifti(brain_mask.astype(np.float32), ref_img, subject_output / 'brain_mask.nii.gz')

    # Save metadata
    metadata = {
        'subject_plds': subject_plds.tolist(),
        'model_plds': model_plds,
        'global_scale': global_scale,
        'cbf_stats': {
            'mean': float(cbf_masked[brain_mask].mean()),
            'std': float(cbf_masked[brain_mask].std()),
            'min': float(cbf_masked[brain_mask].min()),
            'max': float(cbf_masked[brain_mask].max()),
        },
        'att_stats': {
            'mean': float(att_masked[brain_mask].mean()),
            'std': float(att_masked[brain_mask].std()),
            'min': float(att_masked[brain_mask].min()),
            'max': float(att_masked[brain_mask].max()),
        }
    }
    with open(subject_output / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  CBF: {metadata['cbf_stats']['mean']:.1f} ± {metadata['cbf_stats']['std']:.1f} ml/100g/min")
    print(f"  ATT: {metadata['att_stats']['mean']:.0f} ± {metadata['att_stats']['std']:.0f} ms")
    print(f"  Saved to: {subject_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Run spatial SpatialASLNet inference on in vivo ASL data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("invivo_dir", type=str,
                        help="Directory containing subject folders with ASL data")
    parser.add_argument("model_dir", type=str,
                        help="Directory containing trained model (e.g., production_model_v1)")
    parser.add_argument("output_dir", type=str,
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'mps', 'cpu', or 'auto'")
    parser.add_argument("--subjects", type=str, nargs='+', default=None,
                        help="Specific subjects to process (default: all)")

    args = parser.parse_args()

    invivo_dir = Path(args.invivo_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    models, config, norm_stats = load_spatial_model(model_dir, device)

    # Find subjects
    subject_dirs = sorted([d for d in invivo_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

    if args.subjects:
        subject_dirs = [d for d in subject_dirs if d.name in args.subjects]

    print(f"\nFound {len(subject_dirs)} subjects to process")

    # Process each subject
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        try:
            process_subject(subject_dir, models, config, norm_stats, device, output_dir)
        except Exception as e:
            print(f"  ERROR processing {subject_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Complete! Results saved to: {output_dir} ===")


if __name__ == '__main__':
    main()

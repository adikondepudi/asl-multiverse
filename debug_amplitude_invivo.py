#!/usr/bin/env python3
"""
Debug script to investigate why AmplitudeAwareSpatialASLNet fails on in-vivo data.

Compares:
1. Input signal characteristics (synthetic vs in-vivo)
2. Amplitude extractor outputs
3. Intermediate feature activations
4. Output modulation behavior
"""

import torch
import numpy as np
import json
import yaml
from pathlib import Path
import nibabel as nib
import re

from spatial_asl_network import SpatialASLNet
from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet

def load_invivo_slice(subject_dir: Path, global_scale: float = 10.0, target_plds: int = 6):
    """Load a single slice of in-vivo data, padding to match model PLDs."""

    def find_files(pattern):
        files = list(subject_dir.glob(pattern))
        def get_pld(p):
            m = re.search(r'_(\d+)', p.name)
            return int(m.group(1)) if m else -1
        return sorted(files, key=get_pld)

    pcasl_files = find_files('r_normdiff_alldyn_PCASL_*.nii*')
    vsasl_files = find_files('r_normdiff_alldyn_VSASL_*.nii*')

    # Load and average
    pcasl_vols = []
    for f in pcasl_files:
        data = nib.load(f).get_fdata()
        data = np.nan_to_num(data, nan=0.0)  # Replace NaN
        if data.ndim == 4:
            data = np.mean(data, axis=-1)
        pcasl_vols.append(data)

    vsasl_vols = []
    for f in vsasl_files:
        data = nib.load(f).get_fdata()
        data = np.nan_to_num(data, nan=0.0)  # Replace NaN
        if data.ndim == 4:
            data = np.mean(data, axis=-1)
        vsasl_vols.append(data)

    # Stack: (H, W, Z, n_plds)
    pcasl = np.stack(pcasl_vols, axis=-1)
    vsasl = np.stack(vsasl_vols, axis=-1)

    n_subject_plds = pcasl.shape[-1]
    print(f"  Subject has {n_subject_plds} PLDs, model expects {target_plds}")

    # Pad to target PLDs if needed
    if n_subject_plds < target_plds:
        h, w, z = pcasl.shape[:3]
        pcasl_padded = np.zeros((h, w, z, target_plds), dtype=np.float32)
        vsasl_padded = np.zeros((h, w, z, target_plds), dtype=np.float32)
        pcasl_padded[..., :n_subject_plds] = pcasl
        vsasl_padded[..., :n_subject_plds] = vsasl
        pcasl = pcasl_padded
        vsasl = vsasl_padded

    # Get middle slice
    mid_z = pcasl.shape[2] // 2
    pcasl_slice = pcasl[:, :, mid_z, :]  # (H, W, n_plds)
    vsasl_slice = vsasl[:, :, mid_z, :]

    # Combine and transpose to (1, 2*n_plds, H, W)
    combined = np.concatenate([pcasl_slice, vsasl_slice], axis=-1)
    combined = np.transpose(combined, (2, 0, 1))[np.newaxis, ...]  # (1, C, H, W)

    # Replace any remaining NaN
    combined = np.nan_to_num(combined, nan=0.0)

    # Apply global scale
    combined = combined * global_scale

    return combined.astype(np.float32), mid_z, n_subject_plds


def generate_synthetic_sample(n_plds=6):
    """Generate a synthetic sample similar to training data."""
    from enhanced_simulation import SpatialPhantomGenerator, RealisticASLSimulator
    from asl_simulation import ASLParameters

    params = ASLParameters()
    simulator = RealisticASLSimulator(params=params)
    phantom_gen = SpatialPhantomGenerator(size=64)

    plds = np.array([500, 1000, 1500, 2000, 2500, 3000])

    # Generate phantom
    cbf_map, att_map, _ = phantom_gen.generate_phantom()

    # Generate signals
    signals = np.zeros((n_plds * 2, 64, 64), dtype=np.float32)
    for i in range(64):
        for j in range(64):
            if cbf_map[i, j] > 0:
                cbf, att = cbf_map[i, j], att_map[i, j]
                pcasl = simulator._generate_pcasl_signal(plds, att, cbf, params.T1_artery, params.T_tau, params.alpha_PCASL)
                vsasl = simulator._generate_vsasl_signal(plds, att, cbf, params.T1_artery, params.alpha_VSASL)
                signals[:n_plds, i, j] = pcasl
                signals[n_plds:, i, j] = vsasl

    # Apply same scaling as training: *100 (M0) * 10 (global_scale)
    signals = signals * 100.0 * 10.0

    return signals[np.newaxis, ...], cbf_map, att_map


def analyze_amplitude_extractor(model, input_tensor):
    """Extract and analyze amplitude features."""
    model.eval()

    with torch.no_grad():
        # Get raw amplitude features (before MLP)
        n_channels = model.in_channels

        # Per-channel statistics
        channel_means = input_tensor.mean(dim=(2, 3))  # (B, C)
        channel_stds = input_tensor.std(dim=(2, 3))
        channel_maxs = input_tensor.amax(dim=(2, 3))

        # Global features
        global_power = (input_tensor ** 2).mean(dim=(1, 2, 3))

        # PCASL vs VSASL
        pcasl_mean = input_tensor[:, :n_channels//2].mean()
        vsasl_mean = input_tensor[:, n_channels//2:].mean()
        ratio = pcasl_mean / (vsasl_mean + 1e-9)

        return {
            'channel_means': channel_means.numpy(),
            'channel_stds': channel_stds.numpy(),
            'channel_maxs': channel_maxs.numpy(),
            'global_power': global_power.item(),
            'pcasl_mean': pcasl_mean.item(),
            'vsasl_mean': vsasl_mean.item(),
            'pcasl_vsasl_ratio': ratio.item(),
        }


def run_model_debug(model, input_tensor, model_name):
    """Run model and capture intermediate values."""
    model.eval()

    # Hook to capture intermediate activations
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
        return hook

    # Register hooks
    hooks = []
    if hasattr(model, 'amplitude_extractor'):
        hooks.append(model.amplitude_extractor.register_forward_hook(hook_fn('amplitude_extractor')))
    if hasattr(model, 'bottleneck_film'):
        hooks.append(model.bottleneck_film.register_forward_hook(hook_fn('bottleneck_film')))
    if hasattr(model, 'cbf_amplitude_correction'):
        hooks.append(model.cbf_amplitude_correction.register_forward_hook(hook_fn('cbf_amplitude_correction')))
    if hasattr(model, 'spatial_head'):
        hooks.append(model.spatial_head.register_forward_hook(hook_fn('spatial_head')))

    with torch.no_grad():
        output = model(input_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Extract outputs
    if isinstance(output, tuple):
        cbf, att = output[0], output[1]
    else:
        cbf = output[:, 0:1]
        att = output[:, 1:2]

    results = {
        'cbf_mean': cbf.mean().item(),
        'cbf_std': cbf.std().item(),
        'cbf_min': cbf.min().item(),
        'cbf_max': cbf.max().item(),
        'att_mean': att.mean().item(),
        'att_std': att.std().item(),
    }

    # Add activation stats
    for name, act in activations.items():
        results[f'{name}_mean'] = act.mean().item()
        results[f'{name}_std'] = act.std().item()
        results[f'{name}_shape'] = list(act.shape)

    return results


def main():
    print("=" * 70)
    print("DEBUGGING AMPLITUDE-AWARE MODEL ON IN-VIVO DATA")
    print("=" * 70)

    # Load models
    model_dir_amp = Path('amplitude_ablation_v1/02_AmpAware_Full')
    model_dir_base = Path('amplitude_ablation_v1/00_Baseline_SpatialASL')

    with open(model_dir_amp / 'config.yaml') as f:
        amp_config = yaml.safe_load(f)

    # Load AmplitudeAware model
    amp_model = AmplitudeAwareSpatialASLNet(
        n_plds=6,
        features=[32, 64, 128, 256],
        use_film_at_bottleneck=True,
        use_film_at_decoder=True,
        use_amplitude_output_modulation=True,
    )
    state = torch.load(model_dir_amp / 'trained_models/ensemble_model_0.pt', map_location='cpu', weights_only=False)
    amp_model.load_state_dict(state)
    amp_model.eval()

    # Load baseline model
    base_model = SpatialASLNet(n_plds=6)
    state = torch.load(model_dir_base / 'trained_models/ensemble_model_0.pt', map_location='cpu', weights_only=False)
    base_model.load_state_dict(state)
    base_model.eval()

    # Load norm stats
    with open(model_dir_amp / 'norm_stats.json') as f:
        norm_stats = json.load(f)

    print(f"\nNorm stats: CBF mean={norm_stats['y_mean_cbf']:.2f}, std={norm_stats['y_std_cbf']:.2f}")
    print(f"            ATT mean={norm_stats['y_mean_att']:.2f}, std={norm_stats['y_std_att']:.2f}")

    # === 1. SYNTHETIC DATA ===
    print("\n" + "=" * 70)
    print("1. SYNTHETIC DATA ANALYSIS")
    print("=" * 70)

    synth_input, true_cbf, true_att = generate_synthetic_sample()
    synth_tensor = torch.from_numpy(synth_input).float()

    print(f"\nSynthetic input shape: {synth_input.shape}")
    print(f"Synthetic input range: [{synth_input.min():.4f}, {synth_input.max():.4f}]")
    print(f"Synthetic input mean: {synth_input.mean():.4f}")
    print(f"Synthetic input std: {synth_input.std():.4f}")
    print(f"True CBF range: [{true_cbf.min():.1f}, {true_cbf.max():.1f}]")

    synth_amp_features = analyze_amplitude_extractor(amp_model, synth_tensor)
    print(f"\nSynthetic amplitude features:")
    print(f"  Global power: {synth_amp_features['global_power']:.4f}")
    print(f"  PCASL mean: {synth_amp_features['pcasl_mean']:.4f}")
    print(f"  VSASL mean: {synth_amp_features['vsasl_mean']:.4f}")
    print(f"  PCASL/VSASL ratio: {synth_amp_features['pcasl_vsasl_ratio']:.4f}")

    synth_amp_results = run_model_debug(amp_model, synth_tensor, "AmplitudeAware")
    synth_base_results = run_model_debug(base_model, synth_tensor, "Baseline")

    print(f"\nSynthetic - AmplitudeAware output:")
    print(f"  CBF: {synth_amp_results['cbf_mean']:.2f} ± {synth_amp_results['cbf_std']:.2f}")
    print(f"  ATT: {synth_amp_results['att_mean']:.2f} ± {synth_amp_results['att_std']:.2f}")
    if 'spatial_head_mean' in synth_amp_results:
        print(f"  Spatial head output: mean={synth_amp_results['spatial_head_mean']:.4f}, std={synth_amp_results['spatial_head_std']:.4f}")
    if 'cbf_amplitude_correction_mean' in synth_amp_results:
        print(f"  CBF correction: mean={synth_amp_results['cbf_amplitude_correction_mean']:.4f}")

    print(f"\nSynthetic - Baseline output:")
    print(f"  CBF: {synth_base_results['cbf_mean']:.2f} ± {synth_base_results['cbf_std']:.2f}")
    print(f"  ATT: {synth_base_results['att_mean']:.2f} ± {synth_base_results['att_std']:.2f}")

    # === 2. IN-VIVO DATA ===
    print("\n" + "=" * 70)
    print("2. IN-VIVO DATA ANALYSIS")
    print("=" * 70)

    invivo_dir = Path('Multiverse/20231004_MR1_A151')
    invivo_input, slice_idx, n_invivo_plds = load_invivo_slice(invivo_dir, global_scale=10.0, target_plds=6)

    # Pad to 64x64 if needed
    h, w = invivo_input.shape[2], invivo_input.shape[3]
    if h != 64 or w != 64:
        # Pad or crop
        padded = np.zeros((1, invivo_input.shape[1], 64, 64), dtype=np.float32)
        ph, pw = min(h, 64), min(w, 64)
        padded[:, :, :ph, :pw] = invivo_input[:, :, :ph, :pw]
        invivo_input = padded

    invivo_tensor = torch.from_numpy(invivo_input).float()

    print(f"\nIn-vivo input shape: {invivo_input.shape}")
    print(f"In-vivo input range: [{invivo_input.min():.4f}, {invivo_input.max():.4f}]")
    print(f"In-vivo input mean: {invivo_input.mean():.4f}")
    print(f"In-vivo input std: {invivo_input.std():.4f}")

    invivo_amp_features = analyze_amplitude_extractor(amp_model, invivo_tensor)
    print(f"\nIn-vivo amplitude features:")
    print(f"  Global power: {invivo_amp_features['global_power']:.4f}")
    print(f"  PCASL mean: {invivo_amp_features['pcasl_mean']:.4f}")
    print(f"  VSASL mean: {invivo_amp_features['vsasl_mean']:.4f}")
    print(f"  PCASL/VSASL ratio: {invivo_amp_features['pcasl_vsasl_ratio']:.4f}")

    invivo_amp_results = run_model_debug(amp_model, invivo_tensor, "AmplitudeAware")
    invivo_base_results = run_model_debug(base_model, invivo_tensor, "Baseline")

    print(f"\nIn-vivo - AmplitudeAware output:")
    print(f"  CBF: {invivo_amp_results['cbf_mean']:.2f} ± {invivo_amp_results['cbf_std']:.2f}")
    print(f"  ATT: {invivo_amp_results['att_mean']:.2f} ± {invivo_amp_results['att_std']:.2f}")
    if 'spatial_head_mean' in invivo_amp_results:
        print(f"  Spatial head output: mean={invivo_amp_results['spatial_head_mean']:.4f}, std={invivo_amp_results['spatial_head_std']:.4f}")
    if 'cbf_amplitude_correction_mean' in invivo_amp_results:
        print(f"  CBF correction: mean={invivo_amp_results['cbf_amplitude_correction_mean']:.4f}")

    print(f"\nIn-vivo - Baseline output:")
    print(f"  CBF: {invivo_base_results['cbf_mean']:.2f} ± {invivo_base_results['cbf_std']:.2f}")
    print(f"  ATT: {invivo_base_results['att_mean']:.2f} ± {invivo_base_results['att_std']:.2f}")

    # === 3. COMPARE INPUT DISTRIBUTIONS ===
    print("\n" + "=" * 70)
    print("3. INPUT DISTRIBUTION COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Synthetic':>15} {'In-vivo':>15} {'Ratio':>10}")
    print("-" * 70)
    print(f"{'Input mean':<25} {synth_input.mean():>15.4f} {invivo_input.mean():>15.4f} {synth_input.mean()/invivo_input.mean():>10.2f}x")
    print(f"{'Input std':<25} {synth_input.std():>15.4f} {invivo_input.std():>15.4f} {synth_input.std()/invivo_input.std():>10.2f}x")
    print(f"{'Input max':<25} {synth_input.max():>15.4f} {invivo_input.max():>15.4f} {synth_input.max()/invivo_input.max():>10.2f}x")
    print(f"{'Global power':<25} {synth_amp_features['global_power']:>15.4f} {invivo_amp_features['global_power']:>15.4f} {synth_amp_features['global_power']/invivo_amp_features['global_power']:>10.2f}x")

    # === 4. DETAILED AMPLITUDE MECHANISM ANALYSIS ===
    print("\n" + "=" * 70)
    print("4. AMPLITUDE MECHANISM DEEP DIVE")
    print("=" * 70)

    # Manually trace through the amplitude path
    with torch.no_grad():
        # Synthetic
        synth_raw_amp = amp_model.amplitude_extractor.extract_raw_features(synth_tensor)
        synth_conditioning = amp_model.amplitude_extractor(synth_tensor)

        # In-vivo
        invivo_raw_amp = amp_model.amplitude_extractor.extract_raw_features(invivo_tensor)
        invivo_conditioning = amp_model.amplitude_extractor(invivo_tensor)

    print(f"\nRaw amplitude features (before MLP):")
    print(f"  Synthetic: mean={synth_raw_amp.mean():.4f}, std={synth_raw_amp.std():.4f}, range=[{synth_raw_amp.min():.4f}, {synth_raw_amp.max():.4f}]")
    print(f"  In-vivo:   mean={invivo_raw_amp.mean():.4f}, std={invivo_raw_amp.std():.4f}, range=[{invivo_raw_amp.min():.4f}, {invivo_raw_amp.max():.4f}]")

    print(f"\nConditioning vector (after MLP):")
    print(f"  Synthetic: mean={synth_conditioning.mean():.4f}, std={synth_conditioning.std():.4f}")
    print(f"  In-vivo:   mean={invivo_conditioning.mean():.4f}, std={invivo_conditioning.std():.4f}")

    # Check what the CBF amplitude correction produces
    with torch.no_grad():
        synth_log_correction = amp_model.cbf_amplitude_correction(synth_conditioning)
        invivo_log_correction = amp_model.cbf_amplitude_correction(invivo_conditioning)

        synth_correction = torch.exp(synth_log_correction.clamp(-2, 2))
        invivo_correction = torch.exp(invivo_log_correction.clamp(-2, 2))

        # Get base amplitude
        synth_channel_means = synth_tensor.mean(dim=(2, 3))[:, :12]
        synth_base_amp = synth_channel_means.mean(dim=1, keepdim=True) * 100.0

        invivo_channel_means = invivo_tensor.mean(dim=(2, 3))[:, :12]
        invivo_base_amp = invivo_channel_means.mean(dim=1, keepdim=True) * 100.0

    print(f"\nCBF scaling components:")
    print(f"  Synthetic base amplitude: {synth_base_amp.item():.4f}")
    print(f"  In-vivo base amplitude:   {invivo_base_amp.item():.4f}")
    print(f"  Synthetic correction factor: {synth_correction.item():.4f}")
    print(f"  In-vivo correction factor:   {invivo_correction.item():.4f}")
    print(f"  Synthetic total scale: {(synth_base_amp * synth_correction).item():.4f}")
    print(f"  In-vivo total scale:   {(invivo_base_amp * invivo_correction).item():.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
audit_model_weights.py - Audit saved model weights for FiLM/OutputMod training status.

Checks whether amplitude-aware components (AmplitudeExtractor, FiLM layers,
OutputModulation) are present and appear trained in saved checkpoints.

This is important because a known training bug existed where SLURM scripts searched
for wrong filenames, potentially meaning FiLM/OutputMod flags were not respected
during v1 training.

Usage:
    # Single checkpoint
    python audit_model_weights.py --checkpoint path/to/ensemble_model_0.pt

    # Scan all experiments in a directory
    python audit_model_weights.py --scan_dir amplitude_ablation_v1/ --output audit_results.json
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ============================================================================
# Component detection patterns
# ============================================================================

# Each component maps to a list of key prefixes that indicate its presence
COMPONENT_PATTERNS = {
    "amplitude_extractor": [
        "amplitude_extractor.",
    ],
    "film_bottleneck": [
        "bottleneck_film.",
        "film_bottleneck.",
    ],
    "film_decoder": [
        "decoder1_film.",
        "decoder2_film.",
        "decoder3_film.",
        "film_decoder.",
        "decoder_film.",
    ],
    "output_modulation": [
        "cbf_amplitude_correction.",
        "output_modulation.",
        "amplitude_scale.",
    ],
}

# Known initialization patterns for different layer types
# These are used to detect whether a layer has been trained or is still at init
INIT_PATTERNS = {
    "conv2d_kaiming": {
        "description": "Kaiming normal init for Conv2d (fan_out, relu)",
        "check": "kaiming_normal_fan_out",
    },
    "groupnorm_ones_zeros": {
        "description": "GroupNorm init: weight=1, bias=0",
        "weight_val": 1.0,
        "bias_val": 0.0,
    },
    "linear_zeros": {
        "description": "Linear layer initialized to zeros",
        "weight_val": 0.0,
        "bias_val": 0.0,
    },
    "output_small": {
        "description": "Output head init: normal(0, 0.01)",
        "weight_std": 0.01,
    },
}


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """
    Load a state dict from a checkpoint file.

    Handles both raw state dicts and wrapped checkpoints (with 'model_state_dict' key).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        # Check if it looks like a raw state dict (all values are tensors)
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
        # Maybe it has a 'state_dict' key
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        # Fallback: treat as raw state dict, filter to tensor values
        return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    elif hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    else:
        raise ValueError(
            f"Cannot extract state dict from checkpoint at {path}. "
            f"Type: {type(checkpoint)}"
        )


def classify_layer(key: str, shape: torch.Size) -> str:
    """Classify a layer by its type based on key name and shape."""
    if "double_conv" in key and "weight" in key and len(shape) == 4:
        return "Conv2d"
    if "double_conv" in key and "weight" in key and len(shape) == 1:
        return "GroupNorm"
    if "double_conv" in key and "bias" in key and len(shape) == 1:
        return "GroupNorm_bias"
    if ("up" in key or "ConvTranspose" in key) and "weight" in key and len(shape) == 4:
        return "ConvTranspose2d"
    if "spatial_head" in key or "out_conv" in key:
        return "OutputHead"
    if "gamma_generator" in key or "beta_generator" in key:
        return "FiLM_MLP"
    if "feature_mlp" in key:
        return "AmplitudeExtractor_MLP"
    if "cbf_amplitude_correction" in key:
        return "OutputModulation_MLP"
    if len(shape) == 2:
        return "Linear"
    if len(shape) == 1:
        return "Bias"
    if len(shape) == 4:
        return "Conv2d"
    return "Unknown"


def compute_kaiming_expected_std(shape: torch.Size, mode: str = "fan_out") -> float:
    """
    Compute the expected std for Kaiming normal initialization.

    For Conv2d with fan_out mode and ReLU nonlinearity:
        std = sqrt(2 / fan_out)
    where fan_out = out_channels * kernel_h * kernel_w
    """
    if len(shape) < 2:
        return 0.0

    if len(shape) == 4:
        # Conv2d: (out_channels, in_channels, kH, kW)
        if mode == "fan_out":
            fan = shape[0] * shape[2] * shape[3]
        else:
            fan = shape[1] * shape[2] * shape[3]
    elif len(shape) == 2:
        # Linear: (out_features, in_features)
        if mode == "fan_out":
            fan = shape[0]
        else:
            fan = shape[1]
    else:
        return 0.0

    if fan == 0:
        return 0.0

    # Kaiming normal with nonlinearity='relu' uses gain = sqrt(2)
    return math.sqrt(2.0 / fan)


def analyze_layer_training_status(
    key: str, tensor: torch.Tensor
) -> Dict[str, Any]:
    """
    Analyze whether a single layer appears trained or still at initialization.

    Returns a dict with analysis results.
    """
    shape = tensor.shape
    layer_type = classify_layer(key, shape)
    numel = tensor.numel()

    result = {
        "key": key,
        "shape": list(shape),
        "layer_type": layer_type,
        "numel": numel,
        "mean": tensor.mean().item(),
        "std": tensor.std().item() if numel > 1 else 0.0,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "abs_mean": tensor.abs().mean().item(),
        "all_zeros": (tensor == 0).all().item(),
        "all_ones": (tensor == 1).all().item(),
        "fraction_zero": ((tensor == 0).sum().item() / numel) if numel > 0 else 0.0,
    }

    # Determine training status based on layer type
    status = "UNKNOWN"
    confidence = "low"
    reason = ""

    if result["all_zeros"]:
        status = "AT_INIT_OR_UNTRAINED"
        confidence = "high"
        reason = "All weights are exactly zero"
    elif result["all_ones"]:
        status = "AT_INIT_OR_UNTRAINED"
        confidence = "high"
        reason = "All weights are exactly one (GroupNorm/BN init)"
    elif layer_type == "Conv2d" or layer_type == "ConvTranspose2d":
        expected_std = compute_kaiming_expected_std(shape, mode="fan_out")
        actual_std = result["std"]
        # If std is very close to Kaiming init, might not be trained
        # But Kaiming init is also a reasonable starting point that changes during training
        # We consider it "likely trained" if std differs from init by >20%
        if expected_std > 0:
            std_ratio = actual_std / expected_std
            result["kaiming_expected_std"] = expected_std
            result["std_ratio_vs_kaiming"] = std_ratio
            if abs(std_ratio - 1.0) > 0.2:
                status = "TRAINED"
                confidence = "medium"
                reason = f"Std differs from Kaiming init by {abs(std_ratio - 1.0)*100:.1f}% (ratio={std_ratio:.3f})"
            else:
                status = "POSSIBLY_AT_INIT"
                confidence = "low"
                reason = f"Std close to Kaiming init (ratio={std_ratio:.3f})"
        else:
            status = "TRAINED"
            confidence = "low"
            reason = "Cannot compute expected init std"
    elif layer_type == "GroupNorm":
        # GroupNorm weight is initialized to 1.0
        if torch.allclose(tensor, torch.ones_like(tensor), atol=1e-6):
            status = "AT_INIT_OR_UNTRAINED"
            confidence = "high"
            reason = "GroupNorm weight is exactly all-ones (init value)"
        else:
            status = "TRAINED"
            confidence = "high"
            reason = f"GroupNorm weight deviates from all-ones (std={result['std']:.6f})"
    elif layer_type == "GroupNorm_bias":
        if torch.allclose(tensor, torch.zeros_like(tensor), atol=1e-6):
            status = "AT_INIT_OR_UNTRAINED"
            confidence = "medium"
            reason = "GroupNorm bias is exactly all-zeros (init value)"
        else:
            status = "TRAINED"
            confidence = "high"
            reason = f"GroupNorm bias deviates from zeros (abs_mean={result['abs_mean']:.6f})"
    elif layer_type in ("FiLM_MLP", "OutputModulation_MLP", "AmplitudeExtractor_MLP"):
        if "weight" in key:
            if result["all_zeros"]:
                status = "AT_INIT_OR_UNTRAINED"
                confidence = "high"
                reason = "MLP weight is all zeros (matches zero-init for FiLM output layers)"
            elif result["std"] < 1e-7:
                status = "AT_INIT_OR_UNTRAINED"
                confidence = "high"
                reason = f"MLP weight has near-zero std ({result['std']:.2e})"
            else:
                status = "TRAINED"
                confidence = "high"
                reason = f"MLP weight has non-trivial values (std={result['std']:.6f})"
        elif "bias" in key:
            if result["all_zeros"]:
                status = "AT_INIT_OR_UNTRAINED"
                confidence = "medium"
                reason = "Bias is all zeros (common init)"
            elif result["abs_mean"] < 1e-7:
                status = "AT_INIT_OR_UNTRAINED"
                confidence = "medium"
                reason = f"Bias is near-zero (abs_mean={result['abs_mean']:.2e})"
            else:
                status = "TRAINED"
                confidence = "high"
                reason = f"Bias has non-zero values (abs_mean={result['abs_mean']:.6f})"
    elif layer_type == "OutputHead":
        # Output heads are initialized with normal(0, 0.01) for weight, 0 for bias
        if "weight" in key:
            if result["std"] < 0.015:
                status = "POSSIBLY_AT_INIT"
                confidence = "low"
                reason = f"Output head weight std={result['std']:.6f} close to init std=0.01"
            else:
                status = "TRAINED"
                confidence = "medium"
                reason = f"Output head weight std={result['std']:.6f} differs from init std=0.01"
        elif "bias" in key:
            if result["abs_mean"] < 1e-4:
                status = "POSSIBLY_AT_INIT"
                confidence = "low"
                reason = f"Output head bias near zero (abs_mean={result['abs_mean']:.6f})"
            else:
                status = "TRAINED"
                confidence = "medium"
                reason = f"Output head bias non-zero (abs_mean={result['abs_mean']:.6f})"
    elif layer_type == "Linear":
        if "weight" in key:
            if result["all_zeros"]:
                status = "AT_INIT_OR_UNTRAINED"
                confidence = "high"
                reason = "Linear weight is all zeros"
            elif result["std"] < 1e-7:
                status = "AT_INIT_OR_UNTRAINED"
                confidence = "high"
                reason = f"Linear weight has near-zero std ({result['std']:.2e})"
            else:
                status = "TRAINED"
                confidence = "medium"
                reason = f"Linear weight has non-trivial values (std={result['std']:.6f})"
    elif layer_type == "Bias":
        if result["all_zeros"]:
            status = "POSSIBLY_AT_INIT"
            confidence = "low"
            reason = "Bias is all zeros (may or may not be trained)"
        elif result["abs_mean"] < 1e-7:
            status = "POSSIBLY_AT_INIT"
            confidence = "low"
            reason = f"Bias is near-zero (abs_mean={result['abs_mean']:.2e})"
        else:
            status = "TRAINED"
            confidence = "medium"
            reason = f"Bias has non-zero values (abs_mean={result['abs_mean']:.6f})"

    result["status"] = status
    result["confidence"] = confidence
    result["reason"] = reason

    return result


def identify_component(key: str) -> Optional[str]:
    """Identify which amplitude-aware component a key belongs to, if any."""
    for component, patterns in COMPONENT_PATTERNS.items():
        for pattern in patterns:
            if key.startswith(pattern):
                return component
    return None


def audit_checkpoint(checkpoint_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Audit a single checkpoint file.

    Returns a dict with:
    - path: checkpoint path
    - total_params: total parameter count
    - components: which amplitude-aware components are present
    - component_details: per-component analysis
    - layer_analysis: per-layer analysis (if verbose)
    """
    state_dict = load_state_dict(checkpoint_path)

    # Overall statistics
    total_params = sum(t.numel() for t in state_dict.values())

    # Detect which amplitude-aware components are present
    components_present = {comp: False for comp in COMPONENT_PATTERNS}
    component_keys = {comp: [] for comp in COMPONENT_PATTERNS}

    for key in state_dict:
        comp = identify_component(key)
        if comp is not None:
            components_present[comp] = True
            component_keys[comp].append(key)

    # Analyze each component's training status
    component_details = {}
    for comp, keys in component_keys.items():
        if not keys:
            component_details[comp] = {
                "present": False,
                "n_params": 0,
                "trained_status": "N/A",
                "layers": [],
            }
            continue

        layers = []
        n_params = 0
        trained_layers = 0
        untrained_layers = 0
        uncertain_layers = 0

        for key in keys:
            tensor = state_dict[key]
            n_params += tensor.numel()
            analysis = analyze_layer_training_status(key, tensor)
            layers.append(analysis)

            if analysis["status"] == "TRAINED":
                trained_layers += 1
            elif analysis["status"] in ("AT_INIT_OR_UNTRAINED",):
                untrained_layers += 1
            else:
                uncertain_layers += 1

        # Determine overall component training status
        total_analyzed = trained_layers + untrained_layers + uncertain_layers
        if total_analyzed == 0:
            overall_status = "N/A"
        elif untrained_layers == total_analyzed:
            overall_status = "UNTRAINED"
        elif trained_layers == total_analyzed:
            overall_status = "TRAINED"
        elif trained_layers > 0 and untrained_layers > 0:
            # Mixed: some layers trained, some at init
            # This is suspicious - FiLM output layers are zero-init by design,
            # but hidden layers should be trained if the component was used
            # Check if only the zero-init output layers are untrained
            untrained_keys = [
                l["key"] for l in layers if l["status"] == "AT_INIT_OR_UNTRAINED"
            ]
            # FiLM generators have their output layer (.2.) initialized to zeros
            # but hidden layer (.0.) uses default init
            # If only output layers (.2.) are at init, that's the zero-init design
            output_layer_untrained = all(
                ".2." in k or ".4." in k for k in untrained_keys
            )
            if output_layer_untrained and trained_layers > 0:
                overall_status = "PARTIALLY_TRAINED"
            else:
                overall_status = "PARTIALLY_TRAINED"
        elif trained_layers > 0:
            overall_status = "LIKELY_TRAINED"
        else:
            overall_status = "UNCERTAIN"

        component_details[comp] = {
            "present": True,
            "n_params": n_params,
            "n_layers": len(keys),
            "trained_layers": trained_layers,
            "untrained_layers": untrained_layers,
            "uncertain_layers": uncertain_layers,
            "trained_status": overall_status,
            "layers": layers if verbose else [],
        }

    # Analyze all layers if verbose
    all_layers = []
    if verbose:
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            analysis = analyze_layer_training_status(key, tensor)
            all_layers.append(analysis)

    # Detect model architecture type
    has_out_conv = any("out_conv" in k for k in state_dict)
    has_spatial_head = any("spatial_head" in k for k in state_dict)
    has_cbf_head = any("cbf_head" in k for k in state_dict)

    if has_out_conv and not has_spatial_head:
        model_type = "SpatialASLNet"
    elif has_spatial_head:
        model_type = "AmplitudeAwareSpatialASLNet"
    elif has_cbf_head:
        model_type = "DualEncoderSpatialASLNet"
    else:
        model_type = "Unknown"

    result = {
        "path": checkpoint_path,
        "model_type": model_type,
        "total_params": total_params,
        "total_layers": len(state_dict),
        "components_present": components_present,
        "component_details": {
            comp: {k: v for k, v in details.items() if k != "layers"}
            for comp, details in component_details.items()
        },
    }

    if verbose:
        result["component_layer_details"] = {
            comp: details["layers"]
            for comp, details in component_details.items()
            if details["layers"]
        }
        result["all_layers"] = all_layers

    return result


def scan_directory(scan_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Scan a directory for experiment subdirectories and audit all checkpoints.

    Expects structure:
        scan_dir/
            00_ExperimentName/
                trained_models/
                    ensemble_model_0.pt
                    ensemble_model_1.pt
                    ...
            01_ExperimentName/
                ...
    """
    scan_path = Path(scan_dir)
    if not scan_path.exists():
        print(f"ERROR: Directory not found: {scan_dir}")
        sys.exit(1)

    # Find experiment directories
    exp_dirs = sorted(
        [d for d in scan_path.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    results = {}
    experiment_summaries = []

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name

        # Look for checkpoints
        model_dir = exp_dir / "trained_models"
        if not model_dir.exists():
            # Try looking directly in the exp_dir
            checkpoint_files = sorted(exp_dir.glob("ensemble_model_*.pt"))
            if not checkpoint_files:
                checkpoint_files = sorted(exp_dir.glob("*.pt"))
        else:
            checkpoint_files = sorted(model_dir.glob("ensemble_model_*.pt"))
            if not checkpoint_files:
                checkpoint_files = sorted(model_dir.glob("*.pt"))

        if not checkpoint_files:
            continue

        # Audit the first checkpoint (ensemble_model_0) as representative
        representative_ckpt = str(checkpoint_files[0])
        audit = audit_checkpoint(representative_ckpt, verbose=verbose)

        # Also quickly check consistency across ensemble members
        ensemble_consistent = True
        if len(checkpoint_files) > 1:
            ref_keys = set(
                load_state_dict(str(checkpoint_files[0])).keys()
            )
            for ckpt in checkpoint_files[1:]:
                other_keys = set(load_state_dict(str(ckpt)).keys())
                if ref_keys != other_keys:
                    ensemble_consistent = False
                    break

        audit["ensemble_size"] = len(checkpoint_files)
        audit["ensemble_keys_consistent"] = ensemble_consistent

        results[exp_name] = audit

        # Build summary row
        cp = audit["components_present"]
        cd = audit["component_details"]

        summary = {
            "exp": exp_name,
            "model_type": audit["model_type"],
            "has_amp_extractor": cp.get("amplitude_extractor", False),
            "has_film_bottleneck": cp.get("film_bottleneck", False),
            "has_film_decoder": cp.get("film_decoder", False),
            "has_output_mod": cp.get("output_modulation", False),
            "amp_extractor_status": cd.get("amplitude_extractor", {}).get(
                "trained_status", "N/A"
            ),
            "film_bottleneck_status": cd.get("film_bottleneck", {}).get(
                "trained_status", "N/A"
            ),
            "film_decoder_status": cd.get("film_decoder", {}).get(
                "trained_status", "N/A"
            ),
            "output_mod_status": cd.get("output_modulation", {}).get(
                "trained_status", "N/A"
            ),
            "ensemble_size": len(checkpoint_files),
            "ensemble_consistent": ensemble_consistent,
        }
        experiment_summaries.append(summary)

    return {
        "scan_dir": str(scan_dir),
        "n_experiments": len(results),
        "experiments": results,
        "summary": experiment_summaries,
    }


def print_single_audit(audit: Dict[str, Any], verbose: bool = False) -> None:
    """Print audit results for a single checkpoint."""
    print("=" * 80)
    print(f"CHECKPOINT AUDIT: {audit['path']}")
    print("=" * 80)
    print(f"  Model type:    {audit['model_type']}")
    print(f"  Total params:  {audit['total_params']:,}")
    print(f"  Total layers:  {audit['total_layers']}")
    print()

    print("AMPLITUDE-AWARE COMPONENTS:")
    print("-" * 80)
    fmt = "  {:<25s}  {:>8s}  {:>8s}  {:>20s}"
    print(fmt.format("Component", "Present", "Params", "Training Status"))
    print(fmt.format("-" * 25, "-" * 8, "-" * 8, "-" * 20))

    for comp in COMPONENT_PATTERNS:
        present = audit["components_present"].get(comp, False)
        details = audit["component_details"].get(comp, {})
        n_params = details.get("n_params", 0)
        status = details.get("trained_status", "N/A")
        print(
            fmt.format(
                comp,
                "YES" if present else "NO",
                f"{n_params:,}" if present else "-",
                status if present else "N/A",
            )
        )

    print()

    if verbose and "component_layer_details" in audit:
        print("DETAILED LAYER ANALYSIS:")
        print("-" * 80)
        for comp, layers in audit["component_layer_details"].items():
            if layers:
                print(f"\n  [{comp}]")
                for layer in layers:
                    shape_str = "x".join(str(s) for s in layer["shape"])
                    print(
                        f"    {layer['key']:<60s}  shape={shape_str:<20s}  "
                        f"std={layer['std']:.6f}  status={layer['status']}"
                    )
                    print(f"      -> {layer['reason']}")

    if verbose and "all_layers" in audit:
        print("\nALL LAYERS:")
        print("-" * 80)
        for layer in audit["all_layers"]:
            shape_str = "x".join(str(s) for s in layer["shape"])
            comp = identify_component(layer["key"])
            comp_tag = f" [{comp}]" if comp else ""
            print(
                f"  {layer['key']:<60s}  shape={shape_str:<15s}  "
                f"mean={layer['mean']:+.6f}  std={layer['std']:.6f}  "
                f"status={layer['status']}{comp_tag}"
            )


def print_scan_summary(scan_results: Dict[str, Any]) -> None:
    """Print the summary table for a directory scan."""
    summaries = scan_results["summary"]
    if not summaries:
        print("No experiments found.")
        return

    print("=" * 120)
    print(f"WEIGHT AUDIT SUMMARY: {scan_results['scan_dir']}")
    print(f"Experiments found: {scan_results['n_experiments']}")
    print("=" * 120)
    print()

    # Header
    header = (
        f"{'Exp':<40s}  {'Model':<12s}  "
        f"{'AmpExtr':>8s}  {'FiLM_Bn':>8s}  {'FiLM_Dec':>8s}  {'OutMod':>8s}  "
        f"{'AmpExtr':>12s}  {'FiLM_Bn':>12s}  {'FiLM_Dec':>12s}  {'OutMod':>12s}"
    )
    print(header)
    subheader = (
        f"{'':<40s}  {'':<12s}  "
        f"{'Present':>8s}  {'Present':>8s}  {'Present':>8s}  {'Present':>8s}  "
        f"{'Status':>12s}  {'Status':>12s}  {'Status':>12s}  {'Status':>12s}"
    )
    print(subheader)
    print("-" * 120)

    for s in summaries:
        row = (
            f"{s['exp']:<40s}  {s['model_type']:<12s}  "
            f"{'YES' if s['has_amp_extractor'] else 'NO':>8s}  "
            f"{'YES' if s['has_film_bottleneck'] else 'NO':>8s}  "
            f"{'YES' if s['has_film_decoder'] else 'NO':>8s}  "
            f"{'YES' if s['has_output_mod'] else 'NO':>8s}  "
            f"{s['amp_extractor_status']:>12s}  "
            f"{s['film_bottleneck_status']:>12s}  "
            f"{s['film_decoder_status']:>12s}  "
            f"{s['output_mod_status']:>12s}"
        )
        print(row)

    print()

    # Check for anomalies
    print("ANOMALY CHECK:")
    print("-" * 80)
    anomalies_found = False

    for s in summaries:
        exp = s["exp"]
        issues = []

        # Check: component present but appears untrained
        for comp, has_key, status_key in [
            ("AmpExtractor", "has_amp_extractor", "amp_extractor_status"),
            ("FiLM_Bottleneck", "has_film_bottleneck", "film_bottleneck_status"),
            ("FiLM_Decoder", "has_film_decoder", "film_decoder_status"),
            ("OutputMod", "has_output_mod", "output_mod_status"),
        ]:
            if s[has_key] and s[status_key] == "UNTRAINED":
                issues.append(
                    f"{comp} is PRESENT but appears UNTRAINED - "
                    f"flag may not have been respected during training!"
                )

        # Check: ensemble inconsistency
        if not s.get("ensemble_consistent", True):
            issues.append(
                "Ensemble members have DIFFERENT weight keys - "
                "possible training configuration mismatch!"
            )

        if issues:
            anomalies_found = True
            print(f"\n  [{exp}]")
            for issue in issues:
                print(f"    WARNING: {issue}")

    if not anomalies_found:
        print("  No anomalies detected.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Audit model weights for FiLM/OutputMod training status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single checkpoint
  python audit_model_weights.py --checkpoint path/to/ensemble_model_0.pt

  # Verbose single checkpoint (show all layers)
  python audit_model_weights.py --checkpoint path/to/ensemble_model_0.pt --verbose

  # Scan all experiments in a directory
  python audit_model_weights.py --scan_dir amplitude_ablation_v1/

  # Scan and save results to JSON
  python audit_model_weights.py --scan_dir amplitude_ablation_v1/ --output audit_results.json
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a single model checkpoint file (.pt)",
    )
    parser.add_argument(
        "--scan_dir",
        type=str,
        help="Path to parent directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-layer analysis",
    )

    args = parser.parse_args()

    if not args.checkpoint and not args.scan_dir:
        parser.error("Must specify either --checkpoint or --scan_dir")

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        audit = audit_checkpoint(args.checkpoint, verbose=args.verbose)
        print_single_audit(audit, verbose=args.verbose)

        if args.output:
            # Remove non-serializable layer data for JSON output
            output_data = {
                k: v
                for k, v in audit.items()
                if k not in ("all_layers", "component_layer_details")
            }
            if args.verbose:
                # Include layer details but convert to serializable format
                if "all_layers" in audit:
                    output_data["all_layers"] = audit["all_layers"]
                if "component_layer_details" in audit:
                    output_data["component_layer_details"] = audit[
                        "component_layer_details"
                    ]

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")

    elif args.scan_dir:
        scan_results = scan_directory(args.scan_dir, verbose=args.verbose)
        print_scan_summary(scan_results)

        # If verbose, also print per-experiment details
        if args.verbose:
            for exp_name, audit in scan_results["experiments"].items():
                print(f"\n{'#' * 80}")
                print(f"# Experiment: {exp_name}")
                print(f"{'#' * 80}")
                print_single_audit(audit, verbose=True)

        if args.output:
            # Prepare JSON-serializable output
            output_data = {
                "scan_dir": scan_results["scan_dir"],
                "n_experiments": scan_results["n_experiments"],
                "summary": scan_results["summary"],
                "experiments": {},
            }

            for exp_name, audit in scan_results["experiments"].items():
                exp_data = {
                    k: v
                    for k, v in audit.items()
                    if k not in ("all_layers", "component_layer_details")
                }
                if args.verbose:
                    if "all_layers" in audit:
                        exp_data["all_layers"] = audit["all_layers"]
                    if "component_layer_details" in audit:
                        exp_data["component_layer_details"] = audit[
                            "component_layer_details"
                        ]
                output_data["experiments"][exp_name] = exp_data

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualize in-vivo comparison: NN Baseline vs NN AmplitudeAware vs LS Fitting.

Generates:
  1. Axial slice montage: CBF and ATT maps side-by-side for all 3 methods (per subject)
  2. Group bar chart: mean CBF and ATT across subjects
  3. Histogram overlay: voxel-wise CBF/ATT distributions for one representative subject

Usage:
  python visualize_invivo_comparison.py [--results-dir invivo_results_v4] [--output-dir invivo_figures_v4]
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METHODS = {
    "A_Baseline_SpatialASL": {
        "label": "NN Baseline",
        "cbf_file": "nn_cbf.nii.gz",
        "att_file": "nn_att.nii.gz",
        "color": "#4C72B0",
    },
    "B_AmplitudeAware": {
        "label": "NN AmpAware",
        "cbf_file": "nn_cbf.nii.gz",
        "att_file": "nn_att.nii.gz",
        "color": "#DD8452",
    },
    "LS_baseline": {
        "label": "LS Fitting",
        "cbf_file": "ls_cbf.nii.gz",
        "att_file": "ls_att.nii.gz",
        "color": "#55A868",
    },
}

CBF_RANGE = (0, 80)       # ml/100g/min
ATT_RANGE = (0, 3000)     # ms


def load_map(path):
    """Load a NIfTI file, return 3-D numpy array."""
    return nib.load(path).get_fdata()


def pick_slices(mask, n=5):
    """Pick `n` evenly-spaced axial slices with the most brain voxels."""
    counts = [mask[:, :, z].sum() for z in range(mask.shape[2])]
    # Skip the top and bottom 2 slices (often partial)
    margin = 2
    valid = list(range(margin, mask.shape[2] - margin))
    valid.sort(key=lambda z: counts[z], reverse=True)
    # Take top-n by brain voxel count, then sort by slice index
    chosen = sorted(valid[:max(n, len(valid))])
    # Evenly subsample if too many
    if len(chosen) > n:
        idxs = np.linspace(0, len(chosen) - 1, n, dtype=int)
        chosen = [chosen[i] for i in idxs]
    return chosen


# ---------------------------------------------------------------------------
# Figure 1: Axial slice montage per subject
# ---------------------------------------------------------------------------
def plot_slice_montage(results_dir, subject, output_dir):
    """CBF and ATT maps for each method, side by side across axial slices."""
    mask_path = os.path.join(results_dir, "A_Baseline_SpatialASL", subject, "brain_mask.nii.gz")
    mask = load_map(mask_path)
    slices = pick_slices(mask, n=5)

    n_slices = len(slices)
    n_methods = len(METHODS)
    # Layout: rows = methods, cols = slices.  Two panels: CBF (left) and ATT (right)
    fig, axes = plt.subplots(
        n_methods * 2, n_slices,
        figsize=(3 * n_slices, 3 * n_methods * 2),
        facecolor="black",
    )
    fig.suptitle(f"In-Vivo Maps — {subject}", fontsize=16, color="white", y=0.99)

    for m_idx, (method_key, mcfg) in enumerate(METHODS.items()):
        cbf_path = os.path.join(results_dir, method_key, subject, mcfg["cbf_file"])
        att_path = os.path.join(results_dir, method_key, subject, mcfg["att_file"])
        if not os.path.exists(cbf_path):
            continue
        cbf = load_map(cbf_path)
        att = load_map(att_path)

        # Mask out non-brain
        cbf_masked = np.where(mask > 0, cbf, np.nan)
        att_masked = np.where(mask > 0, att, np.nan)

        cbf_row = m_idx * 2
        att_row = m_idx * 2 + 1

        for s_idx, z in enumerate(slices):
            # CBF
            ax_cbf = axes[cbf_row, s_idx]
            ax_cbf.set_facecolor("black")
            im_cbf = ax_cbf.imshow(
                np.rot90(cbf_masked[:, :, z]),
                cmap="hot", vmin=CBF_RANGE[0], vmax=CBF_RANGE[1],
                interpolation="nearest",
            )
            ax_cbf.axis("off")
            if s_idx == 0:
                ax_cbf.set_title(f"{mcfg['label']} — CBF", fontsize=10, color="white", loc="left")
            if cbf_row == 0:
                ax_cbf.set_title(f"Slice {z}", fontsize=9, color="gray")

            # ATT
            ax_att = axes[att_row, s_idx]
            ax_att.set_facecolor("black")
            im_att = ax_att.imshow(
                np.rot90(att_masked[:, :, z]),
                cmap="jet", vmin=ATT_RANGE[0], vmax=ATT_RANGE[1],
                interpolation="nearest",
            )
            ax_att.axis("off")
            if s_idx == 0:
                ax_att.set_title(f"{mcfg['label']} — ATT", fontsize=10, color="white", loc="left")

    # Colorbars
    cbar_cbf = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(*CBF_RANGE), cmap="hot"),
        ax=axes[:, -1], fraction=0.02, pad=0.02,
    )
    cbar_cbf.set_label("CBF (ml/100g/min)", color="white", fontsize=9)
    cbar_cbf.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar_cbf.ax.yaxis.get_ticklabels(), color="white")

    fig.tight_layout(rect=[0, 0, 0.95, 0.97])
    out = os.path.join(output_dir, f"montage_{subject}.png")
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Group bar chart
# ---------------------------------------------------------------------------
def plot_group_bar(results_dir, subjects, output_dir):
    """Bar chart of mean CBF and ATT per method, with per-subject dots."""
    fig, (ax_cbf, ax_att) = plt.subplots(1, 2, figsize=(12, 5))

    method_keys = list(METHODS.keys())
    x = np.arange(len(method_keys))
    width = 0.5

    for metric, ax, ylabel in [
        ("cbf", ax_cbf, "CBF (ml/100g/min)"),
        ("att", ax_att, "ATT (ms)"),
    ]:
        means_per_method = []
        stds_per_method = []
        subject_vals = {mk: [] for mk in method_keys}

        for mk in method_keys:
            for subj in subjects:
                mpath = os.path.join(results_dir, mk, subj, "metadata.json")
                if not os.path.exists(mpath):
                    continue
                with open(mpath) as f:
                    d = json.load(f)
                if f"{metric}_stats" in d:
                    subject_vals[mk].append(d[f"{metric}_stats"]["mean"])
                elif f"{metric}_mean" in d:
                    subject_vals[mk].append(d[f"{metric}_mean"])

            vals = subject_vals[mk]
            means_per_method.append(np.mean(vals) if vals else 0)
            stds_per_method.append(np.std(vals) if vals else 0)

        colors = [METHODS[mk]["color"] for mk in method_keys]
        labels = [METHODS[mk]["label"] for mk in method_keys]

        bars = ax.bar(x, means_per_method, width, yerr=stds_per_method,
                       color=colors, edgecolor="white", linewidth=0.5,
                       capsize=4, error_kw={"linewidth": 1.2})

        # Overlay individual subject dots
        for i, mk in enumerate(method_keys):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(subject_vals[mk]))
            ax.scatter(
                x[i] + jitter, subject_vals[mk],
                color="white", edgecolor=colors[i], s=28, zorder=5, linewidth=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Mean {metric.upper()} Across {len(subjects)} Subjects", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(output_dir, "group_barplot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Voxel-wise histogram overlay
# ---------------------------------------------------------------------------
def plot_histograms(results_dir, subject, output_dir):
    """Overlay histograms of brain-masked CBF and ATT for all methods."""
    mask_path = os.path.join(results_dir, "A_Baseline_SpatialASL", subject, "brain_mask.nii.gz")
    mask = load_map(mask_path).astype(bool)

    fig, (ax_cbf, ax_att) = plt.subplots(1, 2, figsize=(13, 4.5))

    for method_key, mcfg in METHODS.items():
        cbf_path = os.path.join(results_dir, method_key, subject, mcfg["cbf_file"])
        att_path = os.path.join(results_dir, method_key, subject, mcfg["att_file"])
        if not os.path.exists(cbf_path):
            continue
        cbf = load_map(cbf_path)[mask]
        att = load_map(att_path)[mask]

        ax_cbf.hist(cbf, bins=100, range=(0, 100), alpha=0.5,
                     color=mcfg["color"], label=mcfg["label"], density=True)
        ax_att.hist(att, bins=100, range=(0, 4000), alpha=0.5,
                     color=mcfg["color"], label=mcfg["label"], density=True)

    for ax, xlabel, title in [
        (ax_cbf, "CBF (ml/100g/min)", "CBF Distribution"),
        (ax_att, "ATT (ms)", "ATT Distribution"),
    ]:
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{title} — {subject}", fontsize=12)
        ax.legend(frameon=False, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(output_dir, f"histogram_{subject}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Difference maps (AmpAware − Baseline, AmpAware − LS)
# ---------------------------------------------------------------------------
def plot_difference_maps(results_dir, subject, output_dir):
    """Show spatial difference maps between methods for one subject."""
    mask_path = os.path.join(results_dir, "A_Baseline_SpatialASL", subject, "brain_mask.nii.gz")
    mask = load_map(mask_path)
    slices = pick_slices(mask, n=5)

    # Load all CBF maps
    cbf = {}
    for mk, mcfg in METHODS.items():
        p = os.path.join(results_dir, mk, subject, mcfg["cbf_file"])
        if os.path.exists(p):
            cbf[mk] = np.where(mask > 0, load_map(p), np.nan)

    if len(cbf) < 3:
        return

    diffs = {
        "AmpAware − Baseline": cbf["B_AmplitudeAware"] - cbf["A_Baseline_SpatialASL"],
        "AmpAware − LS":       cbf["B_AmplitudeAware"] - cbf["LS_baseline"],
        "Baseline − LS":       cbf["A_Baseline_SpatialASL"] - cbf["LS_baseline"],
    }

    n_slices = len(slices)
    fig, axes = plt.subplots(len(diffs), n_slices, figsize=(3 * n_slices, 3 * len(diffs)),
                              facecolor="black")
    fig.suptitle(f"CBF Difference Maps — {subject}", fontsize=14, color="white", y=0.99)

    for d_idx, (diff_label, diff_map) in enumerate(diffs.items()):
        for s_idx, z in enumerate(slices):
            ax = axes[d_idx, s_idx]
            ax.set_facecolor("black")
            im = ax.imshow(
                np.rot90(diff_map[:, :, z]),
                cmap="RdBu_r", vmin=-40, vmax=40,
                interpolation="nearest",
            )
            ax.axis("off")
            if s_idx == 0:
                ax.set_title(diff_label, fontsize=10, color="white", loc="left")
            if d_idx == 0:
                ax.set_title(f"Slice {z}", fontsize=9, color="gray")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(-40, 40), cmap="RdBu_r"),
        ax=axes[:, -1], fraction=0.02, pad=0.02,
    )
    cbar.set_label("ΔCBF (ml/100g/min)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.tight_layout(rect=[0, 0, 0.95, 0.97])
    out = os.path.join(output_dir, f"diff_maps_{subject}.png")
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: Summary table image
# ---------------------------------------------------------------------------
def plot_summary_table(results_dir, subjects, output_dir):
    """Render a summary stats table as a figure."""
    rows = []
    for subj in subjects:
        row = [subj.replace("_MR1_", "\n")]
        for mk, mcfg in METHODS.items():
            mpath = os.path.join(results_dir, mk, subj, "metadata.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    d = json.load(f)
                if "cbf_stats" in d:
                    cbf_m, att_m = d["cbf_stats"]["mean"], d["att_stats"]["mean"]
                else:
                    cbf_m, att_m = d.get("cbf_mean", 0), d.get("att_mean", 0)
                row.append(f"{cbf_m:.1f}")
                row.append(f"{att_m:.0f}")
            else:
                row.extend(["—", "—"])
        rows.append(row)

    # Grand average row
    avg_row = ["Average"]
    for mk in METHODS:
        cbf_vals, att_vals = [], []
        for subj in subjects:
            mpath = os.path.join(results_dir, mk, subj, "metadata.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    d = json.load(f)
                if "cbf_stats" in d:
                    cbf_vals.append(d["cbf_stats"]["mean"])
                    att_vals.append(d["att_stats"]["mean"])
                else:
                    cbf_vals.append(d.get("cbf_mean", 0))
                    att_vals.append(d.get("att_mean", 0))
        avg_row.append(f"{np.mean(cbf_vals):.1f}" if cbf_vals else "—")
        avg_row.append(f"{np.mean(att_vals):.0f}" if att_vals else "—")
    rows.append(avg_row)

    col_labels = ["Subject"]
    for mcfg in METHODS.values():
        col_labels.extend([f"{mcfg['label']}\nCBF", f"{mcfg['label']}\nATT"])

    fig, ax = plt.subplots(figsize=(14, 0.5 * len(rows) + 2))
    ax.axis("off")

    colors = []
    for mcfg in METHODS.values():
        colors.extend([mcfg["color"], mcfg["color"]])

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j, label in enumerate(col_labels):
        cell = table[0, j]
        if j == 0:
            cell.set_facecolor("#333333")
        else:
            cell.set_facecolor(colors[j - 1])
        cell.set_text_props(color="white", fontweight="bold")

    # Style average row
    for j in range(len(col_labels)):
        cell = table[len(rows), j]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#f0f0f0")

    fig.suptitle("In-Vivo Results Summary (v4 Models)", fontsize=14, y=0.95)
    fig.tight_layout()
    out = os.path.join(output_dir, "summary_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize in-vivo comparison")
    parser.add_argument("--results-dir", default="invivo_results_v4")
    parser.add_argument("--output-dir", default="invivo_figures_v4")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subset of subjects to plot (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover subjects
    baseline_dir = os.path.join(args.results_dir, "A_Baseline_SpatialASL")
    if not os.path.isdir(baseline_dir):
        print(f"ERROR: {baseline_dir} not found"); sys.exit(1)

    subjects = sorted(args.subjects or os.listdir(baseline_dir))
    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}\n")

    # Representative subject for per-subject figures (pick middle one)
    rep_subj = subjects[len(subjects) // 2]

    # --- Generate all figures ---
    print("[1/5] Summary table")
    plot_summary_table(args.results_dir, subjects, args.output_dir)

    print("[2/5] Group bar chart")
    plot_group_bar(args.results_dir, subjects, args.output_dir)

    print(f"[3/5] Axial slice montage (representative: {rep_subj})")
    plot_slice_montage(args.results_dir, rep_subj, args.output_dir)

    print(f"[4/5] Voxel-wise histograms (representative: {rep_subj})")
    plot_histograms(args.results_dir, rep_subj, args.output_dir)

    print(f"[5/5] Difference maps (representative: {rep_subj})")
    plot_difference_maps(args.results_dir, rep_subj, args.output_dir)

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

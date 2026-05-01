"""Post-process test-set conditional generations.

For each scan in the manifest:
  1. Load 100 generated NIfTI binary masks and compute a voxel-wise probability
     map (mean across seeds).
  2. Save the probability map as a NIfTI volume.
  3. Produce two PNG figures with a grid of axial slices:
       - MRI + probability colormap overlay
       - Ground-truth segmentation + probability colormap overlay

Usage:
    # Process all scans
    python postprocess_test_generations.py \
        --manifest test_generation_manifest.json \
        --generation-root /scratch/tc2fh/gam_ai/test_generations

    # Process a single scan
    python postprocess_test_generations.py \
        --manifest test_generation_manifest.json \
        --generation-root /scratch/tc2fh/gam_ai/test_generations \
        --scan-id 123_1_45
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless rendering for HPC
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nifti_volume(path):
    """Load a NIfTI file and return data as float32 array, shape (D, H, W)."""
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32)


def compute_probability_map(generation_dir, n_expected=100):
    """Load all generated NIfTI masks and return the mean probability map.

    Returns:
        prob_map: float32 ndarray, shape (D, H, W), values in [0, 1]
        n_found: number of generation files actually found
    """
    mask_files = sorted(glob.glob(os.path.join(generation_dir, "*.nii.gz")))
    # Exclude probability_map.nii.gz if it already exists from a previous run
    mask_files = [f for f in mask_files if os.path.basename(f) != "probability_map.nii.gz"]

    if not mask_files:
        raise FileNotFoundError(f"No .nii.gz generation files found in {generation_dir}")

    if len(mask_files) != n_expected:
        print(f"  [WARN] Expected {n_expected} masks, found {len(mask_files)}")

    stack = []
    for fpath in mask_files:
        vol = load_nifti_volume(fpath)
        stack.append(vol)

    prob_map = np.mean(np.stack(stack, axis=0), axis=0).astype(np.float32)
    return prob_map, len(mask_files)


def save_nifti(data, out_path, affine=None):
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), out_path)


def select_axial_slices(volume, n_slices=12):
    """Select n_slices evenly-spaced axial (depth) indices that are non-empty.

    Assumes volume shape is (D, H, W). Returns a list of axial indices.
    """
    depth = volume.shape[0]
    # Find depth range that contains signal
    nonzero_depths = np.where(volume.max(axis=(1, 2)) > 0)[0]
    if len(nonzero_depths) == 0:
        # Fall back to evenly-spaced across full depth
        return list(np.linspace(0, depth - 1, n_slices, dtype=int))
    d_min, d_max = nonzero_depths[0], nonzero_depths[-1]
    return list(np.linspace(d_min, d_max, n_slices, dtype=int))


def normalize_for_display(volume):
    """Normalize volume to [0, 1] for display."""
    vmin, vmax = volume.min(), volume.max()
    if vmax == vmin:
        return np.zeros_like(volume)
    return (volume - vmin) / (vmax - vmin)


def make_overlay_figure(background, prob_map, slice_indices, title, cmap_bg="gray"):
    """Create a matplotlib figure with axial slice grid: background + prob overlay.

    Args:
        background: (D, H, W) float array — MRI or GT seg
        prob_map:   (D, H, W) float array in [0, 1]
        slice_indices: list of axial depth indices to display
        title: figure title
        cmap_bg: colormap for background ('gray' for MRI, 'gray' for binary seg)

    Returns:
        matplotlib Figure
    """
    n = len(slice_indices)
    ncols = min(n, 6)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    bg_norm = normalize_for_display(background)

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue
        d = slice_indices[idx]
        ax.imshow(bg_norm[d].T, cmap=cmap_bg, origin="lower", interpolation="nearest")
        overlay = ax.imshow(
            prob_map[d].T,
            cmap="hot",
            origin="lower",
            interpolation="nearest",
            alpha=0.5,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"z={d}", fontsize=8)
        ax.axis("off")

    # Shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.1)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Generation probability")

    fig.suptitle(title, fontsize=11, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# Per-scan processing
# ---------------------------------------------------------------------------

def process_scan(entry, generation_root, n_slices=12, n_seeds=100):
    scan_id = entry["scan_id"]
    gen_dir = os.path.join(generation_root, scan_id)

    print(f"\n--- Processing {scan_id} ---")

    if not os.path.isdir(gen_dir):
        print(f"  [SKIP] Generation directory not found: {gen_dir}")
        return

    # 1. Compute probability map
    prob_map, n_found = compute_probability_map(gen_dir, n_expected=n_seeds)
    print(f"  Probability map: shape={prob_map.shape}, "
          f"mean={prob_map.mean():.4f}, max={prob_map.max():.4f} (from {n_found} masks)")

    # 2. Save probability map as NIfTI
    prob_out = os.path.join(gen_dir, "probability_map.nii.gz")
    # Try to inherit affine from the original MRI
    try:
        ref_img = nib.load(entry["mri_path"])
        affine = ref_img.affine
    except Exception:
        affine = np.eye(4)
    save_nifti(prob_map, prob_out, affine=affine)
    print(f"  Saved: {prob_out}")

    # 3. Load MRI and GT seg
    mri = load_nifti_volume(entry["mri_path"])
    seg = load_nifti_volume(entry["seg_path"])

    # Ensure shapes match the probability map (resize via nearest if they don't)
    if mri.shape != prob_map.shape:
        from scipy.ndimage import zoom as scipy_zoom
        scale = tuple(p / m for p, m in zip(prob_map.shape, mri.shape))
        mri = scipy_zoom(mri, scale, order=1)
        seg = scipy_zoom(seg, scale, order=0)

    # 4. Select axial slices based on where the GT seg has signal
    slice_indices = select_axial_slices(seg if seg.max() > 0 else prob_map, n_slices=n_slices)

    # 5. MRI + probability overlay
    fig_mri = make_overlay_figure(
        mri, prob_map, slice_indices,
        title=f"{scan_id} — MRI + generation probability",
        cmap_bg="gray",
    )
    mri_out = os.path.join(gen_dir, "mri_prob_overlay.png")
    fig_mri.savefig(mri_out, dpi=150, bbox_inches="tight")
    plt.close(fig_mri)
    print(f"  Saved: {mri_out}")

    # 6. GT seg + probability overlay
    fig_seg = make_overlay_figure(
        seg, prob_map, slice_indices,
        title=f"{scan_id} — GT segmentation + generation probability",
        cmap_bg="gray",
    )
    seg_out = os.path.join(gen_dir, "GT_prob_overlay.png")
    fig_seg.savefig(seg_out, dpi=150, bbox_inches="tight")
    plt.close(fig_seg)
    print(f"  Saved: {seg_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute probability maps and visualizations from 100 generated masks"
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to test_generation_manifest.json"
    )
    parser.add_argument(
        "--generation-root", required=True,
        help="Root directory where per-scan generation outputs are stored"
    )
    parser.add_argument(
        "--scan-id", default=None,
        help="Process only this scan_id (default: process all scans in manifest)"
    )
    parser.add_argument(
        "--n-slices", type=int, default=12,
        help="Number of axial slices to show in each PNG figure (default: 12)"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=100,
        help="Expected number of generated masks per scan (default: 100)"
    )
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.scan_id is not None:
        manifest = [e for e in manifest if e["scan_id"] == args.scan_id]
        if not manifest:
            print(f"ERROR: scan_id '{args.scan_id}' not found in manifest.", file=sys.stderr)
            sys.exit(1)

    print(f"Processing {len(manifest)} scan(s)...")
    for entry in manifest:
        process_scan(entry, args.generation_root, n_slices=args.n_slices, n_seeds=args.n_seeds)

    print(f"\nAll done.")


if __name__ == "__main__":
    main()

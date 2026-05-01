"""Build a manifest JSON for test-set conditional generation.

Walks the test embedding CSV directory produced by the GrowthNet
TemporalSpatialEmbedding extractor, pairs each CSV with the corresponding
MRI image and ground-truth segmentation from the raw UVA dataset, reads the
original NIfTI shape, and writes a manifest consumed by the generation SLURM
array job.

Output manifest format (list of dicts):
    [
      {
        "index": 0,
        "scan_id": "123_1_45",
        "patient_id": "123",
        "emb_path": "/path/to/embedding_123_123_1_45.csv",
        "mri_path": "/path/to/.../t2_thin/image/scan.nii.gz",
        "seg_path": "/path/to/.../t2_thin/segmentations/seg.nii.gz",
        "original_shape": [D, H, W]
      },
      ...
    ]

Usage:
    python prepare_test_generation.py \
        --test-embeddings-dir /scratch/tc2fh/gam_ai/GrowthNet/test_data_embeddings \
        --uva-root /scratch/tc2fh/gam_ai/uva_vs_v0003/uva_vs_v0003/split \
        --manifest-out test_generation_manifest.json
"""

import argparse
import glob
import json
import os
import sys

import nibabel as nib

from prepare_data_3d_longitudinal import parse_embedding_filename


def find_mri(uva_root, patient_id, scan_id, modality="t2_thin"):
    """Return path to the MRI image NIfTI for this scan, or None."""
    img_dir = os.path.join(uva_root, "test", patient_id, scan_id, modality, "image")
    matches = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [WARN] Multiple images in {img_dir}, using {os.path.basename(matches[0])}")
    return matches[0]


def find_segmentation(uva_root, patient_id, scan_id, modality="t2_thin"):
    """Return path to the GT segmentation NIfTI for this scan, or None."""
    seg_dir = os.path.join(uva_root, "test", patient_id, scan_id, modality, "segmentations")
    matches = sorted(glob.glob(os.path.join(seg_dir, "*.nii.gz")))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [WARN] Multiple segs in {seg_dir}, using {os.path.basename(matches[0])}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Build test-set generation manifest from embedding CSVs"
    )
    parser.add_argument(
        "--test-embeddings-dir", required=True,
        help="Directory containing test embedding CSV files"
    )
    parser.add_argument(
        "--uva-root", required=True,
        help="Path to uva_vs_v0003/split/ directory (containing train/, val/, test/)"
    )
    parser.add_argument(
        "--manifest-out", default="test_generation_manifest.json",
        help="Output path for the manifest JSON (default: test_generation_manifest.json)"
    )
    parser.add_argument(
        "--modality", default="t2_thin",
        help="MRI modality subdirectory name (default: t2_thin)"
    )
    args = parser.parse_args()

    emb_dir = args.test_embeddings_dir
    uva_root = args.uva_root

    if not os.path.isdir(emb_dir):
        print(f"ERROR: Embeddings directory does not exist: {emb_dir}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(
        f for f in os.listdir(emb_dir)
        if f.startswith("embedding_") and f.endswith(".csv")
    )
    print(f"Found {len(csv_files)} embedding CSVs in {emb_dir}")

    manifest = []
    counts = {"ok": 0, "bad_filename": 0, "missing_mri": 0, "missing_seg": 0}

    for fname in csv_files:
        parsed = parse_embedding_filename(fname)
        if parsed is None:
            print(f"  [SKIP] Bad filename: {fname}")
            counts["bad_filename"] += 1
            continue
        patient_id, scan_id = parsed

        emb_path = os.path.abspath(os.path.join(emb_dir, fname))

        mri_path = find_mri(uva_root, patient_id, scan_id, args.modality)
        if mri_path is None:
            print(f"  [SKIP] No MRI for {patient_id}/{scan_id}")
            counts["missing_mri"] += 1
            continue

        seg_path = find_segmentation(uva_root, patient_id, scan_id, args.modality)
        if seg_path is None:
            print(f"  [SKIP] No seg for {patient_id}/{scan_id}")
            counts["missing_seg"] += 1
            continue

        # Read original spatial shape from NIfTI header (no data loaded)
        img = nib.load(mri_path)
        original_shape = list(img.shape[:3])  # (D, H, W)

        manifest.append({
            "index": len(manifest),
            "scan_id": scan_id,
            "patient_id": patient_id,
            "emb_path": emb_path,
            "mri_path": os.path.abspath(mri_path),
            "seg_path": os.path.abspath(seg_path),
            "original_shape": original_shape,
        })
        counts["ok"] += 1

    print()
    print(f"Manifest entries: {counts['ok']}")
    print(f"Skipped — bad filename:  {counts['bad_filename']}")
    print(f"Skipped — missing MRI:   {counts['missing_mri']}")
    print(f"Skipped — missing seg:   {counts['missing_seg']}")

    if not manifest:
        print("ERROR: No valid entries found. Check paths and embedding filenames.", file=sys.stderr)
        sys.exit(1)

    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to: {args.manifest_out}")
    print(f"\nFor the SLURM array job, set:")
    print(f"  #SBATCH --array=0-{len(manifest) - 1}")


if __name__ == "__main__":
    main()

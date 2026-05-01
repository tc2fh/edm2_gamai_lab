"""Interactive napari viewer for test-set probability maps.

Edit the three path variables below, then run:
    python view_napari.py

Layers shown:
    MRI            — grayscale background image
    GT Seg         — ground-truth binary segmentation (labels layer)
    Probability Map — voxel-wise generation probability, hot colormap

Requirements: napari, nibabel, numpy
    pip install napari[all] nibabel numpy
"""

import json
import os

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these three variables before running
# ---------------------------------------------------------------------------

MANIFEST_PATH = "test_generation_manifest.json"
GENERATION_ROOT = "/path/to/test_generations"
SCAN_ID = "123_1_45"  # scan_id to view (must match an entry in the manifest)

# ---------------------------------------------------------------------------

def load_volume(path):
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32)


def main():
    # --- Load manifest entry ---
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    entry = next((e for e in manifest if e["scan_id"] == SCAN_ID), None)
    if entry is None:
        raise ValueError(
            f"scan_id '{SCAN_ID}' not found in {MANIFEST_PATH}.\n"
            f"Available IDs: {[e['scan_id'] for e in manifest]}"
        )

    prob_path = os.path.join(GENERATION_ROOT, SCAN_ID, "probability_map.nii.gz")
    if not os.path.exists(prob_path):
        raise FileNotFoundError(
            f"Probability map not found: {prob_path}\n"
            "Run postprocess_test_generations.py first."
        )

    # --- Load volumes ---
    print(f"Loading MRI:      {entry['mri_path']}")
    mri = load_volume(entry["mri_path"])

    print(f"Loading GT seg:   {entry['seg_path']}")
    seg = load_volume(entry["seg_path"]).astype(np.uint8)

    print(f"Loading prob map: {prob_path}")
    prob = load_volume(prob_path)

    # Resize MRI and seg to match prob map if shapes differ
    if mri.shape != prob.shape:
        from scipy.ndimage import zoom as scipy_zoom
        scale = tuple(p / m for p, m in zip(prob.shape, mri.shape))
        print(f"  Resizing MRI/seg from {mri.shape} to {prob.shape}")
        mri = scipy_zoom(mri, scale, order=1)
        seg = scipy_zoom(seg, scale, order=0).astype(np.uint8)

    # --- Open napari ---
    import napari

    print(f"\nOpening napari for scan: {SCAN_ID}")
    viewer = napari.Viewer(title=f"GAM-AI Test — {SCAN_ID}")

    viewer.add_image(mri, name="MRI", colormap="gray")
    viewer.add_labels(seg, name="GT Seg")
    viewer.add_image(
        prob,
        name="Probability Map",
        colormap="hot",
        opacity=0.5,
        contrast_limits=[0.0, 1.0],
    )

    napari.run()


if __name__ == "__main__":
    main()

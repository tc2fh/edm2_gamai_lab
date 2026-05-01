"""Prepare 3D binary segmentation mask dataset from NIfTI files and VIVIT embeddings.

Walks a training directory, matches each {A_B_C}_T1_seg.nii.gz to its corresponding
embedding CSV, and outputs a flat dataset directory suitable for EDM2 training.

Output structure:
    outdir/
        00000000.npy   # (1, 128, 128, 128) uint8
        00000001.npy
        ...
        embeddings.npy # (N, 256, 768) float32
        dataset.json   # {"labels": [["00000000.npy", 0], ...]}
"""

import os
import re
import csv
import json
import click
import numpy as np
from scipy.ndimage import zoom

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")


def find_samples(train_dir, embeddings_dir):
    """Find matching NIfTI segmentation files and embedding CSVs."""
    samples = []
    for subdir in sorted(os.listdir(train_dir)):
        seg_path = os.path.join(train_dir, subdir, f'{subdir}_T1_seg.nii.gz')
        if not os.path.isfile(seg_path):
            continue

        # Embedding filename: embedding_A_A_B_C.csv where subdir is A_B_C
        # The first part before the first underscore is repeated
        parts = subdir.split('_')
        emb_name = f'embedding_{parts[0]}_{subdir}.csv'
        emb_path = os.path.join(embeddings_dir, emb_name)

        if not os.path.isfile(emb_path):
            print(f'Warning: No embedding found for {subdir}, expected {emb_path}')
            continue

        samples.append((seg_path, emb_path, subdir))

    return samples


def load_nifti_volume(path, target_shape=(256, 256, 256)):
    """Load a NIfTI file, resize to target_shape, binarize, return (1, D, H, W) uint8 and original shape."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    original_shape = data.shape

    # Resize if needed (use nearest-neighbor to preserve binary labels)
    if data.shape != target_shape:
        scale = tuple(t / s for t, s in zip(target_shape, data.shape))
        data = zoom(data, scale, order=0)  # order=0 = nearest neighbor

    # Binarize
    data = (data > 0).astype(np.uint8)

    return data[np.newaxis, :, :, :], original_shape  # (1, D, H, W), (D, H, W)


def load_embedding_csv(path):
    """Load VIVIT embedding from CSV. Returns (T, D) float32 array."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            rows.append([float(x) for x in row[1:]])  # skip index column
    return np.array(rows, dtype=np.float32)


@click.command()
@click.option('--train-dir', required=True, help='Path to training directory with subject subdirs')
@click.option('--embeddings-dir', required=True, help='Path to directory with embedding CSVs')
@click.option('--outdir', required=True, help='Output dataset directory')
@click.option('--resolution', default=128, help='Expected volume resolution (D=H=W)')
def main(train_dir, embeddings_dir, outdir, resolution):
    """Prepare 3D dataset for EDM2 training."""
    target_shape = (resolution, resolution, resolution)

    print(f'Scanning {train_dir} ...')
    samples = find_samples(train_dir, embeddings_dir)
    print(f'Found {len(samples)} matched samples')

    if len(samples) == 0:
        raise click.ClickException('No samples found')

    os.makedirs(outdir, exist_ok=True)

    labels = []
    all_embeddings = []
    original_shapes = {}

    for idx, (seg_path, emb_path, subdir) in enumerate(samples):
        # Load and save volume
        vol, orig_shape = load_nifti_volume(seg_path, target_shape)
        fname = f'{idx:08d}.npy'
        np.save(os.path.join(outdir, fname), vol)

        original_shapes[fname] = list(orig_shape)

        # Load embedding
        emb = load_embedding_csv(emb_path)
        all_embeddings.append(emb)

        # Label entry: [filename, index into embeddings.npy]
        labels.append([fname, idx])

        if (idx + 1) % 50 == 0 or idx == len(samples) - 1:
            print(f'  Processed {idx + 1}/{len(samples)}')

    # Save embeddings
    embeddings_array = np.stack(all_embeddings, axis=0)  # (N, T, D)
    np.save(os.path.join(outdir, 'embeddings.npy'), embeddings_array)
    print(f'Saved embeddings.npy with shape {embeddings_array.shape}')

    # Save dataset.json with original shapes for rescaling at generation time
    dataset_json = {
        'labels': labels,
        'original_shapes': original_shapes,
        'training_resolution': resolution,
    }
    with open(os.path.join(outdir, 'dataset.json'), 'w') as f:
        json.dump(dataset_json, f)
    print(f'Saved dataset.json with {len(labels)} entries')

    print('Done!')


if __name__ == '__main__':
    main()

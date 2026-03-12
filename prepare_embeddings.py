# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Prepare dataset.json mapping image filenames to VIVIT embeddings.

Reads CSV embedding files from a directory and maps them 1-to-1 to image
filenames in a dataset directory. Each CSV contains a 256x768 matrix where
all rows are nearly identical (single VIVIT embedding replicated); we take
the first row as the representative embedding.

The output dataset.json is written into the image directory in the format
expected by training.dataset.ImageFolderDataset.
"""

import os
import csv
import json
import re
import click


def load_embedding_from_csv(csv_path):
    """Load first row of embedding CSV, skipping index column."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        first_row = next(reader)
        return [float(x) for x in first_row[1:]]  # skip index column


@click.command()
@click.option('--embeddings-dir', help='Directory containing embedding CSV files', metavar='DIR', type=str, required=True)
@click.option('--images-dir', help='Directory containing image files (where dataset.json will be written)', metavar='DIR', type=str, required=True)
@click.option('--output', help='Output path for dataset.json [default: <images-dir>/dataset.json]', metavar='PATH', type=str, default=None)
def main(embeddings_dir, images_dir, output):
    """Map VIVIT embedding CSVs to image filenames and write dataset.json.

    Embedding CSV filenames are expected to follow the pattern:
    embedding_<id1>_<id2>_<batch>_<index>.csv

    Image filenames should contain matching identifiers so that a 1-to-1
    mapping can be established. The script will attempt to match based on
    the full numeric pattern in the filename.

    Examples:

    \b
    # Basic usage
    python prepare_embeddings.py \\
        --embeddings-dir=rivanna_dl/train_data_embeddings/ \\
        --images-dir=datasets/my_images/
    """
    if output is None:
        output = os.path.join(images_dir, 'dataset.json')

    # Discover embedding files
    emb_files = sorted(f for f in os.listdir(embeddings_dir) if f.endswith('.csv'))
    print(f'Found {len(emb_files)} embedding CSV files')

    # Discover image files
    import PIL.Image
    PIL.Image.init()
    supported_ext = set(PIL.Image.EXTENSION.keys()) | {'.npy'}
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in supported_ext
    )
    print(f'Found {len(image_files)} image files')

    # Build mapping: extract numeric identifiers from filenames
    # Embedding pattern: embedding_<id1>_<id2>_<batch>_<index>.csv
    emb_map = {}
    for emb_file in emb_files:
        match = re.findall(r'embedding_(\d+)_(\d+)_(\d+)_(\d+)', emb_file)
        if match:
            key = '_'.join(match[0])  # e.g., "106_106_1_115"
            emb_map[key] = emb_file

    # Try to match images to embeddings by shared numeric identifiers
    labels = []
    matched = 0
    unmatched_images = []

    for img_file in image_files:
        img_base = os.path.splitext(img_file)[0]
        # Try to find matching embedding by looking for shared numeric pattern
        found = False
        for key, emb_file in emb_map.items():
            if key in img_base or img_base in key:
                emb_path = os.path.join(embeddings_dir, emb_file)
                embedding = load_embedding_from_csv(emb_path)
                labels.append([img_file, embedding])
                matched += 1
                found = True
                break
        if not found:
            unmatched_images.append(img_file)

    if unmatched_images:
        print(f'Warning: {len(unmatched_images)} images could not be matched to embeddings')
        if len(unmatched_images) <= 10:
            for f in unmatched_images:
                print(f'  Unmatched: {f}')
        else:
            for f in unmatched_images[:5]:
                print(f'  Unmatched: {f}')
            print(f'  ... and {len(unmatched_images) - 5} more')

    print(f'Matched {matched} image-embedding pairs')

    if matched == 0:
        print('\nNo matches found. Please verify that image filenames contain')
        print('the same numeric identifiers as embedding CSV filenames.')
        print(f'\nExample embedding file: {emb_files[0] if emb_files else "N/A"}')
        print(f'Example image file: {image_files[0] if image_files else "N/A"}')
        print('\nYou may need to customize the matching logic in this script.')
        return

    # Write dataset.json
    dataset_json = {'labels': labels}
    with open(output, 'w') as f:
        json.dump(dataset_json, f)
    print(f'Wrote {output} with {len(labels)} entries')
    print(f'Embedding dimensionality: {len(labels[0][1])}')


if __name__ == '__main__':
    main()

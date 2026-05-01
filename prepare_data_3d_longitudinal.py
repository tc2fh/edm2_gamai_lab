"""Prepare longitudinal 3D dataset by pairing temporal embeddings with segmentation masks.

Walks the train and val embedding CSV directories produced by the GrowthNet
TemporalSpatialEmbedding extractor, pairs each CSV with the segmentation mask
at the same scan_id (looked up in the hierarchical UVA layout):

    <uva_root>/split/<split>/<patient_id>/<scan_id>/t2_thin/segmentations/*.nii.gz

and writes ONE combined EDM2-compatible dataset. The held-out test set is not
touched here. Each `embedding_<patient_id>_<scan_id>.csv` encodes the cumulative
scan history up to but not including `scan_id`, so pairing the embedding with
scan_id's own mask trains the diffusion model to predict the segmentation from
prior history.

Reuses `load_nifti_volume()` and `load_embedding_csv()` from prepare_data_3d.py.
"""

import os
import glob
import json
import click
import numpy as np

from prepare_data_3d import load_nifti_volume, load_embedding_csv


def parse_embedding_filename(fname):
    """Parse `embedding_<patient_id>_<scan_id>.csv` where scan_id starts with patient_id.

    Returns (patient_id, scan_id) as strings, or None if the filename is malformed.
    """
    if not fname.startswith('embedding_') or not fname.endswith('.csv'):
        return None
    stem = fname[len('embedding_'):-len('.csv')]
    parts = stem.split('_')
    if len(parts) < 4:
        return None
    if parts[1] != parts[0]:
        return None
    patient_id = parts[0]
    scan_id = '_'.join(parts[1:])
    return patient_id, scan_id


def find_segmentation(uva_root, split, patient_id, scan_id):
    """Return path to segmentation .nii.gz for (split, patient_id, scan_id), or None."""
    seg_dir = os.path.join(uva_root, 'split', split, patient_id, scan_id, 't2_thin', 'segmentations')
    matches = sorted(glob.glob(os.path.join(seg_dir, '*.nii.gz')))
    if not matches:
        return None
    if len(matches) > 1:
        print(f'Warning: {len(matches)} segs in {seg_dir}, using {os.path.basename(matches[0])}')
    return matches[0]


def find_samples(embeddings_dir, uva_root, split):
    """Walk embeddings_dir and pair each CSV with its corresponding segmentation.

    Returns (samples, counts) where samples is a list of (seg_path, emb_path, sample_key)
    and counts is a dict with scanned/matched/bad_filename/missing_seg tallies.
    """
    samples = []
    counts = {'scanned': 0, 'matched': 0, 'bad_filename': 0, 'missing_seg': 0}

    if not os.path.isdir(embeddings_dir):
        raise click.ClickException(f'Embeddings directory does not exist: {embeddings_dir}')

    entries = sorted(f for f in os.listdir(embeddings_dir) if f.startswith('embedding_') and f.endswith('.csv'))
    for fname in entries:
        counts['scanned'] += 1
        parsed = parse_embedding_filename(fname)
        if parsed is None:
            print(f'Warning: bad filename, skipping: {fname}')
            counts['bad_filename'] += 1
            continue
        patient_id, scan_id = parsed

        seg_path = find_segmentation(uva_root, split, patient_id, scan_id)
        if seg_path is None:
            print(f'Warning: no segmentation for {patient_id}/{scan_id} under {split}, skipping')
            counts['missing_seg'] += 1
            continue

        emb_path = os.path.join(embeddings_dir, fname)
        samples.append((seg_path, emb_path, f'{split}_{patient_id}_{scan_id}'))
        counts['matched'] += 1

    return samples, counts


def write_dataset(samples, outdir, target_shape, resolution):
    """Load all (seg, embedding) pairs and write the EDM2 dataset directory."""
    os.makedirs(outdir, exist_ok=True)

    labels = []
    all_embeddings = []
    original_shapes = {}
    expected_T = None
    load_errors = 0
    written = 0

    for seg_path, emb_path, sample_key in samples:
        try:
            vol, orig_shape = load_nifti_volume(seg_path, target_shape)
            emb = load_embedding_csv(emb_path)
        except Exception as e:
            print(f'Warning: load error for {sample_key} ({e}), skipping')
            load_errors += 1
            continue

        if expected_T is None:
            expected_T = int(emb.shape[0])
            print(f'Detected embedding token count T={expected_T}')
        elif emb.shape[0] != expected_T:
            raise click.ClickException(
                f'Inconsistent token count T at {emb_path}: got {emb.shape[0]}, '
                f'expected {expected_T}. Re-run the extractor with a single config.'
            )

        fname = f'{written:08d}.npy'
        np.save(os.path.join(outdir, fname), vol)
        original_shapes[fname] = list(orig_shape)
        all_embeddings.append(emb)
        labels.append([fname, written])
        written += 1

        if written % 50 == 0 or written == len(samples) - load_errors:
            print(f'  Processed {written}/{len(samples) - load_errors}')

    if written == 0:
        raise click.ClickException(f'All {len(samples)} samples failed to load')

    embeddings_array = np.stack(all_embeddings, axis=0)  # (N, T, 768)
    np.save(os.path.join(outdir, 'embeddings.npy'), embeddings_array)
    print(f'Saved embeddings.npy with shape {embeddings_array.shape}')

    dataset_json = {
        'labels': labels,
        'original_shapes': original_shapes,
        'training_resolution': resolution,
    }
    with open(os.path.join(outdir, 'dataset.json'), 'w') as f:
        json.dump(dataset_json, f)
    print(f'Saved dataset.json with {len(labels)} entries (load_errors={load_errors})')

    return expected_T, written


@click.command()
@click.option('--train-embeddings-dir', default='/scratch/tc2fh/gam_ai/GrowthNet/train_data_embeddings',
              show_default=True, help='Directory of train embedding CSVs')
@click.option('--val-embeddings-dir', default='/scratch/tc2fh/gam_ai/GrowthNet/val_data_embeddings',
              show_default=True, help='Directory of val embedding CSVs')
@click.option('--uva-root', default='/scratch/tc2fh/uva_vs_v0003',
              show_default=True, help='Root of the hierarchical UVA dataset (contains split/)')
@click.option('--outdir', required=True, help='Output directory for the combined dataset')
@click.option('--resolution', default=128, show_default=True, help='Target volume resolution (D=H=W)')
@click.option('--skip-train', is_flag=True, default=False, help='Exclude train split from the combined dataset')
@click.option('--skip-val', is_flag=True, default=False, help='Exclude val split from the combined dataset')
def main(train_embeddings_dir, val_embeddings_dir, uva_root,
         outdir, resolution, skip_train, skip_val):
    """Pair longitudinal embeddings with scans and build a combined EDM2 training dataset.

    The held-out test split is intentionally NOT included.
    """
    if skip_train and skip_val:
        raise click.ClickException('Both splits skipped; nothing to do')

    target_shape = (resolution, resolution, resolution)

    combined_samples = []
    per_split_counts = {}

    if not skip_train:
        print('\n=== Scanning train split ===')
        print(f'Embeddings dir: {train_embeddings_dir}')
        train_samples, train_counts = find_samples(train_embeddings_dir, uva_root, 'train')
        print(
            f'train: scanned={train_counts["scanned"]} matched={train_counts["matched"]} '
            f'skipped={train_counts["bad_filename"] + train_counts["missing_seg"]} '
            f'(bad_filename={train_counts["bad_filename"]}, missing_seg={train_counts["missing_seg"]})'
        )
        combined_samples.extend(train_samples)
        per_split_counts['train'] = train_counts['matched']
    else:
        print('Skipping train split')

    if not skip_val:
        print('\n=== Scanning val split ===')
        print(f'Embeddings dir: {val_embeddings_dir}')
        val_samples, val_counts = find_samples(val_embeddings_dir, uva_root, 'val')
        print(
            f'val: scanned={val_counts["scanned"]} matched={val_counts["matched"]} '
            f'skipped={val_counts["bad_filename"] + val_counts["missing_seg"]} '
            f'(bad_filename={val_counts["bad_filename"]}, missing_seg={val_counts["missing_seg"]})'
        )
        combined_samples.extend(val_samples)
        per_split_counts['val'] = val_counts['matched']
    else:
        print('Skipping val split')

    if not combined_samples:
        raise click.ClickException('No matched samples across train + val; refusing to write empty dataset')

    print(f'\n=== Writing combined dataset ===')
    print(f'Output dir:     {outdir}')
    print(f'Total samples:  {len(combined_samples)} (' +
          ', '.join(f'{k}={v}' for k, v in per_split_counts.items()) + ')')

    final_T, written = write_dataset(combined_samples, outdir, target_shape, resolution)
    print(f'\nDone. N={written} T={final_T}')


if __name__ == '__main__':
    main()

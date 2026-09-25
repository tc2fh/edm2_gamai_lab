"""Tests for prepare_data_vivit_pairs.py and training/encodings.py.

Run from the repo root with the pixi interpreter:
    python -m pytest tests/test_prepare_pairs.py -v

Builds a tiny synthetic token store (contract C2) in tmp_path, runs the pair
builder (contract C3) against it, and checks the file layout, encodings, and
pairing/truncation/split-isolation invariants documented in
docs/vivit_pipeline_contracts.md.
"""

import csv
import json
import math
import os

import numpy as np
import pytest
from click.testing import CliRunner

import prepare_data_vivit_pairs as pdp
from training.encodings import (
    decode_target,
    encode_cond_image,
    encode_target,
    is_degenerate_target,
    time_gap_feature,
)

# Deliberately NOT (1, 1, 2): contract C0 revised the real store's spacing to
# (0.5, 0.5, 1.0), so the builder must read `spacing` from the token npz files
# rather than hard-code a value; using a distinctive non-(1,1,2) spacing here
# means a regression back to a hard-coded constant would fail these tests.
SHAPE = (128, 128, 64)
SPACING = (0.5, 0.5, 1.0)

#----------------------------------------------------------------------------
# Synthetic C2 token store fixture.

def _make_blob_mask(rng, shape=SHAPE, radius=6):
    center = np.array([rng.integers(radius + 4, s - radius - 4) for s in shape])
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    dist2 = (ii - center[0]) ** 2 + (jj - center[1]) ** 2 + (kk - center[2]) ** 2
    mask = (dist2 <= radius ** 2).astype(np.uint8)
    return mask


def _write_scan_npz(path, *, scan_id, patient_id, split, days_since_first, scan_index, rng, spacing=None):
    mask = _make_blob_mask(rng)
    image = rng.standard_normal(SHAPE).astype(np.float16)
    tokens = rng.standard_normal((256, 768)).astype(np.float16)
    affine = np.eye(4, dtype=np.float64)
    np.savez(
        path,
        tokens=tokens,
        tokens_l3=tokens, tokens_l6=tokens, tokens_l9=tokens,
        mask=mask,
        image=image,
        affine=affine,
        spacing=np.array(spacing if spacing is not None else SPACING, dtype=np.float64),
        crop_origin=np.zeros(3, dtype=np.int64),
        resampled_shape=np.array(SHAPE, dtype=np.int64),
        resampled_affine=affine,
        source_image_path=f'/fake/{scan_id}_img.nii.gz',
        source_mask_path=f'/fake/{scan_id}_mask.nii.gz',
        scan_id=scan_id,
        patient_id=patient_id,
        split=split,
        days_since_first=np.float64(days_since_first),
        scan_index=np.int64(scan_index),
    )
    return mask


# (patient_id, split, n_scans)
FIXTURE_PATIENTS = [
    ('patA', 'train', 4),
    ('patB', 'val', 3),
    ('patC', 'test', 2),
]


def build_token_store(tmp_path):
    """Build a synthetic C2 token store; returns (tokens_dir, expected) where
    expected maps patient_id -> list of (scan_id, days_since_first) oldest first."""
    rng = np.random.default_rng(0)
    tokens_dir = tmp_path / 'vit_tokens'
    index = {'splits': {}, 'encoder_checkpoint': 'fake.pth', 'tap': 'spatial_encoder', 'created': '2026-09-11'}
    expected = {}

    for patient_id, split, n_scans in FIXTURE_PATIENTS:
        split_dir = tokens_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        days = sorted(rng.choice(np.arange(0, 400), size=n_scans, replace=False).tolist())
        scans_meta = []
        expected[patient_id] = []
        for k, day in enumerate(days):
            scan_id = f'{patient_id}_scan{k}'
            _write_scan_npz(
                split_dir / f'{scan_id}.npz',
                scan_id=scan_id, patient_id=patient_id, split=split,
                days_since_first=float(day), scan_index=k, rng=rng,
            )
            scans_meta.append({'scan_id': scan_id, 'days_since_first': float(day), 'file': f'{scan_id}.npz'})
            expected[patient_id].append((scan_id, float(day)))
        index['splits'].setdefault(split, []).append({'patient_id': patient_id, 'scans': scans_meta})

    with open(tokens_dir / 'index.json', 'w') as f:
        json.dump(index, f)

    return str(tokens_dir), expected


def run_builder(tokens_dir, out_dir, pairing='consecutive', max_history=4, splits='train,val,test'):
    runner = CliRunner()
    result = runner.invoke(pdp.main, [
        '--tokens-dir', tokens_dir,
        '--out', str(out_dir),
        '--pairing', pairing,
        '--max-history', str(max_history),
        '--splits', splits,
    ])
    assert result.exit_code == 0, result.output
    with open(os.path.join(out_dir, 'dataset.json')) as f:
        return json.load(f)

#----------------------------------------------------------------------------
# File layout and per-sample keys/dtypes/shapes (contract C3).

def test_file_layout_and_sample_keys(tmp_path):
    tokens_dir, expected = build_token_store(tmp_path)
    out = tmp_path / 'pairs_consecutive'
    dataset_json = run_builder(tokens_dir, out, pairing='consecutive', max_history=4)

    assert dataset_json['pairing'] == 'consecutive'
    assert dataset_json['max_history'] == 4
    assert dataset_json['spacing'] == list(SPACING)
    assert dataset_json['shape'] == [128, 128, 64]

    for split in ('train', 'val', 'test'):
        split_dir = out / split
        assert split_dir.is_dir()
        manifest = dataset_json['splits'][split]
        files = sorted(os.listdir(split_dir))
        assert files == [f'{i:08d}.npz' for i in range(manifest['n'])]

        for i, sample_meta in enumerate(manifest['samples']):
            assert sample_meta['idx'] == i
            with np.load(split_dir / f'{i:08d}.npz', allow_pickle=False) as npz:
                assert npz['target_mask'].dtype == np.uint8
                assert npz['target_mask'].shape == SHAPE
                assert npz['cond_mask'].dtype == np.uint8
                assert npz['cond_mask'].shape == SHAPE
                assert npz['cond_image'].dtype == np.float16
                assert npz['cond_image'].shape == SHAPE
                assert np.asarray(npz['delta_days']).dtype == np.float64
                assert npz['history_scan_ids'].ndim == 1
                assert npz['history_ages_days'].dtype == np.float64
                assert npz['history_ages_days'].shape == npz['history_scan_ids'].shape
                assert str(npz['patient_id']) == sample_meta['patient_id']
                assert str(npz['cond_scan_id']) == sample_meta['cond_scan_id']
                assert str(npz['target_scan_id']) == sample_meta['target_scan_id']
                assert str(npz['split']) == split

    # Train stats block is present and self-consistent.
    stats = dataset_json['stats']['train']
    for key in ('target_fraction_mean', 'target_rms_binary', 'target_std_binary',
                'target_rms_sdf', 'target_std_sdf', 'delta_days_max',
                'delta_days_median', 'history_len_max'):
        assert key in stats
    # Binary +-1 encoding always has rms exactly 1.
    assert stats['target_rms_binary'] == pytest.approx(1.0)

#----------------------------------------------------------------------------
# History ordering, ages, and max_history truncation.

def test_history_ordering_and_truncation(tmp_path):
    tokens_dir, expected = build_token_store(tmp_path)
    out = tmp_path / 'pairs_trunc'
    max_history = 2
    dataset_json = run_builder(tokens_dir, out, pairing='consecutive', max_history=max_history)

    patA_scans = expected['patA']  # 4 scans, oldest first: (scan_id, day)

    checked_truncated = False
    for sample_meta in dataset_json['splits']['train']['samples']:
        idx = sample_meta['idx']
        with np.load(out / 'train' / f'{idx:08d}.npz', allow_pickle=False) as npz:
            history_ids = [str(x) for x in npz['history_scan_ids']]
            ages = npz['history_ages_days']
            delta_days = float(npz['delta_days'])

            # oldest-first: ages strictly decreasing as we approach the target's own scan
            assert list(ages) == sorted(ages, reverse=True) or len(ages) == 1
            # last history entry is the conditioning scan; its age equals delta_days
            assert ages[-1] == pytest.approx(delta_days)
            assert history_ids[-1] == sample_meta['cond_scan_id']

            assert len(history_ids) <= max_history
            if sample_meta['n_history'] == max_history and idx > 0:
                # verify truncation actually dropped the oldest scan(s): the kept
                # ids must be a suffix of patA's full chronological scan list
                full_ids = [sid for sid, _ in patA_scans]
                if history_ids[-1] in full_ids:
                    pos = full_ids.index(history_ids[-1])
                    expected_kept = full_ids[max(0, pos - max_history + 1):pos + 1]
                    if history_ids == expected_kept and len(expected_kept) == max_history:
                        checked_truncated = True

    assert checked_truncated, 'expected at least one sample with truncated (most-recent) history'

#----------------------------------------------------------------------------
# Split isolation.

def test_split_isolation(tmp_path):
    tokens_dir, expected = build_token_store(tmp_path)
    out = tmp_path / 'pairs_iso'
    dataset_json = run_builder(tokens_dir, out, pairing='all', max_history=4)

    scan_ids_by_split = {}
    for split in ('train', 'val', 'test'):
        ids = set()
        for sample_meta in dataset_json['splits'][split]['samples']:
            idx = sample_meta['idx']
            with np.load(out / split / f'{idx:08d}.npz', allow_pickle=False) as npz:
                ids.add(str(npz['cond_scan_id']))
                ids.add(str(npz['target_scan_id']))
                ids.update(str(x) for x in npz['history_scan_ids'])
        scan_ids_by_split[split] = ids

    assert scan_ids_by_split['train'] & scan_ids_by_split['val'] == set()
    assert scan_ids_by_split['train'] & scan_ids_by_split['test'] == set()
    assert scan_ids_by_split['val'] & scan_ids_by_split['test'] == set()

    # idx restarts at 0 per split and is contiguous
    for split in ('train', 'val', 'test'):
        idxs = sorted(m['idx'] for m in dataset_json['splits'][split]['samples'])
        assert idxs == list(range(len(idxs)))

#----------------------------------------------------------------------------
# Sample counts for both pairing modes.

def test_sample_counts(tmp_path):
    tokens_dir, expected = build_token_store(tmp_path)
    n_scans_by_patient = {pid: len(scans) for pid, scans in expected.items()}

    out_consecutive = tmp_path / 'pairs_consecutive_count'
    dj_consecutive = run_builder(tokens_dir, out_consecutive, pairing='consecutive', max_history=4)
    expected_consecutive_total = sum(n - 1 for n in n_scans_by_patient.values())
    actual_consecutive_total = sum(dj_consecutive['splits'][s]['n'] for s in ('train', 'val', 'test'))
    assert actual_consecutive_total == expected_consecutive_total

    out_all = tmp_path / 'pairs_all_count'
    dj_all = run_builder(tokens_dir, out_all, pairing='all', max_history=4)
    expected_all_total = sum(math.comb(n, 2) for n in n_scans_by_patient.values())
    actual_all_total = sum(dj_all['splits'][s]['n'] for s in ('train', 'val', 'test'))
    assert actual_all_total == expected_all_total

#----------------------------------------------------------------------------
# Contract C0: spacing comes from the token store, never hard-coded.

def test_spacing_is_read_from_store_not_hardcoded(tmp_path):
    tokens_dir, expected = build_token_store(tmp_path)
    out = tmp_path / 'pairs_spacing'
    dataset_json = run_builder(tokens_dir, out, pairing='consecutive', max_history=4)
    # SPACING here is deliberately not (1, 1, 2); if the builder ever goes
    # back to hard-coding that value this assertion catches it directly.
    assert dataset_json['spacing'] == list(SPACING)
    assert dataset_json['spacing'] != [1.0, 1.0, 2.0]


def test_spacing_mismatch_across_scans_raises(tmp_path):
    """The token store must agree on one voxel spacing (contract C0); a scan
    that disagrees must abort the build with a clear error, not silently
    average or pick one value."""
    rng = np.random.default_rng(0)
    tokens_dir = tmp_path / 'vit_tokens_bad'
    split_dir = tokens_dir / 'train'
    split_dir.mkdir(parents=True)

    _write_scan_npz(
        split_dir / 'p_scan0.npz', scan_id='p_scan0', patient_id='p', split='train',
        days_since_first=0.0, scan_index=0, rng=rng, spacing=SPACING,
    )
    _write_scan_npz(
        split_dir / 'p_scan1.npz', scan_id='p_scan1', patient_id='p', split='train',
        days_since_first=30.0, scan_index=1, rng=rng, spacing=(1.0, 1.0, 2.0),
    )
    index = {
        'splits': {
            'train': [{'patient_id': 'p', 'scans': [
                {'scan_id': 'p_scan0', 'days_since_first': 0.0, 'file': 'p_scan0.npz'},
                {'scan_id': 'p_scan1', 'days_since_first': 30.0, 'file': 'p_scan1.npz'},
            ]}],
        },
        'encoder_checkpoint': 'fake.pth', 'tap': 'spatial_encoder', 'created': '2026-09-11',
    }
    with open(tokens_dir / 'index.json', 'w') as f:
        json.dump(index, f)

    runner = CliRunner()
    result = runner.invoke(pdp.main, [
        '--tokens-dir', str(tokens_dir),
        '--out', str(tmp_path / 'pairs_bad_spacing'),
        '--pairing', 'consecutive',
        '--max-history', '4',
        '--splits', 'train',
    ])
    assert result.exit_code != 0
    assert 'spacing' in result.output.lower()

#----------------------------------------------------------------------------
# training/encodings.py: SDF and binary encoder properties.

def test_binary_encoding_roundtrip():
    rng = np.random.default_rng(1)
    mask = _make_blob_mask(rng)
    x = encode_target(mask, 'binary')
    assert x.dtype == np.float32
    assert set(np.unique(x).tolist()) <= {-1.0, 1.0}
    decoded = decode_target(x, 'binary')
    assert np.array_equal(decoded, mask.astype(bool))


def test_sdf_encoding_properties_and_roundtrip():
    rng = np.random.default_rng(2)
    mask = _make_blob_mask(rng)
    x = encode_target(mask, 'sdf', SPACING)

    assert x.dtype == np.float32
    assert np.all(np.abs(x) <= 1.0 + 1e-6)
    # inside strictly negative, outside strictly positive (no proper mask
    # voxel sits exactly on the zero level set for a discrete EDT)
    assert np.all(x[mask.astype(bool)] < 0)
    assert np.all(x[~mask.astype(bool)] > 0)

    decoded = decode_target(x, 'sdf')
    assert np.array_equal(decoded, mask.astype(bool))


def test_sdf_degenerate_masks():
    empty = np.zeros(SHAPE, dtype=np.uint8)
    full = np.ones(SHAPE, dtype=np.uint8)

    assert is_degenerate_target(empty)
    assert is_degenerate_target(full)

    x_empty = encode_target(empty, 'sdf', SPACING)
    assert np.all(x_empty == 1.0)

    x_full = encode_target(full, 'sdf', SPACING)
    assert np.all(x_full == -1.0)

    rng = np.random.default_rng(3)
    normal_mask = _make_blob_mask(rng)
    assert not is_degenerate_target(normal_mask)


def test_sdf_uses_physical_spacing_mm():
    # A single foreground voxel at the center of a small grid. A neighbor one
    # index step away along the (coarser, 2mm) third axis must be exactly
    # 2mm away; one index step along the (1mm) first axis must be 1mm away.
    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[3, 3, 3] = 1
    x = encode_target(mask, 'sdf', spacing=(1.0, 1.0, 2.0))

    assert x[4, 3, 3] == pytest.approx(1.0 / 8.0)  # 1 mm outside, axis 0
    assert x[3, 4, 3] == pytest.approx(1.0 / 8.0)  # 1 mm outside, axis 1
    assert x[3, 3, 4] == pytest.approx(2.0 / 8.0)  # 2 mm outside, axis 2 (z)
    assert x[3, 3, 3] < 0  # the single foreground voxel itself is inside


def test_time_gap_feature():
    assert time_gap_feature(0) == pytest.approx(0.0)
    assert time_gap_feature(360) == pytest.approx(math.log1p(1.0))
    arr = time_gap_feature(np.array([0.0, 360.0, 720.0]))
    np.testing.assert_allclose(arr, np.log1p(np.array([0.0, 1.0, 2.0])))


#----------------------------------------------------------------------------
# --manifest mode (Phase 5): build the test split from a flow-repo-style
# pair_manifest.csv, restricted to pairs present in the token store's test
# split, history always [baseline].

def _write_flow_manifest_csv(path, rows):
    fieldnames = ['split', 'patient_id', 'baseline_scan_id', 'target_scan_id',
                  'baseline_day', 'target_day', 'delta_days',
                  'baseline_image_path', 'baseline_segmentation_path', 'target_segmentation_path']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_manifest_builder(tokens_dir, manifest_csv, out_dir):
    runner = CliRunner()
    result = runner.invoke(pdp.main, [
        '--tokens-dir', tokens_dir,
        '--out', str(out_dir),
        '--manifest', str(manifest_csv),
    ])
    assert result.exit_code == 0, result.output
    with open(os.path.join(out_dir, 'dataset.json')) as f:
        return json.load(f)


def test_manifest_mode_filters_and_builds_test_split(tmp_path):
    tokens_dir, expected = build_token_store(tmp_path)
    # patC (test split) has 2 scans: patC_scan0, patC_scan1.
    patC_scans = expected['patC']
    good_row = dict(
        split='test', patient_id='patC',
        baseline_scan_id=patC_scans[0][0], target_scan_id=patC_scans[1][0],
        baseline_day=patC_scans[0][1], target_day=patC_scans[1][1],
        delta_days=patC_scans[1][1] - patC_scans[0][1],
        baseline_image_path='/fake/img.nii.gz',
        baseline_segmentation_path='/fake/seg.nii.gz',
        target_segmentation_path='/fake/tseg.nii.gz',
    )
    # Row referencing a scan that only exists in train (patA), not test: must be skipped.
    bad_row = dict(
        split='test', patient_id='patA',
        baseline_scan_id='patA_scan0', target_scan_id='patA_scan1',
        baseline_day=0, target_day=100, delta_days=100,
        baseline_image_path='/fake/img2.nii.gz',
        baseline_segmentation_path='/fake/seg2.nii.gz',
        target_segmentation_path='/fake/tseg2.nii.gz',
    )
    # Row with split != test: must be ignored entirely (not even counted as skipped).
    train_row = dict(good_row, split='train')

    manifest_csv = tmp_path / 'pair_manifest.csv'
    _write_flow_manifest_csv(manifest_csv, [good_row, bad_row, train_row])

    out = tmp_path / 'flow_test_pairs'
    dataset_json = run_manifest_builder(tokens_dir, manifest_csv, out)

    assert dataset_json['pairing'] == 'manifest'
    assert dataset_json['max_history'] == 1
    assert set(dataset_json['splits'].keys()) == {'test'}
    assert 'train' not in dataset_json['splits'] and 'val' not in dataset_json['splits']

    manifest = dataset_json['splits']['test']
    assert manifest['n'] == 1
    assert not (out / 'train').exists()
    assert not (out / 'val').exists()
    assert sorted(os.listdir(out / 'test')) == ['00000000.npz']

    with np.load(out / 'test' / '00000000.npz', allow_pickle=False) as npz:
        assert str(npz['patient_id']) == 'patC'
        assert str(npz['cond_scan_id']) == patC_scans[0][0]
        assert str(npz['target_scan_id']) == patC_scans[1][0]
        assert len(npz['history_scan_ids']) == 1
        assert str(npz['history_scan_ids'][0]) == patC_scans[0][0]
        assert npz['history_ages_days'].shape == (1,)
        assert npz['history_ages_days'][0] == pytest.approx(float(npz['delta_days']))

    prov = dataset_json['manifest']
    assert prov['n_input_rows'] == 2  # only the two split=='test' rows
    assert prov['n_kept'] == 1
    assert prov['n_skipped'] == 1
    assert prov['skipped'][0]['patient_id'] == 'patA'
    assert 'baseline_scan_not_in_vit_test_cohort' in prov['skipped'][0]['reasons']
    assert 'target_scan_not_in_vit_test_cohort' in prov['skipped'][0]['reasons']

    # A test-only pair dir must not fabricate a 'train' stats entry (it has no
    # training data): generate_forecasts.py's --knockout=fixed-time falls back
    # to the checkpoint's own delta_days_median when this is absent, and
    # PairDataset takes an explicit sigma_data override, so nothing downstream
    # needs a fake 'train' alias any more.
    assert set(dataset_json['stats'].keys()) == {'test'}

    # Test-split stats block is present (needed so PairDataset can load split='test').
    stats = dataset_json['stats']['test']
    for key in ('target_fraction_mean', 'target_rms_binary', 'target_std_binary',
                'target_rms_sdf', 'target_std_sdf', 'delta_days_max',
                'delta_days_median', 'history_len_max'):
        assert key in stats
    assert stats['history_len_max'] == 1


def test_encode_cond_image_modes():
    rng = np.random.default_rng(4)
    mask = _make_blob_mask(rng)
    image = rng.standard_normal(SHAPE).astype(np.float32) * 10  # exceeds +-5 clip range

    none_ch = encode_cond_image(mask, image, 'none')
    assert none_ch.shape == (0,) + SHAPE

    mask_ch = encode_cond_image(mask, image, 'mask')
    assert mask_ch.shape == (1,) + SHAPE
    assert set(np.unique(mask_ch).tolist()) <= {-1.0, 1.0}

    both_ch = encode_cond_image(mask, image, 'mask+image')
    assert both_ch.shape == (2,) + SHAPE
    assert np.all(both_ch[1] <= 5.0) and np.all(both_ch[1] >= -5.0)
    np.testing.assert_array_equal(both_ch[0], mask_ch[0])

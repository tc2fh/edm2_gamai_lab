"""Unit tests for training.dataset.PairDataset against contracts C3-C5
(docs/vivit_pipeline_contracts.md), using a synthetic token store + pair
dataset (tests/pair_fixture.py) instead of real ViViT/prepare_data output."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.pair_fixture import build_pair_fixture
from training.dataset import PairDataset
from training.encodings import encode_target, encode_cond_image

SHAPE = (16, 16, 8) # small: dataset-logic tests don't need the real 128x128x64 volume.


@pytest.fixture(scope='module')
def fixture_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('pair_fixture')
    pair_dir, tokens_dir, samples = build_pair_fixture(tmp_path, shape=SHAPE)
    return pair_dir, tokens_dir, samples


def _find_sample(samples, n_history):
    return next(s for s in samples if s['n_history'] == n_history)


@pytest.mark.parametrize('encoding', ['binary', 'sdf'])
@pytest.mark.parametrize('cond_mode,expected_channels', [('none', 0), ('mask', 1), ('mask+image', 2)])
def test_batch_dict_shapes_and_dtypes(fixture_dir, encoding, cond_mode, expected_channels):
    pair_dir, tokens_dir, samples = fixture_dir
    ds = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train',
        target_encoding=encoding, cond_image=cond_mode)
    K = ds.max_history
    assert K == 3

    item = ds[0]
    assert item['image'].shape == (1,) + SHAPE
    assert item['image'].dtype == np.float32
    assert item['cond_image'].shape == (expected_channels,) + SHAPE
    assert item['cond_image'].dtype == np.float32
    assert item['context'].shape == (K, 256, 768)
    assert item['context'].dtype == np.float32
    assert item['context_mask'].shape == (K,)
    assert item['context_mask'].dtype == np.bool_
    assert item['context_ages'].shape == (K,)
    assert item['context_ages'].dtype == np.float32
    assert isinstance(item['delta_days'], np.float32)
    assert isinstance(item['idx'], np.int64)

    # Encoding range and decode round-trip sanity.
    assert item['image'].min() >= -1.0 - 1e-5 and item['image'].max() <= 1.0 + 1e-5
    if cond_mode != 'none':
        assert np.all(np.isin(item['cond_image'][0], [-1.0, 1.0])) # mask channel is +-1


def test_padding_and_mask_correctness(fixture_dir):
    pair_dir, tokens_dir, samples = fixture_dir
    ds = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train', target_encoding='binary', cond_image='mask')
    K = ds.max_history

    s1 = _find_sample(samples, n_history=1)
    item = ds[s1['idx']]
    assert item['context_mask'].tolist() == [True, False, False]
    assert np.array_equal(item['context'][1], np.zeros((256, 768), dtype=np.float32))
    assert np.array_equal(item['context'][2], np.zeros((256, 768), dtype=np.float32))
    assert not np.array_equal(item['context'][0], np.zeros((256, 768), dtype=np.float32))
    assert item['context_ages'][1] == 0.0 and item['context_ages'][2] == 0.0
    assert item['context_ages'][0] == pytest.approx(s1['delta_days'])
    assert item['delta_days'] == pytest.approx(s1['delta_days'])


def test_oldest_first_ordering(fixture_dir):
    pair_dir, tokens_dir, samples = fixture_dir
    ds = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train', target_encoding='binary', cond_image='mask')

    s3 = _find_sample(samples, n_history=3) # p3's last pair: full history, no padding.
    item = ds[s3['idx']]
    assert item['context_mask'].tolist() == [True, True, True]
    # Oldest-first: ages strictly decreasing, last entry equals delta_days.
    ages = item['context_ages']
    assert ages[0] > ages[1] > ages[2]
    assert ages[2] == pytest.approx(item['delta_days'])
    assert ages[2] == pytest.approx(s3['delta_days'])


def test_flips_applied_identically_to_image_and_cond_image(fixture_dir, monkeypatch):
    pair_dir, tokens_dir, samples = fixture_dir
    ds_flip = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train',
        target_encoding='binary', cond_image='mask+image', flip_axes=True)
    ds_noflip = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train',
        target_encoding='binary', cond_image='mask+image', flip_axes=False)

    # Force all three axes to flip (dataset.py checks `np.random.rand() < 0.5`).
    monkeypatch.setattr(np.random, 'rand', lambda: 0.0)
    item_flip = ds_flip[0]
    item_noflip = ds_noflip[0]

    expected_image = np.flip(item_noflip['image'], axis=(1, 2, 3))
    expected_cond = np.flip(item_noflip['cond_image'], axis=(1, 2, 3))
    np.testing.assert_array_equal(item_flip['image'], expected_image)
    np.testing.assert_array_equal(item_flip['cond_image'], expected_cond)

    # flip_axes is honored on the train split ...
    ds_train = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train',
        target_encoding='binary', cond_image='mask', flip_axes=True)
    assert ds_train._flip_axes is True


@pytest.mark.parametrize('split', ['val', 'test'])
@pytest.mark.parametrize('encoding,stat_key', [('binary', 'target_rms_binary'), ('sdf', 'target_rms_sdf')])
def test_val_and_test_splits_use_train_sigma_data(fixture_dir, split, encoding, stat_key):
    # Contract C3: dataset.json only ever has 'train' stats. A val/test
    # PairDataset must pull sigma_data from there, not raise a KeyError
    # trying to read its own (nonexistent) split entry, and must NOT define
    # its own normalization independent of train.
    import json
    pair_dir, tokens_dir, samples = fixture_dir
    with open(Path(pair_dir) / 'dataset.json') as f:
        manifest = json.load(f)

    ds = PairDataset(pair_dir, tokens_dir=tokens_dir, split=split, target_encoding=encoding)
    assert len(ds) >= 1
    assert ds.sigma_data == pytest.approx(manifest['stats']['train'][stat_key])


def test_explicit_sigma_data_override_needs_no_stats(tmp_path):
    # A test-only pair dir (e.g. built from another repo's manifest, as
    # gen-eval's flow-comparison export does) may legitimately have no
    # 'stats' block at all: generation reconstructs the dataset from a
    # checkpoint's C7 metadata and already knows sigma_data from the net,
    # so PairDataset must accept an explicit override and skip the stats
    # lookup entirely rather than raising IOError.
    import json
    pair_dir, tokens_dir, _samples = build_pair_fixture(tmp_path, shape=SHAPE, include_val_test=False)

    with open(Path(pair_dir) / 'dataset.json') as f:
        manifest = json.load(f)
    assert 'stats' in manifest # sanity: the fixture normally has one
    del manifest['stats']
    with open(Path(pair_dir) / 'dataset.json', 'w') as f:
        json.dump(manifest, f)

    ds = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train', target_encoding='binary', sigma_data=0.42)
    assert ds.sigma_data == pytest.approx(0.42)
    item = ds[0]
    assert item['image'].shape == (1,) + SHAPE

    # Without the override, the same stats-less manifest must still raise a
    # clear error rather than a bare KeyError.
    with pytest.raises(IOError):
        PairDataset(pair_dir, tokens_dir=tokens_dir, split='train', target_encoding='binary')

    item = ds[0]
    assert item['image'].shape == (1,) + SHAPE
    assert item['context'].shape[0] == ds.max_history


def test_sdf_uses_dataset_spacing_not_a_hardcoded_default(tmp_path):
    # Contract C0: voxel spacing is data-driven (the re-extracted store uses
    # (0.5, 0.5, 1.0) mm, not the old (1, 1, 2)). PairDataset must read
    # dataset.json's 'spacing' and pass it to encode_target for the sdf
    # encoding, never fall back to a hard-coded default.
    non_default_spacing = (0.5, 0.5, 1.0)
    assert non_default_spacing != (1.0, 1.0, 2.0)
    pair_dir, tokens_dir, samples = build_pair_fixture(tmp_path, shape=SHAPE, spacing=non_default_spacing)

    ds = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train', target_encoding='sdf')
    assert ds.spacing == non_default_spacing
    assert ds.dataset_kwargs['spacing'] == non_default_spacing

    with np.load(Path(pair_dir) / 'train' / '00000000.npz') as z:
        target_mask = z['target_mask']
    item = ds[0]

    expected_with_correct_spacing = encode_target(target_mask, 'sdf', spacing=non_default_spacing)
    expected_with_wrong_default = encode_target(target_mask, 'sdf', spacing=(1.0, 1.0, 2.0))

    np.testing.assert_allclose(item['image'][0], expected_with_correct_spacing)
    # The two spacings must actually give a different encoding for this
    # mask, or the assertion above would pass trivially either way.
    assert not np.allclose(expected_with_correct_spacing, expected_with_wrong_default)
    assert not np.allclose(item['image'][0], expected_with_wrong_default)


@pytest.mark.parametrize('encoding,stat_key', [('binary', 'target_rms_binary'), ('sdf', 'target_rms_sdf')])
def test_sigma_data_pickup(fixture_dir, encoding, stat_key):
    import json
    pair_dir, tokens_dir, samples = fixture_dir
    with open(Path(pair_dir) / 'dataset.json') as f:
        manifest = json.load(f)
    ds = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train', target_encoding=encoding)
    assert ds.sigma_data == pytest.approx(manifest['stats']['train'][stat_key])


def test_dataset_properties_and_kwargs(fixture_dir):
    pair_dir, tokens_dir, samples = fixture_dir
    ds = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train', target_encoding='binary', cond_image='mask+image')
    assert ds.resolution == SHAPE
    assert ds.num_channels == 1
    assert ds.cond_channels == 2
    assert ds.max_history == 3
    assert len(ds) == 6
    kwargs = ds.dataset_kwargs
    assert kwargs['class_name'] == 'training.dataset.PairDataset'
    assert kwargs['path'] == pair_dir
    assert kwargs['tokens_dir'] == tokens_dir
    assert kwargs['target_encoding'] == 'binary'
    assert kwargs['cond_image'] == 'mask+image'
    assert kwargs['max_history'] == 3


def test_encode_target_and_cond_image_match_direct_calls(fixture_dir):
    # Sanity: PairDataset's encoding matches calling training.encodings directly
    # on the same raw npz contents (i.e. dataset.py isn't reimplementing C4).
    import numpy as _np
    pair_dir, tokens_dir, samples = fixture_dir
    ds = PairDataset(pair_dir, tokens_dir=tokens_dir, split='train', target_encoding='sdf', cond_image='mask+image', flip_axes=False)
    item = ds[0]
    with _np.load(Path(pair_dir) / 'train' / '00000000.npz') as z:
        expected_image = encode_target(z['target_mask'], 'sdf')[_np.newaxis, ...]
        expected_cond = encode_cond_image(z['cond_mask'], z['cond_image'], 'mask+image')
    np.testing.assert_allclose(item['image'], expected_image)
    np.testing.assert_allclose(item['cond_image'], expected_cond)

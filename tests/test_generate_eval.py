"""Unit tests for generate_forecasts.py and evaluate_forecasts.py against
contracts C6/C7/C9 (docs/vivit_pipeline_contracts.md, plan Phase 3/4)."""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.pair_fixture import build_pair_fixture
from training.networks_edm2 import Precond
from tools.forecast_common import (
    classify_direction, dice_coeff, pack_samples, surface_dice, unpack_samples, volume_mm3,
)
from evaluate_forecasts import evaluate_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

RES = (128, 128, 64)

NET_KW = dict(
    img_resolution=RES, img_channels=1, cond_channels=1,
    context_dim=768, context_tokens=256, use_time_gap=True, sigma_data=1.0,
    model_channels=16, channel_mult=[1, 2, 2, 4], num_blocks=2, channels_per_head=32,
    attn_resolutions=[16], cross_attn_resolutions=[16, 32], dropout=0.1,
    use_fp16=True, dtype='fp16',
)


@pytest.fixture(scope='module')
def fixture_and_checkpoint(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('gen_eval_fixture')
    pair_dir, tokens_dir, samples = build_pair_fixture(tmp_path / 'data', shape=RES)

    torch.manual_seed(0)
    net = Precond(**NET_KW)
    ckpt = dict(
        ema=net.cpu().eval().requires_grad_(False).to(torch.float16),
        dataset_kwargs=dict(
            class_name='training.dataset.PairDataset', path=pair_dir, tokens_dir=tokens_dir,
            split='train', target_encoding='binary', cond_image='mask', max_history=3,
            token_key='tokens', flip_axes=False, sigma_data=1.0, img_resolution=list(RES),
        ),
    )
    net_path = tmp_path / 'net.pkl'
    with open(net_path, 'wb') as f:
        pickle.dump(ckpt, f)

    return dict(pair_dir=pair_dir, tokens_dir=tokens_dir, samples=samples, net_path=str(net_path), tmp_path=tmp_path)


def test_generate_forecasts_cli(fixture_and_checkpoint):
    from generate_forecasts import cmdline

    fx = fixture_and_checkpoint
    out_dir = fx['tmp_path'] / 'gen'
    runner = CliRunner()
    result = runner.invoke(cmdline, [
        '--net', fx['net_path'], '--data', fx['pair_dir'], '--split', 'train',
        '--out', str(out_dir), '--num-samples', '2', '--steps', '3', '--batch', '2',
        '--ids', '0,1', '--seed', '0', '--device', 'cuda',
    ])
    assert result.exit_code == 0, result.output

    split_dir = out_dir / 'train'
    manifest_path = split_dir / 'manifest.json'
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest['knockout'] == 'none'
    assert manifest['num_samples'] == 2

    for idx in (0, 1):
        npz_path = split_dir / f'{idx:08d}.npz'
        assert npz_path.exists()
        with np.load(npz_path, allow_pickle=True) as z:
            for key in ('samples_packed', 'samples_shape', 'prob', 'target_mask', 'cond_mask',
                        'delta_days', 'patient_id', 'cond_scan_id', 'target_scan_id', 'sampler_settings_json'):
                assert key in z.files, f'missing key {key} in {npz_path}'

            shape = tuple(int(s) for s in z['samples_shape'])
            assert shape == (2,) + RES
            unpacked = unpack_samples(z['samples_packed'], z['samples_shape'])
            assert unpacked.shape == (2,) + RES
            assert unpacked.dtype == bool

            prob = z['prob']
            assert prob.shape == RES
            assert prob.dtype == np.float16
            assert np.all((prob >= 0) & (prob <= 1))
            # Packbits round trip: prob should equal the mean of the unpacked samples exactly.
            expected_prob = unpacked.mean(axis=0).astype(np.float16)
            np.testing.assert_array_equal(prob, expected_prob)

            settings = json.loads(str(z['sampler_settings_json']))
            assert settings['steps'] == 3
            assert settings['num_samples'] == 2


@pytest.fixture(scope='module')
def fixture_without_delta_days_median(fixture_and_checkpoint):
    """A copy of the fixture pair dir whose dataset.json has no
    stats.train.delta_days_median -- simulating a pair dir (e.g. a
    test-only export like datasets/flow_test_pairs) that the fixed-time
    knockout cannot resolve a value from on its own."""
    import shutil

    fx = fixture_and_checkpoint
    src = Path(fx['pair_dir'])
    dst = fx['tmp_path'] / 'pairs_no_delta_median'
    shutil.copytree(src, dst)
    manifest_path = dst / 'dataset.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    del manifest['stats']['train']['delta_days_median']
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    return str(dst)


def test_generate_forecasts_none_knockout_ignores_missing_delta_days_median(
        fixture_and_checkpoint, fixture_without_delta_days_median):
    # The bug this guards: the fixed-time lookup used to run unconditionally
    # for every knockout, so a pair dir without delta_days_median crashed
    # generation even when the knockout never needed that value.
    from generate_forecasts import cmdline

    fx = fixture_and_checkpoint
    out_dir = fx['tmp_path'] / 'gen_none_no_median'
    runner = CliRunner()
    result = runner.invoke(cmdline, [
        '--net', fx['net_path'], '--data', fixture_without_delta_days_median, '--split', 'train',
        '--out', str(out_dir), '--num-samples', '2', '--steps', '3', '--batch', '2',
        '--ids', '0', '--seed', '0', '--knockout', 'none', '--device', 'cuda',
    ])
    assert result.exit_code == 0, result.output
    assert (out_dir / 'train' / '00000000.npz').exists()


def test_generate_forecasts_fixed_time_requires_explicit_value_when_unresolvable(
        fixture_and_checkpoint, fixture_without_delta_days_median):
    from generate_forecasts import cmdline

    fx = fixture_and_checkpoint
    out_dir = fx['tmp_path'] / 'gen_fixed_time_missing'
    runner = CliRunner()
    result = runner.invoke(cmdline, [
        '--net', fx['net_path'], '--data', fixture_without_delta_days_median, '--split', 'train',
        '--out', str(out_dir), '--num-samples', '2', '--steps', '3', '--batch', '2',
        '--ids', '0', '--seed', '0', '--knockout', 'fixed-time', '--device', 'cuda',
    ])
    assert result.exit_code != 0
    assert '--fixed-delta-days' in result.output


def test_generate_forecasts_fixed_time_with_explicit_delta_days(
        fixture_and_checkpoint, fixture_without_delta_days_median):
    from generate_forecasts import cmdline

    fx = fixture_and_checkpoint
    out_dir = fx['tmp_path'] / 'gen_fixed_time_explicit'
    runner = CliRunner()
    result = runner.invoke(cmdline, [
        '--net', fx['net_path'], '--data', fixture_without_delta_days_median, '--split', 'train',
        '--out', str(out_dir), '--num-samples', '2', '--steps', '3', '--batch', '2',
        '--ids', '0', '--seed', '0', '--knockout', 'fixed-time', '--fixed-delta-days', '123.5',
        '--device', 'cuda',
    ])
    assert result.exit_code == 0, result.output
    npz_path = out_dir / 'train' / '00000000.npz'
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=True) as z:
        assert float(np.asarray(z['delta_days'])) == pytest.approx(123.5)
        settings = json.loads(str(z['sampler_settings_json']))
        assert settings['delta_days_used'] == pytest.approx(123.5)
    with open(out_dir / 'train' / 'manifest.json') as f:
        run_manifest = json.load(f)
    assert run_manifest['fixed_delta_days'] == pytest.approx(123.5)


@pytest.fixture(scope='module')
def fixture_without_any_stats(fixture_and_checkpoint):
    """A test-only pair dir with no 'stats' block at all (e.g. a dataset
    exported by another repo, like datasets/flow_test_pairs): dataset
    construction must succeed by falling back to the checkpoint's own
    sigma_data (contract C7), never touching dataset.json stats."""
    import shutil

    fx = fixture_and_checkpoint
    src = Path(fx['pair_dir'])
    dst = fx['tmp_path'] / 'pairs_no_stats'
    shutil.copytree(src, dst)
    manifest_path = dst / 'dataset.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    del manifest['stats']
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    return str(dst)


def test_generate_forecasts_on_pair_dir_with_no_stats_block(fixture_and_checkpoint, fixture_without_any_stats):
    from generate_forecasts import cmdline

    fx = fixture_and_checkpoint
    out_dir = fx['tmp_path'] / 'gen_no_stats'
    runner = CliRunner()
    result = runner.invoke(cmdline, [
        '--net', fx['net_path'], '--data', fixture_without_any_stats, '--split', 'train',
        '--out', str(out_dir), '--num-samples', '2', '--steps', '3', '--batch', '2',
        '--ids', '0', '--seed', '0', '--knockout', 'none', '--device', 'cuda',
    ])
    assert result.exit_code == 0, result.output
    assert (out_dir / 'train' / '00000000.npz').exists()

    # fixed-time still needs an explicit value here (no train stats to fall
    # back to, and the checkpoint in this fixture carries none either).
    out_dir2 = fx['tmp_path'] / 'gen_no_stats_fixed_time'
    result2 = runner.invoke(cmdline, [
        '--net', fx['net_path'], '--data', fixture_without_any_stats, '--split', 'train',
        '--out', str(out_dir2), '--num-samples', '2', '--steps', '3', '--batch', '2',
        '--ids', '0', '--seed', '0', '--knockout', 'fixed-time', '--fixed-delta-days', '90',
        '--device', 'cuda',
    ])
    assert result2.exit_code == 0, result2.output


@pytest.fixture(scope='module')
def fixture_and_checkpoint_non_default_spacing(tmp_path_factory):
    """Same shape/architecture as fixture_and_checkpoint, but the pair dir
    (and its token store) use the real C0 training-frame spacing (0.5, 0.5,
    1.0) instead of the (1,1,2) most other fixtures default to -- an
    end-to-end guard against any hard-coded (1,1,2) surviving somewhere in
    generate_forecasts.py / evaluate_forecasts.py."""
    tmp_path = tmp_path_factory.mktemp('gen_eval_fixture_spacing')
    spacing = (0.5, 0.5, 1.0)
    pair_dir, tokens_dir, samples = build_pair_fixture(tmp_path / 'data', shape=RES, spacing=spacing)

    torch.manual_seed(0)
    net = Precond(**NET_KW)
    ckpt = dict(
        ema=net.cpu().eval().requires_grad_(False).to(torch.float16),
        dataset_kwargs=dict(
            class_name='training.dataset.PairDataset', path=pair_dir, tokens_dir=tokens_dir,
            split='train', target_encoding='binary', cond_image='mask', max_history=3,
            token_key='tokens', flip_axes=False, sigma_data=1.0, img_resolution=list(RES),
        ),
    )
    net_path = tmp_path / 'net.pkl'
    with open(net_path, 'wb') as f:
        pickle.dump(ckpt, f)

    return dict(pair_dir=pair_dir, tokens_dir=tokens_dir, samples=samples, net_path=str(net_path),
                tmp_path=tmp_path, spacing=spacing)


def test_generate_and_evaluate_use_non_default_spacing_end_to_end(fixture_and_checkpoint_non_default_spacing):
    """A hard-coded (1,1,2) anywhere in the generate/evaluate path would make
    this fail: volumes and NIfTI would be computed at 4x/2x the true voxel
    volume for this (0.5,0.5,1.0) fixture."""
    from generate_forecasts import cmdline as gen_cmdline
    from evaluate_forecasts import cmdline as eval_cmdline

    fx = fixture_and_checkpoint_non_default_spacing
    out_dir = fx['tmp_path'] / 'gen'
    runner = CliRunner()
    result = runner.invoke(gen_cmdline, [
        '--net', fx['net_path'], '--data', fx['pair_dir'], '--split', 'train',
        '--out', str(out_dir), '--num-samples', '2', '--steps', '3', '--batch', '2',
        '--ids', '0', '--seed', '0', '--device', 'cuda',
    ])
    assert result.exit_code == 0, result.output

    npz_path = out_dir / 'train' / '00000000.npz'
    with np.load(npz_path, allow_pickle=True) as z:
        assert 'spacing' in z.files
        stored_spacing = tuple(float(s) for s in z['spacing'])
        assert stored_spacing == pytest.approx(fx['spacing'])
        target_mask = z['target_mask'].astype(bool)
        expected_target_vol = float(target_mask.sum()) * np.prod(fx['spacing'])
        expected_target_vol_if_hardcoded = float(target_mask.sum()) * np.prod((1.0, 1.0, 2.0))

    result2 = runner.invoke(eval_cmdline, ['--gen-dir', str(out_dir / 'train')])
    assert result2.exit_code == 0, result2.output
    with open(out_dir / 'train' / 'pair_metrics.csv') as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    assert float(row['target_volume_mm3']) == pytest.approx(expected_target_vol)
    if expected_target_vol > 0:
        assert float(row['target_volume_mm3']) != pytest.approx(expected_target_vol_if_hardcoded)


@pytest.fixture(scope='module')
def no_context_checkpoint(fixture_and_checkpoint):
    """A checkpoint whose C7 metadata says use_context=False (the
    --no-context training path edm2-net is adding): context_dim=0 in the
    network, and dataset_kwargs carries use_context=False so
    tools.forecast_common.build_dataset reconstructs a PairDataset whose
    items have no 'context'/'context_mask'/'context_ages' keys at all."""
    fx = fixture_and_checkpoint
    torch.manual_seed(0)
    net_kw = dict(NET_KW)
    net_kw.update(context_dim=0, context_tokens=0, use_time_gap=True)
    net = Precond(**net_kw)
    ckpt = dict(
        ema=net.cpu().eval().requires_grad_(False).to(torch.float16),
        dataset_kwargs=dict(
            class_name='training.dataset.PairDataset', path=fx['pair_dir'], tokens_dir=fx['tokens_dir'],
            split='train', target_encoding='binary', cond_image='mask', max_history=3,
            token_key='tokens', flip_axes=False, sigma_data=1.0, img_resolution=list(RES),
            use_context=False,
        ),
    )
    net_path = fx['tmp_path'] / 'net_no_context.pkl'
    with open(net_path, 'wb') as f:
        pickle.dump(ckpt, f)
    return str(net_path)


def test_generate_forecasts_no_context_checkpoint_refuses_shuffle_tokens(fixture_and_checkpoint, no_context_checkpoint):
    from generate_forecasts import cmdline

    fx = fixture_and_checkpoint
    out_dir = fx['tmp_path'] / 'gen_no_context_shuffle_refused'
    runner = CliRunner()
    result = runner.invoke(cmdline, [
        '--net', no_context_checkpoint, '--data', fx['pair_dir'], '--split', 'train',
        '--out', str(out_dir), '--num-samples', '2', '--steps', '3', '--batch', '2',
        '--ids', '0', '--seed', '0', '--knockout', 'shuffle-tokens', '--device', 'cuda',
    ])
    assert result.exit_code != 0
    assert 'shuffle-tokens' in result.output
    assert 'use_context' in result.output


def test_generate_forecasts_no_context_checkpoint_none_knockout_works(fixture_and_checkpoint, no_context_checkpoint):
    from generate_forecasts import cmdline

    fx = fixture_and_checkpoint
    out_dir = fx['tmp_path'] / 'gen_no_context_none'
    runner = CliRunner()
    result = runner.invoke(cmdline, [
        '--net', no_context_checkpoint, '--data', fx['pair_dir'], '--split', 'train',
        '--out', str(out_dir), '--num-samples', '2', '--steps', '3', '--batch', '2',
        '--ids', '0', '--seed', '0', '--knockout', 'none', '--device', 'cuda',
    ])
    assert result.exit_code == 0, result.output
    assert (out_dir / 'train' / '00000000.npz').exists()
    with open(out_dir / 'train' / 'manifest.json') as f:
        manifest = json.load(f)
    assert manifest['use_context'] is False


def test_evaluate_forecasts_cli(fixture_and_checkpoint):
    from evaluate_forecasts import cmdline as eval_cmdline

    gen_dir = fixture_and_checkpoint['tmp_path'] / 'gen' / 'train'
    assert gen_dir.exists(), 'run test_generate_forecasts_cli first (module-scoped fixture)'

    runner = CliRunner()
    result = runner.invoke(eval_cmdline, ['--gen-dir', str(gen_dir)])
    assert result.exit_code == 0, result.output

    csv_path = gen_dir / 'pair_metrics.csv'
    summary_path = gen_dir / 'summary.json'
    assert csv_path.exists()
    assert summary_path.exists()

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for row in rows:
        assert 0.0 <= float(row['consensus_dice']) <= 1.0
        assert 0.0 <= float(row['surface_dice_1mm']) <= 1.0
        assert np.isfinite(float(row['abs_volume_error_mm3']))
        assert row['direction_pred'] in ('grew', 'stable', 'shrank')

    with open(summary_path) as f:
        summary = json.load(f)
    assert summary['n_pairs'] == 2
    assert np.isfinite(summary['direction_accuracy'])
    assert np.isfinite(summary['per_pair']['consensus_dice']['mean'])
    # Per-patient aggregation (mean over a patient's pairs first, then over
    # patients) must be reported alongside the flat per-pair means, for both
    # continuous metrics and the accuracy/coverage rates.
    for key in ('consensus_dice', 'direction_correct', 'interval_contains_target', 'cf_dice'):
        assert key in summary['per_patient'], f'missing per_patient[{key!r}]'
        assert np.isfinite(summary['per_patient'][key]['mean'])


#----------------------------------------------------------------------------
# Synthetic evaluation with a known answer: no network, no sampling, just
# evaluate_pair() on hand-built npz files.

def _write_synthetic_pair(path, cond_mask, target_mask, samples, patient_id='p1', spacing=(1.0, 1.0, 2.0)):
    packed, shape = pack_samples(samples)
    np.savez(
        path, samples_packed=packed, samples_shape=shape,
        prob=samples.mean(axis=0).astype(np.float16),
        target_mask=target_mask.astype(np.uint8), cond_mask=cond_mask.astype(np.uint8),
        delta_days=np.float64(30.0), patient_id=patient_id, cond_scan_id=f'{patient_id}_s0',
        target_scan_id=f'{patient_id}_s1', spacing=np.asarray(spacing, dtype=np.float64),
    )


def _block_mask(shape, n_voxels):
    m = np.zeros(shape, dtype=np.uint8)
    flat = m.reshape(-1)
    flat[:n_voxels] = 1
    return flat.reshape(shape)


def test_evaluate_pair_prediction_equals_target_gives_dice_one(tmp_path):
    shape = (8, 8, 4)
    target = _block_mask(shape, 12)
    cond = _block_mask(shape, 10)
    samples = np.stack([target, target, target]) # every sample == target exactly
    path = tmp_path / '00000000.npz'
    _write_synthetic_pair(path, cond, target, samples)

    metrics = evaluate_pair(str(path))
    assert metrics['consensus_dice'] == pytest.approx(1.0)
    assert metrics['surface_dice_1mm'] == pytest.approx(1.0)
    assert metrics['sample_dice_mean'] == pytest.approx(1.0)
    assert metrics['sample_dice_sd'] == pytest.approx(0.0)
    assert metrics['abs_volume_error_mm3'] == pytest.approx(0.0)
    assert metrics['interval_contains_target'] is True


@pytest.mark.parametrize('label,target_n', [('grew', 15), ('stable', 11), ('shrank', 5)])
def test_evaluate_pair_direction_classification_known_answer(tmp_path, label, target_n):
    # baseline = 10 voxels; grew = +50%, stable = +10% (within 20% band), shrank = -50%.
    shape = (8, 8, 4)
    cond = _block_mask(shape, 10)
    target = _block_mask(shape, target_n)
    # Predict "no change": every sample equals the conditioning mask exactly.
    samples = np.stack([cond, cond])
    path = tmp_path / '00000000.npz'
    _write_synthetic_pair(path, cond, target, samples)

    metrics = evaluate_pair(str(path))
    assert metrics['direction_true'] == label
    assert metrics['direction_pred'] == 'stable' # prediction is exactly the baseline: zero change
    assert metrics['direction_correct'] == (label == 'stable')
    # Carry-forward also predicts the baseline unchanged, so it matches the same pattern here.
    assert metrics['cf_direction_pred'] == 'stable'
    assert metrics['cf_direction_correct'] == (label == 'stable')
    # Carry-forward Dice/volume-error must be finite and in a sane range.
    assert 0.0 <= metrics['cf_dice'] <= 1.0
    assert 0.0 <= metrics['cf_surface_dice_1mm'] <= 1.0
    assert np.isfinite(metrics['cf_abs_volume_error_mm3'])
    assert metrics['cf_volume_mm3'] == pytest.approx(volume_mm3(cond, (1.0, 1.0, 2.0)))


def test_classify_direction_thresholds():
    assert classify_direction(100, 121, 0.20) == 'grew'   # +21%
    assert classify_direction(100, 119, 0.20) == 'stable'  # +19%
    assert classify_direction(100, 79, 0.20) == 'shrank'   # -21%
    assert classify_direction(100, 81, 0.20) == 'stable'   # -19%
    assert classify_direction(0, 0, 0.20) == 'stable'
    assert classify_direction(0, 5, 0.20) == 'grew'


def test_dice_and_surface_dice_empty_mask_conventions():
    shape = (8, 8, 4)
    spacing = (1.0, 1.0, 2.0)
    empty = np.zeros(shape, dtype=bool)
    full = _block_mask(shape, 10).astype(bool)
    assert dice_coeff(empty, empty) == 1.0
    assert dice_coeff(empty, full) == 0.0
    assert dice_coeff(full, empty) == 0.0
    assert surface_dice(empty, empty, spacing) == 1.0
    assert surface_dice(empty, full, spacing) == 0.0
    assert surface_dice(full, empty, spacing) == 0.0


#----------------------------------------------------------------------------
# Contract C0: the token store's physical voxel spacing is data-driven, not
# (1,1,2). These guard against any hard-coded spacing creeping back into
# volume_mm3/surface_dice or the generate/evaluate CLIs.

def test_volume_mm3_and_surface_dice_use_the_given_spacing_not_1_1_2():
    shape = (8, 8, 4)
    non_default_spacing = (0.5, 0.5, 1.0) # the real C0 training-frame spacing.
    mask = _block_mask(shape, 40).astype(bool)

    vol_default = volume_mm3(mask, (1.0, 1.0, 2.0))
    vol_actual = volume_mm3(mask, non_default_spacing)
    assert vol_default == pytest.approx(40 * 1.0 * 1.0 * 2.0)
    assert vol_actual == pytest.approx(40 * 0.5 * 0.5 * 1.0)
    assert vol_actual != pytest.approx(vol_default) # a hard-coded (1,1,2) would silently give vol_default here.

    # Surface Dice at a fixed 1mm tolerance must also depend on spacing: two
    # blocks offset by exactly one voxel along the z axis are 2mm apart at
    # spacing (1,1,2) (over the 1mm tolerance -> low surface Dice on the
    # offset faces) but only 1mm apart at (0.5,0.5,1.0) (at the tolerance ->
    # high surface Dice). A hard-coded (1,1,2) would give the same (low)
    # answer in both cases.
    big_shape = (10, 10, 10)
    a = np.zeros(big_shape, dtype=bool)
    a[3:7, 3:7, 3:5] = True
    b = np.zeros(big_shape, dtype=bool)
    b[3:7, 3:7, 4:6] = True # shifted by exactly 1 voxel along axis 2, no wraparound.
    sd_coarse = surface_dice(a, b, (1.0, 1.0, 2.0), tolerance_mm=1.0) # 2mm physical shift > 1mm tol.
    sd_fine = surface_dice(a, b, (0.5, 0.5, 1.0), tolerance_mm=1.0)   # 1mm physical shift <= 1mm tol.
    assert 0.0 <= sd_coarse <= 1.0
    assert 0.0 <= sd_fine <= 1.0
    assert sd_fine > sd_coarse # same voxel data, different spacing -> different (and correctly ordered) result.


def test_evaluate_pair_reads_spacing_from_npz_not_hardcoded(tmp_path):
    shape = (8, 8, 4)
    non_default_spacing = (0.5, 0.5, 1.0)
    cond = _block_mask(shape, 10)
    target = _block_mask(shape, 20)
    samples = np.stack([target, target])
    path = tmp_path / '00000000.npz'
    _write_synthetic_pair(path, cond, target, samples, spacing=non_default_spacing)

    metrics = evaluate_pair(str(path)) # spacing=None: must read the npz's own field.
    assert metrics['target_volume_mm3'] == pytest.approx(20 * 0.5 * 0.5 * 1.0)
    assert metrics['baseline_volume_mm3'] == pytest.approx(10 * 0.5 * 0.5 * 1.0)
    # A (1,1,2)-hardcoded reader would have reported 4x these volumes.
    assert metrics['target_volume_mm3'] != pytest.approx(20 * 1.0 * 1.0 * 2.0)


def test_evaluate_pair_missing_spacing_field_raises_clear_error(tmp_path):
    shape = (8, 8, 4)
    mask = _block_mask(shape, 10)
    samples = np.stack([mask, mask])
    path = tmp_path / '00000000.npz'
    # Write without the 'spacing' field (simulating a pre-C0 generation dir).
    packed, pshape = pack_samples(samples)
    np.savez(path, samples_packed=packed, samples_shape=pshape, prob=samples.mean(axis=0).astype(np.float16),
             target_mask=mask.astype(np.uint8), cond_mask=mask.astype(np.uint8), delta_days=np.float64(30.0),
             patient_id='p1', cond_scan_id='p1_s0', target_scan_id='p1_s1')
    with pytest.raises(ValueError, match='spacing'):
        evaluate_pair(str(path))
    # Passing spacing explicitly still works as an escape hatch.
    metrics = evaluate_pair(str(path), spacing=(1.0, 1.0, 2.0))
    assert np.isfinite(metrics['target_volume_mm3'])

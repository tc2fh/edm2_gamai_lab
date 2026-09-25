"""Tests for tools/make_flow_recipe.py's conditioning-knockout delta helper
(team-lead request, 2026-09-13). Pure numpy/stdlib -- runs under the EDM2
interpreter (no tumor_flow/torch import needed by this tool)."""

import json
import os

import numpy as np
import pytest

import tools.make_flow_recipe as mfr

SHAPE = (6, 6, 6)

#----------------------------------------------------------------------------

def _write_pair(gen_dir, split, idx, *, patient_id, prob, target_mask):
    split_dir = os.path.join(gen_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    np.savez(
        os.path.join(split_dir, f'{idx:08d}.npz'),
        prob=prob.astype(np.float16), target_mask=target_mask.astype(np.uint8),
        patient_id=patient_id,
    )


def _cube_mask(shape, lo, hi):
    m = np.zeros(shape, dtype=np.uint8)
    m[lo:hi, lo:hi, lo:hi] = 1
    return m

#----------------------------------------------------------------------------

def test_dice_conventions():
    empty = np.zeros(SHAPE, dtype=bool)
    full = np.ones(SHAPE, dtype=bool)
    a = _cube_mask(SHAPE, 1, 4).astype(bool)
    assert mfr._dice(empty, empty) == pytest.approx(1.0)
    assert mfr._dice(empty, full) == pytest.approx(0.0)
    assert mfr._dice(a, a) == pytest.approx(1.0)

#----------------------------------------------------------------------------
# knockout_delta: two patients, two pairs each. Full context predicts the
# target exactly (dice 1.0 every pair); the knockout run predicts a disjoint
# region (dice 0.0 every pair) for patient A and the SAME target (dice 1.0)
# for patient B (simulating "the knockout only hurts some patients"), so the
# expected per-patient deltas and the overall/bootstrap statistics are known
# by construction.

def test_knockout_delta_known_answer(tmp_path):
    full_dir = tmp_path / 'full'
    ko_dir = tmp_path / 'knockout'

    target_a = _cube_mask(SHAPE, 1, 4)
    target_b = _cube_mask(SHAPE, 2, 5)
    disjoint = _cube_mask(SHAPE, 0, 0)  # all-zero -> dice 0 against a non-empty target

    # Patient A: 2 pairs, full context = perfect, knockout = empty prediction (dice 0).
    _write_pair(full_dir, 'test', 0, patient_id='A', prob=target_a.astype(np.float32), target_mask=target_a)
    _write_pair(full_dir, 'test', 1, patient_id='A', prob=target_a.astype(np.float32), target_mask=target_a)
    _write_pair(ko_dir, 'test', 0, patient_id='A', prob=disjoint.astype(np.float32), target_mask=target_a)
    _write_pair(ko_dir, 'test', 1, patient_id='A', prob=disjoint.astype(np.float32), target_mask=target_a)

    # Patient B: 1 pair, full context = perfect, knockout = ALSO perfect (dice 1, unaffected).
    _write_pair(full_dir, 'test', 2, patient_id='B', prob=target_b.astype(np.float32), target_mask=target_b)
    _write_pair(ko_dir, 'test', 2, patient_id='B', prob=target_b.astype(np.float32), target_mask=target_b)

    result = mfr.knockout_delta(str(full_dir), str(ko_dir), split='test', n_boot=200, seed=2026)

    assert result['n_patients'] == 2
    assert result['n_pairs'] == 3
    # patient A: full mean dice 1.0, knockout mean dice 0.0 -> delta 1.0
    # patient B: full mean dice 1.0, knockout mean dice 1.0 -> delta 0.0
    # overall (mean over PATIENTS, not pairs): full=1.0, knockout=0.5, delta=0.5
    assert result['consensus_dice_full_mean'] == pytest.approx(1.0)
    assert result['consensus_dice_knockout_mean'] == pytest.approx(0.5)
    assert result['delta_mean'] == pytest.approx(0.5)
    # Bootstrap CI must bracket the observed delta and lie within [0, 1].
    assert result['delta_ci_low'] <= result['delta_mean'] <= result['delta_ci_high']
    assert 0.0 <= result['delta_ci_low'] and result['delta_ci_high'] <= 1.0
    assert result['n_boot'] == 200
    assert result['seed'] == 2026


def test_knockout_delta_is_seed_deterministic(tmp_path):
    full_dir = tmp_path / 'full'
    ko_dir = tmp_path / 'knockout'
    target = _cube_mask(SHAPE, 1, 5)
    for idx, pid in enumerate(['p1', 'p2', 'p3', 'p4']):
        _write_pair(full_dir, 'test', idx, patient_id=pid, prob=target.astype(np.float32), target_mask=target)
        noisy = target.copy()
        noisy[0, 0, 0] = 1  # small perturbation -> knockout dice slightly < 1
        _write_pair(ko_dir, 'test', idx, patient_id=pid, prob=noisy.astype(np.float32), target_mask=target)

    r1 = mfr.knockout_delta(str(full_dir), str(ko_dir), split='test', n_boot=500, seed=2026)
    r2 = mfr.knockout_delta(str(full_dir), str(ko_dir), split='test', n_boot=500, seed=2026)
    assert r1 == r2


def test_knockout_delta_patient_mismatch_raises(tmp_path):
    full_dir = tmp_path / 'full'
    ko_dir = tmp_path / 'knockout'
    target = _cube_mask(SHAPE, 1, 4)
    _write_pair(full_dir, 'test', 0, patient_id='A', prob=target.astype(np.float32), target_mask=target)
    _write_pair(ko_dir, 'test', 0, patient_id='B', prob=target.astype(np.float32), target_mask=target)
    with pytest.raises(ValueError, match='patient mismatch'):
        mfr.knockout_delta(str(full_dir), str(ko_dir), split='test')


def test_knockout_takeaway_mentions_both_arm_labels_and_ci():
    result = dict(
        consensus_dice_full_mean=0.7, consensus_dice_knockout_mean=0.4, delta_mean=0.3,
        delta_ci_low=0.1, delta_ci_high=0.5, n_boot=2000, seed=2026, n_patients=10,
    )
    text = mfr.knockout_takeaway(result, 'arm A (with ViT tokens)', 'arm C (no ViT tokens)')
    # Both the arm the knockout ran on AND the deck's separately-named featured
    # arm must appear -- the whole point of this fix is not to conflate them.
    assert 'arm A (with ViT tokens)' in text
    assert 'arm C (no ViT tokens)' in text
    assert '0.700' in text and '0.400' in text
    assert '+0.300' in text
    assert '2000' in text and '2026' in text
    assert 'n=10' in text
    # CI excludes zero -> a real effect, safe to claim the knockout arm uses its tokens.
    assert 'measurably uses its tokens' in text


def test_knockout_takeaway_null_result_does_not_overclaim():
    # CI straddles zero (the real arm-A run: delta +0.003, CI [-0.003, +0.011]):
    # the takeaway must NOT claim the model measurably uses its tokens, and must
    # still name the featured (different, no-token) arm as uninvolved.
    result = dict(
        consensus_dice_full_mean=0.473, consensus_dice_knockout_mean=0.469, delta_mean=0.0034,
        delta_ci_low=-0.003, delta_ci_high=0.011, n_boot=2000, seed=2026, n_patients=18,
    )
    text = mfr.knockout_takeaway(
        result, 'arm A (binary target + ViT tokens), same kimg 98 checkpoint',
        'EDM2 (mask + time gap, no ViT tokens)')
    assert 'measurably uses its tokens' not in text
    assert 'does not use its tokens' in text
    assert 'straddles zero' in text
    assert 'EDM2 (mask + time gap, no ViT tokens)' in text
    assert 'has no tokens by construction' in text
    assert 'not itself part of this knockout' in text

#----------------------------------------------------------------------------
# CLI-level wiring: --edm2-gen-dir/--knockout-dir must be given together, and
# --knockout-arm-label is required whenever --knockout-dir is given.

def test_cli_requires_both_knockout_args_together(tmp_path, monkeypatch):
    import sys

    old_agg = tmp_path / 'old_agg.json'
    new_agg = tmp_path / 'new_agg.json'
    old_agg.write_text(json.dumps({'overall': {}}))
    new_agg.write_text(json.dumps({'overall': {}}))

    argv = [
        'make_flow_recipe.py',
        '--old-name', 'Old', '--old-checkpoint', 'nope.pt', '--old-aggregate-json', str(old_agg),
        '--new-name', 'New', '--new-checkpoint', 'nope.pkl', '--new-aggregate-json', str(new_agg),
        '--out', str(tmp_path / 'recipe.json'),
        '--knockout-dir', str(tmp_path / 'ko'),  # missing --edm2-gen-dir
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit):
        mfr.main()


def test_cli_requires_knockout_arm_label(tmp_path, monkeypatch):
    import sys

    old_agg = tmp_path / 'old_agg.json'
    new_agg = tmp_path / 'new_agg.json'
    old_agg.write_text(json.dumps({'overall': {}}))
    new_agg.write_text(json.dumps({'overall': {}}))

    argv = [
        'make_flow_recipe.py',
        '--old-name', 'Old', '--old-checkpoint', 'nope.pt', '--old-aggregate-json', str(old_agg),
        '--new-name', 'New', '--new-checkpoint', 'nope.pkl', '--new-aggregate-json', str(new_agg),
        '--out', str(tmp_path / 'recipe.json'),
        '--edm2-gen-dir', str(tmp_path / 'full'), '--knockout-dir', str(tmp_path / 'ko'),
        # --knockout-arm-label deliberately omitted
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit):
        mfr.main()

"""Tests for tools/export_flow_analysis.py.

MUST run with the FLOW repo's interpreter (it imports tumor_flow.*):
    uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet python -m pytest \
        D:/Work/GrowthNet_gamailab/edm2_gamai_lab/tests/test_export_flow_analysis.py -v

Skipped entirely (not failed) if tumor_flow is not importable, e.g. if run
by mistake with the EDM2 repo's own pixi interpreter.
"""

import csv
import os
import sys

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip('tumor_flow')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import tools.export_flow_analysis as efa  # noqa: E402

from tumor_flow.data.crop import CropSpec  # noqa: E402
from tumor_flow.data.spatial import ReferenceGrid  # noqa: E402
from tumor_flow.evaluation.metrics import deterministic_pair_metrics, dice_score  # noqa: E402
from tumor_flow.evaluation.probabilistic import probabilistic_pair_metrics  # noqa: E402
from tumor_flow.flow.solver import VOLUME_QUANTILE_LEVELS  # noqa: E402

REAL_FLOW_ANALYSIS_DIR = (
    'D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/analysis_d040_vflow'
)

#----------------------------------------------------------------------------
# resample_and_crop: known-answer test against a hand-computed translation.
#
# Source: a (6,6,6) XYZ array with identity affine (voxel index == world mm),
# one foreground voxel at XYZ index (3,3,3) -> world (3,3,3) mm.
#
# Target reference grid: same (1,1,1)mm spacing, but its affine places grid
# index (0,0,0) at world (1,0,0), i.e. grid_index = world - (1,0,0). World
# (3,3,3) therefore lands at grid index (2,3,3).
#
# Crop spec: an identity crop over the whole (6,6,6) grid, so apply_crop is a
# straight copy and only the resample + ZYX transpose are exercised.
# xyz_array_to_zyx transposes (X,Y,Z) -> (Z,Y,X), so a voxel at XYZ (2,3,3)
# must land at ZYX (3,3,2).

def _identity_crop_spec(shape_xyz, affine):
    affine_t = tuple(tuple(float(v) for v in row) for row in affine)
    return CropSpec(
        requested_origin_xyz=(0, 0, 0),
        roi_size_xyz=tuple(shape_xyz),
        source_start_xyz=(0, 0, 0),
        source_stop_xyz=tuple(shape_xyz),
        destination_start_xyz=(0, 0, 0),
        destination_stop_xyz=tuple(shape_xyz),
        padding_before_xyz=(0, 0, 0),
        padding_after_xyz=(0, 0, 0),
        reference_shape_xyz=tuple(shape_xyz),
        reference_affine=affine_t,
        cropped_affine=affine_t,
    )


def test_resample_and_crop_known_translation_and_zyx_transpose():
    source = np.zeros((6, 6, 6), dtype=np.uint8)
    source[3, 3, 3] = 1
    source_affine = np.eye(4)

    grid_affine = np.eye(4)
    grid_affine[0, 3] = 1.0  # grid index (0,0,0) -> world (1,0,0)
    grid = ReferenceGrid(
        shape_xyz=(6, 6, 6), spacing_xyz_mm=(1.0, 1.0, 1.0),
        affine=tuple(tuple(float(v) for v in row) for row in grid_affine),
    )
    crop_spec = _identity_crop_spec((6, 6, 6), grid_affine)

    resampled_zyx = efa.resample_and_crop(source, source_affine, grid, crop_spec, order=0)
    assert resampled_zyx.shape == (6, 6, 6)
    assert resampled_zyx.dtype == bool

    nonzero = np.argwhere(resampled_zyx)
    assert len(nonzero) == 1, f'expected exactly one foreground voxel, got {nonzero}'
    z, y, x = nonzero[0]
    assert (z, y, x) == (3, 3, 2)


def test_resample_and_crop_probability_map_is_linear_and_unthresholded():
    # order=1 must return a continuous-valued array, not a boolean mask.
    source = np.zeros((6, 6, 6), dtype=np.float32)
    source[3, 3, 3] = 1.0
    source_affine = np.eye(4)
    grid = ReferenceGrid(
        shape_xyz=(6, 6, 6), spacing_xyz_mm=(1.0, 1.0, 1.0),
        affine=tuple(tuple(float(v) for v in row) for row in np.eye(4)),
    )
    crop_spec = _identity_crop_spec((6, 6, 6), np.eye(4))
    resampled_zyx = efa.resample_and_crop(source, source_affine, grid, crop_spec, order=1)
    assert resampled_zyx.dtype != bool
    assert resampled_zyx.max() <= 1.0 + 1e-6
    assert resampled_zyx[3, 3, 3] == pytest.approx(1.0, abs=1e-5)

#----------------------------------------------------------------------------
# load_source_mask_in_pair_frame: the frame-gate/annotation-agreement helper
# added for the team-lead's 2026-09-12 alignment-check redesign. It must
# reuse an ALREADY-COMPUTED grid/crop_spec (never recompute a fresh,
# mask-centred crop from the file it is given), so the same known-translation
# setup as test_resample_and_crop_known_translation_and_zyx_transpose above
# applies unchanged, this time reading a real NIfTI file from disk through
# tumor_flow's own resample_mask_to_grid rather than nibabel.processing
# directly.

def test_load_source_mask_in_pair_frame_known_translation_and_zyx_transpose(tmp_path):
    source = np.zeros((6, 6, 6), dtype=np.uint8)
    source[3, 3, 3] = 1
    source_affine = np.eye(4)
    seg_path = tmp_path / 'flat05_seg.nii.gz'
    nib.save(nib.Nifti1Image(source, source_affine), str(seg_path))

    grid_affine = np.eye(4)
    grid_affine[0, 3] = 1.0  # grid index (0,0,0) -> world (1,0,0), same as above
    grid = ReferenceGrid(
        shape_xyz=(6, 6, 6), spacing_xyz_mm=(1.0, 1.0, 1.0),
        affine=tuple(tuple(float(v) for v in row) for row in grid_affine),
    )
    crop_spec = _identity_crop_spec((6, 6, 6), grid_affine)

    result_zyx = efa.load_source_mask_in_pair_frame(seg_path, grid, crop_spec)
    assert result_zyx.dtype == bool
    assert result_zyx.shape == (6, 6, 6)
    nonzero = np.argwhere(result_zyx)
    assert len(nonzero) == 1, f'expected exactly one foreground voxel, got {nonzero}'
    z, y, x = nonzero[0]
    assert (z, y, x) == (3, 3, 2)


def test_load_source_mask_in_pair_frame_reuses_grid_not_a_fresh_mask_centred_crop(tmp_path):
    # A crop_spec built (via _identity_crop_spec) for a (6,6,6) reference grid
    # covers the whole grid regardless of where any particular mask's own
    # foreground sits -- so two files with foreground in very different
    # corners of the same grid must both come back at full (6,6,6) shape in
    # the SAME frame, not two different, independently mask-centred crops.
    # This is the property the team lead's design relies on: reusing the
    # pair's own grid/crop_spec rather than calling prepare_baseline_context
    # fresh on each file (which recentres the crop on whatever mask it gets).
    grid = ReferenceGrid(
        shape_xyz=(6, 6, 6), spacing_xyz_mm=(1.0, 1.0, 1.0),
        affine=tuple(tuple(float(v) for v in row) for row in np.eye(4)),
    )
    crop_spec = _identity_crop_spec((6, 6, 6), np.eye(4))

    mask_a = np.zeros((6, 6, 6), dtype=np.uint8)
    mask_a[0:2, 0:2, 0:2] = 1  # near one corner
    mask_b = np.zeros((6, 6, 6), dtype=np.uint8)
    mask_b[4:6, 4:6, 4:6] = 1  # near the opposite corner, same volume

    path_a = tmp_path / 'a.nii.gz'
    path_b = tmp_path / 'b.nii.gz'
    nib.save(nib.Nifti1Image(mask_a, np.eye(4)), str(path_a))
    nib.save(nib.Nifti1Image(mask_b, np.eye(4)), str(path_b))

    zyx_a = efa.load_source_mask_in_pair_frame(path_a, grid, crop_spec)
    zyx_b = efa.load_source_mask_in_pair_frame(path_b, grid, crop_spec)

    assert zyx_a.shape == zyx_b.shape == (6, 6, 6)
    # Both non-empty, in the SAME shared frame, and disjoint (opposite
    # corners) -- exactly what an "annotation disagreement" dice of 0.0 for
    # two totally different masks in the same pair frame should look like.
    assert zyx_a.any() and zyx_b.any()
    assert dice_score(zyx_a, zyx_b) == pytest.approx(0.0)
    assert dice_score(zyx_a, zyx_a) == pytest.approx(1.0)

#----------------------------------------------------------------------------
# Non-(1,1,2) spacing: contract C0 (2026-09-12) moved the token store to
# (0.5, 0.5, 1.0) mm with no reorientation; nothing in export_flow_analysis.py
# may assume any particular spacing. resample_and_crop/load_source_mask_in_
# pair_frame take spacing only via the source/grid affines, so re-running the
# existing known-answer geometry test at a different, non-uniform, non-(1,1,2)
# spacing (and a translation that is NOT axis-aligned with a spacing of 1mm)
# must still reproduce the exact expected voxel.

def test_resample_and_crop_is_spacing_agnostic():
    # A source array at physical spacing (0.5, 0.5, 1.0) mm (the current C0
    # token-store spacing). Foreground voxel chosen so its world position
    # lands exactly on a grid point of the (different-spacing) target grid
    # below, keeping this a known-answer test: XYZ index (3,2,3) -> world
    # (1.5, 1.0, 3.0) mm.
    source = np.zeros((6, 6, 6), dtype=np.uint8)
    source[3, 2, 3] = 1
    source_affine = np.diag([0.5, 0.5, 1.0, 1.0])

    grid_affine = np.eye(4)
    grid_affine[0, 3] = 0.5  # grid index (0,0,0) -> world (0.5, 0, 0)
    grid = ReferenceGrid(
        shape_xyz=(6, 6, 6), spacing_xyz_mm=(1.0, 1.0, 1.0),
        affine=tuple(tuple(float(v) for v in row) for row in grid_affine),
    )
    crop_spec = _identity_crop_spec((6, 6, 6), grid_affine)

    # World (1.5,1.0,3.0) -> grid index ((1.5-0.5)/1, 1.0/1, 3.0/1) = (1,1,3).
    # xyz_array_to_zyx transposes (X,Y,Z)->(Z,Y,X): (1,1,3) -> (3,1,1).
    resampled_zyx = efa.resample_and_crop(source, source_affine, grid, crop_spec, order=0)
    nonzero = np.argwhere(resampled_zyx)
    assert len(nonzero) == 1, f'expected exactly one foreground voxel, got {nonzero}'
    assert tuple(int(v) for v in nonzero[0]) == (3, 1, 1)

#----------------------------------------------------------------------------
# EDM2 field-of-view coverage (2026-09-12 token-store revision: the ViT crop
# shrank to a 64mm cube, generally smaller than the flow crop -- these
# functions must correctly report which flow-frame voxels are genuinely
# covered by the (now smaller) EDM2 source array vs zero-padded).

def test_edm2_fov_coverage_mask_partial_coverage():
    # EDM2 source array covers only world indices [0,4) in every axis (a
    # small "FOV"); the flow grid spans [0,8) in every axis at the same 1mm
    # spacing and orientation -- exactly the "EDM2 FOV smaller than the flow
    # crop" scenario the team lead flagged.
    grid = ReferenceGrid(
        shape_xyz=(8, 8, 8), spacing_xyz_mm=(1.0, 1.0, 1.0),
        affine=tuple(tuple(float(v) for v in row) for row in np.eye(4)),
    )
    crop_spec = _identity_crop_spec((8, 8, 8), np.eye(4))
    source_affine = np.eye(4)

    coverage_zyx = efa.edm2_fov_coverage_mask((4, 4, 4), source_affine, grid, crop_spec)
    assert coverage_zyx.shape == (8, 8, 8)
    assert coverage_zyx.dtype == bool
    # Covered exactly where all three (ZYX) indices are within [0,4).
    assert coverage_zyx[:4, :4, :4].all()
    assert not coverage_zyx[4:, :, :].any()
    assert not coverage_zyx[:, 4:, :].any()
    assert not coverage_zyx[:, :, 4:].any()
    assert not bool(coverage_zyx.all())  # partial coverage -> edm2_fov_covers_flow_crop would be False


def test_edm2_fov_coverage_mask_full_coverage():
    grid = ReferenceGrid(
        shape_xyz=(6, 6, 6), spacing_xyz_mm=(1.0, 1.0, 1.0),
        affine=tuple(tuple(float(v) for v in row) for row in np.eye(4)),
    )
    crop_spec = _identity_crop_spec((6, 6, 6), np.eye(4))
    coverage_zyx = efa.edm2_fov_coverage_mask((6, 6, 6), np.eye(4), grid, crop_spec)
    assert bool(coverage_zyx.all())  # source exactly matches the grid -> fully covered


def test_fraction_outside_fov():
    fov = np.zeros((8, 8, 8), dtype=bool)
    fov[:4, :4, :4] = True  # same partial-coverage region as above

    # Target mask: half inside the FOV, half outside, by construction.
    target = np.zeros((8, 8, 8), dtype=bool)
    target[2:4, 2:4, 2:4] = True   # 8 voxels, fully inside the FOV
    target[5:7, 5:7, 5:7] = True   # 8 voxels, fully outside the FOV

    assert efa.fraction_outside_fov(target, fov) == pytest.approx(0.5)
    assert efa.fraction_outside_fov(np.zeros((8, 8, 8), dtype=bool), fov) == pytest.approx(0.0)
    assert efa.fraction_outside_fov(target, np.ones((8, 8, 8), dtype=bool)) == pytest.approx(0.0)
    assert efa.fraction_outside_fov(target, np.zeros((8, 8, 8), dtype=bool)) == pytest.approx(1.0)

#----------------------------------------------------------------------------
# --alignment-only mode: a full, synthetic end-to-end run of main() (CLI arg
# parsing included, via sys.argv), needing no --generation-dir/--flow-analysis-
# dir/GPU -- exactly the mode/invocation the team lead asked to run against
# datasets/flow_test_pairs_tf. Builds a minimal but complete C2 token store
# entry, C3 pair-dir npz, flow pair_manifest.csv row, and flow config, all in
# perfect agreement (same mask everywhere), so the expected output is
# unambiguous: dice 1.0 on every alignment column and full FOV coverage.

def _write_seg_nifti(path, mask, affine=None):
    affine = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))


def _write_mri_nifti(path, data, affine=None):
    affine = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def _build_alignment_only_fixture(tmp_path):
    shape = (8, 8, 8)
    affine = np.eye(4)

    mri = np.ones(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1  # one shared mask used everywhere -> perfect agreement

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    baseline_image_path = data_dir / 'baseline_image.nii.gz'
    baseline_seg_path = data_dir / 'baseline_seg_v0005.nii.gz'
    target_seg_path = data_dir / 'target_seg_v0005.nii.gz'
    flat05_baseline_seg_path = data_dir / 'baseline_seg_flat05.nii.gz'
    flat05_target_seg_path = data_dir / 'target_seg_flat05.nii.gz'
    _write_mri_nifti(baseline_image_path, mri, affine)
    _write_seg_nifti(baseline_seg_path, mask, affine)
    _write_seg_nifti(target_seg_path, mask, affine)
    _write_seg_nifti(flat05_baseline_seg_path, mask, affine)
    _write_seg_nifti(flat05_target_seg_path, mask, affine)

    manifest_csv = tmp_path / 'pair_manifest.csv'
    with open(manifest_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'patient_id', 'baseline_scan_id', 'target_scan_id', 'baseline_day',
                          'target_day', 'delta_days', 'baseline_image_path', 'baseline_segmentation_path',
                          'target_segmentation_path'])
        writer.writerow(['test', 'p1', 'p1_scanA', 'p1_scanB', '0', '100', '100',
                          str(baseline_image_path), str(baseline_seg_path), str(target_seg_path)])

    flow_config_path = tmp_path / 'flow_config.yaml'
    flow_config_path.write_text(
        'data:\n'
        '  modality: t2_thin\n'
        '  target_spacing_xyz_mm: [1.0, 1.0, 1.0]\n'
        '  roi_size_xyz: [8, 8, 8]\n'
        'evaluation:\n'
        '  surface_dice_tolerance_mm: 1.0\n'
        '  stable_volume_change_fraction: 0.2\n'
    )

    tokens_dir = tmp_path / 'vit_tokens'
    (tokens_dir / 'test').mkdir(parents=True)
    for scan_id, seg_path in (('p1_scanA', flat05_baseline_seg_path), ('p1_scanB', flat05_target_seg_path)):
        np.savez(
            tokens_dir / 'test' / f'{scan_id}.npz',
            affine=affine, source_mask_path=str(seg_path),
        )

    pairs_dir = tmp_path / 'flow_test_pairs_tf'
    (pairs_dir / 'test').mkdir(parents=True)
    np.savez(
        pairs_dir / 'test' / '00000000.npz',
        target_mask=mask, cond_mask=mask, delta_days=np.float64(100),
        history_scan_ids=np.array(['p1_scanA']), history_ages_days=np.array([100.0]),
        patient_id='p1', cond_scan_id='p1_scanA', target_scan_id='p1_scanB', split='test',
    )
    dataset_json = dict(
        tokens_dir=str(tokens_dir), pairing='manifest', max_history=1, spacing=[0.5, 0.5, 1.0],
        shape=[8, 8, 8],
        splits={'test': dict(n=1, patients=['p1'], samples=[
            dict(idx=0, patient_id='p1', cond_scan_id='p1_scanA', target_scan_id='p1_scanB',
                 delta_days=100.0, n_history=1)])},
        stats={'test': dict()},
    )
    import json as _json
    (pairs_dir / 'dataset.json').write_text(_json.dumps(dataset_json))

    return dict(manifest=manifest_csv, flow_config=flow_config_path, tokens_dir=tokens_dir, pairs_dir=pairs_dir)


def test_alignment_only_mode_end_to_end(tmp_path, monkeypatch):
    fx = _build_alignment_only_fixture(tmp_path)
    out_dir = tmp_path / 'out'

    argv = [
        'export_flow_analysis.py',
        '--alignment-only',
        '--pairs-dir', str(fx['pairs_dir']),
        '--tokens-dir', str(fx['tokens_dir']),
        '--manifest', str(fx['manifest']),
        '--flow-config', str(fx['flow_config']),
        '--out', str(out_dir),
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    efa.main()  # must not raise: --generation-dir/--flow-analysis-dir are not required here

    csv_path = out_dir / 'alignment_check.csv'
    summary_path = out_dir / 'alignment_summary.json'
    assert csv_path.is_file()
    assert summary_path.is_file()

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row['patient_id'] == 'p1'
    # Same mask used as the EDM2/ViT-frame mask, the flat_05 source file, and
    # the flow manifest's v0005 file everywhere -> every alignment dice is 1.0.
    for col in ('frame_dice_baseline', 'frame_dice_target', 'annotation_dice_baseline', 'annotation_dice_target'):
        assert float(row[col]) == pytest.approx(1.0), f'{col}={row[col]!r}'
    assert row['edm2_fov_covers_flow_crop'] == 'True'
    assert float(row['frac_target_outside_edm2_fov']) == pytest.approx(0.0)

    import json as _json
    summary = _json.loads(summary_path.read_text())
    assert summary['n_pairs'] == 1
    assert summary['frame_dice_baseline_mean'] == pytest.approx(1.0)
    assert summary['frame_dice_target_mean'] == pytest.approx(1.0)
    assert summary['annotation_disagreements'] == []
    assert summary['edm2_fov_coverage']['n_pairs_fully_covered'] == 1


def test_alignment_only_does_not_require_generation_dir_or_flow_analysis_dir(tmp_path, monkeypatch):
    # The opposite check: full mode DOES require them.
    fx = _build_alignment_only_fixture(tmp_path)
    argv = [
        'export_flow_analysis.py',
        '--pairs-dir', str(fx['pairs_dir']),
        '--tokens-dir', str(fx['tokens_dir']),
        '--manifest', str(fx['manifest']),
        '--flow-config', str(fx['flow_config']),
        '--out', str(tmp_path / 'out2'),
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit):
        efa.main()

#----------------------------------------------------------------------------
# unpack_samples round trip (pure numpy, mirrors tools/forecast_common.py).

def test_unpack_samples_round_trip():
    rng = np.random.default_rng(0)
    masks = rng.random((5, 8, 8, 8)) > 0.5
    packed = np.packbits(masks, axis=-1)
    shape = np.array(masks.shape, dtype=np.int64)
    unpacked = efa.unpack_samples(packed, shape)
    np.testing.assert_array_equal(unpacked, masks)

#----------------------------------------------------------------------------
# pair_metrics.csv column set: every column export_flow_analysis.py can
# populate from its own computation (i.e. every column NOT prefixed
# 'deterministic_', which is instead copied verbatim from the flow repo's own
# analysis) must appear in the real flow analysis's pair_metrics.csv header,
# and vice versa -- so the two directories are structurally interchangeable
# for build_comparison_deck.py and any other pair_metrics.csv consumer.

@pytest.mark.skipif(not os.path.isfile(os.path.join(REAL_FLOW_ANALYSIS_DIR, 'pair_metrics.csv')),
                     reason='real flow analysis directory not present on this machine')
def test_computed_columns_match_real_pair_metrics_header():
    with open(os.path.join(REAL_FLOW_ANALYSIS_DIR, 'pair_metrics.csv'), newline='') as f:
        real_header = next(csv.reader(f))

    shape = (8, 10, 10)
    rng = np.random.default_rng(1)
    baseline = rng.random(shape) > 0.7
    target = rng.random(shape) > 0.7
    predicted = rng.random(shape) > 0.7
    spacing = (1.0, 1.0, 2.0)
    tol = 2.0

    consensus_metrics = deterministic_pair_metrics(predicted, baseline, target, spacing, tol)
    carry_metrics = deterministic_pair_metrics(baseline, baseline, target, spacing, tol)
    sample_masks = np.stack([rng.random(shape) > 0.7 for _ in range(4)])
    sample_volumes = np.array([efa.mask_volume_mm3(m, spacing) for m in sample_masks])
    prob = sample_masks.mean(axis=0).astype(np.float32)
    volume_quantiles = np.quantile(sample_volumes, VOLUME_QUANTILE_LEVELS)
    probabilistic_metrics = probabilistic_pair_metrics(
        prob, sample_masks, sample_volumes, volume_quantiles, VOLUME_QUANTILE_LEVELS,
        target, float(consensus_metrics['target_volume_mm3']),
        float(consensus_metrics['baseline_volume_mm3']), 0.05)

    header_fields = {
        'split', 'patient_id', 'baseline_scan_id', 'target_scan_id', 'baseline_day', 'target_day',
        'delta_days', 'horizon_bin', 'num_flow_samples', 'flow_steps', 'flow_solver',
        'guidance_weight', 'guidance_checkpoint_sha256', 'seed', 'surface_dice_tolerance_mm',
        'crop_touches_boundary', 'baseline_volume_mm3', 'target_volume_mm3',
        'sample_dice_mean', 'sample_dice_sd', 'sample_surface_dice_mean', 'sample_surface_dice_sd',
    }
    computed_keys = set(header_fields)
    computed_keys |= set(efa.prefixed('consensus', consensus_metrics))
    computed_keys |= set(probabilistic_metrics)
    computed_keys |= set(efa.quantile_columns(sample_volumes))
    computed_keys |= set(efa.prefixed('carry_forward', carry_metrics))

    real_non_deterministic = {c for c in real_header if not c.startswith('deterministic_')}
    assert computed_keys == real_non_deterministic, (
        f'mismatch.\nmissing from our computation: {real_non_deterministic - computed_keys}\n'
        f'extra in our computation: {computed_keys - real_non_deterministic}'
    )

    real_deterministic = {c for c in real_header if c.startswith('deterministic_')}
    # Every deterministic_ column name is deterministic_pair_metrics's own key
    # set (minus baseline/target volume) with a 'deterministic_' prefix, since
    # deterministic_passthrough copies them by name from the flow repo's own row.
    metric_keys = set(consensus_metrics) - {'baseline_volume_mm3', 'target_volume_mm3'}
    expected_deterministic = {f'deterministic_{k}' for k in metric_keys}
    assert real_deterministic == expected_deterministic

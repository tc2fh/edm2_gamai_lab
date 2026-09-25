"""Export EDM2 forecasts into a flow-repo-compatible analysis directory
(docs/vivit_conditioning_plan.md Phase 5, docs/vivit_pipeline_contracts.md).

MUST be run with the FLOW repo's interpreter, e.g.:

    uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet python \\
        D:/Work/GrowthNet_gamailab/edm2_gamai_lab/tools/export_flow_analysis.py ...

This script imports `tumor_flow.*` (flow repo, its own torch) to compute every
metric with the exact same code the flow repo's own analysis used, so the two
analysis directories are scored identically. It reaches into the EDM2 repo
only for plain numpy/json/npz reads (never torch), so the two repos' torch
builds never have to coexist in one process.

Per pair: the EDM2 samples (and probability map) live in the ViT crop frame
(128,128,64 XYZ; spacing and affine read from the conditioning scan's C2
token npz -- see docs/vivit_pipeline_contracts.md C0, NEVER hard-coded here,
since the token store's physical spacing/orientation is a per-store choice
that has already changed once). Each is resampled (linear, thresholded at 0.5
for masks; linear, unthresholded for the probability map -- see
resample_and_crop's docstring) into the flow repo's own reference grid for
that pair -- built from the same baseline NIfTI the flow repo used, via
`tumor_flow.data.preprocessing` -- then cropped with the flow's own CropSpec
and transposed to ZYX, landing in exactly the frame `arrays.npz` uses.
Baseline and target masks for scoring are the flow repo's own preprocessed
masks (same annotation files, same frame), not a resampled copy of the EDM2
dataset's own mask, so metrics are computed against the same ground truth the
flow analysis used.

The EDM2/ViT crop's physical field of view need not cover the flow crop's
field of view (it does not: as of the 2026-09-12 token-store revision the ViT
crop is a 64mm cube, generally smaller than the flow crop). Voxels of the
flow-frame array that fall outside the EDM2 source array's bounds come back
as background (0) via nibabel's constant-fill resampling; process_pair()
additionally computes `edm2_fov_covers_flow_crop` and the fraction of the
flow target mask lying outside the EDM2 FOV per pair (alignment_check.csv,
aggregate.json settings.alignment_check), so silent zero-padding of a
tumour that is actually outside EDM2's view is visible rather than read as a
false negative.

`deterministic_*` columns are copied verbatim from the flow repo's own
pair_metrics.csv for the same (patient_id, target_scan_id): the deterministic
forecaster is the flow repo's model, run in the identical frame, so
recomputing it here would just reproduce the same numbers via a slower path.

The mandatory alignment check (see main()) makes two decoupled comparisons per
pair: a strict, never-excused FRAME gate (the same uva_vs_flat_05 source file
run through two pipelines) and a logged-only ANNOTATION agreement check (the
same pipeline run on the uva_vs_flat_05 file vs the flow manifest's
uva_vs_v0005 file). See analysis/README.md's "Alignment check design" section
for the full rationale and the known annotation disagreements.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import yaml
from nibabel.processing import resample_from_to

# --------------------------------------------------------------------------
# Make the EDM2 repo importable by path, for plain-numpy reads only (no torch
# import from this path -- see module docstring).
EDM2_REPO = os.environ.get(
    'EDM2_REPO', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if EDM2_REPO not in sys.path:
    sys.path.insert(0, EDM2_REPO)

# --------------------------------------------------------------------------
# Flow repo imports (this script's own interpreter).
from tumor_flow.config import DataConfig
from tumor_flow.data.crop import apply_crop
from tumor_flow.data.preprocessing import prepare_baseline_context, prepare_future_target
from tumor_flow.data.spatial import load_canonical_nifti, resample_mask_to_grid, xyz_array_to_zyx
from tumor_flow.evaluation.metrics import (
    deterministic_pair_metrics, dice_score, mask_volume_mm3, surface_dice_at_tolerance,
)
from tumor_flow.evaluation.probabilistic import probabilistic_pair_metrics
from tumor_flow.flow.solver import VOLUME_QUANTILE_LEVELS
import tumor_flow.analysis.forecasting as forecasting_mod
from tumor_flow.analysis.forecasting import PartAReport, build_aggregate_reports, write_part_a_report

QUANTILE_COLUMN_NAMES = ('q05', 'q25', 'q50', 'q75', 'q95')
ALIGNMENT_MEAN_THRESHOLD = 0.98
ALIGNMENT_MIN_THRESHOLD = 0.95

#----------------------------------------------------------------------------
# Packbits round trip, duplicated from tools/forecast_common.py (pure numpy)
# rather than imported, since that module imports torch at module scope.

def unpack_samples(packed, shape):
    shape = tuple(int(s) for s in shape)
    bits = np.unpackbits(packed, axis=-1)
    bits = bits[..., :shape[-1]]
    return bits.reshape(shape).astype(bool)

#----------------------------------------------------------------------------

def load_yaml_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_data_config(flow_cfg):
    d = flow_cfg['data']
    intensity_clip = d.get('intensity_clip')
    return DataConfig(
        modality=d.get('modality', 't2_thin'),
        target_spacing_xyz_mm=tuple(float(v) for v in d['target_spacing_xyz_mm']),
        roi_size_xyz=tuple(int(v) for v in d['roi_size_xyz']),
        normalize_nonzero=bool(d.get('normalize_nonzero', True)),
        intensity_clip=tuple(float(v) for v in intensity_clip) if intensity_clip is not None else None,
    )

#----------------------------------------------------------------------------

def read_csv_dicts(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def read_manifest_index(manifest_path):
    """(patient_id, baseline_scan_id, target_scan_id) -> manifest row dict."""
    _, rows = read_csv_dicts(manifest_path)
    return {(r['patient_id'], r['baseline_scan_id'], r['target_scan_id']): r for r in rows}

#----------------------------------------------------------------------------
# Resampling: EDM2's ViT-crop-frame array -> flow reference grid -> flow crop -> ZYX.

def resample_and_crop(array_xyz, source_affine, grid, crop_spec, order):
    """Resample an EDM2/ViT-frame array (its OWN spacing/orientation, taken
    entirely from `source_affine` -- never assumed here) into an
    already-built flow reference `grid`, then crop with `crop_spec`.
    `source_affine` and `grid.affine_array()` fully determine the physical
    relationship between the two grids (any spacing, any relative rotation);
    this function makes no assumption about either one's voxel size or
    field-of-view size.

    order=1: linear resample of a float array (used for the probability map).

    order=0 (masks): the EDM2 mask has already been through one
    nearest-neighbour resample/crop upstream (the ViT token pipeline), so a
    second nearest-neighbour hop here compounds quantisation error -- for a
    small tumour, a fractional-voxel offset between the two independently
    computed grids can flip several boundary voxels even though the frame
    math is correct, which cost a couple of pairs the alignment check's
    0.95 floor during validation. Resampling the float {0,1} mask with
    linear interpolation and thresholding at 0.5 (area-weighted boundary
    reconstruction) removes that avoidable double-quantisation error while
    keeping the exact same transform; this also gives a graceful, non-binary
    blend at the true EDM2 field-of-view boundary now that the two grids do
    not necessarily share spacing or physical extent (see
    edm2_fov_coverage_mask below, which resamples the same way for
    consistency).

    Voxels of `grid` that fall outside `array_xyz`'s own bounds come back as
    background (`cval=0.0`), by nibabel's own constant-fill boundary mode --
    not something this function has to special-case.
    """
    dtype = np.float32
    img = nib.Nifti1Image(np.ascontiguousarray(array_xyz).astype(dtype), np.asarray(source_affine, dtype=np.float64))
    resampled = resample_from_to(
        img, (grid.shape_xyz, grid.affine_array()), order=1, mode='constant', cval=0.0)
    data = np.asarray(resampled.dataobj, dtype=np.float32)
    if order == 0:
        data = data > 0.5
    cropped_xyz = apply_crop(data, crop_spec, fill_value=0)
    return xyz_array_to_zyx(cropped_xyz)


def edm2_fov_coverage_mask(edm2_array_shape, source_affine, grid, crop_spec):
    """True where a flow-frame voxel (after cropping to `crop_spec`) is
    actually covered by the EDM2/ViT source array's field of view; False
    where resample_and_crop would have zero-filled it as out-of-bounds.
    Computed by resampling an all-ones array of the EDM2 array's own shape
    through the EXACT same order=0 pipeline as every mask above, so this is
    consistent by construction with what "background" means for the real
    sample/target/cond masks, rather than a separate geometric bounding-box
    computation that could disagree with the actual resampling behaviour.
    """
    coverage_source = np.ones(edm2_array_shape, dtype=np.uint8)
    return resample_and_crop(coverage_source, source_affine, grid, crop_spec, order=0)


def fraction_outside_fov(target_mask_zyx, fov_coverage_zyx):
    """Fraction of `target_mask_zyx`'s foreground voxels that fall outside
    `fov_coverage_zyx` (True = covered by the EDM2 FOV). 0.0 for an empty
    target mask (nothing to be outside).
    """
    target_fg = np.asarray(target_mask_zyx, dtype=bool)
    count = int(target_fg.sum())
    if count == 0:
        return 0.0
    return float(np.logical_and(target_fg, ~np.asarray(fov_coverage_zyx, dtype=bool)).sum()) / count

#----------------------------------------------------------------------------

def load_edm2_pair(generation_dir, split, idx):
    path = os.path.join(generation_dir, split, f'{idx:08d}.npz')
    with np.load(path, allow_pickle=True) as z:
        samples = unpack_samples(z['samples_packed'], z['samples_shape'])
        return dict(
            samples=samples,
            prob=np.asarray(z['prob'], dtype=np.float32),
            target_mask=z['target_mask'].astype(np.uint8),
            cond_mask=z['cond_mask'].astype(np.uint8),
            delta_days=float(np.asarray(z['delta_days'])),
            patient_id=str(z['patient_id']),
            cond_scan_id=str(z['cond_scan_id']),
            target_scan_id=str(z['target_scan_id']),
            sampler_settings=json.loads(str(z['sampler_settings_json'])),
        )


def load_scan_meta(tokens_dir, split, scan_id):
    """affine (shared per-patient ViT crop frame, contract C1) and the C2
    token store's own `source_mask_path` -- the raw, pre-crop uva_vs_flat_05
    segmentation NIfTI EDM2 was actually conditioned/trained on for this scan.
    """
    path = os.path.join(tokens_dir, split, f'{scan_id}.npz')
    with np.load(path, allow_pickle=True) as z:
        return dict(
            affine=np.asarray(z['affine'], dtype=np.float64),
            source_mask_path=str(z['source_mask_path']),
        )


def load_source_mask_in_pair_frame(segmentation_path, grid, crop_spec):
    """Run a raw uva_vs_flat_05 segmentation NIfTI through the flow repo's own
    resample-to-grid + crop primitives (the same ones prepare_baseline_context/
    prepare_future_target use internally), but reusing an ALREADY-COMPUTED
    grid/crop_spec for the pair rather than recomputing a fresh, mask-centred
    crop from this file's own tumor location (create_crop_spec centres the
    crop on whatever baseline mask it is given, so calling
    prepare_baseline_context fresh on a different segmentation file would
    silently shift the crop window and defeat the whole comparison -- see
    team-lead decision, 2026-09-12). This isolates a pure same-file,
    two-pipeline frame comparison from an annotation-source comparison.
    """
    volume = load_canonical_nifti(Path(segmentation_path), binary=True)
    mask_reference = resample_mask_to_grid(volume, grid)
    return xyz_array_to_zyx(apply_crop(mask_reference, crop_spec))

#----------------------------------------------------------------------------

def compute_alignment_row(idx, patient_id, cond_scan_id, target_scan_id, cond_mask_xyz, target_mask_xyz,
                           manifest_row, tokens_dir, data_config, split, flow_arrays_cache_dir=None):
    """Everything the mandatory alignment check needs for one pair: the flow
    repo's own baseline/target context, our resampled EDM2/ViT-frame masks,
    the same-source-file frame-gate comparison, the annotation-agreement
    comparison, and the FOV-coverage diagnostic.

    Shared by the full export path (process_pair, which additionally resamples
    a generation run's prob/samples arrays) and --alignment-only mode (which
    has no generation dir/trained network at all and passes the C3 pair dir's
    own cond_mask/target_mask in their place -- these ARE the token store's
    per-scan masks, contract C2, just copied into the pair npz).

    Returns (alignment_row, ctx): ctx bundles the intermediate objects the
    full metric computation in process_pair needs, so it is not recomputed.
    """
    baseline_ctx = prepare_baseline_context(
        Path(manifest_row['baseline_image_path']), Path(manifest_row['baseline_segmentation_path']),
        data_config)
    target_ctx = prepare_future_target(baseline_ctx, Path(manifest_row['target_segmentation_path']))

    grid = baseline_ctx.reference_grid
    crop_spec = baseline_ctx.crop_spec
    spacing_xyz_mm = grid.spacing_xyz_mm

    cond_meta = load_scan_meta(tokens_dir, split, cond_scan_id)
    target_meta = load_scan_meta(tokens_dir, split, target_scan_id)
    affine = cond_meta['affine']  # shared per-patient ViT crop frame (contract C1)

    # These two resampled-through-our-own-frame-math arrays are used ONLY by
    # the alignment check. Every scored metric (consensus/sample/carry-forward/
    # probabilistic, in process_pair) uses the flow repo's own preprocessed
    # baseline_mask_zyx/target_mask_zyx instead, so both models are scored
    # against identical ground truth (team-lead follow-up 1).
    our_target_zyx = resample_and_crop(target_mask_xyz, affine, grid, crop_spec, order=0)
    our_baseline_zyx = resample_and_crop(cond_mask_xyz, affine, grid, crop_spec, order=0)

    baseline_mask_zyx = baseline_ctx.baseline_mask_zyx
    target_mask_zyx = target_ctx.future_mask_zyx
    baseline_mri_zyx = baseline_ctx.baseline_mri_zyx

    # --- EDM2 field-of-view coverage -------------------------------------------------------
    # The EDM2/ViT crop's physical FOV need not cover the (generally larger)
    # flow crop's FOV -- true as of the 2026-09-12 token-store revision, where
    # the ViT crop shrank to a 64mm cube. Voxels outside EDM2's FOV are
    # zero-filled by resample_and_crop/nibabel; this makes that visible rather
    # than silently reading as "EDM2 predicted no tumour here".
    fov_coverage_zyx = edm2_fov_coverage_mask(target_mask_xyz.shape, affine, grid, crop_spec)
    edm2_fov_covers_flow_crop = bool(fov_coverage_zyx.all())
    frac_target_outside_fov = fraction_outside_fov(target_mask_zyx, fov_coverage_zyx)

    # --- Mandatory alignment check (team-lead decision, 2026-09-12) -----------------------
    # Two DECOUPLED comparisons, so a frame bug and an annotation-source
    # disagreement can never be mistaken for each other:
    #
    # 1. Frame gate (strict, never excused): the SAME uva_vs_flat_05 source
    #    file EDM2 was conditioned/trained on, run through the flow repo's own
    #    resample-to-grid + crop (load_source_mask_in_pair_frame, reusing this
    #    pair's own grid/crop_spec) vs our resampled EDM2/ViT-frame mask. Same
    #    file, two pipelines -- any disagreement can only be a frame-math bug.
    # 2. Annotation agreement (logged only, never gates): the same
    #    uva_vs_flat_05 file vs the flow manifest's uva_vs_v0005 file, BOTH run
    #    through the identical flow-repo pipeline/grid/crop. Same pipeline, two
    #    annotation sources -- any disagreement is a data-provenance fact
    #    (EDM2 was conditioned on a different mask than the flow model sees),
    #    not something to gate the export on.
    flat05_baseline_zyx = load_source_mask_in_pair_frame(cond_meta['source_mask_path'], grid, crop_spec)
    flat05_target_zyx = load_source_mask_in_pair_frame(target_meta['source_mask_path'], grid, crop_spec)

    shape_ok = len({
        tuple(a.shape) for a in (
            our_target_zyx, our_baseline_zyx, target_mask_zyx, baseline_mask_zyx,
            flat05_baseline_zyx, flat05_target_zyx,
        )
    } | {tuple(data_config.roi_size_xyz[::-1])}) == 1

    frame_dice_baseline = dice_score(our_baseline_zyx, flat05_baseline_zyx)
    frame_dice_target = dice_score(our_target_zyx, flat05_target_zyx)
    annotation_dice_baseline = dice_score(flat05_baseline_zyx, baseline_mask_zyx)
    annotation_dice_target = dice_score(flat05_target_zyx, target_mask_zyx)

    dice_vs_flow_arrays = None
    if flow_arrays_cache_dir is not None:
        flow_arrays_path = flow_arrays_cache_dir(patient_id, target_scan_id)
        if flow_arrays_path is not None and os.path.isfile(flow_arrays_path):
            with np.load(flow_arrays_path) as fz:
                dice_vs_flow_arrays = dice_score(our_target_zyx, fz['target'].astype(bool))

    alignment_row = dict(
        idx=idx, patient_id=patient_id, cond_scan_id=cond_scan_id, target_scan_id=target_scan_id,
        shape_zyx=str(tuple(our_target_zyx.shape)), shape_ok=shape_ok,
        frame_dice_baseline=frame_dice_baseline, frame_dice_target=frame_dice_target,
        annotation_dice_baseline=annotation_dice_baseline, annotation_dice_target=annotation_dice_target,
        dice_vs_flow_arrays_target=dice_vs_flow_arrays,
        edm2_fov_covers_flow_crop=edm2_fov_covers_flow_crop,
        frac_target_outside_edm2_fov=frac_target_outside_fov,
    )
    ctx = dict(
        baseline_ctx=baseline_ctx, target_ctx=target_ctx, grid=grid, crop_spec=crop_spec,
        spacing_xyz_mm=spacing_xyz_mm, affine=affine,
        baseline_mask_zyx=baseline_mask_zyx, target_mask_zyx=target_mask_zyx, baseline_mri_zyx=baseline_mri_zyx,
    )
    return alignment_row, ctx

#----------------------------------------------------------------------------
# Alignment-check aggregation/reporting, shared by full export mode and
# --alignment-only mode.

ALIGNMENT_METHODOLOGY = (
    'Two decoupled comparisons per pair, both baseline and target: '
    '(1) frame gate, strict/never-excused -- the same uva_vs_flat_05 source file '
    'run through two pipelines (EDM2/ViT-frame resample vs the flow repo\'s own '
    'resample-to-grid+crop reusing this pair\'s own grid/crop_spec); a failure can '
    'only be a frame-math bug. (2) annotation agreement, logged only -- the same '
    'flow-repo pipeline run on the uva_vs_flat_05 file vs the uva_vs_v0005 file; a '
    'disagreement is a data-provenance fact (EDM2 conditioned on a different mask '
    'than the flow model is scored against), not something this gate blocks on.')


def write_alignment_check_csv(out_dir, alignment_rows):
    path = os.path.join(out_dir, 'alignment_check.csv')
    with open(path, 'w', newline='') as f:
        fieldnames = ['idx', 'patient_id', 'cond_scan_id', 'target_scan_id', 'shape_zyx', 'shape_ok',
                      'frame_dice_baseline', 'frame_dice_target',
                      'annotation_dice_baseline', 'annotation_dice_target', 'dice_vs_flow_arrays_target',
                      'edm2_fov_covers_flow_crop', 'frac_target_outside_edm2_fov']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in alignment_rows:
            writer.writerow(r)
    return path


def summarize_alignment(alignment_rows):
    def mean_min(key):
        values = [r[key] for r in alignment_rows]
        return float(np.mean(values)), float(np.min(values))

    frame_baseline_mean, frame_baseline_min = mean_min('frame_dice_baseline')
    frame_target_mean, frame_target_min = mean_min('frame_dice_target')
    annotation_baseline_mean, annotation_baseline_min = mean_min('annotation_dice_baseline')
    annotation_target_mean, annotation_target_min = mean_min('annotation_dice_target')
    shapes_ok = all(r['shape_ok'] for r in alignment_rows)

    annotation_disagreements = []
    for r in alignment_rows:
        sides = []
        if r['annotation_dice_baseline'] < ALIGNMENT_MIN_THRESHOLD:
            sides.append('baseline')
        if r['annotation_dice_target'] < ALIGNMENT_MIN_THRESHOLD:
            sides.append('target')
        if sides:
            annotation_disagreements.append(dict(
                patient_id=r['patient_id'], cond_scan_id=r['cond_scan_id'], target_scan_id=r['target_scan_id'],
                sides=sides, annotation_dice_baseline=r['annotation_dice_baseline'],
                annotation_dice_target=r['annotation_dice_target']))

    fov_incomplete = [r for r in alignment_rows if not r['edm2_fov_covers_flow_crop']]
    frac_outside_values = [r['frac_target_outside_edm2_fov'] for r in alignment_rows]
    fov_summary = dict(
        n_pairs_fully_covered=len(alignment_rows) - len(fov_incomplete),
        n_pairs_incomplete_fov=len(fov_incomplete),
        frac_target_outside_fov_mean=float(np.mean(frac_outside_values)),
        frac_target_outside_fov_max=float(np.max(frac_outside_values)),
        incomplete_fov_pairs=[
            dict(patient_id=r['patient_id'], cond_scan_id=r['cond_scan_id'], target_scan_id=r['target_scan_id'],
                 frac_target_outside_edm2_fov=r['frac_target_outside_edm2_fov'])
            for r in fov_incomplete
        ],
    )

    return dict(
        methodology=ALIGNMENT_METHODOLOGY,
        frame_gate_thresholds=dict(mean=ALIGNMENT_MEAN_THRESHOLD, min=ALIGNMENT_MIN_THRESHOLD),
        frame_dice_baseline_mean=frame_baseline_mean, frame_dice_baseline_min=frame_baseline_min,
        frame_dice_target_mean=frame_target_mean, frame_dice_target_min=frame_target_min,
        annotation_dice_baseline_mean=annotation_baseline_mean, annotation_dice_baseline_min=annotation_baseline_min,
        annotation_dice_target_mean=annotation_target_mean, annotation_dice_target_min=annotation_target_min,
        shapes_ok=shapes_ok,
        n_pairs=len(alignment_rows),
        annotation_disagreements=annotation_disagreements,
        edm2_fov_coverage=fov_summary,
    )


def print_alignment_summary(alignment_rows, summary):
    print(f'Alignment check over {len(alignment_rows)} pairs:')
    print(f'  FRAME gate (uva_vs_flat_05 file, two pipelines; strict, never excused): '
          f'baseline mean={summary["frame_dice_baseline_mean"]:.4f} min={summary["frame_dice_baseline_min"]:.4f}, '
          f'target mean={summary["frame_dice_target_mean"]:.4f} min={summary["frame_dice_target_min"]:.4f}, '
          f'shapes_ok={summary["shapes_ok"]}')
    print(f'  ANNOTATION agreement (flat_05 vs v0005 file, same pipeline; logged only): '
          f'baseline mean={summary["annotation_dice_baseline_mean"]:.4f} min={summary["annotation_dice_baseline_min"]:.4f}, '
          f'target mean={summary["annotation_dice_target_mean"]:.4f} min={summary["annotation_dice_target_min"]:.4f}')
    disagreements = summary['annotation_disagreements']
    if disagreements:
        print(f'  {len(disagreements)} pair(s) with an annotation disagreement '
              f'(< {ALIGNMENT_MIN_THRESHOLD} on at least one side; EDM2 was conditioned on a '
              f'different mask than the flow model is scored against for that pair):')
        for r in disagreements:
            print(f"    patient={r['patient_id']} baseline_scan={r['cond_scan_id']} "
                  f"target_scan={r['target_scan_id']} sides={r['sides']} "
                  f"annotation_dice_baseline={r['annotation_dice_baseline']:.4f} "
                  f"annotation_dice_target={r['annotation_dice_target']:.4f}")
    fov = summary['edm2_fov_coverage']
    print(f'  EDM2 field-of-view coverage: {fov["n_pairs_fully_covered"]}/{len(alignment_rows)} pairs '
          f'fully covered by the EDM2 crop; frac_target_outside_fov mean={fov["frac_target_outside_fov_mean"]:.4f} '
          f'max={fov["frac_target_outside_fov_max"]:.4f}')
    if fov['incomplete_fov_pairs']:
        print(f'  {len(fov["incomplete_fov_pairs"])} pair(s) where the flow crop extends beyond the EDM2 FOV:')
        for r in fov['incomplete_fov_pairs']:
            print(f"    patient={r['patient_id']} target={r['target_scan_id']} "
                  f"frac_target_outside_fov={r['frac_target_outside_edm2_fov']:.4f}")


def check_frame_gate(summary, alignment_check_csv_path):
    if (summary['frame_dice_baseline_mean'] < ALIGNMENT_MEAN_THRESHOLD
            or summary['frame_dice_baseline_min'] < ALIGNMENT_MIN_THRESHOLD
            or summary['frame_dice_target_mean'] < ALIGNMENT_MEAN_THRESHOLD
            or summary['frame_dice_target_min'] < ALIGNMENT_MIN_THRESHOLD
            or not summary['shapes_ok']):
        raise SystemExit(
            f'ALIGNMENT CHECK FAILED on the frame gate: baseline mean={summary["frame_dice_baseline_mean"]:.4f} '
            f'min={summary["frame_dice_baseline_min"]:.4f}, target mean={summary["frame_dice_target_mean"]:.4f} '
            f'min={summary["frame_dice_target_min"]:.4f} (need >= {ALIGNMENT_MEAN_THRESHOLD} mean, '
            f'>= {ALIGNMENT_MIN_THRESHOLD} min), shapes_ok={summary["shapes_ok"]}. '
            f'See {alignment_check_csv_path}. Both sides of this comparison '
            f'come from the SAME uva_vs_flat_05 source file (only the resampling pipeline '
            f'differs), so a failure here can only be a frame-math bug; fix it rather than '
            f'lowering this threshold. (An annotation-source disagreement between the '
            f'uva_vs_flat_05 and uva_vs_v0005 files for a scan shows up in the separate, '
            f'non-gating annotation_dice_* columns and settings.alignment_check.'
            f'annotation_disagreements instead -- it is not grounds to relax this gate.)')

#----------------------------------------------------------------------------

def load_pair_dir_pair(pairs_dir, split, idx):
    """Read one pair directly from the C3 pair dataset (prepare_data_vivit_pairs.py
    output) rather than a generate_forecasts.py run. Used by --alignment-only
    mode, which needs no trained network/generation dir at all: cond_mask and
    target_mask here ARE the token store's own per-scan masks (contract C2),
    just copied into the pair npz, so the frame gate/annotation-agreement/FOV
    checks -- which never touch a sample or probability map -- are identical
    to what the full export path would compute for the same checkpoint-
    independent quantities.
    """
    path = os.path.join(pairs_dir, split, f'{idx:08d}.npz')
    with np.load(path, allow_pickle=True) as z:
        return dict(
            target_mask=z['target_mask'].astype(np.uint8),
            cond_mask=z['cond_mask'].astype(np.uint8),
            patient_id=str(z['patient_id']),
            cond_scan_id=str(z['cond_scan_id']),
            target_scan_id=str(z['target_scan_id']),
        )

#----------------------------------------------------------------------------

def build_sample_rows(patient_id, target_scan_id, sample_dice, sample_surface_dice, sample_volumes):
    return [
        dict(patient_id=patient_id, target_scan_id=target_scan_id, sample_idx=i,
             dice=float(sample_dice[i]), surface_dice=float(sample_surface_dice[i]),
             volume_mm3=float(sample_volumes[i]))
        for i in range(len(sample_dice))
    ]


def quantile_columns(sample_volumes):
    q = np.quantile(np.asarray(sample_volumes, dtype=np.float64), VOLUME_QUANTILE_LEVELS)
    return {f'sample_volume_{name}_mm3': float(v) for name, v in zip(QUANTILE_COLUMN_NAMES, q)}


def prefixed(prefix, metrics):
    return {f'{prefix}_{k}': v for k, v in metrics.items()
            if k not in ('baseline_volume_mm3', 'target_volume_mm3')}


def deterministic_passthrough(flow_row, pair_fields):
    out = {}
    for col in pair_fields:
        if not col.startswith('deterministic_'):
            continue
        v = flow_row.get(col, '')
        out[col] = float(v) if v not in ('', None) else None
    return out

#----------------------------------------------------------------------------

def process_pair(args, idx, edm2_pair, manifest_row, tokens_dir, data_config, split,
                  surface_dice_tolerance_mm, stable_volume_change_fraction, alignment_rows,
                  flow_arrays_cache_dir):
    patient_id = edm2_pair['patient_id']
    cond_scan_id = edm2_pair['cond_scan_id']
    target_scan_id = edm2_pair['target_scan_id']

    alignment_row, ctx = compute_alignment_row(
        idx, patient_id, cond_scan_id, target_scan_id, edm2_pair['cond_mask'], edm2_pair['target_mask'],
        manifest_row, tokens_dir, data_config, split, flow_arrays_cache_dir)
    alignment_rows.append(alignment_row)

    grid, crop_spec, affine = ctx['grid'], ctx['crop_spec'], ctx['affine']
    spacing_xyz_mm = ctx['spacing_xyz_mm']
    baseline_mask_zyx, target_mask_zyx = ctx['baseline_mask_zyx'], ctx['target_mask_zyx']
    baseline_mri_zyx = ctx['baseline_mri_zyx']
    target_ctx = ctx['target_ctx']

    # Generation-run-specific arrays (prob/samples): not needed by the
    # alignment check above, only by the metrics below, so resampled only here.
    prob_zyx = resample_and_crop(edm2_pair['prob'], affine, grid, crop_spec, order=1)
    samples_zyx = np.stack([
        resample_and_crop(s.astype(np.uint8), affine, grid, crop_spec, order=0)
        for s in edm2_pair['samples']
    ])

    consensus_mask_zyx = prob_zyx > 0.5
    consensus_metrics = deterministic_pair_metrics(
        consensus_mask_zyx, baseline_mask_zyx, target_mask_zyx, spacing_xyz_mm, surface_dice_tolerance_mm)
    carry_metrics = deterministic_pair_metrics(
        baseline_mask_zyx, baseline_mask_zyx, target_mask_zyx, spacing_xyz_mm, surface_dice_tolerance_mm)

    sample_dice = np.array([dice_score(s, target_mask_zyx) for s in samples_zyx])
    sample_surface_dice = np.array([
        surface_dice_at_tolerance(s, target_mask_zyx, spacing_xyz_mm, surface_dice_tolerance_mm)
        for s in samples_zyx
    ])
    sample_volumes = np.array([mask_volume_mm3(s, spacing_xyz_mm) for s in samples_zyx])

    volume_quantiles = np.quantile(sample_volumes, VOLUME_QUANTILE_LEVELS)
    probabilistic_metrics = probabilistic_pair_metrics(
        prob_zyx, samples_zyx, sample_volumes, volume_quantiles, VOLUME_QUANTILE_LEVELS,
        target_mask_zyx, float(consensus_metrics['target_volume_mm3']),
        float(consensus_metrics['baseline_volume_mm3']), stable_volume_change_fraction)

    settings = edm2_pair['sampler_settings']
    computed = dict(
        split=manifest_row['split'], patient_id=patient_id,
        baseline_scan_id=cond_scan_id, target_scan_id=target_scan_id,
        baseline_day=int(manifest_row['baseline_day']), target_day=int(manifest_row['target_day']),
        delta_days=int(manifest_row['delta_days']),
        horizon_bin=forecasting_mod.horizon_bin(int(manifest_row['delta_days'])),
        num_flow_samples=settings.get('num_samples'), flow_steps=settings.get('steps'),
        flow_solver='edm_heun', guidance_weight=None, guidance_checkpoint_sha256=None,
        seed=settings.get('seed'), surface_dice_tolerance_mm=surface_dice_tolerance_mm,
        crop_touches_boundary=target_ctx.crop_quality.future_touches_crop_boundary,
        baseline_volume_mm3=consensus_metrics['baseline_volume_mm3'],
        target_volume_mm3=consensus_metrics['target_volume_mm3'],
        sample_dice_mean=float(sample_dice.mean()), sample_dice_sd=float(sample_dice.std()),
        sample_surface_dice_mean=float(sample_surface_dice.mean()),
        sample_surface_dice_sd=float(sample_surface_dice.std()),
    )
    computed.update(prefixed('consensus', consensus_metrics))
    computed.update(probabilistic_metrics)
    computed.update(quantile_columns(sample_volumes))
    computed.update(prefixed('carry_forward', carry_metrics))

    viz_arrays = dict(
        mri=np.asarray(baseline_mri_zyx, dtype=np.float32),
        density=np.asarray(prob_zyx, dtype=np.float32),
        target=np.asarray(target_mask_zyx, dtype=np.uint8),
        dice=np.asarray(sample_dice, dtype=np.float64),
        volumes=np.asarray(sample_volumes, dtype=np.float64),
        spacing_xyz_mm=np.asarray(spacing_xyz_mm, dtype=np.float64),
    )
    viz_metrics = dict(
        sample_dice_mean=float(sample_dice.mean()), sample_dice_sd=float(sample_dice.std()),
        sample_surface_dice_mean=float(sample_surface_dice.mean()),
        sample_surface_dice_sd=float(sample_surface_dice.std()),
        consensus_dice=float(consensus_metrics['dice']),
        consensus_surface_dice=float(consensus_metrics['surface_dice']),
        sample_volume_mean_mm3=float(sample_volumes.mean()),
        sample_volume_sd_mm3=float(sample_volumes.std()),
        consensus_volume_mm3=float(consensus_metrics['predicted_volume_mm3']),
        target_volume_mm3=float(consensus_metrics['target_volume_mm3']),
    )
    sample_rows = build_sample_rows(patient_id, target_scan_id, sample_dice, sample_surface_dice, sample_volumes)

    return computed, sample_rows, viz_arrays, viz_metrics

#----------------------------------------------------------------------------

def write_viz_scan(out_dir, patient_id, target_scan_id, arrays, metrics_dict, pair_meta, z_levels_override=None):
    scan_dir = os.path.join(out_dir, 'viz', f'patient{patient_id}', f'scan{target_scan_id}')
    os.makedirs(scan_dir, exist_ok=True)
    arrays = dict(arrays)
    if z_levels_override is not None:
        arrays['z_levels'] = np.asarray(z_levels_override, dtype=np.int64)
    else:
        depth = arrays['target'].shape[0]
        supported = np.flatnonzero(np.any((arrays['target'] > 0) | (arrays['density'] >= 0.5), axis=(1, 2)))
        if supported.size == 0:
            mid = depth // 2
            arrays['z_levels'] = np.asarray([mid, mid, mid], dtype=np.int64)
        else:
            positions = np.linspace(float(supported[0]), float(supported[-1]), 5)[1:4]
            arrays['z_levels'] = np.asarray([int(round(p)) for p in positions], dtype=np.int64)
    np.savez_compressed(os.path.join(scan_dir, 'arrays.npz'), **arrays)

    # render_3d_gif/render_3d_slow_gif match the flow repo's own files-dict
    # keys (tumor_flow.analysis.visualizations) and this repo's fixed naming
    # convention for them (tools/render_edm2_3d.py, which writes into this
    # same scan_dir) -- declared here even though that separate step runs
    # after this one and may not have produced them yet, exactly as the flow
    # repo's own metrics.json declares its GIF paths before rendering them.
    files = {
        'arrays_npz': f'viz/patient{patient_id}/scan{target_scan_id}/arrays.npz',
        'metrics_json': f'viz/patient{patient_id}/scan{target_scan_id}/metrics.json',
        'render_3d_gif': f'viz/patient{patient_id}/scan{target_scan_id}/render_3d.gif',
        'render_3d_slow_gif': f'viz/patient{patient_id}/scan{target_scan_id}/render_3d_slow.gif',
    }
    # Same convention as tumor_flow.analysis.visualizations._build_scan (the
    # flow repo's own Part B writer), so tools/render_edm2_3d.py's captions and
    # any other consumer of metrics.json's 'title' see identical text for the
    # same patient/scan on both models.
    title = f"Patient {patient_id}: {pair_meta['baseline_scan_id']} -> {target_scan_id} ({pair_meta['delta_days']} days)"
    metrics_document = dict(
        schema_version=2, analysis='EDM2 export (mirrors ANALYSIS_01 Part B)',
        patient_id=patient_id, target_scan_id=target_scan_id, **pair_meta,
        title=title,
        metrics=metrics_dict,
        visualization=dict(
            z_indices_zyx=[int(v) for v in arrays['z_levels']],
            density_support_threshold=0.5, density_colormap='viridis',
        ),
        files=files,
    )
    with open(os.path.join(scan_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_document, f, indent=2)

    index_entry = dict(
        baseline_scan_id=pair_meta['baseline_scan_id'], target_scan_id=target_scan_id,
        delta_days=pair_meta['delta_days'], files=files, metrics=metrics_dict,
        z_indices_zyx=[int(v) for v in arrays['z_levels']],
        density_support_threshold=0.5, density_colormap='viridis',
    )
    return index_entry

#----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-dir', default=None, help='Required unless --alignment-only.')
    p.add_argument('--pairs-dir', required=True)
    p.add_argument('--tokens-dir', required=True)
    p.add_argument('--manifest', required=True, help='Flow repo pair_manifest.csv')
    p.add_argument('--flow-analysis-dir', default=None, help='Required unless --alignment-only.')
    p.add_argument('--flow-config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--label', default='EDM2 + ViT tokens')
    p.add_argument('--split', default='test')
    p.add_argument(
        '--alignment-only', action='store_true',
        help='Run only the frame gate / annotation-agreement / FOV-coverage diagnostic, '
             'reading cond_mask/target_mask directly from --pairs-dir (the token store\'s '
             'own per-scan masks, contract C2/C3) instead of a generate_forecasts.py run. '
             'No --generation-dir, no trained network, no GPU, no --flow-analysis-dir. '
             'Writes alignment_check.csv and alignment_summary.json only; skips the full '
             'Part A/B report (pair_metrics.csv, viz/, etc.) entirely.')
    args = p.parse_args()

    if not args.alignment_only:
        if args.generation_dir is None:
            p.error('--generation-dir is required unless --alignment-only')
        if args.flow_analysis_dir is None:
            p.error('--flow-analysis-dir is required unless --alignment-only')

    os.makedirs(args.out, exist_ok=True)
    started = time.perf_counter()

    flow_cfg = load_yaml_config(args.flow_config)
    data_config = build_data_config(flow_cfg)
    surface_dice_tolerance_mm = float(flow_cfg['evaluation']['surface_dice_tolerance_mm'])
    stable_volume_change_fraction = float(flow_cfg['evaluation']['stable_volume_change_fraction'])

    manifest_index = read_manifest_index(args.manifest)

    with open(os.path.join(args.pairs_dir, 'dataset.json')) as f:
        pairs_manifest = json.load(f)
    samples_meta = pairs_manifest['splits'][args.split]['samples']

    if args.alignment_only:
        alignment_rows = []
        for sm in samples_meta:
            idx = sm['idx']
            pair = load_pair_dir_pair(args.pairs_dir, args.split, idx)
            key = (pair['patient_id'], pair['cond_scan_id'], pair['target_scan_id'])
            manifest_row = manifest_index.get(key)
            if manifest_row is None:
                raise SystemExit(
                    f'idx={idx}: no manifest row for {key}; is --manifest the flow '
                    f'pair_manifest.csv used to build --pairs-dir?')
            alignment_row, _ctx = compute_alignment_row(
                idx, pair['patient_id'], pair['cond_scan_id'], pair['target_scan_id'],
                pair['cond_mask'], pair['target_mask'], manifest_row, args.tokens_dir, data_config, args.split)
            alignment_rows.append(alignment_row)

        csv_path = write_alignment_check_csv(args.out, alignment_rows)
        summary = summarize_alignment(alignment_rows)
        print_alignment_summary(alignment_rows, summary)
        summary_path = os.path.join(args.out, 'alignment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(dict(
                summary, pairs_dir=args.pairs_dir, split=args.split,
                manifest=args.manifest, tokens_dir=args.tokens_dir,
                elapsed_seconds=time.perf_counter() - started,
            ), f, indent=2)
        print(f'Wrote {csv_path} and {summary_path}')
        check_frame_gate(summary, csv_path)
        return

    pair_fields, flow_pair_rows_list = read_csv_dicts(os.path.join(args.flow_analysis_dir, 'pair_metrics.csv'))
    flow_pair_rows = {(r['patient_id'], r['target_scan_id']): r for r in flow_pair_rows_list}

    with open(os.path.join(args.generation_dir, args.split, 'manifest.json')) as f:
        gen_manifest = json.load(f)

    with open(os.path.join(args.flow_analysis_dir, 'selected_patients.json')) as f:
        flow_selected = json.load(f)
    with open(os.path.join(args.flow_analysis_dir, 'viz_index.json')) as f:
        flow_viz_index = json.load(f)

    def flow_arrays_path(patient_id, target_scan_id):
        path = os.path.join(args.flow_analysis_dir, 'viz', f'patient{patient_id}', f'scan{target_scan_id}', 'arrays.npz')
        return path

    pair_rows = []
    sample_rows = []
    alignment_rows = []
    viz_payloads = {}  # (patient_id, target_scan_id) -> (arrays, metrics_dict, pair_meta)
    missing_deterministic = []

    for sm in samples_meta:
        idx = sm['idx']
        edm2_pair = load_edm2_pair(args.generation_dir, args.split, idx)
        key = (edm2_pair['patient_id'], edm2_pair['cond_scan_id'], edm2_pair['target_scan_id'])
        manifest_row = manifest_index.get(key)
        if manifest_row is None:
            raise SystemExit(f'idx={idx}: no manifest row for {key}; is --manifest the flow pair_manifest.csv used to build --pairs-dir?')

        computed, s_rows, viz_arrays, viz_metrics = process_pair(
            args, idx, edm2_pair, manifest_row, args.tokens_dir, data_config, args.split,
            surface_dice_tolerance_mm, stable_volume_change_fraction, alignment_rows, flow_arrays_path)

        flow_key = (computed['patient_id'], computed['target_scan_id'])
        flow_row = flow_pair_rows.get(flow_key)
        if flow_row is None:
            missing_deterministic.append(flow_key)
            det_cols = {c: None for c in pair_fields if c.startswith('deterministic_')}
        else:
            det_cols = deterministic_passthrough(flow_row, pair_fields)
        computed.update(det_cols)

        row = {col: computed.get(col) for col in pair_fields}
        pair_rows.append(row)
        sample_rows.extend(s_rows)

        viz_payloads[(computed['patient_id'], computed['target_scan_id'])] = (
            viz_arrays, viz_metrics,
            dict(baseline_scan_id=computed['baseline_scan_id'], delta_days=computed['delta_days']),
        )

    if missing_deterministic:
        print(f'[warn] {len(missing_deterministic)} pair(s) had no matching row in the flow '
              f'analysis pair_metrics.csv (deterministic_* left as None): {missing_deterministic}')

    # --- Alignment check (team-lead decision, 2026-09-12) ----------------------------------
    # See compute_alignment_row/summarize_alignment for the frame gate / annotation
    # agreement / FOV-coverage methodology (shared with --alignment-only mode above).
    csv_path = write_alignment_check_csv(args.out, alignment_rows)
    summary = summarize_alignment(alignment_rows)
    print_alignment_summary(alignment_rows, summary)
    check_frame_gate(summary, csv_path)

    # --- Part A report: reuse the flow repo's own aggregation/report writer ---------------
    settings0 = samples_meta and load_edm2_pair(args.generation_dir, args.split, samples_meta[0]['idx'])['sampler_settings']
    num_samples = gen_manifest.get('num_samples', settings0.get('num_samples') if settings0 else None)
    steps = gen_manifest.get('steps', settings0.get('steps') if settings0 else None)
    seed = gen_manifest.get('seed', settings0.get('seed') if settings0 else None)

    forecasting_mod.NUM_FLOW_SAMPLES = num_samples
    forecasting_mod.NUM_FLOW_STEPS = steps
    forecasting_mod.FLOW_SOLVER = 'edm_heun'
    forecasting_mod.ANALYSIS_SEED = seed

    aggregate, horizon_rows, method_rows = build_aggregate_reports(
        pair_rows, elapsed_seconds=time.perf_counter() - started, guidance=None,
        volume_flow_checkpoint_path=None, volume_flow_source=None)
    aggregate['analysis'] = 'EDM2 export (mirrors ANALYSIS_01 Part A)'
    aggregate['counts']['samples'] = len(pair_rows) * num_samples
    aggregate['settings']['label'] = args.label
    aggregate['settings']['edm2_checkpoint'] = gen_manifest.get('checkpoint')
    aggregate['settings']['edm2_checkpoint_sha256'] = gen_manifest.get('checkpoint_sha256')
    aggregate['settings']['deterministic_provenance'] = (
        f'deterministic_* columns copied verbatim from '
        f'{os.path.join(args.flow_analysis_dir, "pair_metrics.csv")} for the same '
        f'(patient_id, target_scan_id): the deterministic forecaster is the flow repo\'s own '
        f'model, evaluated in the identical flow-crop frame, so it is not recomputed here.'
    )
    aggregate['settings']['scoring_reference'] = (
        'Every scored metric (consensus/*, sample_dice_*/sample_surface_dice_*, '
        'carry_forward_*, and the probabilistic_pair_metrics block) is computed against '
        'the flow repo\'s own preprocessed baseline_mask_zyx / future_mask_zyx for this pair '
        '(prepare_baseline_context/prepare_future_target on --manifest\'s NIfTI paths), never '
        'against the ViT-frame masks resampled from generate_forecasts.py\'s npz -- both '
        'models are therefore scored against identical ground truth. The resampled EDM2/ViT-'
        'frame target and baseline masks are used ONLY for the alignment_check below. EDM2 is '
        'trained on the uva_vs_flat_05 annotations; the alignment check\'s target column (not '
        'the scoring itself) is how a difference from the flow manifest\'s v0005 annotation '
        'files, for a given pair, would show up.'
    )
    aggregate['settings']['alignment_check'] = dict(summary, path='alignment_check.csv')

    report = PartAReport(
        pair_rows=tuple(pair_rows), sample_rows=tuple(sample_rows), aggregate=aggregate,
        horizon_rows=horizon_rows, method_rows=method_rows)
    paths = write_part_a_report(report, Path(args.out))
    print(f'Wrote Part A report: {len(pair_rows)} pairs, {len(sample_rows)} samples')
    for path in paths.values():
        print(' ', path)

    # --- Part B: viz/ tree, viz_index.json, selected_patients.json ------------------------
    present_keys = set(viz_payloads.keys())

    filtered_patients = []
    for entry in flow_selected['patients']:
        pid = entry['patient_id']
        kept_targets = [t for t in entry['target_scan_ids'] if (pid, t) in present_keys]
        if kept_targets:
            new_entry = dict(entry)
            new_entry['target_scan_ids'] = kept_targets
            filtered_patients.append(new_entry)
    selected_out = dict(flow_selected)
    selected_out['patients'] = filtered_patients
    with open(os.path.join(args.out, 'selected_patients.json'), 'w') as f:
        json.dump(selected_out, f, indent=2)

    flow_patients = flow_viz_index.get('patients', {})
    viz_patients = {}
    n_scans = 0
    for pid, block in flow_patients.items():
        for sid, flow_scan_entry in block.get('scans', {}).items():
            key = (pid, sid)
            if key not in viz_payloads:
                continue
            arrays, viz_metrics, pair_meta = viz_payloads[key]
            # Share z-levels with the flow analysis for this scan when its own
            # arrays.npz is available, so slices line up across models.
            z_override = None
            fp = flow_arrays_path(pid, sid)
            if os.path.isfile(fp):
                with np.load(fp) as fz:
                    if 'z_levels' in fz.files:
                        z_override = fz['z_levels']
            index_entry = write_viz_scan(args.out, pid, sid, arrays, viz_metrics, pair_meta, z_override)
            index_entry['category'] = flow_scan_entry.get('category', 'stable')
            for k in ('patient_growth_fraction', 'initial_volume_mm3', 'final_volume_mm3', 'target_volume_mm3',
                      'scan_growth_fraction'):
                if k in flow_scan_entry:
                    index_entry[k] = flow_scan_entry[k]
            patient_block = viz_patients.setdefault(pid, dict(
                category=flow_scan_entry.get('category', 'stable'),
                patient_growth_fraction=flow_scan_entry.get('patient_growth_fraction'),
                scans={}))
            patient_block['scans'][sid] = index_entry
            n_scans += 1

    viz_index = dict(
        schema_version=2, analysis='EDM2 export (mirrors ANALYSIS_01 Part B)',
        selection=flow_viz_index.get('selection', {}),
        settings=dict(
            num_flow_samples=num_samples, num_flow_steps=steps, solver='edm_heun', seed=seed,
            surface_dice_tolerance_mm=surface_dice_tolerance_mm, label=args.label,
        ),
        counts=dict(patients=len(viz_patients), scans=n_scans),
        elapsed_seconds=time.perf_counter() - started,
        patients=viz_patients,
    )
    with open(os.path.join(args.out, 'viz_index.json'), 'w') as f:
        json.dump(viz_index, f, indent=2)
    print(f'Wrote viz/ tree: {len(viz_patients)} patients, {n_scans} scans, viz_index.json, selected_patients.json')


if __name__ == '__main__':
    main()

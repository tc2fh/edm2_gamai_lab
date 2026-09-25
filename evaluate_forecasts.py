"""Score a generate_forecasts.py output directory against the
FlowMatchingGrowthNet metric set (docs/vivit_conditioning_plan.md decision 9,
Phase 4). Writes {out}/pair_metrics.csv (one row per pair) and
{out}/summary.json (aggregate stats)."""

import csv
import glob
import json
import os

import click
import numpy as np

from tools.forecast_common import classify_direction, dice_coeff, surface_dice, unpack_samples, volume_mm3

#----------------------------------------------------------------------------

PAIR_FIELDS = [
    'idx', 'patient_id', 'cond_scan_id', 'target_scan_id', 'delta_days',
    'target_volume_mm3', 'baseline_volume_mm3',
    'consensus_dice', 'surface_dice_1mm', 'sample_dice_mean', 'sample_dice_sd',
    'pred_volume_consensus_mm3', 'pred_volume_sample_mean_mm3',
    'abs_volume_error_mm3', 'rel_volume_error', 'signed_volume_change_error_mm3',
    'direction_true', 'direction_pred', 'direction_correct',
    'interval_low_mm3', 'interval_high_mm3', 'interval_contains_target',
    'cf_dice', 'cf_surface_dice_1mm', 'cf_volume_mm3',
    'cf_abs_volume_error_mm3', 'cf_rel_volume_error', 'cf_signed_volume_change_error_mm3',
    'cf_direction_pred', 'cf_direction_correct',
]

#----------------------------------------------------------------------------

def evaluate_pair(npz_path, spacing=None, direction_threshold=0.20, tolerance_mm=1.0):
    """spacing=None (default) reads the pair's own 'spacing' field, written by
    generate_forecasts.py from the pair dataset's spacing (contract C0: the
    token store's physical voxel spacing, never assumed to be (1,1,2)).
    Pass an explicit spacing only to override a generation dir predating that
    field."""
    with np.load(npz_path, allow_pickle=True) as z:
        target_mask = z['target_mask'].astype(bool)
        cond_mask = z['cond_mask'].astype(bool)
        prob = z['prob'].astype(np.float32)
        samples = unpack_samples(z['samples_packed'], z['samples_shape'])
        delta_days = float(np.asarray(z['delta_days']))
        patient_id = str(z['patient_id'])
        cond_scan_id = str(z['cond_scan_id'])
        target_scan_id = str(z['target_scan_id'])
        idx = int(os.path.splitext(os.path.basename(npz_path))[0])
        if spacing is None:
            if 'spacing' not in z.files:
                raise ValueError(
                    f"{npz_path}: no 'spacing' field (predates the C0 fix) and no explicit spacing "
                    "given -- regenerate with the current generate_forecasts.py, or pass spacing= explicitly.")
            spacing = tuple(float(s) for s in z['spacing'])

    consensus_mask = prob > 0.5
    target_vol = volume_mm3(target_mask, spacing)
    baseline_vol = volume_mm3(cond_mask, spacing)

    consensus_dice = dice_coeff(consensus_mask, target_mask)
    surf_dice = surface_dice(consensus_mask, target_mask, spacing=spacing, tolerance_mm=tolerance_mm)

    sample_dices = np.array([dice_coeff(s, target_mask) for s in samples])
    sample_volumes = np.array([volume_mm3(s, spacing) for s in samples])

    pred_vol_consensus = volume_mm3(consensus_mask, spacing)
    pred_vol_sample_mean = float(sample_volumes.mean())

    abs_vol_err = abs(pred_vol_consensus - target_vol)
    rel_vol_err = abs_vol_err / target_vol if target_vol > 0 else (0.0 if abs_vol_err == 0 else float('nan'))
    signed_change_err = (pred_vol_consensus - baseline_vol) - (target_vol - baseline_vol)

    direction_true = classify_direction(baseline_vol, target_vol, direction_threshold)
    direction_pred = classify_direction(baseline_vol, pred_vol_consensus, direction_threshold)

    interval_low, interval_high = np.percentile(sample_volumes, [5, 95])
    interval_contains_target = bool(interval_low <= target_vol <= interval_high)

    # Carry-forward baseline: predict the conditioning mask unchanged.
    cf_dice = dice_coeff(cond_mask, target_mask)
    cf_surf_dice = surface_dice(cond_mask, target_mask, spacing=spacing, tolerance_mm=tolerance_mm)
    cf_abs_err = abs(baseline_vol - target_vol)
    cf_rel_err = cf_abs_err / target_vol if target_vol > 0 else (0.0 if cf_abs_err == 0 else float('nan'))
    cf_signed_err = 0.0 - (target_vol - baseline_vol)
    cf_direction_pred = classify_direction(baseline_vol, baseline_vol, direction_threshold) # always 'stable'
    cf_direction_correct = (cf_direction_pred == direction_true)

    return dict(
        idx=idx, patient_id=patient_id, cond_scan_id=cond_scan_id, target_scan_id=target_scan_id,
        delta_days=delta_days,
        target_volume_mm3=target_vol, baseline_volume_mm3=baseline_vol,
        consensus_dice=consensus_dice, surface_dice_1mm=surf_dice,
        sample_dice_mean=float(sample_dices.mean()), sample_dice_sd=float(sample_dices.std()),
        pred_volume_consensus_mm3=pred_vol_consensus, pred_volume_sample_mean_mm3=pred_vol_sample_mean,
        abs_volume_error_mm3=abs_vol_err, rel_volume_error=rel_vol_err,
        signed_volume_change_error_mm3=signed_change_err,
        direction_true=direction_true, direction_pred=direction_pred, direction_correct=(direction_true == direction_pred),
        interval_low_mm3=float(interval_low), interval_high_mm3=float(interval_high),
        interval_contains_target=interval_contains_target,
        cf_dice=cf_dice, cf_surface_dice_1mm=cf_surf_dice, cf_volume_mm3=baseline_vol,
        cf_abs_volume_error_mm3=cf_abs_err, cf_rel_volume_error=cf_rel_err,
        cf_signed_volume_change_error_mm3=cf_signed_err,
        cf_direction_pred=cf_direction_pred, cf_direction_correct=cf_direction_correct,
    )

#----------------------------------------------------------------------------

def _mean_sd(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return dict(mean=float('nan'), sd=float('nan'), n=0)
    return dict(mean=float(values.mean()), sd=float(values.std()), n=int(values.size))


def summarize(rows, key):
    return _mean_sd([r[key] for r in rows])


def summarize_by_patient(rows, key):
    by_patient = {}
    for r in rows:
        by_patient.setdefault(r['patient_id'], []).append(r[key])
    per_patient_means = [float(np.nanmean(v)) for v in by_patient.values()]
    return _mean_sd(per_patient_means)

#----------------------------------------------------------------------------

@click.command()
@click.option('--gen-dir', required=True, help='Directory produced by generate_forecasts.py (contains manifest.json and *.npz pairs).')
@click.option('--out', default=None, help='Where to write pair_metrics.csv/summary.json (default: --gen-dir).')
@click.option('--direction-threshold', default=0.20, show_default=True, type=float)
@click.option('--surface-tolerance-mm', default=1.0, show_default=True, type=float)
def cmdline(gen_dir, out, direction_threshold, surface_tolerance_mm):
    """Evaluate a generation directory and write pair_metrics.csv / summary.json."""
    out = out or gen_dir
    os.makedirs(out, exist_ok=True)

    manifest_path = os.path.join(gen_dir, 'manifest.json')
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            gen_manifest = json.load(f)
    else:
        gen_manifest = {}

    npz_paths = sorted(glob.glob(os.path.join(gen_dir, '[0-9]' * 8 + '.npz')))
    if not npz_paths:
        raise click.ClickException(f'no pair npz files found in {gen_dir!r}')

    # spacing=None: each pair's own 'spacing' field (contract C0) is used,
    # not a hard-coded value -- see evaluate_pair's docstring.
    rows = [evaluate_pair(p, spacing=None, direction_threshold=direction_threshold,
                           tolerance_mm=surface_tolerance_mm) for p in npz_paths]

    csv_path = os.path.join(out, 'pair_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Both accuracy/coverage rates (direction_correct etc.) and continuous
    # error metrics go through the same two aggregations: "per_pair" is a
    # flat mean/sd over every pair, "per_patient" averages a patient's own
    # pairs first and then takes mean/sd across patients (so patients with
    # more follow-up pairs don't dominate) -- matching how the flow repo
    # reports both.
    metric_keys = [
        'consensus_dice', 'surface_dice_1mm', 'sample_dice_mean', 'sample_dice_sd',
        'abs_volume_error_mm3', 'rel_volume_error', 'signed_volume_change_error_mm3',
        'direction_correct', 'interval_contains_target',
        'cf_dice', 'cf_surface_dice_1mm', 'cf_abs_volume_error_mm3', 'cf_rel_volume_error',
        'cf_signed_volume_change_error_mm3', 'cf_direction_correct',
    ]
    summary = dict(
        n_pairs=len(rows),
        n_patients=len({r['patient_id'] for r in rows}),
        knockout=gen_manifest.get('knockout'),
        checkpoint=gen_manifest.get('checkpoint'),
        per_pair={k: summarize(rows, k) for k in metric_keys},
        per_patient={k: summarize_by_patient(rows, k) for k in metric_keys},
        # Convenience top-level aliases for the three rate metrics (same
        # numbers as per_pair[...]['mean']) so callers don't have to dig
        # into per_pair for the headline accuracy/coverage figures.
        direction_accuracy=float(np.mean([r['direction_correct'] for r in rows])),
        carry_forward_direction_accuracy=float(np.mean([r['cf_direction_correct'] for r in rows])),
        interval_coverage=float(np.mean([r['interval_contains_target'] for r in rows])),
    )
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    click.echo(f'Wrote {csv_path} and summary.json ({len(rows)} pairs, {summary["n_patients"]} patients)')
    click.echo(f'  consensus dice: {summary["per_pair"]["consensus_dice"]["mean"]:.3f} '
               f'(carry-forward: {summary["per_pair"]["cf_dice"]["mean"]:.3f})')
    click.echo(f'  direction accuracy: {summary["direction_accuracy"]:.3f} '
               f'(carry-forward: {summary["carry_forward_direction_accuracy"]:.3f})')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    cmdline()

#----------------------------------------------------------------------------

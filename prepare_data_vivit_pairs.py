"""Build the (history, target) pair dataset for ViViT-conditioned EDM2.

Reads the Phase 0 ViT token store (contract C2 in
docs/vivit_pipeline_contracts.md) and writes the Phase 1 pair dataset
(contract C3). Tokens themselves are not copied; only masks/images/metadata
needed to train are materialized per sample. The PyTorch dataset class
loads `{tokens_dir}/{split}/{scan_id}.npz` for each history scan at run
time.

Sample definition, per patient, scans chronologically ordered 0..n-1:
    - 'consecutive' pairing: (history 0..i, target i+1) for every i.
    - 'all' pairing: (history 0..i, target j) for every i < j.
`--max-history` truncates a sample's history to its most recent scans; the
conditioning scan (`cond_mask` / `cond_image`) is always history scan i,
the most recent one, regardless of truncation.
"""

import os
import csv
import json
import collections
import click
import numpy as np

from training.encodings import encode_target, is_degenerate_target

#----------------------------------------------------------------------------

DEFAULT_TOKENS_DIR = (
    'D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/'
    'growth_classifier_v0005/out/vit_tokens'
)
SHAPE = (128, 128, 64)

#----------------------------------------------------------------------------
# Voxel spacing (contract C0): the token store, not this script, owns the
# physical voxel spacing. Every scan's npz carries its own `spacing`; this
# tracker asserts the whole store agrees on one value and is the single
# source of truth threaded through SDF stats and dataset.json.

class SpacingTracker:
    def __init__(self):
        self._value = None

    def check(self, spacing, context):
        spacing = tuple(float(x) for x in np.asarray(spacing).reshape(-1))
        if len(spacing) != 3:
            raise click.ClickException(f'{context}: spacing must have 3 elements, got {spacing}')
        if self._value is None:
            self._value = spacing
        elif not np.allclose(self._value, spacing, rtol=0, atol=1e-6):
            raise click.ClickException(
                f'{context}: spacing {spacing} disagrees with the store spacing '
                f'{self._value} established from an earlier scan; the token store '
                f'must use one consistent voxel spacing (contract C0)')

    @property
    def value(self):
        if self._value is None:
            raise click.ClickException('no scans were loaded; spacing is unknown')
        return self._value

#----------------------------------------------------------------------------
# Token store access.

def load_index(tokens_dir):
    index_path = os.path.join(tokens_dir, 'index.json')
    if not os.path.isfile(index_path):
        raise click.ClickException(f'index.json not found under {tokens_dir}')
    with open(index_path) as f:
        return json.load(f)


def load_scan(tokens_dir, split, scan_id, spacing_tracker=None):
    path = os.path.join(tokens_dir, split, f'{scan_id}.npz')
    if not os.path.isfile(path):
        raise click.ClickException(f'scan npz not found: {path}')
    with np.load(path) as npz:
        if 'spacing' not in npz.files:
            raise click.ClickException(f'{path}: missing required "spacing" key (contract C2/C0)')
        if spacing_tracker is not None:
            spacing_tracker.check(npz['spacing'], context=path)
        return {
            'mask': npz['mask'].astype(np.uint8),
            'image': npz['image'].astype(np.float16),
            'days_since_first': float(npz['days_since_first']),
        }


def ordered_scans(patient_entry):
    """Chronologically sort a patient's scan list; ties keep list order."""
    scans = list(patient_entry['scans'])
    indexed = list(enumerate(scans))
    indexed.sort(key=lambda t: (float(t[1]['days_since_first']), t[0]))
    return [s for _, s in indexed]

#----------------------------------------------------------------------------
# Pairing logic.

def build_pairs(n, pairing):
    """Return [(history_end_i, target_j), ...], 0-based into a patient's
    chronologically sorted scan list."""
    pairs = []
    if pairing == 'consecutive':
        for i in range(n - 1):
            pairs.append((i, i + 1))
    elif pairing == 'all':
        for i in range(n - 1):
            for j in range(i + 1, n):
                pairs.append((i, j))
    else:
        raise ValueError(f'unknown pairing: {pairing!r}')
    return pairs


def truncate_history(history_idxs, max_history):
    if max_history is not None and max_history > 0 and len(history_idxs) > max_history:
        return history_idxs[-max_history:]
    return history_idxs

#----------------------------------------------------------------------------
# Running mean/rms accumulator (avoids materializing all train voxels at once).

class RunningStats:
    def __init__(self):
        self.sum = 0.0
        self.sumsq = 0.0
        self.count = 0

    def add(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.sum += float(x.sum())
        self.sumsq += float(np.square(x).sum())
        self.count += x.size

    @property
    def mean(self):
        return self.sum / self.count if self.count else float('nan')

    @property
    def rms(self):
        return float(np.sqrt(self.sumsq / self.count)) if self.count else float('nan')

    @property
    def std(self):
        if not self.count:
            return float('nan')
        var = self.sumsq / self.count - self.mean ** 2
        return float(np.sqrt(max(var, 0.0)))

#----------------------------------------------------------------------------
# Per-split processing.

def process_split(tokens_dir, split, patients, pairing, max_history, split_out_dir, spacing_tracker):
    os.makedirs(split_out_dir, exist_ok=True)

    samples_manifest = []
    idx = 0
    n_patients_used = 0
    n_patients_skipped = 0
    degenerate_target_count = 0
    history_lens = []
    delta_days_list = []
    target_fractions = []
    binary_stats = RunningStats()
    sdf_stats = RunningStats()
    is_train = (split == 'train')

    for patient in patients:
        patient_id = patient['patient_id']
        scans = ordered_scans(patient)
        n = len(scans)
        if n < 2:
            n_patients_skipped += 1
            print(f'  [warn] {split}/{patient_id}: only {n} scan(s), no valid pairs, skipping')
            continue
        n_patients_used += 1

        # Eagerly load all of this patient's scans (patients have only a
        # handful of scans, so this is cheap and avoids re-reading npz files
        # across the multiple pairs an 'all' pairing generates).
        scan_data = {s['scan_id']: load_scan(tokens_dir, split, s['scan_id'], spacing_tracker) for s in scans}

        for i, j in build_pairs(n, pairing):
            history_idxs = truncate_history(list(range(0, i + 1)), max_history)
            cond_scan = scans[i]
            target_scan = scans[j]
            cond_data = scan_data[cond_scan['scan_id']]
            target_data = scan_data[target_scan['scan_id']]

            day_target = target_data['days_since_first']
            day_cond = cond_data['days_since_first']
            delta_days = day_target - day_cond

            history_scan_ids = np.array([scans[h]['scan_id'] for h in history_idxs])
            history_ages_days = np.array(
                [day_target - scan_data[scans[h]['scan_id']]['days_since_first'] for h in history_idxs],
                dtype=np.float64,
            )

            target_mask = target_data['mask']
            if is_degenerate_target(target_mask):
                degenerate_target_count += 1
                print(f'  [warn] {split}/{patient_id}: target scan {target_scan["scan_id"]} '
                      f'has an empty or fully-filled mask')

            np.savez(
                os.path.join(split_out_dir, f'{idx:08d}.npz'),
                target_mask=target_mask.astype(np.uint8),
                cond_mask=cond_data['mask'].astype(np.uint8),
                cond_image=cond_data['image'].astype(np.float16),
                delta_days=np.float64(delta_days),
                history_scan_ids=history_scan_ids,
                history_ages_days=history_ages_days,
                patient_id=patient_id,
                cond_scan_id=cond_scan['scan_id'],
                target_scan_id=target_scan['scan_id'],
                split=split,
            )

            samples_manifest.append(dict(
                idx=idx,
                patient_id=patient_id,
                cond_scan_id=cond_scan['scan_id'],
                target_scan_id=target_scan['scan_id'],
                delta_days=float(delta_days),
                n_history=len(history_idxs),
            ))
            history_lens.append(len(history_idxs))
            delta_days_list.append(float(delta_days))
            target_fractions.append(float(target_mask.sum()) / target_mask.size)

            if is_train:
                binary_stats.add(encode_target(target_mask, 'binary'))
                sdf_stats.add(encode_target(target_mask, 'sdf', spacing_tracker.value))

            idx += 1

        del scan_data

    patients_with_samples = sorted({m['patient_id'] for m in samples_manifest})

    manifest = dict(n=idx, patients=patients_with_samples, samples=samples_manifest)

    summary = dict(
        n_patients_used=n_patients_used,
        n_patients_skipped=n_patients_skipped,
        degenerate_target_count=degenerate_target_count,
        history_lens=history_lens,
        delta_days_list=delta_days_list,
        target_fractions=target_fractions,
    )

    train_stats = None
    if is_train:
        train_stats = dict(
            target_fraction_mean=float(np.mean(target_fractions)) if target_fractions else float('nan'),
            target_rms_binary=binary_stats.rms,
            target_std_binary=binary_stats.std,
            target_rms_sdf=sdf_stats.rms,
            target_std_sdf=sdf_stats.std,
            delta_days_max=float(np.max(delta_days_list)) if delta_days_list else float('nan'),
            delta_days_median=float(np.median(delta_days_list)) if delta_days_list else float('nan'),
            history_len_max=int(np.max(history_lens)) if history_lens else 0,
        )

    return manifest, summary, train_stats

#----------------------------------------------------------------------------
# Manifest mode (Phase 5): build the 'test' split directly from the flow
# repo's pair_manifest.csv rows, restricted to pairs whose baseline and
# target scan both have a file in the token store's test split. History is
# always [baseline] only (K=1), regardless of --max-history.

def read_manifest_rows(manifest_path):
    with open(manifest_path, newline='') as f:
        return list(csv.DictReader(f))


def process_manifest(tokens_dir, manifest_path, split_out_dir, index, spacing_tracker):
    os.makedirs(split_out_dir, exist_ok=True)

    test_patients = index.get('splits', {}).get('test', [])
    test_scan_ids = {s['scan_id'] for p in test_patients for s in p['scans']}

    rows = read_manifest_rows(manifest_path)
    test_rows = [r for r in rows if r['split'] == 'test']

    samples_manifest = []
    skipped = []
    idx = 0
    degenerate_target_count = 0
    history_lens = []
    delta_days_list = []
    target_fractions = []
    binary_stats = RunningStats()
    sdf_stats = RunningStats()

    for r in test_rows:
        patient_id = r['patient_id']
        baseline_id = r['baseline_scan_id']
        target_id = r['target_scan_id']

        reasons = []
        if baseline_id not in test_scan_ids:
            reasons.append('baseline_scan_not_in_vit_test_cohort')
        if target_id not in test_scan_ids:
            reasons.append('target_scan_not_in_vit_test_cohort')
        if reasons:
            skipped.append(dict(
                patient_id=patient_id, baseline_scan_id=baseline_id, target_scan_id=target_id,
                manifest_delta_days=float(r['delta_days']), reasons=reasons,
            ))
            print(f'  [skip] patient={patient_id} {baseline_id}->{target_id}: {", ".join(reasons)}')
            continue

        cond_data = load_scan(tokens_dir, 'test', baseline_id, spacing_tracker)
        target_data = load_scan(tokens_dir, 'test', target_id, spacing_tracker)
        delta_days = target_data['days_since_first'] - cond_data['days_since_first']

        manifest_delta_days = float(r['delta_days'])
        if abs(delta_days - manifest_delta_days) > 1.0:
            print(f'  [warn] patient={patient_id} {baseline_id}->{target_id}: '
                  f'token-store delta_days={delta_days} disagrees with manifest delta_days={manifest_delta_days}')

        target_mask = target_data['mask']
        if is_degenerate_target(target_mask):
            degenerate_target_count += 1
            print(f'  [warn] test/{patient_id}: target scan {target_id} has an empty or fully-filled mask')

        np.savez(
            os.path.join(split_out_dir, f'{idx:08d}.npz'),
            target_mask=target_mask.astype(np.uint8),
            cond_mask=cond_data['mask'].astype(np.uint8),
            cond_image=cond_data['image'].astype(np.float16),
            delta_days=np.float64(delta_days),
            history_scan_ids=np.array([baseline_id]),
            history_ages_days=np.array([delta_days], dtype=np.float64),
            patient_id=patient_id,
            cond_scan_id=baseline_id,
            target_scan_id=target_id,
            split='test',
        )

        samples_manifest.append(dict(
            idx=idx, patient_id=patient_id, cond_scan_id=baseline_id, target_scan_id=target_id,
            delta_days=float(delta_days), n_history=1,
        ))
        history_lens.append(1)
        delta_days_list.append(float(delta_days))
        target_fractions.append(float(target_mask.sum()) / target_mask.size)
        binary_stats.add(encode_target(target_mask, 'binary'))
        sdf_stats.add(encode_target(target_mask, 'sdf', spacing_tracker.value))
        idx += 1

    patients_with_samples = sorted({m['patient_id'] for m in samples_manifest})
    manifest = dict(n=idx, patients=patients_with_samples, samples=samples_manifest)
    summary = dict(
        n_patients_used=len(patients_with_samples),
        n_patients_skipped=len({s['patient_id'] for s in skipped}),
        degenerate_target_count=degenerate_target_count,
        history_lens=history_lens,
        delta_days_list=delta_days_list,
        target_fractions=target_fractions,
    )
    stats = dict(
        target_fraction_mean=float(np.mean(target_fractions)) if target_fractions else float('nan'),
        target_rms_binary=binary_stats.rms,
        target_std_binary=binary_stats.std,
        target_rms_sdf=sdf_stats.rms,
        target_std_sdf=sdf_stats.std,
        delta_days_max=float(np.max(delta_days_list)) if delta_days_list else float('nan'),
        delta_days_median=float(np.median(delta_days_list)) if delta_days_list else float('nan'),
        history_len_max=1 if history_lens else 0,
    )
    provenance = dict(
        source=manifest_path,
        filter="split == 'test'",
        n_input_rows=len(test_rows),
        n_kept=idx,
        n_skipped=len(skipped),
        skipped=skipped,
    )
    return manifest, summary, stats, provenance

#----------------------------------------------------------------------------

def print_summary(split, manifest, summary):
    print(f'\n=== {split} ===')
    print(f'  samples: {manifest["n"]}  patients: {len(manifest["patients"])} '
          f'(used={summary["n_patients_used"]}, skipped_too_few_scans={summary["n_patients_skipped"]})')
    if summary['degenerate_target_count']:
        print(f'  [warn] {summary["degenerate_target_count"]} target(s) with empty/full mask')
    if summary['history_lens']:
        hist = collections.Counter(summary['history_lens'])
        hist_str = ', '.join(f'{k}:{hist[k]}' for k in sorted(hist))
        print(f'  history length histogram: {hist_str}')
        dd = summary['delta_days_list']
        print(f'  delta_days: min={min(dd):.1f} median={np.median(dd):.1f} max={max(dd):.1f}')
    else:
        print('  (no samples)')

#----------------------------------------------------------------------------

@click.command()
@click.option('--tokens-dir', default=DEFAULT_TOKENS_DIR, show_default=True,
              help='Phase 0 ViT token store (contains index.json and {split}/{scan_id}.npz)')
@click.option('--out', required=True, help='Output directory for the pair dataset')
@click.option('--pairing', type=click.Choice(['consecutive', 'all']), default='consecutive',
              show_default=True, help='consecutive: (0..i, i+1). all: every i<j with history 0..i')
@click.option('--max-history', type=int, default=4, show_default=True,
              help='Cap history length; keeps the most recent max-history scans')
@click.option('--splits', default='train,val,test', show_default=True,
              help='Comma-separated splits to process')
@click.option('--manifest', default=None,
              help='Flow-repo pair_manifest.csv. When given, builds ONLY the test split '
                   'from this manifest\'s split==test rows (Phase 5), ignoring --pairing/'
                   '--splits/--max-history; history is always [baseline] (K=1). See '
                   'docs/vivit_conditioning_plan.md Phase 5.')
def main(tokens_dir, out, pairing, max_history, splits, manifest):
    """Build the ViViT-conditioned EDM2 pair dataset (contract C3)."""
    os.makedirs(out, exist_ok=True)
    index = load_index(tokens_dir)
    available = index.get('splits', {})
    spacing_tracker = SpacingTracker()

    if manifest is not None:
        print(f'\nManifest mode: building test split from {manifest}')
        split_out_dir = os.path.join(out, 'test')
        pair_manifest, summary, test_stats, provenance = process_manifest(
            tokens_dir, manifest, split_out_dir, index, spacing_tracker)
        dataset_json = dict(
            tokens_dir=tokens_dir,
            pairing='manifest',
            max_history=1,
            spacing=list(spacing_tracker.value),
            shape=list(SHAPE),
            splits={'test': pair_manifest},
            # This mode only ever produces a 'test' split, so this is the only
            # stats entry (contract C3: 'stats' is keyed by whichever splits
            # were actually built). generate_forecasts.py's --knockout=fixed-
            # time now falls back to the checkpoint's own delta_days_median
            # (contract C7) when a pair dir has no 'train' stats, and
            # PairDataset accepts an explicit sigma_data override, so nothing
            # downstream needs a fabricated 'train' alias here any more; a
            # test-only pair dir must not claim to have training statistics.
            stats={'test': test_stats},
            manifest=provenance,
        )
        print_summary('test', pair_manifest, summary)
        print(f'\nManifest survival: {provenance["n_kept"]}/{provenance["n_input_rows"]} '
              f'test pairs kept, {provenance["n_skipped"]} skipped')
        for s in provenance['skipped']:
            print(f'  skipped patient={s["patient_id"]} {s["baseline_scan_id"]}->{s["target_scan_id"]}: '
                  f'{", ".join(s["reasons"])}')
        with open(os.path.join(out, 'dataset.json'), 'w') as f:
            json.dump(dataset_json, f, indent=2)
        print(f'\nWrote {os.path.join(out, "dataset.json")}')
        return

    split_list = [s.strip() for s in splits.split(',') if s.strip()]
    if not split_list:
        raise click.ClickException('--splits produced an empty list')

    for s in split_list:
        if s not in available:
            raise click.ClickException(f'split {s!r} not found in {tokens_dir}/index.json')

    dataset_json = dict(
        tokens_dir=tokens_dir,
        pairing=pairing,
        max_history=max_history,
        shape=list(SHAPE),
        splits={},
        stats={},
    )

    for split in split_list:
        print(f'\nProcessing split: {split}')
        split_out_dir = os.path.join(out, split)
        split_manifest, summary, train_stats = process_split(
            tokens_dir, split, available[split], pairing, max_history, split_out_dir, spacing_tracker)
        dataset_json['splits'][split] = split_manifest
        if train_stats is not None:
            dataset_json['stats']['train'] = train_stats
        print_summary(split, split_manifest, summary)

    dataset_json['spacing'] = list(spacing_tracker.value)

    if 'train' in split_list:
        ts = dataset_json['stats']['train']
        print('\n=== train stats (used for sigma_data etc.) ===')
        for k, v in ts.items():
            print(f'  {k}: {v}')
    else:
        print('\n[warn] train split not processed; dataset.json has no "stats" block')

    with open(os.path.join(out, 'dataset.json'), 'w') as f:
        json.dump(dataset_json, f, indent=2)
    print(f'\nWrote {os.path.join(out, "dataset.json")}')


if __name__ == '__main__':
    main()

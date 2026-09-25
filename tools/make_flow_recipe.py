"""Write the recipe.json consumed by the flow repo's build_comparison_deck.py
(docs/vivit_conditioning_plan.md Phase 5). Pure stdlib/json/numpy; runs fine
under either the EDM2 or the flow interpreter (deliberately does not import
tumor_flow/torch, so it never needs either repo's own env).

Headline metrics are read from both models' aggregate.json (written by
tumor_flow.analysis.forecasting.write_part_a_report / this repo's
tools/export_flow_analysis.py), using the same `overall` block shape both
produce (flow_sample_dice / flow_sample_surface_dice / flow_consensus_dice /
flow_consensus_surface_dice, each {pair_mean, pair_sd, patient_mean,
patient_sd[, bootstrap ci]}).

Optionally (--knockout-dir + --edm2-gen-dir) computes the conditioning-
knockout delta cited in the plan's Phase 4 acceptance requirement: consensus
Dice with full conditioning vs. with the shuffle-tokens knockout, paired by
patient, with a patient-level bootstrap CI (2000 resamples, seed 2026 --
same methodology as the flow-repo headline table's own bootstrap). Computed
directly from generate_forecasts.py's raw per-pair npz (prob, target_mask),
in the EDM2/ViT frame -- deliberately NOT the flow-repo frame, since the
knockout comparison only needs both runs scored against the SAME reference
consistently with each other, not against the flow model's own frame.
"""

import argparse
import hashlib
import json
import os

import numpy as np


def sha256_of(path):
    if path is None or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_sum, b_sum = int(a.sum()), int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    return float(2.0 * np.logical_and(a, b).sum() / (a_sum + b_sum))


def _load_consensus_and_target(gen_dir, split, idx):
    path = os.path.join(gen_dir, split, f'{idx:08d}.npz')
    with np.load(path, allow_pickle=True) as z:
        consensus = np.asarray(z['prob'], dtype=np.float32) > 0.5
        target = z['target_mask'].astype(bool)
        patient_id = str(z['patient_id'])
    return consensus, target, patient_id


def knockout_delta(edm2_gen_dir, knockout_gen_dir, split='test', n_boot=2000, seed=2026):
    """Consensus Dice, full context vs. shuffle-tokens knockout, paired by
    patient (mean over that patient's pairs in each arm), with a patient-level
    bootstrap 95% CI on the delta. Both generation dirs must come from the
    SAME pair dataset/split (so idx i means the same pair in both), which is
    the case for a --knockout=shuffle-tokens rerun of the same --data/--split.
    """
    full_dir = os.path.join(edm2_gen_dir, split)
    idxs = sorted(
        int(os.path.splitext(f)[0]) for f in os.listdir(full_dir)
        if f.endswith('.npz') and os.path.splitext(f)[0].isdigit()
    )
    if not idxs:
        raise ValueError(f'no {{idx:08d}}.npz pairs found under {full_dir!r}')

    per_patient_full, per_patient_knockout = {}, {}
    for idx in idxs:
        cons_full, target_full, patient_full = _load_consensus_and_target(edm2_gen_dir, split, idx)
        cons_ko, target_ko, patient_ko = _load_consensus_and_target(knockout_gen_dir, split, idx)
        if patient_full != patient_ko:
            raise ValueError(
                f'idx={idx}: patient mismatch between --edm2-gen-dir ({patient_full!r}) and '
                f'--knockout-dir ({patient_ko!r}) -- are they the same --data/--split?')
        per_patient_full.setdefault(patient_full, []).append(_dice(cons_full, target_full))
        per_patient_knockout.setdefault(patient_ko, []).append(_dice(cons_ko, target_ko))

    patients = sorted(per_patient_full)
    patient_mean_full = np.array([float(np.mean(per_patient_full[p])) for p in patients])
    patient_mean_ko = np.array([float(np.mean(per_patient_knockout[p])) for p in patients])
    patient_delta = patient_mean_full - patient_mean_ko

    rng = np.random.default_rng(seed)
    n = len(patients)
    boot_deltas = np.array([
        patient_delta[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)
    ])
    ci_low, ci_high = (float(v) for v in np.percentile(boot_deltas, [2.5, 97.5]))

    return dict(
        n_patients=n, n_pairs=len(idxs),
        consensus_dice_full_mean=float(patient_mean_full.mean()),
        consensus_dice_knockout_mean=float(patient_mean_ko.mean()),
        delta_mean=float(patient_delta.mean()),
        delta_ci_low=ci_low, delta_ci_high=ci_high,
        n_boot=n_boot, seed=seed,
    )


def knockout_takeaway(result, knockout_arm_label, featured_arm_label):
    """knockout_arm_label identifies the arm the shuffle-tokens knockout was
    actually computed on (e.g. 'arm A (binary target + ViT tokens), same
    kimg 98 checkpoint') -- this is NOT necessarily featured_arm_label, the
    deck's featured/'new' model, which may have no tokens at all (and so
    cannot itself be shuffle-tokens-knocked-out). Naming both, explicitly,
    in the same sentence avoids a reader assuming the knockout was run on
    whichever model the deck happens to be about.
    """
    # The CI excluding zero is what would demonstrate the model reacts to its
    # tokens; a CI straddling zero is a null result and must be stated as
    # one, not asserted away -- an unconditional "uses its conditioning"
    # claim would be wrong whenever the interval crosses zero.
    ci_excludes_zero = result['delta_ci_low'] > 0 or result['delta_ci_high'] < 0
    conclusion = (
        f'{knockout_arm_label} measurably uses its tokens.' if ci_excludes_zero
        else 'the token arm does not use its tokens (95% CI straddles zero, no significant effect).'
    )
    return (
        f'Conditioning-knockout check on {knockout_arm_label}: consensus Dice '
        f'{result["consensus_dice_full_mean"]:.3f} (full context) -> '
        f'{result["consensus_dice_knockout_mean"]:.3f} (another patient\'s tokens substituted), '
        f'delta {result["delta_mean"]:+.3f}, 95% CI [{result["delta_ci_low"]:+.3f}, '
        f'{result["delta_ci_high"]:+.3f}] (patient-level bootstrap, {result["n_boot"]} resamples, '
        f'seed {result["seed"]}, n={result["n_patients"]} patients): {conclusion} '
        f'The featured model here, {featured_arm_label}, has no tokens by construction and was '
        f'not itself part of this knockout.'
    )


def headline_from_aggregates(old_agg, new_agg):
    old_overall = old_agg.get('overall', {})
    new_overall = new_agg.get('overall', {})
    lines = []
    for key, label in (
        ('flow_consensus_dice', 'consensus Dice'),
        ('flow_consensus_surface_dice', 'consensus surface Dice'),
        ('flow_sample_dice', 'per-sample Dice'),
    ):
        if key in old_overall and key in new_overall:
            old_v = old_overall[key]['pair_mean']
            new_v = new_overall[key]['pair_mean']
            lines.append(f'{label} {old_v:.3f} -> {new_v:.3f} ({new_v - old_v:+.3f})')
    return '; '.join(lines) if lines else 'see headline table'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-name', required=True)
    p.add_argument('--old-checkpoint', required=True)
    p.add_argument('--old-aggregate-json', required=True)
    p.add_argument('--old-bullet', action='append', default=[], dest='old_bullets')
    p.add_argument('--new-name', required=True)
    p.add_argument('--new-checkpoint', required=True)
    p.add_argument('--new-aggregate-json', required=True)
    p.add_argument('--new-bullet', action='append', default=[], dest='new_bullets')
    p.add_argument('--eyebrow', default='PHASE 5 - MODEL COMPARISON')
    p.add_argument('--headline', default=None, help='Override the auto-computed headline sentence.')
    p.add_argument('--takeaway', action='append', default=[], dest='takeaways')
    p.add_argument('--knockout-arm-label', default=None,
                    help="Label for the arm the shuffle-tokens knockout was actually run on (e.g. "
                         "'arm A (binary target + ViT tokens), same kimg 98 checkpoint'). Required "
                         "with --knockout-dir. This is deliberately separate from --new-name: the "
                         "knockout needs tokens to shuffle, so it is often run on a DIFFERENT arm "
                         "than the deck's featured/--new-name model (e.g. a no-token model), and "
                         "the takeaway must say so rather than implying they are the same model.")
    p.add_argument('--intermediate-kimg', type=float, default=None,
                    help='If given, adds a bullet noting this is an INTERMEDIATE checkpoint '
                         '(training paused at this kimg), for --new-bullets.')
    p.add_argument('--edm2-gen-dir', default=None,
                    help='Full-context generate_forecasts.py output (raw per-pair npz), required '
                         'with --knockout-dir to compute the conditioning-knockout delta.')
    p.add_argument('--knockout-dir', default=None,
                    help='--knockout=shuffle-tokens generate_forecasts.py output over the SAME '
                         '--data/--split as --edm2-gen-dir. Adds a knockout-delta takeaway and '
                         'extra_metrics entries; requires --edm2-gen-dir.')
    p.add_argument('--knockout-split', default='test')
    p.add_argument('--knockout-n-boot', type=int, default=2000)
    p.add_argument('--knockout-seed', type=int, default=2026)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    if bool(args.edm2_gen_dir) != bool(args.knockout_dir):
        raise SystemExit('--edm2-gen-dir and --knockout-dir must be given together')
    if args.knockout_dir and not args.knockout_arm_label:
        raise SystemExit('--knockout-arm-label is required with --knockout-dir')

    old_agg = load_json(args.old_aggregate_json)
    new_agg = load_json(args.new_aggregate_json)

    old_sha = sha256_of(args.old_checkpoint) if os.path.isfile(args.old_checkpoint) else None
    new_sha = sha256_of(args.new_checkpoint) if os.path.isfile(args.new_checkpoint) else None

    old_bullets = list(args.old_bullets)
    if old_sha:
        old_bullets.append(f'Checkpoint sha256 {old_sha[:12]}...')
    new_bullets = list(args.new_bullets)
    if new_sha:
        new_bullets.append(f'Checkpoint sha256 {new_sha[:12]}...')
    if args.intermediate_kimg is not None:
        new_bullets.append(
            f'INTERMEDIATE checkpoint: training paused at {args.intermediate_kimg:.0f} kimg')

    headline = args.headline or headline_from_aggregates(old_agg, new_agg)
    takeaways = list(args.takeaways)
    extra_metrics_new = {}

    if args.knockout_dir:
        result = knockout_delta(
            args.edm2_gen_dir, args.knockout_dir, split=args.knockout_split,
            n_boot=args.knockout_n_boot, seed=args.knockout_seed)
        takeaways.append(knockout_takeaway(result, args.knockout_arm_label, args.new_name))
        extra_metrics_new['conditioning_knockout'] = result
        print(f'Knockout delta: {result["consensus_dice_full_mean"]:.4f} -> '
              f'{result["consensus_dice_knockout_mean"]:.4f} '
              f'(delta={result["delta_mean"]:+.4f}, 95% CI '
              f'[{result["delta_ci_low"]:+.4f}, {result["delta_ci_high"]:+.4f}], '
              f'n={result["n_patients"]} patients)')

    recipe = {
        'old': {'name': args.old_name, 'checkpoint': args.old_checkpoint, 'bullets': old_bullets},
        'new': {'name': args.new_name, 'checkpoint': args.new_checkpoint, 'bullets': new_bullets},
        'eyebrow': args.eyebrow,
        'headline': headline,
        'takeaways': takeaways,
        'extra_metrics': {'old': {}, 'new': extra_metrics_new},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(recipe, f, indent=1)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()

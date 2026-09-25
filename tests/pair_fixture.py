# Synthetic C2 token store + C3 pair dataset builder, shared by tests/test_dataset.py,
# tests/test_training_loop.py, and tests/test_train_cli.py. Not a test module itself
# (no test_ prefix).
#
# Builds 3 patients with 2, 3, and 4 scans respectively, consecutive pairing
# (j = i + 1), giving history lengths 1, 1, 2, 1, 2, 3 across 6 train samples
# (max_history = 3), matching contracts C2/C3 in docs/vivit_pipeline_contracts.md.
# Also builds one small val and one small test split (1 patient, 2 scans, 1
# pair each) so PairDataset(split='val'/'test') can be exercised: per
# contract C3, dataset.json only ever carries 'train' stats, so val/test
# datasets must pull sigma_data from the train entry, never their own.

import json
import numpy as np

from training.encodings import encode_target

PATIENTS = {'p1': 2, 'p2': 3, 'p3': 4}
EVAL_PATIENTS = {'val': {'v1': 2}, 'test': {'t1': 2}}


def _scan_id(pid, si):
    return f'{pid}_s{si}'


def _day(si):
    return float(30 * si)


def _make_mask(shape, si):
    m = np.zeros(shape, dtype=np.uint8)
    cx = (2 + si * 3) % max(shape[0] - 4, 1)
    cy = (1 + si * 2) % max(shape[1] - 4, 1)
    cz = (1 + si) % max(shape[2] - 2, 1)
    m[cx:cx + 4, cy:cy + 4, cz:cz + 2] = 1
    return m


def _build_split(rng, tokens_dir, pair_dir, split, patients, shape, spacing):
    """Writes one split's token files and pair .npz files.

    Returns (samples, max_history, target_masks) for this split alone; idx
    restarts at 0 per split, per contract C3.
    """
    (tokens_dir / split).mkdir(parents=True)
    (pair_dir / split).mkdir(parents=True)

    all_masks = {}
    all_images = {}
    for pid, n in patients.items():
        for si in range(n):
            all_masks[(pid, si)] = _make_mask(shape, si)
            all_images[(pid, si)] = rng.randn(*shape).astype(np.float32)
            tokens = rng.randn(256, 768).astype(np.float16)
            np.savez(tokens_dir / split / f'{_scan_id(pid, si)}.npz',
                tokens=tokens, mask=all_masks[(pid, si)], image=all_images[(pid, si)],
                spacing=np.array(spacing, dtype=np.float64))

    samples = []
    idx = 0
    max_history = 0
    target_masks = []
    for pid, n in patients.items():
        for j in range(1, n):
            i = j - 1
            history_scan_ids = [_scan_id(pid, k) for k in range(i + 1)]
            history_ages_days = [_day(j) - _day(k) for k in range(i + 1)]
            max_history = max(max_history, len(history_scan_ids))
            target_mask = all_masks[(pid, j)]
            cond_mask = all_masks[(pid, i)]
            cond_image = all_images[(pid, i)].astype(np.float16)
            delta_days = _day(j) - _day(i)
            target_masks.append(target_mask)

            np.savez(pair_dir / split / f'{idx:08d}.npz',
                target_mask=target_mask, cond_mask=cond_mask, cond_image=cond_image,
                delta_days=np.float64(delta_days),
                history_scan_ids=np.array(history_scan_ids),
                history_ages_days=np.array(history_ages_days, dtype=np.float64),
                patient_id=pid, cond_scan_id=_scan_id(pid, i), target_scan_id=_scan_id(pid, j), split=split)

            samples.append(dict(idx=idx, patient_id=pid, cond_scan_id=_scan_id(pid, i),
                target_scan_id=_scan_id(pid, j), delta_days=delta_days, n_history=len(history_scan_ids)))
            idx += 1

    return samples, max_history, target_masks


def build_pair_fixture(tmp_path, shape=(128, 128, 64), seed=0, include_val_test=True,
                        spacing=(1.0, 1.0, 2.0)):
    """Writes a synthetic token store and pair dataset under tmp_path.

    Returns (pair_dir, tokens_dir, samples) where `samples` is the list of
    TRAIN-split manifest sample dicts (idx, patient_id, cond_scan_id,
    target_scan_id, delta_days, n_history) in build order, for tests to
    index against (unchanged shape from before `include_val_test` existed).

    When include_val_test (default), also writes a 1-patient/2-scan/1-pair
    val split and test split, with NO stats entries of their own -- only
    'train' ever has stats, per contract C3.

    `spacing` defaults to (1, 1, 2) for backward compatibility with existing
    callers; pass a different value (contract C0: the real store now uses
    (0.5, 0.5, 1.0)) to exercise consumers that must read spacing from the
    data rather than hard-code it.
    """
    rng = np.random.RandomState(seed)
    tokens_dir = tmp_path / 'vit_tokens'
    pair_dir = tmp_path / 'pairs'

    train_samples, max_history, target_masks = _build_split(
        rng, tokens_dir, pair_dir, 'train', PATIENTS, shape, spacing)

    splits = {'train': dict(n=len(train_samples), patients=list(PATIENTS.keys()), samples=train_samples)}
    if include_val_test:
        for split, patients in EVAL_PATIENTS.items():
            eval_samples, eval_max_history, _ = _build_split(
                rng, tokens_dir, pair_dir, split, patients, shape, spacing)
            max_history = max(max_history, eval_max_history)
            splits[split] = dict(n=len(eval_samples), patients=list(patients.keys()), samples=eval_samples)

    binary_vals = np.stack([encode_target(m, 'binary') for m in target_masks]).astype(np.float64)
    sdf_vals = np.stack([encode_target(m, 'sdf', spacing) for m in target_masks]).astype(np.float64)
    stats = dict(
        target_fraction_mean=float(np.mean([m.mean() for m in target_masks])),
        target_rms_binary=float(np.sqrt(np.mean(binary_vals ** 2))),
        target_std_binary=float(np.std(binary_vals)),
        target_rms_sdf=float(np.sqrt(np.mean(sdf_vals ** 2))),
        target_std_sdf=float(np.std(sdf_vals)),
        delta_days_max=float(max(s['delta_days'] for s in train_samples)),
        delta_days_median=float(np.median([s['delta_days'] for s in train_samples])),
        history_len_max=max_history,
    )

    manifest = dict(
        tokens_dir=str(tokens_dir),
        pairing='consecutive',
        max_history=max_history,
        spacing=list(spacing),
        shape=list(shape),
        splits=splits,
        stats={'train': stats}, # contract C3: val/test never get their own stats
    )
    with open(pair_dir / 'dataset.json', 'w') as f:
        json.dump(manifest, f)

    return str(pair_dir), str(tokens_dir), train_samples

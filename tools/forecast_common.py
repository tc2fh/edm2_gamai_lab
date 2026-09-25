# Shared helpers for generate_forecasts.py, evaluate_forecasts.py and
# select_checkpoint.py: checkpoint loading (contract C7), the EDM sampler
# adapted to the batch-dict interface of contract C5/C6, packbits round trip
# for sample masks, and the FlowMatchingGrowthNet-style metric set (plan
# decision 9 / Phase 4).
#
# generate_images.py is deleted as part of this same change set (plan
# Phase 2 cleanup), so the sampler is reimplemented here rather than
# imported from it.

import glob
import hashlib
import os
import re

import numpy as np
import torch

from training.encodings import decode_target

#----------------------------------------------------------------------------
# Checkpoint loading (contract C7).

def load_checkpoint(net_path, device):
    """Loads a network-snapshot-*.pkl and returns (net, dataset_kwargs, sha256)."""
    sha256 = hashlib.sha256()
    with open(net_path, 'rb') as f:
        raw = f.read()
    sha256.update(raw)
    import pickle
    data = pickle.loads(raw)
    net = data['ema'].to(device)
    net.eval()
    dataset_kwargs = data.get('dataset_kwargs')
    if dataset_kwargs is None:
        raise ValueError(f'{net_path}: pickle has no dataset_kwargs (contract C7)')
    for key in ('target_encoding', 'cond_image', 'max_history', 'tokens_dir', 'sigma_data', 'img_resolution'):
        if key not in dataset_kwargs:
            raise ValueError(f'{net_path}: dataset_kwargs missing contract C7 key {key!r}')
    return net, dataset_kwargs, sha256.hexdigest()


def resolve_net_path(net, run_dir, snapshot_pattern):
    """Resolve --net, or a --run-dir + --snapshot glob pattern, to one .pkl path."""
    if net is not None:
        return net
    if run_dir is None:
        raise ValueError('must pass either --net or --run-dir')
    pattern = snapshot_pattern or 'network-snapshot-*.pkl'
    matches = sorted(glob.glob(os.path.join(run_dir, pattern)))
    if not matches:
        raise ValueError(f'no snapshot in {run_dir!r} matching {pattern!r}')
    if len(matches) > 1:
        # Pick the highest kimg; if several EMA variants tie at that kimg,
        # refuse rather than silently guessing which one the caller wanted.
        def kimg_of(path):
            m = re.search(r'network-snapshot-(\d+)', os.path.basename(path))
            return int(m.group(1)) if m else -1
        best_kimg = max(kimg_of(p) for p in matches)
        at_best = [p for p in matches if kimg_of(p) == best_kimg]
        if len(at_best) > 1:
            raise ValueError(
                f'{len(at_best)} snapshots tie at kimg={best_kimg} matching {pattern!r} in {run_dir!r}: '
                f'{at_best}. Narrow --snapshot to a single file (e.g. include the EMA suffix).')
        return at_best[0]
    return matches[0]

#----------------------------------------------------------------------------
# Dataset reconstruction from checkpoint metadata (contract C7), overriding
# only the path/split the caller wants to generate on. Never augments
# (flip_axes=False) at generation time.

def build_dataset(pair_dir, split, ckpt_dataset_kwargs, tokens_dir_override=None, max_history_override=None):
    from training.dataset import PairDataset
    # Always pass the checkpoint's own sigma_data (contract C7 guarantees
    # it's present): this is what the network was actually trained with, and
    # it lets PairDataset skip its dataset.json stats.train lookup entirely,
    # so generation/evaluation/selection work on a pair dir with no train
    # stats block (e.g. a test-only export) as long as --data has the
    # requested split's samples.
    return PairDataset(
        pair_dir,
        tokens_dir=tokens_dir_override or ckpt_dataset_kwargs['tokens_dir'],
        split=split,
        target_encoding=ckpt_dataset_kwargs['target_encoding'],
        cond_image=ckpt_dataset_kwargs['cond_image'],
        max_history=max_history_override or ckpt_dataset_kwargs['max_history'],
        flip_axes=False,
        sigma_data=ckpt_dataset_kwargs['sigma_data'],
        # Older checkpoints (pre --no-context) carry no 'use_context' key;
        # True matches PairDataset's own default and their actual behavior.
        use_context=ckpt_dataset_kwargs.get('use_context', True),
    )

#----------------------------------------------------------------------------
# EDM sampler (Heun, 2nd order), adapted from generate_images.py to the
# batch-dict / Precond.forward interface of contract C5/C6.

def edm_sampler(
    net, noise, cond_image=None, context=None, context_mask=None, context_ages=None, delta_days=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    def denoise(x, t):
        return net(x, t, cond_image=cond_image, context=context, context_mask=context_mask,
                    context_ages=context_ages, delta_days=delta_days, force_fp32=True).to(dtype)

    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Per-sample-seeded noise, matching generate_images.py's StackedRandomGenerator.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

#----------------------------------------------------------------------------
# Packbits round trip for the sample-mask stack (N,128,128,64) bool -> packed
# uint8 along the last axis, plus the original shape to unpack exactly.

def pack_samples(bool_masks):
    shape = np.array(bool_masks.shape, dtype=np.int64)
    packed = np.packbits(bool_masks, axis=-1)
    return packed, shape


def unpack_samples(packed, shape):
    shape = tuple(int(s) for s in shape)
    bits = np.unpackbits(packed, axis=-1)
    bits = bits[..., :shape[-1]]
    return bits.reshape(shape).astype(bool)

#----------------------------------------------------------------------------
# Metrics (plan decision 9 / Phase 4), matching FlowMatchingGrowthNet's
# empty-mask conventions: both masks empty -> Dice 1, exactly one empty -> 0.

def dice_coeff(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return float(2.0 * intersection / (a_sum + b_sum))


def volume_mm3(mask, spacing):
    # spacing is required, not defaulted: contract C0 -- the token store's
    # physical voxel spacing is not (1,1,2) and callers must read it from
    # the data (dataset.json / PairDataset.spacing, or a generation npz's
    # own 'spacing' field), never assume it.
    voxel_vol = float(np.prod(spacing))
    return float(np.asarray(mask, dtype=bool).sum()) * voxel_vol


def surface_dice(a, b, spacing, tolerance_mm=1.0):
    """Surface Dice at the given tolerance, via scipy distance transforms.

    `spacing` is required (contract C0: never assume (1,1,2), read it from
    the data). Empty-mask convention matches dice_coeff: both empty -> 1, one empty -> 0.
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0

    def surface_voxels(mask):
        eroded = binary_erosion(mask, border_value=0)
        return np.logical_and(mask, np.logical_not(eroded))

    surf_a = surface_voxels(a)
    surf_b = surface_voxels(b)
    if not surf_a.any() or not surf_b.any():
        # A mask can be foreground everywhere within the volume (no
        # background to erode against): fall back to dice's convention via
        # its area comparison, since "surface" is degenerate.
        return dice_coeff(a, b)

    # Distance from every voxel to the nearest surface voxel of the *other*
    # mask, sampled only at the surface voxels of each mask.
    dist_to_a = distance_transform_edt(np.logical_not(surf_a), sampling=spacing)
    dist_to_b = distance_transform_edt(np.logical_not(surf_b), sampling=spacing)

    d_a_to_b = dist_to_b[surf_a] # distance from each surface-a voxel to nearest surface-b voxel
    d_b_to_a = dist_to_a[surf_b]

    n_close = int((d_a_to_b <= tolerance_mm).sum()) + int((d_b_to_a <= tolerance_mm).sum())
    n_total = int(surf_a.sum()) + int(surf_b.sum())
    return float(n_close / n_total) if n_total > 0 else 1.0


def classify_direction(baseline_vol, target_vol, threshold=0.20):
    """grew / stable / shrank, +-`threshold` relative to baseline volume."""
    if baseline_vol <= 0:
        return 'grew' if target_vol > 0 else 'stable'
    rel_change = (target_vol - baseline_vol) / baseline_vol
    if rel_change > threshold:
        return 'grew'
    if rel_change < -threshold:
        return 'shrank'
    return 'stable'

#----------------------------------------------------------------------------
# Deterministic cross-patient shuffle map for the shuffle-tokens knockout:
# rotate the (sorted) patient list by an amount derived from --seed, and map
# each sample to the same-position sample of its rotated target patient
# (wrapping if that patient has fewer samples).

def build_patient_shuffle_map(patient_ids, seed=0):
    patients_sorted = sorted(set(patient_ids))
    n_patients = len(patients_sorted)
    patient_to_indices = {p: [i for i, pp in enumerate(patient_ids) if pp == p] for p in patients_sorted}
    if n_patients < 2:
        return list(range(len(patient_ids))) # cannot swap to "another patient"; identity map.
    rot = 1 + (seed % (n_patients - 1)) if n_patients > 1 else 1
    rotated = patients_sorted[rot:] + patients_sorted[:rot]
    patient_rotate_map = dict(zip(patients_sorted, rotated))
    shuffle_map = []
    within_patient_pos = {p: 0 for p in patients_sorted}
    for p in patient_ids:
        target_patient = patient_rotate_map[p]
        candidates = patient_to_indices[target_patient]
        pos = within_patient_pos[p] % len(candidates)
        shuffle_map.append(candidates[pos])
        within_patient_pos[p] += 1
    return shuffle_map

#----------------------------------------------------------------------------

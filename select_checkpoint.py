"""Cheap val-split checkpoint selection for a training run
(docs/vivit_conditioning_plan.md Phase 4). Evaluates every network snapshot
(or every k-th) in a run directory with a small sample/step budget and
writes {run}/val_selection.csv, printing the best snapshot by consensus
Dice. Handles the EMA variants training_loop.py saves, one pickle per EMA
std (e.g. network-snapshot-0000010-0.050.pkl)."""

import csv
import glob
import os
import re

import click
import numpy as np
import torch

from tools.forecast_common import (
    StackedRandomGenerator, build_dataset, classify_direction, dice_coeff, edm_sampler,
    load_checkpoint, surface_dice, volume_mm3,
)
from training.encodings import decode_target

#----------------------------------------------------------------------------

SNAPSHOT_RE = re.compile(r'network-snapshot-(\d+)(-[\d.]+)?\.pkl$')


def list_snapshots(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, 'network-snapshot-*.pkl')))
    out = []
    for p in paths:
        m = SNAPSHOT_RE.search(os.path.basename(p))
        if not m:
            continue
        kimg = int(m.group(1))
        ema_std = m.group(2)[1:] if m.group(2) else 'final'
        out.append((kimg, ema_std, p))
    out.sort(key=lambda t: (t[0], t[1]))
    return out

#----------------------------------------------------------------------------

def evaluate_snapshot(net_path, data, split, tokens_dir_override, num_samples, steps, device,
                       sigma_min, sigma_max, rho, seed, direction_threshold, surface_tolerance_mm):
    model, ckpt_dataset_kwargs, sha256 = load_checkpoint(net_path, device)
    dataset = build_dataset(data, split, ckpt_dataset_kwargs, tokens_dir_override=tokens_dir_override)
    img_resolution = tuple(dataset.resolution)
    spacing = tuple(dataset.spacing) # contract C0: data-driven, never (1,1,2) by assumption.

    def to_batch_or_none(x, n):
        if x is None:
            return None
        t = torch.as_tensor(x, device=device).unsqueeze(0)
        return t.expand(n, *([-1] * (t.ndim - 1))).contiguous()

    consensus_dices, surf_dices, direction_correct, cf_dices = [], [], [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        cond_image = torch.as_tensor(item['cond_image'], device=device).unsqueeze(0)
        # .get(): a checkpoint trained with use_context=False reconstructs a
        # PairDataset whose items omit these keys (nothing to condition on).
        context = item.get('context')
        context_mask = item.get('context_mask')
        context_ages = item.get('context_ages')
        delta_days = torch.as_tensor([item['delta_days']], device=device, dtype=torch.float32)

        seeds = [seed * 1_000_003 + idx * 131 + s for s in range(num_samples)]
        rnd = StackedRandomGenerator(device, seeds)
        noise = rnd.randn([num_samples, 1] + list(img_resolution), device=device)
        cond_image_b = cond_image.expand(num_samples, *([-1] * (cond_image.ndim - 1))).contiguous()
        context_b = to_batch_or_none(context, num_samples)
        context_mask_b = to_batch_or_none(context_mask, num_samples)
        context_ages_b = to_batch_or_none(context_ages, num_samples)
        delta_days_b = delta_days.expand(num_samples).contiguous()

        with torch.no_grad():
            latents = edm_sampler(
                model, noise, cond_image=cond_image_b, context=context_b, context_mask=context_mask_b,
                context_ages=context_ages_b, delta_days=delta_days_b,
                num_steps=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, randn_like=rnd.randn_like,
            )
        decoded = decode_target(latents, ckpt_dataset_kwargs['target_encoding'])[:, 0].cpu().numpy().astype(bool)
        prob = decoded.mean(axis=0)
        consensus = prob > 0.5

        target_mask = decode_target(item['image'][0], ckpt_dataset_kwargs['target_encoding']).astype(bool)
        cond_mask = item['cond_image'][0] > 0 if item['cond_image'].shape[0] > 0 else np.zeros(img_resolution, dtype=bool)

        consensus_dices.append(dice_coeff(consensus, target_mask))
        surf_dices.append(surface_dice(consensus, target_mask, spacing=spacing, tolerance_mm=surface_tolerance_mm))
        cf_dices.append(dice_coeff(cond_mask, target_mask))

        baseline_vol = volume_mm3(cond_mask, spacing)
        target_vol = volume_mm3(target_mask, spacing)
        pred_vol = volume_mm3(consensus, spacing)
        d_true = classify_direction(baseline_vol, target_vol, direction_threshold)
        d_pred = classify_direction(baseline_vol, pred_vol, direction_threshold)
        direction_correct.append(d_true == d_pred)

    return dict(
        consensus_dice_mean=float(np.mean(consensus_dices)),
        surface_dice_mean=float(np.mean(surf_dices)),
        direction_accuracy=float(np.mean(direction_correct)),
        carry_forward_dice_mean=float(np.mean(cf_dices)),
        n_pairs=len(dataset),
    )

#----------------------------------------------------------------------------

@click.command()
@click.option('--run-dir', required=True, help='Training run directory with network-snapshot-*.pkl files.')
@click.option('--data', required=True, help='Pair dataset root.')
@click.option('--split', default='val', show_default=True, type=click.Choice(['train', 'val', 'test']))
@click.option('--tokens-dir', default=None, help='Override the C2 token store dir.')
@click.option('--num-samples', default=8, show_default=True, type=int)
@click.option('--steps', default=18, show_default=True, type=int)
@click.option('--every-k', default=1, show_default=True, type=int, help='Evaluate every k-th snapshot (by kimg order).')
@click.option('--seed', default=0, show_default=True, type=int)
@click.option('--sigma-min', default=0.002, show_default=True, type=float)
@click.option('--sigma-max', default=80.0, show_default=True, type=float)
@click.option('--rho', default=7.0, show_default=True, type=float)
@click.option('--direction-threshold', default=0.20, show_default=True, type=float)
@click.option('--surface-tolerance-mm', default=1.0, show_default=True, type=float)
@click.option('--out', default=None, help='Output CSV path (default: {run-dir}/val_selection.csv).')
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu', show_default=True)
def cmdline(run_dir, data, split, tokens_dir, num_samples, steps, every_k, seed,
            sigma_min, sigma_max, rho, direction_threshold, surface_tolerance_mm, out, device):
    """Evaluate every (or every k-th) snapshot in a run dir on the val split
    with a cheap sample/step budget, and report the best by consensus Dice."""
    device = torch.device(device)
    snapshots = list_snapshots(run_dir)
    if not snapshots:
        raise click.ClickException(f'no network-snapshot-*.pkl found in {run_dir!r}')
    snapshots = snapshots[::every_k] if every_k > 1 else snapshots

    out = out or os.path.join(run_dir, 'val_selection.csv')
    rows = []
    for kimg, ema_std, path in snapshots:
        click.echo(f'Evaluating {os.path.basename(path)} (kimg={kimg}, ema={ema_std}) ...')
        metrics = evaluate_snapshot(path, data, split, tokens_dir, num_samples, steps, device,
                                     sigma_min, sigma_max, rho, seed, direction_threshold, surface_tolerance_mm)
        row = dict(snapshot=os.path.basename(path), path=path, kimg=kimg, ema_std=ema_std, **metrics)
        rows.append(row)
        click.echo(f'  consensus_dice={metrics["consensus_dice_mean"]:.4f} '
                   f'surface_dice={metrics["surface_dice_mean"]:.4f} '
                   f'direction_acc={metrics["direction_accuracy"]:.4f} '
                   f'carry_forward_dice={metrics["carry_forward_dice_mean"]:.4f}')

    fieldnames = ['snapshot', 'path', 'kimg', 'ema_std', 'consensus_dice_mean', 'surface_dice_mean',
                  'direction_accuracy', 'carry_forward_dice_mean', 'n_pairs']
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    best = max(rows, key=lambda r: r['consensus_dice_mean'])
    click.echo(f'\nWrote {out}')
    click.echo(f'Best snapshot by consensus Dice: {best["snapshot"]} '
               f'(consensus_dice={best["consensus_dice_mean"]:.4f}, path={best["path"]})')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    cmdline()

#----------------------------------------------------------------------------

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate future-mask forecasts from a trained ViViT-conditioned EDM2
network (docs/vivit_conditioning_plan.md Phase 2/4, contracts C6/C7 in
docs/vivit_pipeline_contracts.md). Replaces generate_images.py for this
pipeline.

Per pair, draws --num-samples masks with the EDM Heun sampler and writes
{out}/{split}/{idx:08d}.npz (packed sample masks, mean probability map,
ground-truth/baseline masks, and metadata) plus one
{out}/{split}/manifest.json for the whole run.
"""

import hashlib
import json
import os

import click
import numpy as np
import torch

from tools.forecast_common import (
    StackedRandomGenerator, build_dataset, build_patient_shuffle_map,
    edm_sampler, pack_samples, resolve_net_path, load_checkpoint,
)
from training.encodings import decode_target

#----------------------------------------------------------------------------

def parse_int_list(s):
    if not s:
        return None
    return [int(x) for x in str(s).split(',') if x != '']

#----------------------------------------------------------------------------

def load_raw_pair(pair_dir, split, idx):
    """Raw fields straight from the pair npz (contract C3), independent of
    the diffusion encoding a PairDataset applies."""
    path = os.path.join(pair_dir, split, f'{idx:08d}.npz')
    with np.load(path) as z:
        return dict(
            target_mask=z['target_mask'].astype(np.uint8),
            cond_mask=z['cond_mask'].astype(np.uint8),
            delta_days=float(np.asarray(z['delta_days'])),
            patient_id=str(z['patient_id']),
            cond_scan_id=str(z['cond_scan_id']),
            target_scan_id=str(z['target_scan_id']),
        )

#----------------------------------------------------------------------------

def apply_knockout(item, knockout, cond_channels, shuffle_item, fixed_delta_days):
    """Mutates a copy of a PairDataset.__getitem__ dict in place per the
    plan's Phase-4 ablations; returns (context, context_mask, context_ages,
    cond_image, delta_days) ready to feed the network (context_* may be None)."""
    # .get(), not [...]: a checkpoint trained with use_context=False (C7
    # metadata) reconstructs a PairDataset whose items omit these keys
    # entirely (contract: no context to knock out either).
    context = item.get('context')
    context_mask = item.get('context_mask')
    context_ages = item.get('context_ages')
    cond_image = item['cond_image'].copy()
    delta_days = float(item['delta_days'])

    drop_context = knockout in ('no-context', 'no-context-no-cond')
    drop_cond_image = knockout in ('no-cond-image', 'no-context-no-cond')

    if drop_context:
        context, context_mask, context_ages = None, None, None
    elif knockout == 'shuffle-tokens':
        context = shuffle_item['context']
        context_mask = shuffle_item['context_mask']
        context_ages = shuffle_item['context_ages']

    if drop_cond_image and cond_channels > 0:
        # "empty tumor": mask channel filled with -1 (encode_cond_image's
        # background value), image channel (if present) zeroed. Documented
        # in the plan's generation deliverable as the chosen convention.
        cond_image[0, ...] = -1.0
        if cond_channels > 1:
            cond_image[1, ...] = 0.0

    if knockout == 'fixed-time':
        delta_days = float(fixed_delta_days)

    return context, context_mask, context_ages, cond_image, delta_days

#----------------------------------------------------------------------------

def to_batch(x_np, device, chunk):
    """(...) numpy -> (chunk, ...) torch tensor on device, or None."""
    if x_np is None:
        return None
    t = torch.as_tensor(x_np, device=device).unsqueeze(0)
    return t.expand(chunk, *([-1] * (t.ndim - 1))).contiguous()

#----------------------------------------------------------------------------

def write_nifti_outputs(out_dir, split, idx, tokens_dir, cond_scan_id, prob, consensus_mask):
    # No spacing argument: the crop-frame affine read below from the token
    # store already fully encodes voxel spacing and orientation (contract
    # C1/C2), and resample_from_to works from affines directly, so there is
    # nothing here that could hard-code the wrong spacing.
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except ImportError:
        click.echo(f'[warn] nibabel not available; skipping NIfTI output for idx={idx}', err=True)
        return

    tok_path = os.path.join(tokens_dir, split, f'{cond_scan_id}.npz')
    if not os.path.isfile(tok_path):
        click.echo(f'[warn] token store file not found for NIfTI affine: {tok_path}', err=True)
        return
    with np.load(tok_path, allow_pickle=True) as z:
        affine = np.asarray(z['affine'], dtype=np.float64)
        source_image_path = str(z['source_image_path']) if 'source_image_path' in z.files else None

    nii_dir = os.path.join(out_dir, split, 'nifti')
    os.makedirs(nii_dir, exist_ok=True)

    prob_img = nib.Nifti1Image(prob.astype(np.float32), affine)
    mask_img = nib.Nifti1Image(consensus_mask.astype(np.uint8), affine)
    nib.save(prob_img, os.path.join(nii_dir, f'{idx:08d}_prob_crop.nii.gz'))
    nib.save(mask_img, os.path.join(nii_dir, f'{idx:08d}_mask_crop.nii.gz'))

    if source_image_path and os.path.isfile(source_image_path):
        orig = nib.load(source_image_path)
        prob_resampled = resample_from_to(prob_img, orig, order=1) # linear for probability
        mask_resampled = resample_from_to(mask_img, orig, order=0) # nearest for masks
        nib.save(prob_resampled, os.path.join(nii_dir, f'{idx:08d}_prob_orig.nii.gz'))
        nib.save(mask_resampled, os.path.join(nii_dir, f'{idx:08d}_mask_orig.nii.gz'))
    else:
        click.echo(f'[warn] source_image_path missing/unreadable for idx={idx}; no orig-grid NIfTI written', err=True)

#----------------------------------------------------------------------------

@click.command()
@click.option('--net', default=None, help='Network snapshot pickle. If omitted, resolved from --run-dir/--snapshot.')
@click.option('--run-dir', default=None, help='Training run directory (used with --snapshot if --net is not given).')
@click.option('--snapshot', 'snapshot_pattern', default=None, help='Glob pattern for --run-dir, e.g. "network-snapshot-*-0.050.pkl".')
@click.option('--data', required=True, help='Pair dataset root (prepare_data_vivit_pairs.py output).')
@click.option('--split', default='test', show_default=True, type=click.Choice(['train', 'val', 'test']))
@click.option('--out', required=True, help='Output directory.')
@click.option('--num-samples', default=32, show_default=True, type=int)
@click.option('--steps', default=32, show_default=True, type=int)
@click.option('--batch', default=8, show_default=True, type=int, help='Samples per forward pass.')
@click.option('--seed', default=0, show_default=True, type=int)
@click.option('--sigma-min', default=0.002, show_default=True, type=float)
@click.option('--sigma-max', default=80.0, show_default=True, type=float)
@click.option('--rho', default=7.0, show_default=True, type=float)
@click.option('--knockout', default='none', show_default=True,
              type=click.Choice(['none', 'no-context', 'no-cond-image', 'shuffle-tokens', 'fixed-time', 'no-context-no-cond']))
@click.option('--write-nifti/--no-write-nifti', default=False, show_default=True)
@click.option('--ids', default=None, help='Comma-separated subset of sample indices (default: all).')
@click.option('--tokens-dir', default=None, help='Override the C2 token store dir (default: checkpoint metadata).')
@click.option('--fixed-delta-days', default=None, type=float,
              help="delta_days value for --knockout=fixed-time. Default: the pair dir's train "
                   "delta_days_median (dataset.json), falling back to the checkpoint's own metadata "
                   "if the dataset has no train stats (e.g. a test-only pair dir); required explicitly "
                   "if neither is available.")
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu', show_default=True)
def cmdline(net, run_dir, snapshot_pattern, data, split, out, num_samples, steps, batch, seed,
            sigma_min, sigma_max, rho, knockout, write_nifti, ids, tokens_dir, fixed_delta_days, device):
    """Generate forecast masks for a pair dataset split from a trained checkpoint."""
    device = torch.device(device)
    net_path = resolve_net_path(net, run_dir, snapshot_pattern)
    click.echo(f'Loading checkpoint {net_path} ...')
    model, ckpt_dataset_kwargs, sha256 = load_checkpoint(net_path, device)

    dataset = build_dataset(data, split, ckpt_dataset_kwargs, tokens_dir_override=tokens_dir)
    n = len(dataset)
    id_list = parse_int_list(ids) if ids else list(range(n))
    spacing = tuple(dataset.spacing) # contract C0: data-driven, never (1,1,2) by assumption.

    if knockout == 'shuffle-tokens' and not dataset.use_context:
        raise click.ClickException(
            "--knockout=shuffle-tokens needs a checkpoint trained with context "
            "(dataset_kwargs['use_context']=True in its C7 metadata); this checkpoint has none "
            "to shuffle. Use --knockout=none or another knockout instead.")

    manifest_path = os.path.join(data, 'dataset.json')
    with open(manifest_path) as f:
        pair_manifest = json.load(f)
    split_samples = pair_manifest['splits'][split]['samples']
    patient_ids = [s['patient_id'] for s in split_samples]
    shuffle_map = build_patient_shuffle_map(patient_ids, seed=seed) if knockout == 'shuffle-tokens' else None

    # Only resolved when actually needed: a test-only pair dir (e.g.
    # datasets/flow_test_pairs) has no train stats block by design (contract
    # C3), so this must not be looked up unconditionally.
    if knockout == 'fixed-time':
        if fixed_delta_days is not None:
            pass # explicit CLI override wins.
        elif pair_manifest.get('stats', {}).get('train', {}).get('delta_days_median') is not None:
            fixed_delta_days = float(pair_manifest['stats']['train']['delta_days_median'])
        elif ckpt_dataset_kwargs.get('delta_days_median') is not None:
            fixed_delta_days = float(ckpt_dataset_kwargs['delta_days_median'])
        else:
            raise click.ClickException(
                "--knockout=fixed-time needs a delta_days value: the pair dir at --data has no "
                "dataset.json stats.train.delta_days_median (e.g. a test-only pair dir) and the "
                "checkpoint carries none either. Pass --fixed-delta-days explicitly.")

    cond_channels = dataset.cond_channels
    img_resolution = tuple(dataset.resolution)
    split_out_dir = os.path.join(out, split)
    os.makedirs(split_out_dir, exist_ok=True)

    click.echo(f'Generating {len(id_list)} pair(s) x {num_samples} samples, knockout={knockout!r} ...')
    for idx in id_list:
        item = dataset[idx]
        shuffle_item = dataset[shuffle_map[idx]] if shuffle_map is not None else None
        context, context_mask, context_ages, cond_image, delta_days = apply_knockout(
            item, knockout, cond_channels, shuffle_item, fixed_delta_days)

        raw = load_raw_pair(data, split, idx)

        all_masks = np.zeros((num_samples,) + img_resolution, dtype=bool)
        done = 0
        while done < num_samples:
            chunk = min(batch, num_samples - done)
            seeds = [seed * 1_000_003 + idx * 131 + done + s for s in range(chunk)]
            rnd = StackedRandomGenerator(device, seeds)

            noise = rnd.randn([chunk, 1] + list(img_resolution), device=device)
            cond_image_b = to_batch(cond_image, device, chunk)
            context_b = to_batch(context, device, chunk)
            context_mask_b = to_batch(context_mask, device, chunk)
            context_ages_b = to_batch(context_ages, device, chunk)
            delta_days_b = torch.full((chunk,), delta_days, device=device, dtype=torch.float32)

            with torch.no_grad():
                latents = edm_sampler(
                    model, noise, cond_image=cond_image_b, context=context_b,
                    context_mask=context_mask_b, context_ages=context_ages_b, delta_days=delta_days_b,
                    num_steps=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
                    randn_like=rnd.randn_like,
                )
            decoded = decode_target(latents, ckpt_dataset_kwargs['target_encoding'])
            all_masks[done:done + chunk] = decoded[:, 0].cpu().numpy().astype(bool)
            done += chunk

        prob = all_masks.mean(axis=0).astype(np.float16)
        packed, packed_shape = pack_samples(all_masks)

        sampler_settings = dict(
            num_samples=num_samples, steps=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            seed=seed, knockout=knockout, delta_days_used=delta_days, checkpoint=net_path, checkpoint_sha256=sha256,
        )

        np.savez(
            os.path.join(split_out_dir, f'{idx:08d}.npz'),
            samples_packed=packed,
            samples_shape=packed_shape,
            prob=prob,
            target_mask=raw['target_mask'],
            cond_mask=raw['cond_mask'],
            delta_days=np.float64(delta_days),
            delta_days_original=np.float64(raw['delta_days']),
            patient_id=raw['patient_id'],
            cond_scan_id=raw['cond_scan_id'],
            target_scan_id=raw['target_scan_id'],
            sampler_settings_json=json.dumps(sampler_settings),
            spacing=np.asarray(spacing, dtype=np.float64),
        )

        if write_nifti:
            consensus = prob > 0.5
            write_nifti_outputs(out, split, idx, ckpt_dataset_kwargs['tokens_dir'], raw['cond_scan_id'],
                                 prob.astype(np.float32), consensus)

    run_manifest = dict(
        checkpoint=net_path,
        checkpoint_sha256=sha256,
        data=data,
        split=split,
        knockout=knockout,
        num_samples=num_samples,
        steps=steps,
        sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
        seed=seed,
        ids=id_list,
        cond_image_mode=ckpt_dataset_kwargs['cond_image'],
        target_encoding=ckpt_dataset_kwargs['target_encoding'],
        max_history=ckpt_dataset_kwargs['max_history'],
        tokens_dir=ckpt_dataset_kwargs['tokens_dir'],
        write_nifti=write_nifti,
        fixed_delta_days=fixed_delta_days if knockout == 'fixed-time' else None,
        spacing=list(spacing),
        use_context=dataset.use_context,
    )
    with open(os.path.join(split_out_dir, 'manifest.json'), 'w') as f:
        json.dump(run_manifest, f, indent=2)
    click.echo(f'Wrote {len(id_list)} pair(s) to {split_out_dir}')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    cmdline()

#----------------------------------------------------------------------------

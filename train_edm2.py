# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion models according to the EDM2 recipe from the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models",
extended with ViViT token conditioning (docs/vivit_conditioning_plan.md)."""

import os
import json
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Legacy architecture/hyperparameter presets. These predate the ViViT pair
# dataset and were tuned for the old whole-volume 128^3 longitudinal setup on
# Rivanna; they still run against the current PairDataset pipeline (just
# pass --data/--tokens-dir etc. as usual) but are not the recommended
# starting point for the local ViT-token pipeline -- see the plan's section
# 2.8 for the small local architecture (pass architecture flags directly,
# no --preset).

config_presets = {
    'edm2-img512-xxs':  dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=64,  lr=0.0170, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xs':   dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-s':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-m':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-l':    dnnlib.EasyDict(duration=1792<<20, batch=2048, channels=320, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xl':   dnnlib.EasyDict(duration=1280<<20, batch=2048, channels=384, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xxl':  dnnlib.EasyDict(duration=896<<20,  batch=2048, channels=448, lr=0.0065, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img64-xs':    dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=128, lr=0.0120, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-s':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=192, lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-m':     dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-l':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=320, lr=0.0080, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-xl':    dnnlib.EasyDict(duration=640<<20,  batch=2048, channels=384, lr=0.0070, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),

    # 3D volume presets (Rivanna, legacy whole-volume 128^3 longitudinal setup).
    'edm2-vol128-xxs':  dnnlib.EasyDict(duration=256<<20, batch=8,  channels=32,  channel_mult=[1,2,4,8], lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-vol128-xs':   dnnlib.EasyDict(duration=256<<20, batch=8,  channels=48,  channel_mult=[1,2,4,8], lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-vol128-s':    dnnlib.EasyDict(duration=256<<20, batch=4,  channels=64,  channel_mult=[1,2,4,8], lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.4, P_std=1.0),
}

#----------------------------------------------------------------------------

def parse_int_list(s):
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    return [int(x) for x in str(s).split(',') if x != '']

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset=None, **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()

    # Preset (optional; fills in only options not already given on the command line).
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts.get(key, None) is None:
                opts[key] = value

    if opts.get('channels', None) is None:
        raise click.ClickException('--channels is required when --preset is not given')
    if opts.get('duration', None) is None:
        raise click.ClickException('--duration is required when --preset is not given')
    if opts.get('batch', None) is None:
        raise click.ClickException('--batch is required when --preset is not given')
    for key in ('lr', 'decay', 'dropout', 'P_mean', 'P_std'):
        if opts.get(key, None) is None:
            raise click.ClickException(f'--{key.replace("_", "-").lower()} is required when --preset is not given')

    # Dataset (contract C3/C5): PairDataset over the prepared pair directory.
    dataset_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.PairDataset',
        path=opts.data,
        split=opts.get('split', None) or 'train',
        target_encoding=opts.get('target_encoding', None) or 'binary',
        cond_image=opts.get('cond_image', None) or 'mask',
    )
    if opts.get('tokens_dir', None):
        dataset_kwargs.tokens_dir = opts.tokens_dir
    if opts.get('max_history', None):
        dataset_kwargs.max_history = opts.max_history
    if opts.get('flip_axes', None) is not None:
        dataset_kwargs.flip_axes = opts.flip_axes
    use_context = opts.get('context', True)
    if use_context is None:
        use_context = True
    dataset_kwargs.use_context = use_context
    c.dataset_kwargs = dataset_kwargs

    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    sigma_data = opts.get('sigma_data', None)
    if sigma_data is None:
        sigma_data = dataset_obj.sigma_data

    # Architecture.
    channel_mult = parse_int_list(opts.get('channel_mult', None))
    attn_resolutions = parse_int_list(opts.get('attn_resolutions', None))
    cross_attn_resolutions = parse_int_list(opts.get('cross_attn_resolutions', None))
    if cross_attn_resolutions is None:
        cross_attn_resolutions = [16, 32]

    network_kwargs = dnnlib.EasyDict(
        class_name='training.networks_edm2.Precond',
        model_channels=opts.channels,
        dropout=opts.dropout,
        sigma_data=sigma_data,
        channels_per_head=opts.get('channels_per_head', None) or 32,
        cross_attn_resolutions=cross_attn_resolutions,
    )
    if channel_mult is not None:
        network_kwargs.channel_mult = channel_mult
    if attn_resolutions is not None:
        network_kwargs.attn_resolutions = attn_resolutions
    if opts.get('num_blocks', None) is not None:
        network_kwargs.num_blocks = opts.num_blocks
    network_kwargs.use_fp16 = opts.get('fp16', True)
    network_kwargs.dtype = opts.get('dtype', None) or 'fp16'
    c.network_kwargs = network_kwargs

    # Fail fast on a zero-heads configuration (or any other constructor
    # error) before launching training, with a readable message.
    try:
        probe_net = dnnlib.util.construct_class_by_name(
            img_resolution=dataset_obj.resolution,
            img_channels=dataset_obj.num_channels,
            cond_channels=dataset_obj.cond_channels,
            context_dim=(768 if use_context else 0),
            context_tokens=(256 if use_context else 0),
            **network_kwargs,
        )
        del probe_net
    except ValueError as err:
        raise click.ClickException(str(err))

    # Hyperparameters.
    c.update(total_nimg=opts.duration, batch_size=opts.batch)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.training_loop.EDM2Loss', P_mean=opts.P_mean, P_std=opts.P_std, sigma_data=sigma_data)
    c.lr_kwargs = dnnlib.EasyDict(func_name='training.training_loop.learning_rate_schedule', ref_lr=opts.lr, ref_batches=opts.decay)
    if opts.get('rampup', None) is not None:
        c.lr_kwargs.rampup_nimg = opts.rampup

    # Performance-related options.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    # I/O-related options.
    c.status_nimg = opts.get('status', 0) or None
    c.snapshot_nimg = opts.get('snapshot', 0) or None
    c.checkpoint_nimg = opts.get('checkpoint', 0) or None
    c.seed = opts.get('seed', 0)
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16} ({c.network_kwargs.dtype})')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    if dist.get_world_size() > 1:
        torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--data',             help='Path to the prepared pair dataset (prepare_data_vivit_pairs.py output)', metavar='DIR', type=str, required=True)
@click.option('--preset',           help='Configuration preset (optional; see --channels etc.)', metavar='STR', type=str, default=None)

# Hyperparameters.
@click.option('--duration',         help='Training duration. With N train pairs, 1 epoch = N samples (e.g. --duration=160Ki for ~1000 epochs on a 160-pair set)', metavar='NIMG', type=parse_nimg, default=None)
@click.option('--batch',            help='Total batch size', metavar='NIMG',                    type=parse_nimg, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=8), default=None)
@click.option('--channel-mult',     help='Per-resolution channel multipliers, e.g. 1,2,2,4', metavar='LIST', type=str, default=None)
@click.option('--num-blocks',       help='Number of residual blocks per resolution', metavar='INT', type=int, default=None)
@click.option('--attn-resolutions', help='Self-attention resolutions, e.g. 16', metavar='LIST',  type=str, default=None)
@click.option('--cross-attn-resolutions', help='Cross-attention resolutions to context tokens', metavar='LIST', type=str, default='16,32', show_default=True)
@click.option('--channels-per-head', help='Channels per attention head (self- and cross-attention)', metavar='INT', type=int, default=32, show_default=True)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--P_mean', 'P_mean', help='Noise level mean', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std', 'P_std',   help='Noise level standard deviation', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref): number of optimizer batches (nimg/batch_size) after which inverse-sqrt decay starts', metavar='BATCHES', type=click.FloatRange(min=0), default=None)
@click.option('--rampup',           help='Learning rate rampup duration in samples, parse_nimg syntax (e.g. 4Ki). Default: 10,000,000 (upstream\'s 10 Mimg default) -- with a short --duration this needs lowering or the run ends before ramping up', metavar='NIMG', type=parse_nimg, default=None)

# Pipeline options (contract C3/C4/C5).
@click.option('--context/--no-context', help='ViViT token conditioning (cross-attention + pooled FiLM). --no-context builds a token-free network (context_dim=0) and skips all token I/O -- a required comparison arm now that the Phase 0 classifier probe came back at chance for every ViT tap', default=True, show_default=True)
@click.option('--tokens-dir',       help='C2 token store root. Default: read from dataset.json', metavar='DIR', type=str, default=None)
@click.option('--target-encoding',  help='Target encoding', metavar='STR',                       type=click.Choice(['binary', 'sdf']), default='binary', show_default=True)
@click.option('--cond-image',       help='Baseline conditioning channels concatenated to the input', metavar='STR', type=click.Choice(['none', 'mask', 'mask+image']), default='mask', show_default=True)
@click.option('--max-history',      help='Cap the number of history scans (default: dataset.json max_history)', metavar='INT', type=int, default=None)
@click.option('--flip-axes/--no-flip-axes', help='Random flips in all three spatial axes (train split only)', default=True, show_default=True)
@click.option('--dtype',            help='Reduced-precision dtype used when --fp16', metavar='STR', type=click.Choice(['fp16', 'bf16']), default='fp16', show_default=True)
@click.option('--sigma-data',       help='Override sigma_data (default: dataset stats for --target-encoding)', metavar='FLOAT', type=float, default=None)
@click.option('--split',            help='Dataset split to train on', metavar='STR',             type=str, default='train', show_default=True)

# Performance-related options.
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NIMG',            type=parse_nimg, default=0, show_default=True)
@click.option('--fp16',             help='Enable reduced-precision training', metavar='BOOL',   type=bool, default=True, show_default=True)
@click.option('--ls',               help='Loss scaling (not needed with --dtype=bf16)', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints', metavar='NIMG',           type=parse_nimg, default='128Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NIMG',       type=parse_nimg, default='8Mi', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NIMG',    type=parse_nimg, default='128Mi', show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)

def cmdline(outdir, dry_run, **opts):
    """Train diffusion models according to the EDM2 recipe from the paper
    "Analyzing and Improving the Training Dynamics of Diffusion Models",
    conditioned on ViViT tokens (docs/vivit_conditioning_plan.md).

    Examples:

    \b
    # Local single-GPU run with the plan's initial small architecture.
    python train_edm2.py --outdir=training-runs/00000 \\
        --data=datasets/vivit_pairs --tokens-dir=.../vit_tokens \\
        --channels=16 --channel-mult=1,2,2,4 --num-blocks=2 \\
        --channels-per-head=32 --attn-resolutions=16 --cross-attn-resolutions=16,32 \\
        --dropout=0.1 --duration=160Ki --batch=4 --batch-gpu=4 \\
        --lr=0.01 --decay=35000 --P_mean=-0.4 --P_std=1.0

    \b
    # To resume training, run the same command again.

    \b
    # Multi-GPU (Rivanna): torchrun --standalone --nproc_per_node=8 train_edm2.py ...
    """
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass # already set (e.g. re-entrant call, or platform default)
    dist.init() # single-process on Windows works without torchrun: see torch_utils/distributed.py init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)
    print_training_config(run_dir=outdir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=outdir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------

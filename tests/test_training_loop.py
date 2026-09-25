"""Unit tests for training.training_loop (EDM2Loss and a mini end-to-end run),
against contracts C5-C7 (docs/vivit_pipeline_contracts.md)."""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.pair_fixture import build_pair_fixture
from training.networks_edm2 import Precond
from training.training_loop import EDM2Loss, training_loop
from torch_utils import distributed as dist

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

RES = (128, 128, 64)


def test_edm2loss_finite_and_backward():
    dev = torch.device('cuda')
    net = Precond(img_resolution=RES, img_channels=1, cond_channels=1, context_dim=768,
        context_tokens=256, use_time_gap=True, sigma_data=1.0, model_channels=16,
        channel_mult=[1, 2, 2, 4], num_blocks=2, channels_per_head=32,
        attn_resolutions=[16], cross_attn_resolutions=[16, 32], dropout=0.1).to(dev)
    net.train()

    B, K = 2, 2
    batch = dict(
        image=torch.randn(B, 1, *RES, device=dev),
        cond_image=torch.randn(B, 1, *RES, device=dev),
        context=torch.randn(B, K, 256, 768, device=dev),
        context_mask=torch.ones(B, K, dtype=torch.bool, device=dev),
        context_ages=torch.rand(B, K, device=dev) * 100,
        delta_days=torch.rand(B, device=dev) * 50,
    )
    loss_fn = EDM2Loss(sigma_data=1.0)
    loss = loss_fn(net, batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_training_loop_mini_run(tmp_path):
    dist.init() # single-process (see torch_utils/distributed.py); required before DistributedDataParallel.
    pair_dir, tokens_dir, samples = build_pair_fixture(tmp_path / 'data', shape=RES)
    run_dir = str(tmp_path / 'run')
    os.makedirs(run_dir, exist_ok=True)

    dataset_kwargs = dict(class_name='training.dataset.PairDataset', path=pair_dir,
        tokens_dir=tokens_dir, split='train', target_encoding='binary', cond_image='mask')
    network_kwargs = dict(class_name='training.networks_edm2.Precond', model_channels=16,
        channel_mult=[1, 2, 2, 4], num_blocks=2, channels_per_head=32,
        attn_resolutions=[16], cross_attn_resolutions=[16, 32], dropout=0.1,
        use_fp16=True, dtype='fp16')
    loss_kwargs = dict(class_name='training.training_loop.EDM2Loss', sigma_data=1.0)
    lr_kwargs = dict(func_name='training.training_loop.learning_rate_schedule', ref_lr=1e-3, ref_batches=100)
    data_loader_kwargs = dict(class_name='torch.utils.data.DataLoader', pin_memory=False, num_workers=0)

    # batch=2, batch_gpu=1: exercises gradient accumulation (2 rounds/iter).
    # 5 iterations => total_nimg=10. snapshot_nimg must be a multiple of
    # 1024 (training_loop's granularity assert), so within 10 images the
    # only snapshot written is the initial one at cur_nimg=0 -- sufficient
    # to test the save/unpickle/forward round trip end to end.
    training_loop(
        dataset_kwargs=dataset_kwargs,
        data_loader_kwargs=data_loader_kwargs,
        network_kwargs=network_kwargs,
        loss_kwargs=loss_kwargs,
        lr_kwargs=lr_kwargs,
        run_dir=run_dir,
        seed=0,
        batch_size=2,
        batch_gpu=1,
        total_nimg=10,
        status_nimg=2,
        snapshot_nimg=1024,
        checkpoint_nimg=None,
        device=torch.device('cuda'),
    )

    snapshot_files = sorted(Path(run_dir).glob('network-snapshot-*.pkl'))
    assert len(snapshot_files) >= 1, 'expected at least one network snapshot'

    with open(snapshot_files[0], 'rb') as f:
        data = pickle.load(f)
    assert 'ema' in data and 'dataset_kwargs' in data

    ds_kwargs = data['dataset_kwargs']
    for key in ('target_encoding', 'cond_image', 'max_history', 'tokens_dir', 'sigma_data', 'img_resolution'):
        assert key in ds_kwargs, f'missing contract C7 key: {key}'
    assert ds_kwargs['target_encoding'] == 'binary'
    assert ds_kwargs['cond_image'] == 'mask'
    assert tuple(ds_kwargs['img_resolution']) == RES

    ema_net = data['ema'].cuda().eval()
    x = torch.randn(1, 1, *RES, device='cuda')
    sigma = torch.rand(1, device='cuda') + 0.1
    cond_image = torch.randn(1, 1, *RES, device='cuda')
    with torch.no_grad():
        out = ema_net(x, sigma, cond_image=cond_image, context=None, delta_days=torch.zeros(1, device='cuda'), force_fp32=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    assert (Path(run_dir) / 'stats.jsonl').exists()
    lines = (Path(run_dir) / 'stats.jsonl').read_text().strip().splitlines()
    assert len(lines) >= 5 # one status report per iteration (status_nimg=batch_size=2)

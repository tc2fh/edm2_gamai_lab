"""Unit tests for training.training_loop.learning_rate_schedule. Pure
arithmetic, no CUDA/network/dataset involved, so this file is not gated by
the CUDA skipif the other test modules use.

Regression test for a real training blocker found by the trainer: the
schedule's rampup defaulted to 10 Mimg (upstream's rampup_Mimg=10) and
train_edm2.py never overrode it, so a short local run (e.g. 96Ki samples)
never finished ramping up and trained at ~1% of --lr the whole time."""

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.training_loop import learning_rate_schedule


def test_reaches_ref_lr_at_rampup_end():
    ref_lr = 0.01
    rampup_nimg = 4096
    ref_batches = 1e6 # far beyond anything reached in this test: decay must not have started yet
    batch_size = 2

    lr_before = learning_rate_schedule(rampup_nimg - batch_size, batch_size, ref_lr=ref_lr,
        ref_batches=ref_batches, rampup_nimg=rampup_nimg)
    lr_at = learning_rate_schedule(rampup_nimg, batch_size, ref_lr=ref_lr,
        ref_batches=ref_batches, rampup_nimg=rampup_nimg)
    lr_after = learning_rate_schedule(rampup_nimg + batch_size, batch_size, ref_lr=ref_lr,
        ref_batches=ref_batches, rampup_nimg=rampup_nimg)

    assert lr_before < ref_lr
    assert lr_at == ref_lr
    assert lr_after == ref_lr # rampup factor clamps to 1, stays there


def test_decays_after_ref_batches():
    ref_lr = 0.01
    batch_size = 2
    ref_batches = 100
    rampup_nimg = 1 # rampup finishes immediately, isolating the decay behaviour

    nimg_at_ref_batches = ref_batches * batch_size
    lr_at_ref_batches = learning_rate_schedule(nimg_at_ref_batches, batch_size, ref_lr=ref_lr,
        ref_batches=ref_batches, rampup_nimg=rampup_nimg)
    lr_before = learning_rate_schedule(nimg_at_ref_batches // 2, batch_size, ref_lr=ref_lr,
        ref_batches=ref_batches, rampup_nimg=rampup_nimg)
    lr_after = learning_rate_schedule(nimg_at_ref_batches * 4, batch_size, ref_lr=ref_lr,
        ref_batches=ref_batches, rampup_nimg=rampup_nimg)

    # No decay yet at or before ref_batches (max(...,1) clamps the divisor to 1).
    assert lr_before == ref_lr
    assert lr_at_ref_batches == ref_lr
    # Strictly decaying (inverse sqrt) beyond ref_batches.
    assert lr_after < lr_at_ref_batches
    expected_after = ref_lr / np.sqrt(4)
    assert lr_after == expected_after


def test_default_rampup_matches_upstreams_10_mimg():
    # train_edm2.py's --rampup is optional and, when omitted, must leave
    # old behaviour (and old presets) unchanged: upstream's default was
    # rampup_Mimg=10, i.e. 10 * 1e6 samples.
    default_rampup_nimg = inspect.signature(learning_rate_schedule).parameters['rampup_nimg'].default
    assert default_rampup_nimg == 10_000_000


def test_rampup_disabled_when_zero():
    ref_lr = 0.01
    lr = learning_rate_schedule(0, 2, ref_lr=ref_lr, ref_batches=0, rampup_nimg=0)
    assert lr == ref_lr


def _minimal_config_opts(pair_dir, tokens_dir, **overrides):
    opts = dict(
        data=str(pair_dir), tokens_dir=str(tokens_dir),
        channels=16, channel_mult='1,2', num_blocks=1, channels_per_head=8,
        attn_resolutions=None, cross_attn_resolutions='', dropout=0.0,
        duration=8, batch=2, lr=0.01, decay=100, P_mean=-0.4, P_std=1.0,
    )
    opts.update(overrides)
    return opts


def test_cli_rampup_flag_wires_into_lr_kwargs(tmp_path):
    # setup_training_config only, no actual training: --rampup, when given,
    # must land in lr_kwargs as rampup_nimg (parsed via parse_nimg); when
    # omitted, rampup_nimg must be absent so learning_rate_schedule's own
    # default (10,000,000, matching upstream) applies untouched.
    from tests.pair_fixture import build_pair_fixture
    import train_edm2

    pair_dir, tokens_dir, _samples = build_pair_fixture(tmp_path, shape=(16, 16, 8))

    c_default = train_edm2.setup_training_config(**_minimal_config_opts(pair_dir, tokens_dir))
    assert 'rampup_nimg' not in c_default.lr_kwargs

    # setup_training_config receives already-parsed values (click applies
    # `type=parse_nimg` before calling it); parse_nimg('4Ki') == 4096.
    c_rampup = train_edm2.setup_training_config(**_minimal_config_opts(pair_dir, tokens_dir, rampup=4096))
    assert c_rampup.lr_kwargs['rampup_nimg'] == 4096
    assert train_edm2.parse_nimg('4Ki') == 4096

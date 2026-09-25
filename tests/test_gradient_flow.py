"""Deterministic, CPU-only regression test for a real gradient deadlock
found in review of training.networks_edm2.

CrossAttention's `to_out` has a zero-initialised gain (so a fresh network's
output is context-independent, see
test_network.py::test_fresh_network_context_equals_no_context), and
Block.cross_attn_balance mixes its output into `x` via mp_sum. If BOTH were
zero-initialised, mp_sum(x, y, t)'s derivative wrt `t` at t=0 equals `y`
(=0, since the gain is 0) and its derivative wrt `y` equals `t` (=0) -- so
neither could ever move: a permanent deadlock. The fix keeps `out_gain`
zero-initialised but starts `cross_attn_balance` at a nonzero value (0.3,
matching Block.attn_balance's default), with Block.forward mixing in an
explicit zero placeholder for the "no context" case so the zero-init
invariant still holds exactly (see networks_edm2.py).

This lives in its own CPU-only module, separate from test_network.py's
CUDA tests, for two reasons found during review:

1. Two more zero-init gates sit *between* the loss and every internal
   parameter, both expected EDM2 behaviour and not specific to this fork:
   `unet.out_gain` (the final output conv) blocks ALL internal gradient on
   the very first step (only out_gain and the separate logvar head can
   move then), and EVERY Block's own `emb_gain` independently gates that
   block's `emb`-conditioning term. So the question this test asks is not
   "is step-1 gradient nonzero" (it provably isn't, for anything but
   out_gain/logvar, regardless of whether the deadlock is fixed) but "does
   each parameter actually move over a run of optimizer steps".

2. A single-step gradient-is-nonzero assertion on a near-zero true
   gradient (the true values here are of order 1e-8, see
   test_network.py::test_mixed_precision_forward_backward's neighbourhood
   for the fp16 angle) proved flaky under CUDA/cuDNN algorithm
   nondeterminism in a full-suite run. CPU execution in PyTorch is
   reproducible bit-for-bit given a fixed seed, so this test runs on CPU
   at a reduced spatial size (fast enough despite CPU being much slower
   per-step) and checks the much more robust question: after N optimizer
   steps, has each listed parameter moved from its initial value by more
   than a small tolerance -- rather than asserting anything about a single
   gradient's exact value.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.networks_edm2 import Precond

RES = (32, 32, 16)
# channels_per_head=8 (not the plan's 32): at this reduced resolution the
# 4-level channel schedule (model_channels * [1,2,2,4] = [16,32,32,64])
# would otherwise put a cross-attention resample block at 16 channels
# (below channels_per_head=32), which Block correctly rejects as a
# zero-heads config. That constructor check is exercised elsewhere
# (test_network.py::test_zero_heads_config_raises); this test is only
# about gradient flow, so a smaller channels_per_head sidesteps it.
KW = dict(
    img_resolution=RES, img_channels=1, cond_channels=1,
    context_dim=768, context_tokens=256, use_time_gap=True, sigma_data=1.0,
    model_channels=16, channel_mult=[1, 2, 2, 4], num_blocks=2, channels_per_head=8,
    attn_resolutions=[16], cross_attn_resolutions=[16], dropout=0.1, use_fp16=False,
)
N_WARMUP_STEPS = 10
MOVEMENT_TOL = 1e-4


def _checked_param_names(net):
    names = []
    for name, _p in net.named_parameters():
        is_cross_attn_out = 'cross_attn.to_out' in name or 'cross_attn.out_gain' in name
        is_balance = name.endswith('cross_attn_balance') or name.endswith('context_balance')
        is_context_proj = name.endswith('ctx_proj.weight') or name.endswith('emb_context.weight')
        is_time_gap = name.endswith('time_gap_linear.weight')
        if is_cross_attn_out or is_balance or is_context_proj or is_time_gap:
            names.append(name)
    return names


def test_no_permanent_gradient_deadlock():
    device = torch.device('cpu')
    generator = torch.Generator(device=device).manual_seed(0)
    torch.manual_seed(0) # module-level init (MPConv/MPFourier) also draws from the global RNG

    net = Precond(**KW).to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    checked_names = _checked_param_names(net)
    params = dict(net.named_parameters())
    assert len(checked_names) >= 4 * 1 + 3 # at least one cross-attn block's (to_out, out_gain, balance) + 3 emb-path params
    init_values = {name: params[name].detach().clone() for name in checked_names}

    def one_step():
        B, K = 2, 2
        x = torch.randn(B, 1, *RES, generator=generator, device=device)
        sigma = torch.rand(B, generator=generator, device=device) + 0.1
        cond_image = torch.randn(B, 1, *RES, generator=generator, device=device)
        context = torch.randn(B, K, 256, 768, generator=generator, device=device)
        context_mask = torch.ones(B, K, dtype=torch.bool, device=device)
        context_ages = torch.rand(B, K, generator=generator, device=device) * 100
        delta_days = torch.rand(B, generator=generator, device=device) * 50
        out, logvar = net(x, sigma, cond_image=cond_image, context=context, context_mask=context_mask,
            context_ages=context_ages, delta_days=delta_days, return_logvar=True, force_fp32=True)
        loss = ((out - x) ** 2 / logvar.exp() + logvar).mean()
        loss.backward()
        return loss

    for _ in range(N_WARMUP_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = one_step()
        assert torch.isfinite(loss)
        opt.step()

    for name in checked_names:
        moved = (params[name].detach() - init_values[name]).abs().max().item()
        assert moved > MOVEMENT_TOL, (
            f'{name} moved only {moved:.3e} over {N_WARMUP_STEPS} steps '
            f'(expected > {MOVEMENT_TOL:.0e}) -- possible gradient deadlock')

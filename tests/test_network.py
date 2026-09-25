"""Unit tests for training.networks_edm2.Precond against contract C6
(docs/vivit_pipeline_contracts.md), at the plan's initial architecture
(docs/vivit_conditioning_plan.md section 2.8): channels=16,
channel_mult=[1,2,2,4], num_blocks=2, channels_per_head=32,
attn_resolutions=[16], cross_attn_resolutions=[16,32], dropout=0.1,
at (128,128,64)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.networks_edm2 import Precond, Block

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

RES = (128, 128, 64)
BASE_KW = dict(
    img_resolution=RES, img_channels=1, cond_channels=1,
    context_dim=768, context_tokens=256, use_time_gap=True, sigma_data=1.0,
    model_channels=16, channel_mult=[1, 2, 2, 4], num_blocks=2, channels_per_head=32,
    attn_resolutions=[16], cross_attn_resolutions=[16, 32], dropout=0.1,
)


def _make_net(**overrides):
    kw = dict(BASE_KW)
    kw.update(overrides)
    torch.manual_seed(0)
    return Precond(**kw)


def _dummy_inputs(dev, B, K, cond_channels=1):
    x = torch.randn(B, 1, *RES, device=dev)
    sigma = torch.rand(B, device=dev) + 0.1
    cond_image = torch.randn(B, cond_channels, *RES, device=dev) if cond_channels > 0 else None
    context = torch.randn(B, K, 256, 768, device=dev)
    context_mask = torch.ones(B, K, dtype=torch.bool, device=dev)
    context_ages = torch.rand(B, K, device=dev) * 100
    delta_days = torch.rand(B, device=dev) * 50
    return x, sigma, cond_image, context, context_mask, context_ages, delta_days


@pytest.fixture(scope='module')
def device():
    return torch.device('cuda')


def test_param_count_sane(device):
    net = _make_net().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    # The plan's pre-Phase-2 table (section 2.8) measured ~4.6M for this
    # config, but that used a per-block 768-dim cross-attention projection.
    # Phase 2's shared MPConv(768 -> model_channels*4) token projection
    # (contract C6) is far cheaper, so the real count is lower; just check
    # it's in a sane small-model range.
    assert 2e6 < n_params < 6e6


@pytest.mark.parametrize('K', [1, 3])
def test_forward_shapes(device, K):
    net = _make_net().to(device)
    x, sigma, cond_image, context, context_mask, context_ages, delta_days = _dummy_inputs(device, B=2, K=K)
    out = net(x, sigma, cond_image=cond_image, context=context, context_mask=context_mask,
        context_ages=context_ages, delta_days=delta_days, force_fp32=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_fresh_network_context_equals_no_context(device):
    net = _make_net().to(device)
    B = 2
    x, sigma, cond_image, context, context_mask, context_ages, delta_days = _dummy_inputs(device, B=B, K=2)

    with torch.no_grad():
        out_ctx = net(x, sigma, cond_image=cond_image, context=context, context_mask=context_mask,
            context_ages=context_ages, delta_days=delta_days, force_fp32=True)
        out_none = net(x, sigma, cond_image=cond_image, context=None,
            delta_days=delta_days, force_fp32=True)
    torch.testing.assert_close(out_ctx, out_none, atol=1e-5, rtol=0)


def test_context_mask_all_false_equals_no_context(device):
    net = _make_net().to(device)
    B = 2
    x, sigma, cond_image, context, _, context_ages, delta_days = _dummy_inputs(device, B=B, K=2)
    mask_false = torch.zeros(B, 2, dtype=torch.bool, device=device)

    with torch.no_grad():
        out_masked = net(x, sigma, cond_image=cond_image, context=context, context_mask=mask_false,
            context_ages=context_ages, delta_days=delta_days, force_fp32=True)
        out_none = net(x, sigma, cond_image=cond_image, context=None,
            delta_days=delta_days, force_fp32=True)
    torch.testing.assert_close(out_masked, out_none, atol=1e-5, rtol=0)


def test_padded_slots_do_not_change_output(device):
    net = _make_net().to(device)
    B = 2
    x, sigma, cond_image, context1, mask1, ages1, delta_days = _dummy_inputs(device, B=B, K=1)

    context3 = torch.zeros(B, 3, 256, 768, device=device)
    context3[:, 0] = context1[:, 0]
    mask3 = torch.zeros(B, 3, dtype=torch.bool, device=device)
    mask3[:, 0] = True
    ages3 = torch.zeros(B, 3, device=device)
    ages3[:, 0] = ages1[:, 0]

    with torch.no_grad():
        out1 = net(x, sigma, cond_image=cond_image, context=context1, context_mask=mask1,
            context_ages=ages1, delta_days=delta_days, force_fp32=True)
        out3 = net(x, sigma, cond_image=cond_image, context=context3, context_mask=mask3,
            context_ages=ages3, delta_days=delta_days, force_fp32=True)
    torch.testing.assert_close(out1, out3, atol=1e-5, rtol=0)


def test_activation_magnitude_near_unit_variance(device):
    net = _make_net().to(device)
    stds = {}
    hooks = []
    def make_hook(name):
        def hook(_mod, _inp, out):
            stds[name] = out.float().std().item()
        return hook
    count = 0
    for name, mod in net.unet.enc.items():
        if isinstance(mod, Block):
            hooks.append(mod.register_forward_hook(make_hook(f'enc.{name}')))
            count += 1
        if count >= 3:
            break
    count = 0
    for name, mod in net.unet.dec.items():
        if isinstance(mod, Block):
            hooks.append(mod.register_forward_hook(make_hook(f'dec.{name}')))
            count += 1
        if count >= 3:
            break

    x, sigma, cond_image, context, context_mask, context_ages, delta_days = _dummy_inputs(device, B=2, K=2)
    with torch.no_grad():
        net(x, sigma, cond_image=cond_image, context=context, context_mask=context_mask,
            context_ages=context_ages, delta_days=delta_days, force_fp32=True)
    for h in hooks:
        h.remove()

    assert len(stds) >= 4
    for name, std in stds.items():
        assert 0.5 <= std <= 2.0, f'{name} activation std {std:.3f} outside [0.5, 2.0]'


def test_zero_heads_config_raises():
    # channels_per_head=128 exceeds every level's channel count (max 64 at
    # the deepest level with model_channels=16, channel_mult=[1,2,2,4]).
    with pytest.raises(ValueError):
        _make_net(channels_per_head=128, model_channels=16, cross_attn_resolutions=[16])
    with pytest.raises(ValueError):
        _make_net(channels_per_head=128, model_channels=16, attn_resolutions=[16], cross_attn_resolutions=[])


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_mixed_precision_forward_backward(device, dtype):
    net = _make_net(dtype=dtype, use_fp16=True).to(device)
    net.train()
    x, sigma, cond_image, context, context_mask, context_ages, delta_days = _dummy_inputs(device, B=2, K=2)
    out, logvar = net(x, sigma, cond_image=cond_image, context=context, context_mask=context_mask,
        context_ages=context_ages, delta_days=delta_days, return_logvar=True)
    assert out.dtype == torch.float32 # Precond always returns fp32
    loss = ((out - x) ** 2 / logvar.exp() + logvar).mean()
    loss.backward()
    assert torch.isfinite(loss)
    for p in net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_training_learns_context_dependence(device):
    """20 Adam steps on a fixed synthetic batch whose target depends on the
    context content (sign of the mean token value, broadcast over the
    volume) must reduce the loss and make the trained network's output
    depend measurably on context -- the conditioning-knockout check the
    plan requires (section 2, decision 9) starts here: a model that ignores
    its conditioning would fail this trivially."""
    torch.manual_seed(0)
    net = _make_net().to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    B, K = 2, 2
    cond_image = torch.randn(B, 1, *RES, device=device)
    context = torch.randn(B, K, 256, 768, device=device)
    context_mask = torch.ones(B, K, dtype=torch.bool, device=device)
    context_ages = torch.rand(B, K, device=device) * 100
    delta_days = torch.rand(B, device=device) * 50

    ctx_signal = context.mean(dim=(1, 2, 3)).sign().view(B, 1, 1, 1, 1)
    target = ctx_signal.expand(B, 1, *RES).clone()
    sigma = torch.full((B,), 0.5, device=device)

    def compute_loss():
        noise = torch.randn_like(target) * sigma.reshape(-1, 1, 1, 1, 1)
        out, logvar = net(target + noise, sigma, cond_image=cond_image, context=context,
            context_mask=context_mask, context_ages=context_ages, delta_days=delta_days, return_logvar=True)
        return ((out - target) ** 2 / logvar.exp() + logvar).mean()

    first_loss = compute_loss().item()
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        loss = compute_loss()
        loss.backward()
        opt.step()
    last_loss = compute_loss().item()
    assert last_loss < first_loss

    net.eval()
    with torch.no_grad():
        out_ctx = net(target, sigma, cond_image=cond_image, context=context, context_mask=context_mask,
            context_ages=context_ages, delta_days=delta_days, force_fp32=True)
        out_none = net(target, sigma, cond_image=cond_image, context=None, delta_days=delta_days, force_fp32=True)
    diff = (out_ctx - out_none).abs().max().item()
    assert diff > 1e-3

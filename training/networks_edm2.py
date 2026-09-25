# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models",
extended with ViViT token cross-attention conditioning
(see docs/vivit_conditioning_plan.md and docs/vivit_pipeline_contracts.md,
contract C6 for the exact Precond/UNet API)."""

import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc
from einops import rearrange

#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    is_3d = (x.ndim == 5)
    if is_3d:
        f = np.einsum('i,j,k->ijk', f, f, f)[np.newaxis, np.newaxis, :, :, :]
    else:
        f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = misc.const_like(x, f)
    c = x.shape[1]
    if is_3d:
        if mode == 'down':
            return torch.nn.functional.conv3d(x, f.tile([c, 1, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
        assert mode == 'up'
        return torch.nn.functional.conv_transpose3d(x, (f * 8).tile([c, 1, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    else:
        if mode == 'down':
            return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
        assert mode == 'up'
        return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).
#
# `t` may be a python float (the common case) or a 0-dim torch tensor/
# Parameter (used for the learned, zero-initialised cross-attention and
# context-FiLM balances below). `np.sqrt` on a tensor that requires_grad
# raises (it tries `.numpy()`), so the denominator is computed with `** 0.5`,
# which is autograd-safe for both python floats and tensors and numerically
# identical to `np.sqrt` for floats.

def mp_sum(a, b, t=0.5):
    denom = ((1 - t) ** 2 + t ** 2) ** 0.5
    return a.lerp(b, t) / denom

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

@persistence.persistent_class
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66). Also used as a plain
# linear projection (kernel=[]) for the token/context path below; matmul
# broadcasts over any number of leading batch dims, so it accepts inputs of
# shape (..., in_channels), not just (B, in_channels).

@persistence.persistent_class
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim in [4, 5]
        if w.ndim == 5:
            return torch.nn.functional.conv3d(x, w, padding=(w.shape[-1]//2,)*3)
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

#----------------------------------------------------------------------------
# Cross-attention from block spatial activations (query) to projected ViViT
# context tokens (key/value). Uses scaled_dot_product_attention and lets
# PyTorch pick the backend: Flash has no sm_120 kernel on Blackwell in this
# build, but the cuDNN and memory-efficient backends do (plan section 2.10).
#
# The output projection's contribution is gated to exactly zero at
# construction via `out_gain` (a learned scalar starting at 0, same trick as
# Block.emb_gain / UNet.out_gain elsewhere in this file). The residual mix
# in Block uses a separate learned `cross_attn_balance`, starting NOT at 0
# but at 0.3 (matching Block.attn_balance's default): mp_sum(x, y, t)'s
# derivatives are dt = y and dy = t, so if both `out_gain` and this balance
# started at 0 simultaneously, y would stay exactly 0 and neither could
# ever receive a gradient -- a permanent deadlock (found in review). With
# only `out_gain` zero-init, y is exactly 0 at construction regardless of
# `t`, and Block.forward mixes in an explicit zero placeholder for the
# "no context" case through that same mp_sum call, so a fresh network's
# output still exactly matches the no-context case (0 == 0 for any t),
# while the balance itself is free to move once `out_gain` (and every
# other zero-gate upstream of it) has, unlocking real gradient to `y` and
# hence to `out_gain`, `to_out`, `to_q`/`to_k`/`to_v`. See Block.forward
# and Block.__init__ for the exact mechanism, and
# tests/test_gradient_flow.py::test_no_permanent_gradient_deadlock for the
# regression test.

@persistence.persistent_class
class CrossAttention(torch.nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, dropout=0.):
        super().__init__()
        if heads <= 0:
            raise ValueError(
                f'CrossAttention needs at least one head, got heads={heads} '
                f'(query_dim={query_dim}, context_dim={context_dim}, dim_head={dim_head}). '
                'Lower --channels-per-head or drop this resolution from --cross-attn-resolutions.')
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.to_q = MPConv(query_dim, inner_dim, kernel=[])
        self.to_k = MPConv(context_dim, inner_dim, kernel=[])
        self.to_v = MPConv(context_dim, inner_dim, kernel=[])
        self.to_out = MPConv(inner_dim, query_dim, kernel=[])
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x, context, key_padding_mask=None):
        # x: (B, N, query_dim). context: (B, M, context_dim).
        # key_padding_mask: (B, M) bool, True = attend, False = ignore.
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.heads)
        attn_mask = key_padding_mask[:, None, None, :] if key_padding_mask is not None else None
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=(self.dropout if self.training else 0.0))
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out, gain=self.out_gain)
        return out

#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21) and
# optional cross-attention to ViViT context tokens. Self-attention
# (`attention`) and cross-attention (`context_dim != 0`) are independent
# switches (plan decision 7): a level can have either, both, or neither.

@persistence.persistent_class
class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        context_dim         = 0,        # Projected context dimensionality. 0 = no cross-attention.
        flavor               = 'enc',   # Flavor: 'enc' or 'dec'.
        resample_mode        = 'keep',  # Resampling: 'keep', 'up', or 'down'.
        resample_filter       = [1,1],  # Resampling filter.
        attention            = False,   # Include self-attention?
        channels_per_head    = 64,      # Number of channels per attention head.
        dropout              = 0,       # Dropout probability.
        res_balance          = 0.3,     # Balance between main branch (0) and residual branch (1).
        attn_balance         = 0.3,     # Balance between main branch (0) and self-attention (1).
        cross_attn_balance_init = 0.3,  # Initial value of the learned cross-attention balance.
        clip_act             = 256,     # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1,1]) if in_channels != out_channels else None

        # Self-attention.
        self.num_heads = out_channels // channels_per_head if attention else 0
        if attention and self.num_heads == 0:
            raise ValueError(
                f'self-attention requested for a block with out_channels={out_channels} < '
                f'channels_per_head={channels_per_head} (zero heads). Lower --channels-per-head '
                'or drop this resolution from --attn-resolutions.')
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1,1]) if self.num_heads != 0 else None

        # Cross-attention to ViViT context tokens.
        self.cross_attn = None
        self.cross_attn_balance = None
        if context_dim != 0:
            crossattn_heads = out_channels // channels_per_head
            if crossattn_heads == 0:
                raise ValueError(
                    f'cross-attention requested for a block with out_channels={out_channels} < '
                    f'channels_per_head={channels_per_head} (zero heads). Lower --channels-per-head '
                    'or drop this resolution from --cross-attn-resolutions.')
            self.cross_attn = CrossAttention(query_dim=out_channels, context_dim=context_dim,
                heads=crossattn_heads, dim_head=channels_per_head, dropout=dropout)
            # Learned, initialised to a nonzero value (matching attn_balance
            # above), NOT zero: mp_sum(x, y, t) has df/dt|_{t} = y and
            # df/dy|_{t} = t, so if both `t` (this balance) and `y` (the
            # cross-attention output, gated to exactly 0 by CrossAttention's
            # zero-init out_gain) started at zero simultaneously, neither
            # could ever receive a gradient -- a permanent deadlock. Only
            # `out_gain` is zero-initialised; this balance starts nonzero so
            # its gradient (equal to `y`, which is exactly 0 at a fresh
            # init) still lets it move once y becomes nonzero during
            # training. See Block.forward for how this stays exactly
            # magnitude-preserving and context-independent at a fresh init
            # despite `t` being nonzero: the "no context" path is mixed with
            # an explicit zero placeholder through the same mp_sum, rather
            # than bypassing the branch, so both paths produce identical
            # output at any `t` as long as `y` is 0 in both.
            self.cross_attn_balance = torch.nn.Parameter(torch.tensor(float(cross_attn_balance_init)))

    def forward(self, x, emb, context=None, context_mask=None):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).unsqueeze(4).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, int(np.prod(y.shape[2:])))
            q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Cross-attention to ViViT context tokens. Always mixed in through
        # the same mp_sum, even when there is no context: `cross_attn_balance`
        # is a nonzero-initialised learned Parameter (see __init__), so
        # mp_sum(x, y, t) does NOT reduce to x when t != 0 -- it only matches
        # the "no context" case because y is exactly 0 in both (attn_out is
        # gated to 0 by CrossAttention's zero-init out_gain at a fresh init;
        # the placeholder below is exactly 0 by construction). Bypassing the
        # branch entirely for context=None would break that equality once
        # training moves the balance away from its initial value.
        if self.cross_attn is not None:
            if context is not None:
                b, c_, d, h, w_ = x.shape
                spatial_x = rearrange(x, 'b c d h w -> b (d h w) c')
                attn_out = self.cross_attn(spatial_x, context=context.to(x.dtype), key_padding_mask=context_mask)
                attn_out = rearrange(attn_out, 'b (d h w) c -> b c d h w', d=d, h=h, w=w_)
                attn_out = normalize(attn_out, dim=1) # pixel norm, matches self-attention treatment
            else:
                attn_out = torch.zeros_like(x)
            x = mp_sum(x, attn_out, t=self.cross_attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21) with cond_image input concatenation and
# ViViT token cross-attention / FiLM conditioning (contract C6).

@persistence.persistent_class
class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution, (D, H, W).
        img_channels,                       # Image channels (target channels).
        cond_channels        = 0,           # Extra input-concatenation channels (baseline mask/image). 0 = none.
        context_dim           = 0,           # Raw per-token context dimensionality (768). 0 = no context path.
        context_tokens        = 256,         # Tokens per history scan (sanity check only).
        use_time_gap          = True,        # Build the delta_days MPFourier + MPConv branch into emb.
        model_channels        = 192,         # Base multiplier for the number of channels.
        channel_mult          = [1,2,3,4],   # Per-resolution multipliers for the number of channels.
        channel_mult_noise    = None,        # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb      = None,        # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks            = 3,           # Number of residual blocks per resolution.
        attn_resolutions      = [16,8],      # List of resolutions with self-attention.
        cross_attn_resolutions = [],         # List of resolutions with cross-attention to context tokens.
        context_proj_mult     = 4,           # context_proj_dim = model_channels * context_proj_mult.
        ctx_age_balance       = 0.5,         # Balance mixing the per-scan age embedding into projected tokens.
        time_gap_balance      = 0.5,         # Balance mixing the delta_days embedding into emb.
        concat_balance        = 0.5,         # Balance between skip connections (0) and main path (1).
        **block_kwargs,                      # Arguments for Block (channels_per_head, dropout, ...).
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.img_resolution = tuple(img_resolution)
        self.cond_channels = cond_channels
        self.context_dim = context_dim
        self.context_tokens = context_tokens
        self.use_time_gap = use_time_gap
        self.ctx_age_balance = ctx_age_balance
        self.time_gap_balance = time_gap_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Noise embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])

        # Context path: shared token projection + scan-age embedding + FiLM.
        # Replaces the learnable LayerNorm(768) with a fixed `normalize` after
        # a forced-weight-normalized MPConv projection (plan Phase 2).
        self.context_proj_dim = model_channels * context_proj_mult
        if context_dim > 0:
            self.ctx_proj = MPConv(context_dim, self.context_proj_dim, kernel=[])
            self.ctx_age_fourier = MPFourier(self.context_proj_dim)
            self.emb_context = MPConv(self.context_proj_dim, cemb, kernel=[])
            # Learned, zero-initialised: mp_sum(emb, film, t=0) == emb exactly.
            self.context_balance = torch.nn.Parameter(torch.zeros([]))
        else:
            self.ctx_proj = None
            self.ctx_age_fourier = None
            self.emb_context = None
            self.context_balance = None

        # Time-gap (delta_days) embedding.
        if use_time_gap:
            self.time_gap_fourier = MPFourier(cnoise)
            self.time_gap_linear = MPConv(cnoise, cemb, kernel=[])
        else:
            self.time_gap_fourier = None
            self.time_gap_linear = None

        res0 = self.img_resolution[0]

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + cond_channels + 1
        for level, channels in enumerate(cblock):
            res = res0 >> level
            use_ctx = self.context_proj_dim if (context_dim > 0 and res in cross_attn_resolutions) else 0
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', context_dim=use_ctx, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), context_dim=use_ctx, **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = res0 >> level
            use_ctx = self.context_proj_dim if (context_dim > 0 and res in cross_attn_resolutions) else 0
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, context_dim=use_ctx, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', context_dim=use_ctx, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', context_dim=use_ctx, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), context_dim=use_ctx, **block_kwargs)
        self.out_conv = MPConv(cout, img_channels, kernel=[3,3,3])

    def _process_context(self, context, context_mask, context_ages, dtype):
        """Project raw (B,K,T,context_dim) tokens once for the whole UNet.

        Returns:
            ctx_flat:  (B, K*T, context_proj_dim) at `dtype`, for cross-attention K/V.
            key_mask:  (B, K*T) bool, True = valid, for cross-attention masking.
            pooled:    (B, context_proj_dim) fp32, masked mean for the FiLM path.
        """
        B, K, T, D = context.shape
        assert D == self.context_dim, f'context has {D} channels, expected {self.context_dim}'
        assert T == self.context_tokens, f'context has {T} tokens per scan, expected {self.context_tokens}'
        context = context.to(torch.float32)
        if context_mask is None:
            context_mask = torch.ones(B, K, dtype=torch.bool, device=context.device)
        else:
            context_mask = context_mask.to(torch.bool)
        if context_ages is None:
            context_ages = torch.zeros(B, K, dtype=torch.float32, device=context.device)
        else:
            context_ages = context_ages.to(torch.float32)

        proj = self.ctx_proj(context)               # (B,K,T,P) fp32
        proj = normalize(proj, dim=-1)               # per-token unit norm (replaces LayerNorm)
        age_feat = torch.log1p(context_ages / 360.0).reshape(B * K)
        age_emb = self.ctx_age_fourier(age_feat).reshape(B, K, 1, -1) # (B,K,1,P) fp32
        proj = mp_sum(proj, age_emb.expand(-1, -1, T, -1), t=self.ctx_age_balance)

        proj_flat = proj.reshape(B, K * T, -1)                                   # (B,K*T,P) fp32
        mask_flat = context_mask.unsqueeze(-1).expand(-1, -1, T).reshape(B, K * T) # (B,K*T) bool

        mask_f = mask_flat.to(torch.float32).unsqueeze(-1)  # (B,K*T,1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)            # (B,1)
        pooled = (proj_flat * mask_f).sum(dim=1) / denom    # (B,P) fp32

        return proj_flat.to(dtype), mask_flat, pooled

    def forward(self, x, noise_labels, context=None, context_mask=None, context_ages=None, delta_days=None):
        # Noise embedding. Stays fp32 throughout, like upstream; only `x` and
        # the cross-attention context get cast to the compute dtype.
        emb = self.emb_noise(self.emb_fourier(noise_labels))

        ctx_flat, ctx_key_mask = None, None
        if context is not None:
            assert self.ctx_proj is not None, 'UNet was built with context_dim=0; cannot accept context'
            ctx_flat, ctx_key_mask, pooled = self._process_context(context, context_mask, context_ages, dtype=x.dtype)
            film = self.emb_context(pooled) # fp32 in, fp32 out: no dtype mismatch in mp_sum below
            emb = mp_sum(emb, film, t=self.context_balance)

        if self.use_time_gap and delta_days is not None:
            tg_feat = torch.log1p(delta_days.to(torch.float32) / 360.0)
            tg_emb = self.time_gap_linear(self.time_gap_fourier(tg_feat))
            emb = mp_sum(emb, tg_emb, t=self.time_gap_balance)

        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb, context=ctx_flat, context_mask=ctx_key_mask)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb, context=ctx_flat, context_mask=ctx_key_mask)

        x = self.out_conv(x, gain=self.out_gain)
        return x

#----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

@persistence.persistent_class
class Precond(torch.nn.Module):
    def __init__(self,
        img_resolution,          # Image resolution, (D, H, W).
        img_channels,            # Image channels (target channels).
        cond_channels    = 0,    # Extra input-concatenation channels (baseline mask/image). 0 = none.
        context_dim      = 768,  # Raw per-token context dimensionality. 0 = no context path.
        context_tokens   = 256,  # Tokens per history scan.
        use_time_gap     = True, # Condition on delta_days.
        use_fp16         = True, # Run the model at reduced precision?
        dtype            = 'fp16', # 'fp16' or 'bf16'; which reduced precision to use when use_fp16.
        sigma_data       = 1.0,  # Expected standard deviation of the training data.
        logvar_channels  = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,           # Keyword arguments for UNet.
    ):
        super().__init__()
        if dtype not in ('fp16', 'bf16'):
            raise ValueError(f"dtype must be 'fp16' or 'bf16', got {dtype!r}")
        self.img_resolution = tuple(img_resolution)
        self.img_channels = img_channels
        self.cond_channels = cond_channels
        self.context_dim = context_dim
        self.use_fp16 = use_fp16
        self.dtype = dtype
        self.sigma_data = sigma_data
        self.unet = UNet(img_resolution=self.img_resolution, img_channels=img_channels,
            cond_channels=cond_channels, context_dim=context_dim, context_tokens=context_tokens,
            use_time_gap=use_time_gap, **unet_kwargs)
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, sigma, cond_image=None, context=None, context_mask=None,
                context_ages=None, delta_days=None, force_fp32=False, return_logvar=False):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1, 1)

        compute_dtype = torch.float16 if self.dtype == 'fp16' else torch.bfloat16
        run_dtype = compute_dtype if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        if self.cond_channels > 0:
            if cond_image is None:
                raise ValueError(f'cond_channels={self.cond_channels} but cond_image is None')
            cond_image = cond_image.to(torch.float32)
            if cond_image.shape[1] != self.cond_channels:
                raise ValueError(f'cond_image has {cond_image.shape[1]} channels, expected {self.cond_channels}')

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model. cond_image is concatenated to c_in * x, unnoised and unscaled.
        x_scaled = c_in * x
        if self.cond_channels > 0:
            x_in = torch.cat([x_scaled, cond_image], dim=1).to(run_dtype)
        else:
            x_in = x_scaled.to(run_dtype)
        F_x = self.unet(x_in, c_noise, context=context, context_mask=context_mask,
                         context_ages=context_ages, delta_days=delta_days)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1, 1)
            return D_x, logvar # u(sigma) in Equation 21
        return D_x

#----------------------------------------------------------------------------

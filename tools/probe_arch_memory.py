"""Memory/speed probe of the fork's 3D Precond at the plan's initial config
(docs/vivit_conditioning_plan.md section 2.8), against the Phase 2 API
(docs/vivit_pipeline_contracts.md, contract C6).

Runs a few forward/backward/optimizer steps on random data at the ViT crop
size (1 x 128 x 128 x 64), with a baseline-mask cond_image channel and a
2-scan context, for a range of batch_gpu sizes and both --dtype settings.
Used to decide --batch-gpu for gradient accumulation on the RTX 5090 (32 GB).

Run from the repo root with a Python that has torch (CUDA) and einops, e.g.
the ViViT repo's pixi interpreter:
  GrowthNet/projects/vivit/tien_rivanna_repo/.pixi/envs/default/python.exe tools/probe_arch_memory.py

Both workarounds noted in a previous version of this file are gone: the
pooled-token FiLM branch and the noise embedding now share a single fp32
emb path (no dtype mismatch in mp_sum), and Block/CrossAttention raise a
clear ValueError at construction for a zero-heads configuration instead of
silently building a zero-size attention weight.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.networks_edm2 import Precond  # noqa: E402

SHAPE = (128, 128, 64)  # D, H, W of the ViT crop
COND_CHANNELS = 1       # baseline mask concatenated to the input (contract C4 'mask' mode)
CONTEXT_K = 2           # history scans in the context for this probe

CONFIG = dict(
    model_channels=16, channel_mult=[1, 2, 2, 4], num_blocks=2,
    channels_per_head=32, attn_resolutions=[16], cross_attn_resolutions=[16, 32], dropout=0.1,
)
BATCH_GPU_SIZES = (4, 8)
DTYPES = ('fp16', 'bf16')


def main() -> None:
    dev = torch.device('cuda')
    print(f"{'dtype':6s} {'batch_gpu':>9s} {'params':>10s} {'peak GB':>8s} {'ms/step':>8s}")
    for dtype in DTYPES:
        net = Precond(img_resolution=SHAPE, img_channels=1, cond_channels=COND_CHANNELS,
            context_dim=768, context_tokens=256, use_time_gap=True, sigma_data=1.0,
            use_fp16=True, dtype=dtype, **CONFIG).to(dev)
        n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)

        for batch in BATCH_GPU_SIZES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(batch, 1, *SHAPE, device=dev)
            sigma = torch.rand(batch, device=dev) + 0.1
            cond_image = torch.randn(batch, COND_CHANNELS, *SHAPE, device=dev)
            context = torch.randn(batch, CONTEXT_K, 256, 768, device=dev)
            context_mask = torch.ones(batch, CONTEXT_K, dtype=torch.bool, device=dev)
            context_ages = torch.rand(batch, CONTEXT_K, device=dev) * 100
            delta_days = torch.rand(batch, device=dev) * 50

            def step():
                opt.zero_grad(set_to_none=True)
                noisy = x + torch.randn_like(x) * sigma.reshape(-1, 1, 1, 1, 1)
                d_x, logvar = net(noisy, sigma, cond_image=cond_image, context=context,
                    context_mask=context_mask, context_ages=context_ages, delta_days=delta_days,
                    return_logvar=True)
                loss = ((d_x - x) ** 2 / logvar.exp() + logvar).mean()
                loss.backward()
                opt.step()

            step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 3 * 1000
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"{dtype:6s} {batch:9d} {n_params / 1e6:9.2f}M {peak:8.2f} {ms:8.0f}")
            del x, sigma, cond_image, context, context_mask, context_ages, delta_days

        del net, opt


if __name__ == "__main__":
    main()

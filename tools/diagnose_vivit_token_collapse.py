"""Read-only diagnostic: where does the v0005 encoder representation collapse?

Hooks the ViT output, the temporal encoder, and every temporal block of the frozen
encoder on three train patients plus a noise input, and prints the relative
input-dependent signal at each tap. Result (2026-09-11): the ViT output is healthy,
the eight temporal blocks attenuate it to ~1e-4 of the norm. See
docs/vivit_conditioning_plan.md section 1.2. Run with the ViViT repo pixi
interpreter: GrowthNet/projects/vivit/tien_rivanna_repo/.pixi/envs/default/python.exe."""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path(r"D:\Work\GrowthNet_gamailab\GrowthNet\projects\vivit\tien_rivanna_repo")
sys.path.insert(0, str(REPO))
from monai.data import Dataset
from monai.utils import set_determinism
from growth_classifier_v0005 import config
from growth_classifier_v0005.extract_embeddings import load_encoder
from src.data.temporal_loader import load_temporal_splits_from_json
from src.data.transforms import build_eval_transform
from src.data.utils import TransformSequence

set_determinism(config.SEED)
dev = torch.device("cuda")
net = load_encoder(dev)
raw = load_temporal_splits_from_json(config.FLAT_ROOT)["train"]
tf = TransformSequence(keys=["images","labels","dates"], spatial_transforms=build_eval_transform(target_spacing=config.TARGET_SPACING, roi_size=config.ROI_SIZE))
ds = Dataset(data=raw, transform=tf)

taps = {}
def hook(name):
    def f(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        taps[name] = t.detach().float().cpu()
    return f
net.spatial_encoder.register_forward_hook(hook("vit_out"))
net.temporal_encoder.register_forward_hook(hook("temporal_encoder"))
for i, blk in enumerate(net.temporal_blocks):
    blk.register_forward_hook(hook(f"tblock{i}"))
# also tap the ViT's internal blocks (MONAI ViT: net.spatial_encoder.blocks)
for i, blk in enumerate(net.spatial_encoder.blocks):
    if i in (2, 5, 8, 11):
        blk.register_forward_hook(hook(f"vit_block{i}"))

def run(images, dates, T_use):
    """images (T,C,H,W,D) on cpu, dates (T,). Use first T_use scans, dates T_use+1."""
    taps.clear()
    x = images[:T_use].unsqueeze(0).to(dev)
    d = dates[:T_use+1].unsqueeze(0).to(dev)
    sl = torch.tensor([T_use], device=dev)
    with torch.no_grad():
        out, skips = net(x, sl, d)
    res = {k: v.clone() for k, v in taps.items()}
    res["final"] = out.detach().float().cpu()
    res["skips"] = [s.detach().float().cpu() for s in skips]
    return res

def tokens(t):
    """Return (N_tokens, E) for the LAST timestep in the tap, whatever the layout."""
    t = t.reshape(-1, 256, 768) if t.numel() % (256*768) == 0 else t
    return t[-1] if t.ndim == 3 else t

idx = [0, 1, 2]
items = [ds[i] for i in idx]
for i, it in zip(idx, items):
    print("patient", raw[i]["patient_id"], "T=", it["images"].shape, "dates", it["dates"].tolist())
runs = [run(it["images"], it["dates"], 1) for it in items]           # baseline-only step
runs_noise = run(torch.randn_like(items[0]["images"]), items[0]["dates"], 1)
runs_step2 = run(items[0]["images"], items[0]["dates"], 2) if items[0]["images"].shape[0] >= 2 else None

names = ["vit_block2","vit_block5","vit_block8","vit_block11","vit_out","temporal_encoder"] + [f"tblock{i}" for i in range(8)] + ["final"]
print(f"\n{'tap':18s} {'norm':>8s} {'tokstd/featstd':>15s} {'pat-vs-pat rel':>15s} {'noise-vs-pat rel':>17s} {'step2-vs-step1 rel':>19s}")
for n in names:
    A = [tokens(r[n]) for r in runs]
    m = A[0].mean(0)
    norm = m.norm().item()
    tokstd = A[0].std(0).mean().item(); featstd = A[0].std(1).mean().item()
    pp = (A[0].mean(0) - A[1].mean(0)).norm().item() / norm
    nz = (tokens(runs_noise[n]).mean(0) - m).norm().item() / norm
    s2 = (tokens(runs_step2[n]).mean(0) - m).norm().item() / norm if runs_step2 else float('nan')
    print(f"{n:18s} {norm:8.2f} {tokstd/featstd:15.2e} {pp:15.2e} {nz:17.2e} {s2:19.2e}")

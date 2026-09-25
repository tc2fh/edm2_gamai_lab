## EDM2 fork: ViViT-conditioned 3D diffusion forecasting of vestibular-schwannoma masks

Fork of NVIDIA's EDM2 and Autoguidance reference implementation (citation and
license at the bottom of this file). Upstream trained 2D image diffusion
models on ImageNet; this fork trains a 3D diffusion model that forecasts a
patient's *future* tumor mask from their *current* mask, the time gap to the
target scan, and spatial ViT tokens from a frozen ViViT encoder trained
elsewhere in the lab. There is no 2D path left, no ImageNet dataset tooling,
and no image (pixel) synthesis: the target is always a binary or signed-
distance mask volume.

The plan and the exact data/API contracts this code implements live in
`docs/vivit_conditioning_plan.md` and `docs/vivit_pipeline_contracts.md`.
Read those before changing conditioning, encodings, or the batch dict shape;
this README only describes how to run what is already implemented.

## Environment

- This repo (dataset prep, training, generation, evaluation, checkpoint
  selection) is run with the ViViT repo's pixi interpreter, which has the
  matching torch/CUDA build and the packages the token store and encoder
  code need:
  `D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/.pixi/envs/default/python.exe`
  (Python 3.12, torch 2.11.0+cu130, monai, einops, nibabel, scipy, sklearn).
  Do not use a bare `python`/`python3` on Windows; it may resolve to the
  Microsoft Store stub.
- Training runs single-GPU on a local RTX 5090, no Slurm, no `torchrun`
  (see "What changed from upstream" below).
- Phase 5 (comparison against FlowMatchingGrowthNet) scores forecasts with
  the flow repo's own metrics code and deck builder, so those specific steps
  run under the flow repo's `uv` environment instead:
  `uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet python ...`

## Pipeline overview

The pipeline has six steps, each documented below in order: token store,
pair dataset, training, checkpoint selection, generation (with knockouts),
evaluation, and (Phase 5 only) a comparison deck against the flow repo. Set
`PY` once for convenience:

```bash
PY=D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/.pixi/envs/default/python.exe
```

### 1. Token store (ViViT repo, not this repo)

A frozen ViViT encoder produces per-scan ViT spatial tokens, tapped before
its temporal blocks (the temporal blocks collapse the representation to
near-constant, see `docs/vivit_conditioning_plan.md` section 1.2). The
store is built by a script in the ViViT repo and lives outside this repo;
this repo only reads it. Layout and contents (256 x 768 float16 tokens per
scan, the scan's own mask/image in the crop frame, affine, spacing, and
split membership) are pinned in `docs/vivit_pipeline_contracts.md` contract
C2. Every consumer in this repo reads `spacing` from the store rather than
assuming a fixed value (see contract C0: the store's true voxel spacing is
0.5 x 0.5 x 1.0 mm, not the originally intended 1 x 1 x 2 mm).

### 2. Pair dataset

```bash
$PY prepare_data_vivit_pairs.py \
    --tokens-dir D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/growth_classifier_v0005/out/vit_tokens_trainframe \
    --out datasets/vivit_pairs \
    --pairing consecutive --max-history 4
```

Builds `{split}/{idx:08d}.npz` samples (history scans paired with a later
target scan, contract C3) plus `dataset.json` with the manifest and target
statistics used as `sigma_data`. `--pairing all` uses every `i < j` history/
target combination per patient instead of only consecutive scans, to enlarge
the training set. Pass `--manifest <flow pair_manifest.csv>` instead of
`--pairing`/`--splits`/`--max-history` to build only a test split aligned to
the flow repo's own baseline/target pairs (Phase 5, history is always the
single baseline scan); see `analysis/README.md` for that invocation.

### 3. Training

```bash
$PY train_edm2.py --outdir=training-runs/00000 \
    --data=datasets/vivit_pairs \
    --tokens-dir=D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/growth_classifier_v0005/out/vit_tokens_trainframe \
    --channels=16 --channel-mult=1,2,2,4 --num-blocks=2 \
    --channels-per-head=32 --attn-resolutions=16 --cross-attn-resolutions=16,32 \
    --dropout=0.1 --duration=160Ki --batch=4 --batch-gpu=4 \
    --lr=0.01 --decay=35000 --P_mean=-0.4 --P_std=1.0
```

To resume, run the exact same command again; the script finds the
highest-numbered checkpoint in `--outdir` automatically. Run `python
train_edm2.py --help` for the full flag list; it is reproduced below where
it documents pipeline- and architecture-specific behavior not obvious from
upstream EDM2.

**Architecture flags and the plan's initial configuration.** Upstream EDM2
only exposes architecture through named `--preset` bundles; this fork adds
the underlying flags directly since the training set here (on the order of
160 pairs) needs a much smaller network than any upstream preset, tuned by
hand instead of picked from a table:

| flag | initial value | meaning |
|---|---|---|
| `--channels` | `16` | base channel count (upstream minimum was 16; this fork allows down to 8) |
| `--channel-mult` | `1,2,2,4` | per-resolution channel multiplier, comma list |
| `--num-blocks` | `2` | residual blocks per resolution |
| `--attn-resolutions` | `16` | self-attention resolutions, comma list |
| `--cross-attn-resolutions` | `16,32` | resolutions that get a cross-attention step against the ViViT tokens (decoupled from self-attention; see `docs/vivit_conditioning_plan.md` decision 7) |
| `--channels-per-head` | `32` | channels per attention head, self- and cross-attention; the upstream default of 64 gives this small a model zero attention heads |
| `--dropout` | `0.1` | |

This is about 4.2M parameters before the shared token projection, 3.55M
after it (see `docs/vivit_conditioning_plan.md` section 2.8 and its revision
log for the measured cost of each cross-attention configuration, and for the
scale-up/scale-down path if this configuration under- or overfits).

**Gradient accumulation: `--batch` vs `--batch-gpu`.** `--batch` is the total
(logical) batch size and determines the effective step; changing it changes
training dynamics (loss scale, learning-rate schedule interaction) exactly
as in upstream EDM2. `--batch-gpu` limits the physical micro-batch size per
forward/backward pass; the script accumulates gradients over `--batch /
--batch-gpu` micro-batches before each optimizer step. Changing `--batch-gpu`
alone (to fit GPU memory) does not change training dynamics and is always
safe; changing `--batch` does. On the RTX 5090 (32 GB) `--batch=4
--batch-gpu=4` (no accumulation) fits the plan's initial architecture at
about 11 GB peak.

**`--context` / `--no-context`.** Token conditioning (cross-attention plus
the pooled FiLM term) is on by default. `--no-context` builds a token-free
network (`context_dim=0`) and skips all token I/O entirely, rather than just
zeroing the tokens at run time. This is not an ablation convenience: the
Phase 0 classifier probe on the re-extracted ViT tokens came back at chance
(see `docs/vivit_pipeline_contracts.md` C0 and the plan's revision log), so a
no-context trained arm is a required comparison, not optional, until token
conditioning is shown to help.

### 4. Checkpoint selection

```bash
$PY select_checkpoint.py \
    --run-dir training-runs/00000 \
    --data datasets/vivit_pairs --split val \
    --num-samples 8 --steps 18
```

Evaluates every (or every k-th, via `--every-k`) snapshot in a run directory
on the validation split with a cheap sample/step budget and reports the best
by consensus Dice, writing `val_selection.csv`. Checkpoint selection is
always on val Dice, never on the diffusion training loss.

### 5. Generation (with required knockouts)

```bash
$PY generate_forecasts.py \
    --net training-runs/00000/network-snapshot-XXXXXXX-0.XXX.pkl \
    --data datasets/vivit_pairs --split test \
    --out generations/test_none \
    --num-samples 32 --steps 32 --knockout none
```

`--knockout` controls what conditioning the network actually receives at
sampling time, independent of how it was trained:

| value | effect |
|---|---|
| `none` | full conditioning (default) |
| `no-context` | drop the ViViT tokens (cross-attention and pooled FiLM skipped) |
| `no-cond-image` | drop the concatenated baseline mask/image channels |
| `shuffle-tokens` | swap in another patient's tokens, same time gap |
| `fixed-time` | freeze `delta_days` to a fixed value (train median by default, `--fixed-delta-days` to override) |
| `no-context-no-cond` | drop both tokens and the conditioning image |

Every one of these knockouts must be run and compared against `none` before
any claim that the model uses its conditioning: the FlowMatchingGrowthNet
project (the lab's related flow-matching effort) found a model that could
ignore its conditioning entirely and still score well, purely from the
target distribution's own structure. A favorable Dice with no knockout
comparison is not evidence of conditioning use.

### 6. Evaluation

```bash
$PY evaluate_forecasts.py --gen-dir generations/test_none
```

Writes `pair_metrics.csv` and `summary.json` next to (or under `--out`) the
generation directory: Dice, surface Dice, volume error, growth/shrink
direction accuracy, and probabilistic metrics over the sampled forecasts,
compared against carry-forward and (where available) the flow repo's
deterministic forecaster.

### 7. Flow-repo comparison deck (Phase 5, optional)

Once a trained checkpoint clears the knockout checks, `tools/export_flow_analysis.py`
(run under the flow repo's `uv` environment; it imports `tumor_flow.*` to
score both models with identical code), `tools/make_flow_recipe.py`, and
`tools/build_flow_deck.sh` build a side-by-side PowerPoint comparing this
fork's best EDM2 model against the best FlowMatchingGrowthNet model on
shared test scans. Full commands, the alignment-check design, and known
data-provenance caveats (the two cohorts' annotations do not always agree)
are in `analysis/README.md`; that file is the place results eventually get
reported, not this one.

## Testing

```bash
$PY -m pytest tests/
```

Covers dataset shapes and split isolation, network forward passes at
128x128x64 with variable history length, a CPU-only regression test for a
zero-init gradient deadlock in cross-attention, the training loop and the
real `train_edm2.py` CLI path, and `prepare_data_vivit_pairs.py`. Tests that
build a real network are `@pytest.mark.skipif(not torch.cuda.is_available())`.

`tests/test_export_flow_analysis.py` is the one exception: it imports
`tumor_flow` (the flow repo's package) and must be run with the flow repo's
own interpreter instead, or it skips itself:

```bash
uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet python -m pytest \
    tests/test_export_flow_analysis.py -v
```

## What changed from upstream

- **3D-only UNet.** `training/networks_edm2.py` uses 3D convolutions, 3D
  resampling, and 5D (`B, C, X, Y, Z`) preconditioning throughout. The 2D
  path is gone; the network can no longer run on 2D images.
- **`PairDataset` replaces `ImageFolderDataset`.** `training/dataset.py`
  loads `(history scans, target scan)` pairs produced by
  `prepare_data_vivit_pairs.py`, not a flat labeled image folder, and reads
  ViViT tokens from the C2 token store at run time rather than from a
  dataset zip.
- **Conditioning paths added throughout the network and loss**: a
  concatenated baseline-mask/image input channel, cross-attention against
  variable-length ViViT token histories with a key-padding mask, a pooled-
  token FiLM term, and an explicit time-gap (and per-history-scan age)
  embedding. See `docs/vivit_pipeline_contracts.md` contracts C5-C6 for the
  exact batch dict and network API.
- **Deleted scripts** (ImageNet/2D-specific, no longer applicable): `dataset_tool.py`,
  `calculate_metrics.py`, `generate_images.py`, `count_flops.py`,
  `prepare_data_3d.py`, `prepare_data_3d_longitudinal.py`,
  `prepare_embeddings.py`, `prepare_test_generation.py`,
  `postprocess_test_generations.py`, `run_train_3d.sh`, `view_napari.py`.
  Their replacements are `prepare_data_vivit_pairs.py`,
  `generate_forecasts.py`, `evaluate_forecasts.py`, and `select_checkpoint.py`.
- **Single-process launch on Windows.** `torch_utils/distributed.py`'s
  `init()` works without `torchrun` when there is exactly one process, so
  `train_edm2.py` runs directly (`python train_edm2.py ...`) for the local,
  single-GPU RTX 5090 setup this fork targets; multi-GPU (Rivanna) still
  uses `torchrun --standalone --nproc_per_node=N`.

For the full design rationale (why these choices, what was tried and
rejected, known defects fixed along the way, and the phased implementation
plan) see `docs/vivit_conditioning_plan.md`; for the frozen data/API
contracts every component implements see `docs/vivit_pipeline_contracts.md`.

## Citation

```
@inproceedings{Karras2024edm2,
  title     = {Analyzing and Improving the Training Dynamics of Diffusion Models},
  author    = {Tero Karras and Miika Aittala and Jaakko Lehtinen and
               Janne Hellsten and Timo Aila and Samuli Laine},
  booktitle = {Proc. CVPR},
  year      = {2024},
}

@inproceedings{Karras2024autoguidance,
  title     = {Guiding a Diffusion Model with a Bad Version of Itself},
  author    = {Tero Karras and Miika Aittala and Tuomas Kynk\"a\"anniemi and
               Jaakko Lehtinen and Timo Aila and Samuli Laine},
  booktitle = {Proc. NeurIPS},
  year      = {2024},
}
```

## License

Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

All material, including source code and pre-trained models, is licensed
under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
This fork inherits that license; it is a research reference implementation
for internal lab use, not a redistributed product.

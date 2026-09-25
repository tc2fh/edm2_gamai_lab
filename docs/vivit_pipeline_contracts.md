# Data and API contracts for the ViViT-conditioned EDM2 pipeline

Companion to `vivit_conditioning_plan.md`. These contracts are shared by three
independently implemented components (token extractor in the ViViT repo, pair
builder, network/training in this repo). Do not deviate without updating this
file.

Interpreter for everything in this repo and the ViViT repo:
`D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/.pixi/envs/default/python.exe`
(Python 3.12, torch 2.11.0+cu130, monai 1.5.2, einops, nibabel, scipy, sklearn, xgboost).
Phase 5 (flow-repo comparison) uses `D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/.venv/Scripts/python.exe`.

## C0. Frame revision (2026-09-12): the ViT training frame

Discovered during Phase 0: the ViViT repo's loader (`src/data/temporal_loader.py`,
`np.asarray` on the loaded MetaTensor) drops the NIfTI affine before
`Orientationd`/`Spacingd`, so the frozen encoder was TRAINED on native-array
volumes (no reorientation) with an identity affine: `Spacingd(pixdim=(1,1,2))`
therefore left x/y at native 0.5 mm and halved z to 1.0 mm. The physically
correct 1 x 1 x 2 mm frame used by the first token store (`out/vit_tokens/`) is
a 2x scale shift relative to what the ViT saw in training.

Binding change: the token store is re-extracted in the ViT training frame
(`out/vit_tokens_trainframe/`): native array axes as stored on disk (no
reorientation; the extractor records each file's axcodes and asserts they are
identical across the cohort), x/y at native voxel spacing, z at 2x native, crop
128 x 128 x 64 voxels from the first scan's mask exactly as `build_eval_transform`
plus the training loader would produce it. The true voxel spacing (0.5, 0.5, 1.0)
mm and the true affine (native affine composed with the z-subsampling and the
crop offset) are stored in every npz, so resampling to other grids stays exact.

Every consumer reads `spacing` from the data (token npz `spacing`, and
`dataset.json["spacing"]` copied from the store) and must not hard-code
(1, 1, 2). This applies to: SDF encoding (`encode_target(..., spacing)`),
surface Dice tolerance, volumes in mm^3, NIfTI writing, the flow-frame
resampling, and any physical-size assertion. The `spacing` field of C2 and
C3 is now (0.5, 0.5, 1.0) for this store. The extractor must assert the whole
tumour mask of every scan lies inside the 128 x 128 x 64 crop (64 x 64 x 64 mm)
and report any that do not.

## C1. Spatial frame

- Per patient there is exactly one crop frame: the ViViT eval crop computed from
  the patient's FIRST scan (in `uva_vs_flat_05/train_val_test_split.json` order)
  after RAS reorientation and resampling to 1 x 1 x 2 mm, tumor-centred
  128 x 128 x 64, snapped as the ViViT eval transform does. All scans of the
  patient (images, masks, tokens) are expressed in that same frame.
- Array axis order for every stored volume is the crop frame's (X, Y, Z) =
  (128, 128, 64) with voxel spacing (1, 1, 2) mm, exactly the tensor layout the
  ViViT eval transform produces (no transposes added). The network consumes it
  as (C, 128, 128, 64).
- `affine` (4 x 4, float64) maps crop voxel index (i, j, k, 1) to RAS world mm.
  With it and the original scan's NIfTI affine, any volume can be resampled back
  to the original grid with `scipy.ndimage.map_coordinates` /
  `nibabel.processing.resample_from_to` (nearest for masks, linear for
  probabilities). Never use `scipy.ndimage.zoom`.

## C2. Token store (Phase 0 output, ViViT repo)

Directory: `D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/growth_classifier_v0005/out/vit_tokens/`
Layout: `{split}/{scan_id}.npz` for split in train, val, test, plus `index.json`.

`{scan_id}.npz` keys:

| key | dtype / shape | meaning |
|---|---|---|
| `tokens` | float16 (256, 768) | ViT final output (`net.spatial_encoder` output, before the temporal blocks), one row per 16^3 patch |
| `tokens_l3`, `tokens_l6`, `tokens_l9` | float16 (256, 768) | ViT skip-layer outputs (blocks 3, 6, 9, 1-indexed as MONAI ViT `hidden_states`) |
| `mask` | uint8 (128, 128, 64) | this scan's own tumor mask in the patient crop frame |
| `image` | float16 (128, 128, 64) | this scan's z-scored T2 in the patient crop frame (the exact tensor the ViT saw) |
| `affine` | float64 (4, 4) | crop frame voxel -> RAS mm |
| `spacing` | float64 (3,) | true voxel spacing in mm, (0.5, 0.5, 1.0) for the training-frame store (see C0) |
| `crop_origin` | int64 (3,) | crop start voxel index in the resampled RAS volume |
| `resampled_shape` | int64 (3,) | shape of the resampled RAS volume the crop was taken from |
| `resampled_affine` | float64 (4, 4) | affine of that resampled RAS volume |
| `source_image_path`, `source_mask_path` | str | original NIfTI paths |
| `scan_id`, `patient_id`, `split` | str | |
| `days_since_first` | float64 | from the split JSON |
| `scan_index` | int64 | position of the scan in the patient's list |

`index.json`: `{"splits": {"train": [{"patient_id", "scans": [{"scan_id", "days_since_first", "file"}]}], ...}, "encoder_checkpoint": ..., "tap": "spatial_encoder", "created": ISO date}`.

Acceptance for the extractor: (a) resampling `mask` back to the original NIfTI
grid via `affine` gives Dice >= 0.95 with the original mask; (b) `tokens`
token-to-token std / feature std >= 0.1 and the patient-vs-patient relative L2
distance >= 0.2 (the plan's section 1.2 numbers for the ViT output are 0.25 /
0.32); (c) every scan in the split JSON has a file.

## C3. Pair dataset (Phase 1 output, this repo)

`prepare_data_vivit_pairs.py --tokens-dir <C2 dir> --out <dir> [--all-pairs] [--max-history K]`

Layout: `{out}/{split}/{idx:08d}.npz` for split in train, val, test and
`{out}/dataset.json`. Samples are never shared across splits and idx restarts
at 0 per split.

`{idx:08d}.npz` keys:

| key | dtype / shape | meaning |
|---|---|---|
| `target_mask` | uint8 (128, 128, 64) | mask of target scan j |
| `cond_mask` | uint8 (128, 128, 64) | mask of the most recent history scan i |
| `cond_image` | float16 (128, 128, 64) | z-scored image of scan i |
| `delta_days` | float64 scalar | day(j) - day(i) |
| `history_scan_ids` | str array (K_i,) | scan ids 0..i, oldest first |
| `history_ages_days` | float64 (K_i,) | day(j) - day(h) for each history scan, oldest first (last entry equals delta_days) |
| `patient_id`, `cond_scan_id`, `target_scan_id`, `split` | str | |

Tokens are NOT copied; the dataset loads `{tokens_dir}/{split}/{scan_id}.npz`
for each history scan at run time (with an in-process cache).

`dataset.json`: `{"tokens_dir", "pairing": "consecutive" | "all", "max_history",
"spacing": [<from token store>], "shape": [128,128,64], "splits": {"train": {"n": ..., "patients": [...], "samples": [{"idx", "patient_id", "cond_scan_id", "target_scan_id", "delta_days", "n_history"}]}, ...},
"stats": {"train": {"target_fraction_mean", "target_rms_binary", "target_std_binary", "target_rms_sdf", "target_std_sdf", "delta_days_max", "delta_days_median", "history_len_max"}}}`.

`target_rms_*` is sqrt(mean(x^2)) of the encoded target over all train voxels
and is what training uses as `sigma_data` by default (binary +-1 gives exactly
1.0; that is intended: the preconditioner cares about signal magnitude).

## C4. Target and conditioning encodings

- `binary`: mask {0,1} -> {-1,+1} (inside +1). Decode: x > 0.
- `sdf`: signed Euclidean distance in mm with the dataset `spacing` (see C0), negative
  inside, clipped to [-8, 8], divided by 8 -> [-1, 1]. Decode: x <= 0.
- `cond_image` option: `none` (0 channels), `mask` (1 channel, +-1 encoded
  cond_mask), `mask+image` (2 channels: +-1 cond_mask, then cond_image as
  float, clipped to [-5, 5]).
- Time gap scalar fed to the net: `log1p(delta_days / 360)`. Scan age for
  history scan h: `log1p(age_days / 360)`.

## C5. Batch dict (dataset -> training loop -> loss -> network)

`PairDataset.__getitem__` returns a dict; the collated batch has:

| key | dtype / shape |
|---|---|
| `image` | float32 (B, 1, 128, 128, 64) encoded target |
| `cond_image` | float32 (B, C_cond, 128, 128, 64), C_cond in {0,1,2}; present even when C_cond == 0 |
| `context` | float32 (B, K, 256, 768), K = max_history, zero-padded at the END |
| `context_mask` | bool (B, K), True = valid history slot |
| `context_ages` | float32 (B, K), raw days before target (0 for padding) |
| `delta_days` | float32 (B,) raw days |
| `idx` | int64 (B,) |

Augmentation (train only, flag `--flip-axes`): random flips in any of the three
spatial axes applied identically to `image` and `cond_image`; tokens untouched.

## C6. Network API

```
Precond(img_resolution=(128,128,64), img_channels=1, cond_channels=C_cond,
        context_dim=768, context_tokens=256, use_time_gap=True, sigma_data=1.0,
        model_channels=16, channel_mult=[1,2,2,4], num_blocks=2,
        channels_per_head=32, attn_resolutions=[16], cross_attn_resolutions=[16,32],
        dropout=0.1, use_fp16=True, dtype='fp16'|'bf16', ...)

Precond.forward(x, sigma, cond_image=None, context=None, context_mask=None,
                context_ages=None, delta_days=None, force_fp32=False,
                return_logvar=False)
```

- `x` is the noisy encoded target (B, 1, 128, 128, 64). `cond_image` is
  concatenated to `c_in * x` along channels before the UNet (it is NOT noised
  and NOT scaled by c_in).
- `context=None` skips cross-attention and the pooled FiLM term entirely;
  `delta_days=None` skips the time-gap term. Training never passes None.
- Cross-attention happens only in blocks whose resolution (the spatial size of
  the FIRST axis, i.e. 128 -> 64 -> 32 -> 16) is in `cross_attn_resolutions`;
  resample blocks included, matching the existing fork.
- `EDM2Loss.__call__(net, batch)` builds `x + n` from `batch['image']` and
  forwards the rest of the batch dict.

## C7. Checkpoint metadata

`training_state` / network pickles must carry a `dataset_kwargs`-style dict with
`target_encoding`, `cond_image`, `max_history`, `tokens_dir`, `sigma_data`,
`img_resolution`, so generation and evaluation scripts can reconstruct the
dataset and decode samples without extra flags.

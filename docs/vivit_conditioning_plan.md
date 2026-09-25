# Plan: ViViT-conditioned EDM2 forecasting of future tumor masks

Status: Phases 0-5 complete (2026-09-14); final deck under `analysis/final/`, arm C (no ViT tokens) is the model of record, ViT-token conditioning found inert. See the revision log.

## 1. What exists today

### 1.1 The embeddings (ViViT side)

| item | location |
|---|---|
| per-transition token embeddings (CSV, 256 x 768) | `GrowthNet/projects/vivit/tien_rivanna_repo/growth_classifier_v0005/out/embeddings/{train,val,test}_data_embeddings/` (159 / 34 / 28 files) |
| pooled variants (768-d) | `.../growth_classifier_v0005/out/tumor_pool/pooled_*.csv` |
| classifier table and labels | `.../growth_classifier_v0005/out/{dataset.csv,dataset_meta.csv,growth_labels.csv}` |
| extractor | `.../growth_classifier_v0005/extract_embeddings.py` |
| encoder definition | `.../tien_rivanna_repo/src/networks/t_unetr.py` (`TemporalSpatialEmbedding`, line 135) |
| checkpoint (bare state_dict, 173.5M params) | `D:/Work/GrowthNet_gamailab/epochs200_td8_dual_mean_lr1e4_tverskyce_warmup_lam10_128x128x64_31_pretrain_v0005pre_embedding.pth` |
| flat T2 cohort the embeddings cover | `GrowthNet/uva_vs_flat_05/` (train 104 pts / 263 scans, val 25 / 59, test 21 / 49) |

File naming: `embedding_{pid}_{pid}_{scan_index}_{days}.csv` is computed
from scans `0..i` and named after scan `i+1`. The target scan's date is fed to
the temporal encoder, so the forecast interval is part of the input.

Each CSV is the encoder's final output after the 12-layer ViT and all 8
temporal blocks, averaged over time. Rows are the 256 spatial tokens of an
8 x 8 x 4 grid of 16^3 patches over the 128 x 128 x 64 crop. Preprocessing:
RAS, resample to 1 x 1 x 2 mm, z-score over nonzero voxels, tumor-centred
128 x 128 x 64 crop snapped to multiples of 16.

### 1.2 The saved embeddings are unusable as conditioning

Measured on 2026-09-11 (script: `tools/diagnose_vivit_token_collapse.py` in this repo).

Static check on the CSVs: token-to-token std within a file is about 3e-5,
the across-patient std of the pooled vector is about 8e-5, against a
per-dimension std of 0.83 and a vector norm of 23.0. Every file is the same
vector to about 5 parts per million.

Forward-pass check with taps on the frozen encoder, three train patients,
baseline-only input, relative L2 distance to the vector norm:

| tap | token std / feature std | patient vs patient | noise input vs patient |
|---|---:|---:|---:|
| ViT final output (`spatial_encoder`) | 0.25 | 0.32 | 0.82 |
| after temporal encoder (dates added) | 0.19 | 0.89 | 0.60 |
| temporal block 0 | 0.13 | 0.64 | 0.44 |
| temporal block 2 | 0.020 | 0.087 | 0.060 |
| temporal block 4 | 0.0020 | 0.0077 | 0.0052 |
| temporal block 7 (saved CSVs) | 0.00005 | 0.00017 | 0.00012 |

The ViT output is healthy: patient-specific and spatially varying. Each
temporal block attenuates the input-dependent part by 3 to 4x, and after
eight blocks the output is input-independent. The mechanism is in
`TemporalTransformerBlock.forward` (`t_unetr.py:105-133`): the residual is
taken from the normalized input (`x = norm1(x); x = x + attn(x)`), so every
block rescales the stream and the fixed MLP/attention biases dominate. This
also explains the chance-level growth classifier (CV ROC-AUC 0.57) and why
tumor-focused pooling could not help: there was nothing spatial left to pool.

Consequence for this plan: do not use the CSVs. Re-extract from the ViT
output (or ViT skip layers) and supply the time gap to EDM2 explicitly.

### 1.3 The EDM2 fork (this repo, branch `compvis_embeddings`)

Already implemented on top of upstream EDM2:

- 3D-only UNet (`training/networks_edm2.py`): 3D convs, 3D resampling,
  5D preconditioning. It can no longer run 2D.
- `BinaryMask3DEncoder` (`training/encoders.py:76-91`): mask {0,1} to {-1,+1}.
- Dataset (`training/dataset.py`): loads `{idx:08d}.npy` volumes and an
  `embeddings.npy` of shape (N, T, 768); `label_dim` is the last axis.
- Conditioning entry (`networks_edm2.py:415-431`): `LayerNorm(768)` on the
  tokens, then two paths: FiLM from the token mean into the noise embedding
  (line 363-366) and `CrossAttention` inside every `Block` at the resolutions
  in `attn_resolutions` (only res 16 materializes for a 128 input with
  `channel_mult=[1,2,4,8]`). Output is pixel-normalized and mixed in with
  `mp_sum(..., t=0.3)`.
- Data prep (`prepare_data_3d_longitudinal.py`): pairs the CSV of the history
  up to scan `i` with scan `i`'s own mask, zoomed to a 128^3 cube with
  `order=0`, spacing not preserved. Target is the mask only. No time gap.
- Generation and post-processing (`generate_images.py`,
  `prepare_test_generation.py`, `postprocess_test_generations.py`), Slurm
  scripts with Rivanna paths under `/scratch/tc2fh/`.

Known defects to fix while implementing this plan:

- `networks_edm2.py:365`: `pooled * sqrt(768)` over-scales an already
  unit-variance, LayerNorm'd vector by about 27.7 before `mp_sum`.
- `networks_edm2.py:410`: learnable `torch.nn.LayerNorm` on `Precond` sits
  outside EDM2's forced weight normalization; `generate_images.py:191-195`
  monkey-patches its dtype at load time.
- `Precond.forward` asserts labels exist and `--cond=False` crashes on
  `LayerNorm(0)`, so the network has no unconditional mode. Not a blocker
  for this plan (classifier-free guidance is deferred, see section 5), but
  the assert should go so a no-context forward is possible for tests.
- Cross-attention runs in the `_down` / `_up` resample blocks too, not only
  in attention blocks.
- `sigma_data` stays 0.5 although a sparse +-1 mask has a much smaller std.
- Dataset asserts D == H == W; the ViT crop is 128 x 128 x 64.
- `prepare_embeddings.py` keeps only row 0 of each CSV; it is dead code and
  should be deleted.
- `dataset.py:236-239` disambiguates 3D `.npy` by `shape[0] <= 4`.
- `networks_edm2.py:367`: with `use_fp16=True` the pooled-token FiLM branch
  is fp16 while the noise embedding is fp32, and `mp_sum` (a `lerp`) raises
  a dtype error on torch 2.11. Found by `tools/probe_arch_memory.py`.
- `networks_edm2.py:202`: attention heads are `out_channels //
  channels_per_head`. Any block with fewer channels than `channels_per_head`
  builds a zero-size cross-attention weight and `normalize` divides by zero.
  Upstream only silently disables self-attention in that case. Small models
  need `channels_per_head` exposed and an explicit check.

## 2. Design decisions

1. Conditioning signal: ViT spatial tokens from the frozen v0005 encoder,
   tapped before the temporal blocks, one (256, 768) token set per history
   scan. The saved CSVs are not used. Which ViT layer to tap (final output vs
   skip layers 3/6/9) is an experiment in Phase 0.
2. Time gap: explicit scalar input to EDM2, `log1p(delta_days / 360)`,
   encoded by an `MPFourier` and mixed into the noise embedding with
   `mp_sum`, matching the ViT's own `MLPEncoder` normalization. Each history
   scan's token set also gets a scan-age embedding (days before the target)
   added after projection, so multi-scan histories are ordered.
3. Spatial frame: the ViT's eval crop (RAS, 1 x 1 x 2 mm, tumor-centred
   128 x 128 x 64 computed from the baseline mask). All follow-ups in
   uva_vs_v0005 share the baseline grid and affine and passed a registration
   review, so the future mask is resampled into the baseline crop directly.
   This replaces the 128^3 whole-brain zoom.
4. Target: the future mask as a 1-channel volume. Keep the +-1 binary
   encoding as the default for continuity with the existing generations, and
   add a clipped signed-distance encoding (mm, clipped to +-8, scaled to
   +-1) as a config option; the SDF is a better fit for Gaussian noise and is
   what FlowMatchingGrowthNet used. Image synthesis is out of scope for the
   first model: the ViT already sees the T2 image, and mask forecasting is
   what the lab evaluates.
5. Spatial conditioning in addition to cross-attention: concatenate the
   baseline mask (and optionally the z-scored baseline T2) as extra input
   channels to the UNet. FlowMatchingGrowthNet's knockout diagnosis showed a
   token-only image branch was ignored; input concatenation is the cheapest
   way to make the baseline geometry unmissable.
6. Guidance: none in the first model. Training is purely conditional, with
   no condition dropout and no null token. Autoguidance through `--gnet`
   remains available from upstream if sample quality needs it; classifier-
   free guidance is deferred (section 5).
7. Cross-attention placement: decoupled from self-attention. A new
   `--cross-attn-resolutions` flag lists the UNet levels whose residual
   blocks (including the resample blocks, as the fork already does) get a
   cross-attention step after the residual branch; `--attn-resolutions`
   keeps controlling self-attention only. Default `16,32`. Rationale: the
   median tumor is about 168 mm^3, roughly 5 voxels across at 1 x 1 x 2 mm,
   so at the 16 level (8 x 8 x 16 mm cells) the whole tumor is one cell and
   cross-attention there can only modulate it as a unit; the 32 level
   (4 x 4 x 8 mm) lets boundary voxels query the tokens directly. Measured
   cost in the decision 8 model at batch 4 (2026-09-11, fork code, 16 ch/head
   for the three-level row because of the zero-heads defect):

   | cross-attn levels | blocks | params | peak | ms/step |
   |---|---:|---:|---:|---:|
   | 16 | 8 | 4.17M | 10.6 GB | 380 |
   | 16, 32 | 15 | 4.58M | 11.0 GB | 422 |
   | 16, 32, 64 | 22 | 4.91M | 16.2 GB | 470 |

   The 128 level is excluded (1M queries per sample, and the concatenated
   baseline mask already conditions that level spatially). Each
   cross-attention has a zero-initialized output projection, so adding
   levels cannot change the function at initialization; the balance in
   `mp_sum` becomes a learned scalar. Phase 4 ablates `16` vs `16,32` vs
   `16,32,64` next to the token knockout; the extra K/V projections from
   768 dims are the main added overfitting surface, about 0.4M parameters
   per level.
8. Architecture: no preset. All training runs on the local RTX 5090 (32 GB),
   and the training set is on the order of 160 pairs, so the model is
   deliberately small and every architectural knob is set on the command
   line. Measured on 2026-09-11 with `tools/probe_arch_memory.py` (fp16,
   1 x 128 x 128 x 64 input, 256 x 768 context, one Adam step):

   | channels | channel_mult | blocks | ch/head | params | batch 4 peak | ms/step |
   |---:|---|---:|---:|---:|---:|---:|
   | 32 (xxs preset) | 1,2,4,8 | 3 | 64 | 63.7M | 29.0 GB | 1082 |
   | 16 | 1,2,4,8 | 3 | 64 | 16.9M | 14.4 GB | 1054 |
   | 32 | 1,2,2,4 | 2 | 32 | 15.1M | 21.2 GB | 360 |
   | 24 | 1,2,2,4 | 2 | 32 | 8.8M | 15.9 GB | 598 |
   | 16 | 1,2,4,4 | 2 | 32 | 5.7M | 11.1 GB | 405 |
   | **16** | **1,2,2,4** | **2** | **32** | **4.2M** | **10.6 GB** | **397** |
   | 16 | 1,2,2,4 | 1 | 32 | 3.0M | 7.8 GB | 260 |
   | 16 | 1,1,2,2 | 1 | 16 | 1.2M | 6.3 GB | 314 |

   Initial preference: `--channels=16 --channel-mult=1,2,2,4 --num-blocks=2
   --channels-per-head=32 --attn-resolutions=16 --dropout=0.1`, about 4.2M
   parameters, of which about 0.85M are the cross-attention and token
   projection. At batch 4 one epoch of 160 pairs is about 16 s, so 1000
   epochs is under 5 hours. Scale-up path if it underfits: 1,2,4,4 (5.7M),
   then channels 24 (8.8M). Scale-down if it overfits: one block per level
   (3.0M). The `edm2-vol128-*` presets stay in the file for the Rivanna runs
   but are not used here.

   Self-attention in this configuration lives entirely at the 16 x 16 x 8
   level (64 channels, 2 heads of 32, 2048 query tokens): 6 blocks (encoder
   block0/1, in0, decoder block0/1/2). Cross-attention follows decision 7:
   8 blocks at the 16 level and 7 at the 32 level (32 channels, 1 head of
   32, 16384 query tokens). `channels_per_head=32` is what makes those heads
   exist; at the upstream default of 64 the 16-channel model has no
   attention at all.
10. Precision and kernels on the RTX 5090 (capability 12.0, torch 2.11
   cu130, checked 2026-09-11): keep EDM2's fp16 activations with fp32
   weights as the default, add a `--dtype {fp16,bf16}` flag (bf16 is
   supported and removes the need for `--ls` loss scaling and the
   `clip_act=256` guard; try both in the Phase 3 smoke run). Upstream
   disables TF32 on purpose; leave it. FP8 and FP4 tensor cores are the only
   Blackwell-specific dtypes and do not apply: the model is 3D convolutions
   under forced weight normalization, torchao float8 wraps only `nn.Linear`,
   and at 4M parameters the step is bound by activation traffic, not
   matmul throughput. Flash SDPA has no sm_120 kernel in this build, but the
   cuDNN and memory-efficient SDPA backends work, so the `CrossAttention`
   rewrite in Phase 2 should call `scaled_dot_product_attention` and let
   PyTorch pick. Measure `channels_last_3d` and `torch.compile` in Phase 3;
   both are plausible speedups but untested here.
9. Evaluation mirrors FlowMatchingGrowthNet so results are comparable:
   Dice, surface Dice, volume error, growth/shrink direction accuracy,
   prediction intervals over 32-100 samples, against carry-forward and the
   deterministic forecaster. A conditioning knockout (swap another patient's
   tokens, freeze the time gap) is a required check before any claim.

## 3. Phases

### Phase 0: re-extract embeddings and prove they carry signal (ViViT repo)

1. New script `growth_classifier_v0005/extract_vit_tokens.py`:
   - Loads the encoder with `load_encoder`, registers a forward hook on
     `net.spatial_encoder` (and optionally on ViT blocks 3, 6, 9).
   - Runs every scan once with `T = 1` (no history needed; the tap is before
     the temporal blocks), same eval transform, `set_determinism(42)`.
   - Writes one `.npz` per scan: `tokens` (256, 768) float16, `crop_origin`,
     `crop_shape`, `affine`, `spacing`, `scan_id`, `days_since_first`, plus
     the crop-frame baseline mask so Phase 1 can reuse the identical crop.
   - Keyed by scan id, not by transition, so any (history, target) pairing
     can be assembled later.
2. Sanity probe: rerun `train_classifiers.py` on the mean-pooled ViT tokens
   and on tumor-weighted pooling. Expected: CV ROC-AUC clearly above the
   0.57 obtained from the collapsed vectors. If it is not, stop and revisit
   the tap (skip layers) before building the diffusion pipeline.
3. Document the collapse finding in `growth_classifier_v0005/RESULTS.md`
   and note that the temporal stack needs a standard pre-norm residual
   (`x = x + attn(norm1(x))`) before any retraining of the ViViT.

### Phase 1: dataset builder (this repo)

New `prepare_data_vivit_pairs.py` replacing `prepare_data_3d_longitudinal.py`:

- Inputs: uva_vs_v0005 split root, the Phase 0 token directory, the
  patient-level split from `uva_vs_flat_05/train_val_test_split.json`.
- Sample = (history scans `0..i`, target scan `j > i`). Default pairs are
  consecutive (`j = i + 1`), matching the ViViT training; a flag enables all
  `i < j` pairs to enlarge the training set (171 of 257 patients have only
  two scans, so consecutive pairing gives about 160 train samples).
- Per sample, in the baseline crop frame: target mask (128, 128, 64) uint8,
  baseline mask, baseline image (float16), `delta_days`, history scan ids
  and their ages relative to the target.
- Output layout: `{idx:08d}.npz` per sample plus `dataset.json` with the
  manifest, the split, and dataset statistics (`sigma_data` for the chosen
  target encoding, delta_days max on train). Tokens are referenced by scan
  id and loaded from the Phase 0 store, not copied per sample.
- Keep test strictly separate; build train and val separately (no merged
  trainval).

### Phase 2: network and loss changes (this repo)

`training/dataset.py`

- Replace `ImageFolderDataset` with a `PairDataset` that returns
  `image` (target, C x 128 x 128 x 64), `cond_image` (baseline mask/image
  channels), `context` (K x 256 x 768, padded to K max history scans with a
  key mask), `context_ages` (K,), `delta_days` (1,).
- Drop the cubic assertion; `resolution` becomes a tuple.

`training/networks_edm2.py`

- `UNet.__init__` gains `cond_channels` (input concat), `context_dim`,
  `context_tokens`, `use_time_gap`. `img_resolution` becomes a 3-tuple.
- Context projection: `MPConv(768 -> model_channels * 4, kernel=[])`
  followed by a fixed `normalize` (replaces the learnable `LayerNorm`), then
  add an `MPFourier` scan-age embedding per history scan.
- Time gap: `MPFourier` + `MPConv` into `emb`, mixed with `mp_sum`.
- Remove the `sqrt(768)` factor; keep a pooled-token FiLM path but computed
  from the normalized projected tokens.
- `CrossAttention`: key padding mask for variable history length, SDPA
  instead of the explicit einsum, zero-init `to_out`, learned balance.
- Apply cross-attention only when `attention=True` for the block.
- `Precond.forward(x, sigma, context=None, ...)` tolerates a missing context
  (skips cross-attention and the pooled FiLM term) so tests and the
  knockout ablation can run; training never drops the context.
- `Precond` accepts `sigma_data` from the dataset stats.

`training/training_loop.py`

- `EDM2Loss` receives the batch dict; passes `cond_image` concatenated to the
  noisy target, `context`, `context_mask`, `delta_days`.
- Network summary dummy inputs follow the new shapes.

`train_edm2.py`

- No new presets. Make `--preset` optional and add architecture flags that
  upstream keeps preset-only: `--channel-mult` (comma list), `--num-blocks`,
  `--attn-resolutions` (comma list), `--channels-per-head`; lower the
  `--channels` minimum from 16 to 8. Refuse a configuration where any
  attention block would get zero heads.
- Pipeline flags: `--tokens-dir`, `--target-encoding {binary,sdf}`,
  `--cond-image {mask,mask+image,none}`, `--max-history`.
- Express `--duration`, `--snapshot`, `--status` in samples as upstream does
  but document them in epochs for a 160-pair set (1 epoch = 160 samples,
  so `--duration=160Ki` is about 1000 epochs).
- The full initial command line is recorded in section 2.8.

`generate_images.py`

- Load samples by manifest id, write NIfTI in
  the baseline crop frame and a second file resampled to the original grid
  using the stored crop origin and affine (not `zoom`).

Delete `prepare_embeddings.py` and `prepare_data_3d.py`; keep
`prepare_data_3d_longitudinal.py` only until Phase 1 lands, then delete.

### Phase 3: verification before any real training

- Unit tests (new `tests/`): dataset shapes and split isolation; network
  forward at 128 x 128 x 64 with K = 1 and K = 3 history scans; freshly
  built network with zero-init cross-attention reproduces the no-context
  output; magnitude
  check that activations stay near unit variance through the new blocks.
- Local smoke run on the RTX 5090 for a few hundred steps on 8 samples only
  to check loss goes down and sampling produces a mask. This is a plumbing
  check, not training, and is scheduled after the user signs off on the plan.

### Phase 4: training and evaluation (local RTX 5090)

- All training runs on this workstation with the section 2.8 architecture,
  batch 4 (about 11 GB), single process, no Slurm. Snapshots every few
  epochs, post-hoc EMA and checkpoint selection on val Dice, never on the
  diffusion loss.
- Overfitting controls, in order of preference: all `i < j` pairs, flips in
  all three axes, `--dropout=0.1`, then a smaller model (section 2.8).
- The Rivanna Slurm scripts are left as they are for now; parameterizing
  their paths is only needed if a larger run is ever wanted.
- Evaluation script producing the FlowMatchingGrowthNet metric set on the
  test split, 32 samples per pair, consensus mask by mean probability > 0.5.
- Required ablations: no-context (spatial cond + time gap only), no
  spatial cond (tokens + time gap only), token knockout (another patient's
  tokens), time gap fixed. The FlowMatchingGrowthNet knockout found that
  models could ignore their conditioning entirely; this pipeline must show
  the tokens change the output before any result is reported.

### Phase 5: side-by-side comparison deck against the best flow-matching model

Goal: one PowerPoint that shows the best EDM2 model from Phase 4 and the
best FlowMatchingGrowthNet model on the same selected test examples, same
slices, same crop windows, same metrics, built by the flow repo's existing
deck tool rather than a new one.

What the flow repo already provides (all under
`D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet`, surveyed 2026-09-11):

- Deck builder `src/tumor_flow/analysis/build_comparison_deck.py`, run as
  `python -m tumor_flow.analysis.build_comparison_deck --old-dir A --new-dir B
  --old-label .. --new-label .. --recipe-json recipe.json --out deck.pptx`
  (needs `python-pptx`). It takes two analysis directories with identical
  layout and produces: title, recipe bullets, headline metric table with
  paired bootstrap CIs, horizon-bin bars, paired consensus-Dice scatter,
  per-sample Dice distribution, one slide per shared scan, takeaways.
- Per-scan slide: model A row above model B row, each row three axial
  slices at the same z levels and the same square crop, each slice a
  composite of baseline MRI, forecast probability map in viridis, and the
  ground-truth future mask as a red contour; right column has per-sample
  Dice and volume histograms; caption with consensus Dice, sample Dice
  mean and sd, consensus vs target volume. Slice levels are three evenly
  spaced z values over the support where either model's density is at
  least 0.5 or the target is present, so they are shared across models.
- Example selection `src/tumor_flow/analysis/visualizations.py`
  (`select_growth_pairs`): test patients classified grew / stable / shrank
  by +-20 percent volume change, top 5 per class by final volume, every
  follow-up of those patients. Stored in
  `artifacts/analysis_phase20/selected_patients.json`: 15 patients, 21
  scans (grew 743, 406, 289, 527, 537; stable 475, 399, 425, 114, 300;
  shrank 50, 339, 55, 209, 651). The deck intersects the two directories,
  so an EDM2 scan missing from the ViT cohort simply drops out. Checked
  2026-09-11 against `uva_vs_flat_05/train_val_test_split.json`: 20 of the
  21 scans are in the ViT test cohort with their baseline; only patient
  651 (shrank) is absent, so the deck will have 20 shared scan slides and
  all three growth categories stay represented.
- Analysis directory contract (Part A `forecasting.py:write_part_a_report`
  and Part B `visualizations.py`): `pair_metrics.csv`, `sample_metrics.csv`,
  `aggregate.json`, `method_comparison.csv`, `horizon_binned.csv`,
  `viz_index.json`, `selected_patients.json`, and per scan
  `viz/patient<id>/scan<id>/{arrays.npz,metrics.json}` where `arrays.npz`
  holds `mri`, `density`, `target`, `dice` (per sample), `volumes` (mm^3),
  `z_levels`, `spacing_xyz_mm`, all in the flow repo's baseline-crop ZYX
  frame.
- Test pairs: `artifacts/data_audit/pair_manifest.csv`, 38 baseline-to-
  future pairs over 26 patients, columns `split, patient_id,
  baseline_scan_id, target_scan_id, baseline_day, target_day, delta_days`
  plus absolute NIfTI paths into uva_vs_v0005.
- Flow model to compare against: the flow repo has three "best" layers
  that disagree (README production D031, the `release/best_flow_model`
  bundle from Phase 15, and the D040 selection with the volume flow, plus
  the D041 candidate). Default to the D040 model, whose finished analysis
  directory is `artifacts/analysis_d040_vflow/`, and name the checkpoint
  digest on the recipe slide; switching to another candidate is a one-line
  change of `--old-dir`.

Work in this repo:

1. `tools/export_flow_analysis.py`: given the EDM2 checkpoint, the flow
   pair manifest, and the flow analysis directory to mirror, run the EDM2
   model on the 38 test pairs (history = baseline scan only, so the
   context is one 256-token set and `delta_days` comes from the manifest),
   draw N samples per pair (default 256; the flow analysis used 1000,
   configurable if time allows), and write a complete analysis directory
   in the contract above. Metrics are computed by importing
   `tumor_flow.evaluation.metrics` and the Part A / Part B writers from
   the flow package rather than re-implementing them, so both directories
   are scored by the same code. Run it inside the flow repo's `uv`
   environment (`uv run --project <flow repo> python tools/export_flow_analysis.py ...`).
2. Frame alignment: EDM2 predicts in the ViT crop (1 x 1 x 2 mm). Each
   sample and the probability map are resampled to the full baseline
   reference grid using the stored crop origin and affine (the Phase 2
   generator already does this), then cropped with the flow repo's
   `CropSpec` for that pair so `arrays.npz` lands in the flow repo's crop
   frame. Nearest-neighbour for masks, linear for the probability map.
   The mask convention is the flow repo's (`mask = sdf <= 0` for their
   samples, threshold 0.5 on the mean for EDM2 consensus); record both in
   `metrics.json`.
3. Recipe JSON for the deck (model names, checkpoints, bullets, headline
   metrics, takeaways), following `artifacts/hpsearch/tools/phase20_recipe.py`.
4. Build the deck:
   `--old-dir <flow analysis dir> --new-dir <edm2 analysis dir>`, labels
   "Flow matching (D040)" and "EDM2 + ViT tokens", output
   `analysis/EDM2_vs_FlowMatching.pptx` in this repo with its `panels/`
   PNGs. Add the carry-forward and deterministic rows to the headline
   table as the flow deck does, so the reader sees all four.
5. A short `analysis/README.md` stating which EDM2 checkpoint, which flow
   checkpoint, sample counts, seeds, and the number of shared scans.

Acceptance: the deck opens with one slide per shared selected scan showing
the two models on identical slices and crops; the headline table reports
paired deltas with bootstrap CIs computed by the flow repo's code; the
EDM2 conditioning knockout result from Phase 4 is cited on the takeaways
slide so a favourable Dice cannot be read without it.

## 4. Risks and open questions

- Data size: roughly 160 training pairs. Diffusion at 128 x 128 x 64 will
  overfit; the 4.2M-parameter model, flips in all three axes and all
  `i < j` pairs help, and the carry-forward Dice of about 0.64 from
  FlowMatchingGrowthNet is the bar.
- Tap choice: the ViT final layer carries patient identity, but whether it
  carries growth-predictive information is untested; Phase 0 step 2 decides.
- The ViT was trained on a reshaped (not permuted) volume
  (`t_unetr.py:262`), so token index does not map to an anatomical 8 x 8 x 4
  grid. Cross-attention is permutation-invariant over keys, so this only
  matters if positional correspondence is added later.
- Mask annotation noise: same-scan masks agree at Dice 0.71-0.85, so
  volume-change and direction metrics matter more than Dice.
- 54 patients are T1-only and have no ViT tokens; they are excluded.
- If Phase 0 shows weak signal, the fallback is to retrain the ViViT with a
  corrected temporal block, which is outside this repo.

## 5. Deferred

- Classifier-free guidance (condition dropout, learned null token, guided
  sampling in a limited sigma interval). Deliberately left out of the first
  model on 2026-09-11. Revisit only if the conditioning knockout in Phase 4
  shows the tokens are used but adherence is weak; the FlowMatchingGrowthNet
  D035 work is the reference implementation if it is picked up.

## Revision log

- 2026-09-12: measured parameter count for the section 2.8 initial
  configuration is 3.55M after the shared token projection, not the 4.6M
  originally estimated in the section 2.8 table.
- 2026-09-12: found the ViT training-frame bug (see
  `docs/vivit_pipeline_contracts.md` C0): the ViViT repo's loader drops the
  NIfTI affine before `Orientationd`/`Spacingd`, so the frozen encoder was
  trained on native-array volumes at native x/y spacing and 2x native z
  spacing, not the physically-correct 1 x 1 x 2 mm frame the first token
  store assumed. The token store was re-extracted in the ViT training frame
  (`out/vit_tokens_trainframe/`), and every consumer in this repo now reads
  `spacing` from the data instead of hard-coding it. This is the spacing
  change referenced in decision 2 and contracts C0-C1.
- 2026-09-12: the Phase 0 classifier probe on the re-extracted ViT tokens
  came back at chance on the 1 x 1 x 2 mm store (before the C0 frame fix
  above), consistent with the CV ROC-AUC 0.57 seen on the collapsed
  temporal-block CSVs in section 1.2. This is why a `--no-context` trained
  arm is required, not optional, until token conditioning is shown to help.
- 2026-09-12: added a no-context trained arm to Phase 4 (`train_edm2.py
  --no-context` builds a token-free network with `context_dim=0` and skips
  all token I/O) as a required comparison alongside the Phase 4 knockouts,
  given the chance-level probe result above.
- 2026-09-14: arm B (SDF target + ViT tokens) completed the full 245 kimg
  budget after a resume from kimg 131. Val selection (`select_checkpoint.py`,
  same settings as arms A and C, 46 val pairs, carry-forward 0.721) picks
  kimg 12 EMA 0.05 at consensus Dice 0.558; every later snapshot scores
  below 0.35 and the run degrades steadily. Arm C (binary target, no
  tokens) stays the model of record at 0.579 (kimg 98 EMA 0.05); arm A
  (binary target + tokens) 0.460 (kimg 98 EMA 0.05). The SDF target did
  not help. For the record, B's selected checkpoint on the 25 test pairs
  of `datasets/flow_test_pairs_tf` (64 samples, 32 steps, seed 2026, EDM2
  scoring frame) gives consensus Dice 0.577 pair-level / 0.604
  patient-level (C 0.580 / 0.611, A 0.466 / 0.473, carry-forward 0.677);
  see `runs/intermediate/B_gen/metrics/summary.json`.
- 2026-09-14: the two remaining Phase 4 knockouts are done, so all four
  ablations are in (`runs/intermediate/knockouts_summary.md`; patient-level
  means, delta = knockout minus full context, paired patient-level
  bootstrap, 2000 resamples, seed 2026, 25 test pairs over 18 patients).
  no-cond-image (mask channel filled with the background value): A 0.473
  -> 0.000, delta -0.473, 95% CI [-0.587, -0.354]; C 0.611 -> 0.000, delta
  -0.611, CI [-0.701, -0.499]. fixed-time (delta_days = train median 682
  for every pair): A delta +0.014, CI [-0.010, +0.038]; C delta -0.006, CI
  [-0.021, +0.010]. Together with the earlier shuffle-tokens result on A
  (delta -0.003, CI [-0.011, +0.003]): both arms rely on the baseline mask
  alone, and neither the time gap nor the ViT tokens is measurably used.
  The "no spatial cond" ablation listed in Phase 4 is covered by
  no-cond-image; the CI-excludes-zero requirement before reporting a
  token result was never met, so no token-conditioned result is reported.
- 2026-09-14: Phase 5 final deck built under `analysis/final/`
  (`recipe.json`, `EDM2_vs_FlowMatching.pptx` 28 slides,
  `EDM2_vs_FlowMatching_3d.pptx` 48 slides), reusing the arm C export and 3D
  renders from `analysis/intermediate/edm2_analysis`; commands and the
  three-arm tables are in `analysis/README.md` ("Final run"). Bottom line
  on the takeaways slide: the diffusion forecaster trails flow D040
  (consensus Dice 0.576 vs 0.639, flow frame; paired on the 25 shared
  pairs 0.576 vs 0.677, delta -0.101, CI [-0.230, -0.007]) and its own
  carry-forward baseline (0.675), and ViT-token conditioning is inert.
  Phases 4 and 5 are complete; the plan's status line at the top reflects
  this. The Phase 5 deck name differs from the plan's original
  "EDM2 + ViT tokens" label because the featured model is the no-token arm.
- 2026-09-15: follow-up moved into the flow repo. To separate the network
  from the objective and the data frame, the EDM2 magnitude-preserving U-Net
  (`velocity_field.kind: edm2`) and EDM2's exact training objective
  (`flow_training.objective: diffusion`, log-normal noise levels, EDM
  preconditioning, uncertainty-weighted loss, EDM Heun sampler) were added to
  `FlowMatchingGrowthNet` as config switches (its decision D042, handoff
  `handoffs/PHASE_21_IMPLEMENTATION.md`) and trained on D040's exact data,
  latent, conditioning, and budget (`artifacts/hpsearch/stage30`-`stage32`
  there). This repo is unchanged by that work.

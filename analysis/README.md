# EDM2 vs Flow Matching comparison deck

Phase 5 deliverable (`docs/vivit_conditioning_plan.md` Phase 5,
`docs/vivit_pipeline_contracts.md`). This file documents the final run
(`analysis/final/`, 2026-09-14) first, then the superseded intermediate
run it was built on top of (`analysis/intermediate/`, 2026-09-13), which
still holds the featured model's export and the full five-step toolchain
with its per-step explanations.

The dry run used to validate this toolchain (untrained network, garbage
forecasts, complete pipeline) lives under `runs/dryrun_flow_export/`.

## Final run (2026-09-14)

All three Phase 4 arms have now trained the full 245 kimg budget on
`datasets/vivit_pairs_all_tf`, every Phase 4 ablation is done, and the deck
under `analysis/final/` is the Phase 5 deliverable. The intermediate run
below is kept as history only; nothing in it is the result of record any
more except the arm C export it produced, which the final deck reuses
unchanged (same checkpoint, same 25-pair generation, same 3D renders).

### Models compared

The featured EDM2 model is unchanged from the intermediate run: arm C,
`runs/C_binary_notokens/network-snapshot-0000098-0.050.pkl` (sha256
`7f0db34e9c45...`), val-selected over the full 245 kimg run, compared with
flow matching D040 (`flow_ema_00025500_0.100.pt`, sha256 `4e8e52357161...`).
Sample counts, steps and seeds are as in the intermediate table below
(EDM2 64 samples / 32 steps / seed 2026; flow 1000 / 16 / 2026).

### Three-arm val selection (`select_checkpoint.py`, same settings for all arms)

Consensus Dice on the 46 val pairs of `datasets/vivit_pairs_all_tf`;
carry-forward on the same pairs is 0.721 for every arm.

| arm | target | ViT tokens | best snapshot | val consensus Dice | val surface Dice | note |
|---|---|---|---|---:|---:|---|
| C (featured) | binary | no | kimg 98, EMA 0.05 | 0.579 | 0.682 | `runs/C_binary_notokens/val_selection.csv` |
| B | SDF | yes | kimg 12, EMA 0.05 | 0.558 | 0.699 | degrades steadily after kimg 12; all 114 later snapshots score below 0.35 (`runs/B_sdf_tokens/val_selection.csv`, finished 2026-09-14 after a resume from kimg 131) |
| A | binary | yes | kimg 98, EMA 0.05 | 0.460 | 0.528 | kimg 0-98 in `runs/A_binary_tokens/val_selection.csv`; kimg 98-245 at a coarser stride in `val_selection_pass1_ext_raw.csv`, none above 0.446 |

The SDF target (arm B) did not help: its best snapshot is below arm C and it
is the only arm whose val Dice collapses with more training.

### Arm B test evaluation (for the record only)

Arm B's selected checkpoint was run on the same 25 test pairs of
`datasets/flow_test_pairs_tf` with exactly the C_gen settings (64 samples,
32 steps, seed 2026, trainframe token store), output
`runs/intermediate/B_gen`, log `runs/intermediate/B_gen.log`, metrics from
`evaluate_forecasts.py` in `runs/intermediate/B_gen/metrics/summary.json`.
These are EDM2's own scoring frame (same as the A row and the knockout
table), not the flow-frame headline numbers.

| arm (kimg, EMA) | consensus Dice, pair mean | consensus Dice, patient mean | surface Dice 1mm | direction acc | carry-forward Dice |
|---|---:|---:|---:|---:|---:|
| C (98, 0.05) | 0.580 | 0.611 | 0.635 | 0.12 | 0.677 |
| B (12, 0.05) | 0.577 | 0.604 | 0.625 | 0.32 | 0.677 |
| A (98, 0.05) | 0.466 | 0.473 | 0.498 | 0.08 | 0.677 |

Neither A nor C was evaluated on the 36-pair test split of
`datasets/vivit_pairs_all_tf` (no generation manifest under `runs/` names
that data dir), so B was not either; the three-arm test table is the
25-pair `flow_test_pairs_tf` split only.

### Phase 4 conditioning knockouts (all four done)

From `runs/intermediate/knockouts_summary.md` / `.json`
(`runs/eval/paired_bootstrap.py`). Convention: **every Dice in this table is
a patient-level mean** (each patient's pairs averaged first), which is why
arm C reads 0.611 here versus 0.580 pair-level above. Delta = knockout
minus full context, paired by patient; 95% CI from a patient-level
bootstrap, 2000 resamples, seed 2026; 25 test pairs over 18 patients.
Generation settings identical to the full-context runs except the knockout.

| arm | knockout | full | knockout | delta | 95% CI | reading |
|---|---|---:|---:|---:|---|---|
| A | shuffle-tokens (another patient's ViT tokens) | 0.473 | 0.469 | -0.003 | [-0.011, +0.003] | straddles zero: tokens not measurably used |
| A | no-cond-image (mask channel filled with background) | 0.473 | 0.000 | -0.473 | [-0.587, -0.354] | excludes zero: mask used |
| A | fixed-time (delta_days = train median 682) | 0.473 | 0.486 | +0.014 | [-0.010, +0.038] | straddles zero: time gap not measurably used |
| C | no-cond-image | 0.611 | 0.000 | -0.611 | [-0.701, -0.499] | excludes zero: mask used |
| C | fixed-time | 0.611 | 0.604 | -0.006 | [-0.021, +0.010] | straddles zero: time gap not measurably used |

Arm C has no shuffle-tokens row because it has no tokens. Conclusion: both
arms rely entirely on the baseline mask; without it they forecast an empty
tumour. The time gap and the ViT tokens are not measurably used. Note that
the deck's auto-generated knockout sentence (from `tools/make_flow_recipe.py`)
reports the same shuffle-tokens result as full minus knockout, i.e. delta
+0.003, CI [-0.003, +0.011]; the sign convention differs, the finding does
not.

### Headline numbers (unchanged, flow-frame)

The featured export is reused, so the headline table in the intermediate
section still holds: consensus Dice 0.576 (EDM2 arm C) vs 0.639 (flow
D040, mean over its own 38 test pairs); paired on the 25 shared pairs the
deck's headline slide shows 0.677 (flow) vs 0.576 (EDM2), delta -0.101,
95% CI [-0.230, -0.007]. EDM2's own carry-forward on those 25 pairs is
0.675 and the deterministic forecaster 0.667, both above the diffusion
model.

### Outputs

- `analysis/final/recipe.json`
- `analysis/final/EDM2_vs_FlowMatching.pptx` -- 28 slides (2D only).
- `analysis/final/EDM2_vs_FlowMatching_3d.pptx` -- 48 slides (28 base +
  20 3D-rotation companion slides). About 115 MB; ignored by git via
  `analysis/**/*.pptx`, as is `analysis/final/panels/`. Regenerate with the
  commands below.
- `runs/intermediate/B_gen/` (arm B test generation + metrics) and
  `runs/intermediate/B_gen.log`.

The takeaways slide states: the three-arm val selection above; the full
four-ablation result with deltas and CIs and the mask-only interpretation;
the Phase 0 probe (best ViT-tap CV ROC-AUC 0.52 vs 0.57 control); and the
bottom line that the diffusion forecaster trails both flow D040 and
carry-forward on these pairs, with ViT-token conditioning inert.

### Exact commands used for the final run

Interpreters as in the intermediate section. Steps 1 and 2 of the toolchain
(export, 3D renders) were NOT rerun; `analysis/intermediate/edm2_analysis`
is reused as the EDM2 analysis dir.

Arm B test generation and evaluation (EDM2 interpreter, GPU):

```
<edm2-python> generate_forecasts.py \
  --net runs/B_sdf_tokens/network-snapshot-0000012-0.050.pkl \
  --data datasets/flow_test_pairs_tf --split test --num-samples 64 --steps 32 --seed 2026 \
  --tokens-dir D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/growth_classifier_v0005/out/vit_tokens_trainframe \
  --out runs/intermediate/B_gen
<edm2-python> evaluate_forecasts.py --gen-dir runs/intermediate/B_gen/test --out runs/intermediate/B_gen/metrics
```

Recipe (step 3):

```
<edm2-python> tools/make_flow_recipe.py \
  --old-name "Flow matching (D040)" \
  --old-checkpoint D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/hpsearch/stage23/ref300/recon_step00025500/flow_ema_00025500_0.100.pt \
  --old-aggregate-json D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/analysis_d040_vflow/aggregate.json \
  --new-name "EDM2 (mask + time gap, no ViT tokens)" \
  --new-checkpoint runs/C_binary_notokens/network-snapshot-0000098-0.050.pkl \
  --new-aggregate-json analysis/intermediate/edm2_analysis/aggregate.json \
  --new-bullet "3.32M params, binary target, baseline mask + time gap conditioning, no ViT tokens" \
  --new-bullet "Arm C of three: trained the full 245 kimg budget, val-selected at kimg 98 (EMA 0.05)" \
  --edm2-gen-dir runs/intermediate/A_gen \
  --knockout-dir runs/intermediate/A_knockout_shuffle \
  --knockout-arm-label "arm A (binary target + ViT tokens), same kimg 98 checkpoint" \
  --eyebrow "PHASE 5 - FINAL COMPARISON" \
  --takeaway "<(a) three-arm val selection and B test Dice>" \
  --takeaway "<(b) four-ablation deltas and CIs>" \
  --takeaway "<(b) interpretation: mask only>" \
  --takeaway "<(c) Phase 0 probe>" \
  --takeaway "<(d) bottom line>" \
  --out analysis/final/recipe.json
```

The full takeaway strings are in `analysis/final/recipe.json` (the file is
committed; the deck is rebuilt from it).

Deck build (step 4):

```
tools/build_flow_deck.sh \
  D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet \
  D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/analysis_d040_vflow \
  analysis/intermediate/edm2_analysis \
  "Flow matching (D040)" "EDM2 (mask + time gap, no ViT tokens)" \
  analysis/final/recipe.json \
  analysis/final/EDM2_vs_FlowMatching.pptx
```

3D augmentation (step 5):

```
uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet --with python-pptx python \
  tools/augment_deck_3d.py \
  --pptx analysis/final/EDM2_vs_FlowMatching.pptx \
  --old-analysis-dir D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/analysis_d040_vflow \
  --new-analysis-dir analysis/intermediate/edm2_analysis \
  --old-label "Flow matching (D040)" --new-label "EDM2 (mask + time gap, no ViT tokens)" \
  --out analysis/final/EDM2_vs_FlowMatching_3d.pptx
```

Tests run after the build (all passing, 2026-09-14): `python -m pytest tests -q`
with the EDM2 interpreter (89 passed, 2 skipped: the flow-interpreter
tests), and `uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet
--with python-pptx python -m pytest tests/test_export_flow_analysis.py
tests/test_augment_deck_3d.py -q` (18 passed).

## Intermediate run (2026-09-13) -- superseded history

**Superseded by the final run above.** Kept because the final deck reuses
this run's arm C export (`analysis/intermediate/edm2_analysis`) and
because the five-step toolchain is documented here in full. Training was
paused at kimg 98 (of a planned 245) to produce this comparison; the
featured checkpoint turned out to be the val-selected one over the full
run as well, so the headline numbers below still stand. The knockout and
arm-status statements in this section reflect what was known on
2026-09-13; the final section above has the complete picture.

### Models compared

| | old (flow) | new/featured (EDM2, arm C) |
|---|---|---|
| name | Flow matching (D040) | EDM2 (mask + time gap, no ViT tokens) |
| checkpoint | `FlowMatchingGrowthNet/artifacts/hpsearch/stage23/ref300/recon_step00025500/flow_ema_00025500_0.100.pt` | `edm2_gamai_lab/runs/C_binary_notokens/network-snapshot-0000098-0.050.pkl` |
| checkpoint sha256 | `4e8e52357161be1e87e91ed6a50470853cea734c783addd4a9015977f5f1a01a` | `7f0db34e9c45e394514a864c82633f6839806abe900e55d9f94e9204cf545119` |
| architecture | volume flow + `flow_ema_00025500` recon flow (`artifacts/hpsearch/stage12/volume_flow/volume_flow.pt`) | 3.32M params, binary target encoding, baseline-mask + time-gap conditioning, **no ViT tokens** (`--no-context`) |
| samples / steps | 1000 / 16 (Heun) | 64 / 32 (`edm_heun`) |
| seed | 2026 | 2026 |

Arm A (same kimg-98 checkpoint stage, but WITH ViT tokens) was trained in
parallel for the conditioning ablation cited in the takeaways, not as a
deck model: checkpoint `edm2_gamai_lab/runs/A_binary_tokens/network-snapshot-0000098-0.050.pkl`,
consensus Dice 0.460 on val, **0.466 on these 25 test pairs (EDM2's own
scoring frame, not the flow-frame headline table -- the two are not
directly comparable numbers, only the qualitative token-vs-no-token
direction is)**.

### Data coverage

- Pair dataset: `datasets/flow_test_pairs_tf` (built from the ViT
  training-frame token store, contract C0 -- 0.5x0.5x1.0mm spacing, 64mm
  FOV). 25 of the flow repo's 38 test pairs survive (13 skipped, all
  `*_not_in_vit_test_cohort`); see `datasets/flow_test_pairs_tf/dataset.json`'s
  `"manifest"` provenance block for the full skip list.
- Deck slides: 20 of the flow deck's 21 selected scans across 15 patients
  (14 patients survive). Patient 651's only test pair drops out because
  that patient isn't in the ViT token store's test split at all -- a
  **permanent gap**, independent of which EDM2 checkpoint is used (see
  below).

### Headline numbers (25 shared test pairs, flow-frame metrics)

| metric | flow (D040) | EDM2 (arm C) | delta |
|---|---:|---:|---:|
| consensus Dice | 0.639 | 0.576 | -0.064 |
| consensus surface Dice | 0.888 | 0.786 | -0.102 |
| per-sample Dice | 0.555 | 0.533 | -0.022 |
| carry-forward (single, same 25 pairs) | -- | 0.675 | -- |
| deterministic (single, same 25 pairs, copied from flow's own model) | -- | 0.667 | -- |

Both models score below their own carry-forward baseline at this
checkpoint stage; EDM2 arm C additionally scores below the deterministic
forecaster.

### Conditioning-knockout result (arm A, not arm C)

The shuffle-tokens knockout needs tokens to shuffle, so it was run on arm A
(the token arm), not on the deck's featured no-token arm C:

> Conditioning-knockout check on arm A (binary target + ViT tokens), same
> kimg 98 checkpoint: consensus Dice 0.473 (full context) -> 0.469 (another
> patient's tokens substituted), delta +0.003, 95% CI [-0.003, +0.011]
> (patient-level bootstrap, 2000 resamples, seed 2026, n=18 patients): the
> token arm does not use its tokens (95% CI straddles zero, no significant
> effect). The featured model here, EDM2 (mask + time gap, no ViT tokens),
> has no tokens by construction and was not itself part of this knockout.

Combined with the Phase 0 classifier probe (best ViT-tap CV ROC-AUC 0.52 vs
0.57 for the collapsed-embedding control -- no tap carries growth-predictive
signal) and arm A scoring below arm C, this is a consistent three-way
finding: ViT tokens do not help this pipeline at the current checkpoint
stage.

### Annotation-disagreement note (unchanged from the dry run, re-verified against this run's own `alignment_check.csv`)

Same 3 pairs, same values -- this is a data-provenance fact about the two
cohorts' annotation pipelines, independent of the checkpoint or the frame
revision (contract C0):

| patient | baseline scan | target scan | annotation dice (baseline) | annotation dice (target) | side(s) |
|---|---|---|---:|---:|---|
| 191 | 191_2_558 | 191_3_922 | 0.78 | 1.00 | baseline |
| 475 | 475_2_854 | 475_3_1326 | 0.87 | 0.84 | baseline, target |
| 527 | 527_0_0 | 527_1_267 | 1.00 | 0.80 | target |

The frame gate itself is exact on every one of the 25 pairs (baseline and
target `frame_dice_*` both 1.0000 mean/min) -- these 3 disagreements are
real differences between the `uva_vs_flat_05` (EDM2's training annotations)
and `uva_vs_v0005` (flow's annotations) source files, not a resampling bug.
EDM2's field of view (64mm cube) does not fully contain the flow's larger
crop for any of the 25 pairs (`edm2_fov_covers_flow_crop` is False on all
25), but the fraction of each target tumour lying outside EDM2's FOV is
0.0000 on every pair -- no pathology is lost to zero-padding.

### Outputs

- `analysis/intermediate/edm2_analysis/` -- full `export_flow_analysis.py`
  output (pair_metrics.csv, sample_metrics.csv, aggregate.json,
  horizon_binned.csv, method_comparison.csv, alignment_check.csv,
  viz_index.json, selected_patients.json, viz/ with arrays.npz,
  metrics.json, render_3d.gif, render_3d_slow.gif per scan).
- `analysis/intermediate/recipe.json`
- `analysis/intermediate/EDM2_vs_FlowMatching_intermediate.pptx` -- 28
  slides (2D only).
- `analysis/intermediate/EDM2_vs_FlowMatching_intermediate_3d.pptx` -- 48
  slides (28 base + 20 3D-rotation companion slides, one per shared scan,
  each immediately after its 2D slide). **115 MB -- do not commit to git**;
  regenerate from the commands below instead of trying to version it.

### Exact commands used for this run

Interpreters:
- EDM2 repo (pixi): `D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/.pixi/envs/default/python.exe`
- Flow repo (uv): `uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet python ...`

Generation (`runs/intermediate/C_gen`) was run by the evaluator, not by
this tooling -- see `runs/intermediate/C_gen/test/manifest.json` for its
exact settings (64 samples, 32 steps, seed 2026, `--data
datasets/flow_test_pairs_tf --split test`).

**1. Export** (flow interpreter, CPU, no torch/GPU used by this script):

```
uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet python \
  tools/export_flow_analysis.py \
  --generation-dir runs/intermediate/C_gen \
  --pairs-dir datasets/flow_test_pairs_tf \
  --tokens-dir D:/Work/GrowthNet_gamailab/GrowthNet/projects/vivit/tien_rivanna_repo/growth_classifier_v0005/out/vit_tokens_trainframe \
  --manifest D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/data_audit/pair_manifest.csv \
  --flow-analysis-dir D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/analysis_d040_vflow \
  --flow-config D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/hpsearch/stage22/configs/ref300.yaml \
  --out analysis/intermediate/edm2_analysis \
  --label "EDM2 (mask + time gap, no ViT tokens)"
```

(`--alignment-only` -- no `--generation-dir`/`--flow-analysis-dir` needed --
is available for a CPU-only frame/annotation/FOV check straight from a C3
pair dir, without any generation run; see `tools/export_flow_analysis.py --help`.)

**2. 3D renders** (any interpreter with `uv` on PATH; each scan spawns an
isolated `uv run --isolated --no-project --with pyvista==0.44.1 ...`):

```
<edm2-python> tools/render_edm2_3d.py --analysis-dir analysis/intermediate/edm2_analysis
```

**3. Recipe** (either interpreter; pure stdlib/numpy):

```
<edm2-python> tools/make_flow_recipe.py \
  --old-name "Flow matching (D040)" \
  --old-checkpoint D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/hpsearch/stage23/ref300/recon_step00025500/flow_ema_00025500_0.100.pt \
  --old-aggregate-json D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/analysis_d040_vflow/aggregate.json \
  --new-name "EDM2 (mask + time gap, no ViT tokens)" \
  --new-checkpoint runs/C_binary_notokens/network-snapshot-0000098-0.050.pkl \
  --new-aggregate-json analysis/intermediate/edm2_analysis/aggregate.json \
  --new-bullet "3.32M params, binary target, baseline mask + time gap conditioning, no ViT tokens" \
  --intermediate-kimg 98 \
  --edm2-gen-dir runs/intermediate/A_gen \
  --knockout-dir runs/intermediate/A_knockout_shuffle \
  --knockout-arm-label "arm A (binary target + ViT tokens), same kimg 98 checkpoint" \
  --eyebrow "PHASE 5 - INTERMEDIATE CHECKPOINT COMPARISON" \
  --takeaway "Conditioning ablation: the ViT-token arm (A, same kimg 98 checkpoint stage) scored consensus Dice 0.466 on these 25 test pairs in EDM2's own scoring frame, versus 0.576 (flow-frame headline) for the no-tokens arm shown here -- ViT tokens did not help at this checkpoint." \
  --takeaway "Phase 0 conditioning-signal probe (ViViT repo): no ViT tap showed growth-predictive signal on the frozen-encoder classifier probe -- best cross-validated ROC-AUC 0.52 vs 0.57 for the collapsed-embedding control -- consistent with tokens not improving forecast Dice here." \
  --takeaway "INTERMEDIATE checkpoint: training paused at kimg 98 of a planned 245 (selected on val); evaluated here with 64 samples per pair vs the flow model's 1000." \
  --out analysis/intermediate/recipe.json
```

`--knockout-arm-label` is required whenever `--knockout-dir` is given, and
is deliberately separate from `--new-name`: the knockout needs tokens to
shuffle, so it is commonly run on a different arm than the deck's featured
model (as here). Omit `--edm2-gen-dir`/`--knockout-dir` entirely to build
without a knockout takeaway (e.g. while the knockout run is still pending);
rerun step 3 onward once it lands.

**4. Deck build:**

```
tools/build_flow_deck.sh \
  D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet \
  D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/analysis_d040_vflow \
  analysis/intermediate/edm2_analysis \
  "Flow matching (D040)" "EDM2 (mask + time gap, no ViT tokens)" \
  analysis/intermediate/recipe.json \
  analysis/intermediate/EDM2_vs_FlowMatching_intermediate.pptx
```

**5. 3D augmentation** (flow interpreter + python-pptx; reuses the GIFs
from step 2, does not re-render):

```
uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet --with python-pptx python \
  tools/augment_deck_3d.py \
  --pptx analysis/intermediate/EDM2_vs_FlowMatching_intermediate.pptx \
  --old-analysis-dir D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/analysis_d040_vflow \
  --new-analysis-dir analysis/intermediate/edm2_analysis \
  --old-label "Flow matching (D040)" --new-label "EDM2 (mask + time gap, no ViT tokens)" \
  --out analysis/intermediate/EDM2_vs_FlowMatching_intermediate_3d.pptx
```

### To rerun for a final/later checkpoint

Same five commands, with: a fresh `--generation-dir` from
`generate_forecasts.py` (owned by the evaluator, not run by this tooling);
`--new-checkpoint`/`--new-aggregate-json` pointed at the new run;
`--intermediate-kimg` updated or dropped once training is no longer paused;
the takeaway text updated with whatever the final ablation/knockout/probe
results are at that point; and the `--out` paths given a new name (e.g.
drop `_intermediate` once this is the final comparison) so the intermediate
artifacts above aren't silently overwritten.

## Scoring reference (team-lead follow-up, 2026-09-12)

Every scored metric in `pair_metrics.csv`/`sample_metrics.csv`/`viz/*/metrics.json`
(consensus, per-sample, carry-forward, probabilistic) is computed against the
flow repo's own preprocessed baseline/target masks for the pair (built from
`--manifest`'s NIfTI paths through `tumor_flow.data.preprocessing`), never
against the ViT-frame masks resampled from the generation npz. The
resampled EDM2/ViT-frame masks are used ONLY for `alignment_check.csv`. This
is stated in every run's `aggregate.json` under `settings.scoring_reference`.

EDM2 is trained on the `uva_vs_flat_05` (T1-derived) annotations via the ViT
token store; the flow repo's masks come from `uva_vs_v0005`'s T2-thin manual
segmentations. See "Annotation-disagreement note" above for the current
known differences.

## Alignment check design (team-lead decision, 2026-09-12)

`tools/export_flow_analysis.py`'s alignment check (also available standalone
via `--alignment-only`, no generation run needed) makes two DECOUPLED
comparisons per pair, for both the baseline and target scan, so a frame bug
and an annotation-source difference can never be mistaken for each other:

1. **Frame gate** (`frame_dice_baseline` / `frame_dice_target`, strict,
   `>= 0.98` mean / `>= 0.95` min, never excused, no exceptions flag): the
   SAME `uva_vs_flat_05` source file EDM2 was conditioned/trained on, run
   through two pipelines -- the EDM2/ViT-frame resample vs the flow repo's
   own resample-to-grid + crop, reusing that pair's own reference
   grid/crop_spec (not a freshly mask-centred one). Same file, two
   pipelines: any disagreement can only be a frame-math bug.
2. **Annotation agreement** (`annotation_dice_baseline` / `annotation_dice_target`,
   logged only, never gates): the same flow-repo pipeline/grid/crop run on the
   `uva_vs_flat_05` file vs the flow manifest's `uva_vs_v0005` file. Same
   pipeline, two annotation sources: a disagreement is a data-provenance fact,
   collected into `aggregate.json`'s `settings.alignment_check.annotation_disagreements`
   rather than blocking the export.
3. **EDM2 field-of-view coverage** (`edm2_fov_covers_flow_crop`,
   `frac_target_outside_edm2_fov`, logged only): whether the flow crop's
   physical extent is fully contained in EDM2's (contract C0: 64mm cube),
   and what fraction of the flow's own target mask, if any, would be
   zero-padded as a result. See "Annotation-disagreement note" above for
   this run's values.

## Known, permanent gap

Patient 651's test pair (`651_1_1351` -> `651_3_1653`) is not in the ViT
token store's test split (the ViViT cohort excludes this patient), so it
never appears in `datasets/flow_test_pairs_tf` or the deck, regardless of
which EDM2 checkpoint is used.

## Tests

- `tests/test_prepare_pairs.py` (EDM2 interpreter): includes the
  `--manifest` mode tests, and asserts a test-only pair dir's `dataset.json`
  has only a `'test'` stats entry (no fabricated `'train'` alias).
- `tests/test_export_flow_analysis.py` (flow interpreter; skips if
  `tumor_flow` is not importable): resample/crop/ZYX-transpose known-answer
  tests for `resample_and_crop`, `load_source_mask_in_pair_frame`,
  `edm2_fov_coverage_mask`, `fraction_outside_fov`, a non-uniform-spacing
  known-answer test (contract C0), a full `--alignment-only` end-to-end CLI
  test, and a header-parity check against the real
  `analysis_d040_vflow/pair_metrics.csv`.
- `tests/test_render_edm2_3d.py` (EDM2 interpreter; does not invoke the
  real PyVista subprocess): title-resolution and `--only`-filtering logic.
- `tests/test_augment_deck_3d.py` (flow interpreter + python-pptx): GIF
  downsampling, slide-reordering, and a full synthetic end-to-end run of
  `main()` checking companion-slide placement and captioned Dice values.
- `tests/test_make_flow_recipe.py` (EDM2 interpreter): the knockout-delta
  known-answer test, the CI-straddles-zero vs CI-excludes-zero takeaway
  wording (must not overclaim a null result), and CLI argument wiring
  (`--edm2-gen-dir`/`--knockout-dir`/`--knockout-arm-label`).

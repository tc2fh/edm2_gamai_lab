#!/usr/bin/env bash
# Build the EDM2-vs-flow-matching comparison deck (docs/vivit_conditioning_plan.md
# Phase 5) with the flow repo's own deck builder. python-pptx is not a flow-repo
# dependency, so it is added ad hoc via `uv run --with` rather than the repo's
# pyproject.toml.
#
# Usage:
#   tools/build_flow_deck.sh <flow-repo-dir> <old-analysis-dir> <new-analysis-dir> \
#       <old-label> <new-label> <recipe-json> <out-pptx>
#
# Example (final run):
#   tools/build_flow_deck.sh \
#     D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet \
#     D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet/artifacts/analysis_d040_vflow \
#     analysis/edm2_analysis \
#     "Flow matching (D040)" "EDM2 + ViT tokens" \
#     analysis/recipe.json analysis/EDM2_vs_FlowMatching.pptx

set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: $0 <flow-repo-dir> <old-analysis-dir> <new-analysis-dir> <old-label> <new-label> <recipe-json> <out-pptx>" >&2
  exit 1
fi

FLOW_REPO="$1"
OLD_DIR="$2"
NEW_DIR="$3"
OLD_LABEL="$4"
NEW_LABEL="$5"
RECIPE_JSON="$6"
OUT_PPTX="$7"

uv run --project "$FLOW_REPO" --with python-pptx python -m tumor_flow.analysis.build_comparison_deck \
  --old-dir "$OLD_DIR" \
  --new-dir "$NEW_DIR" \
  --old-label "$OLD_LABEL" \
  --new-label "$NEW_LABEL" \
  --recipe-json "$RECIPE_JSON" \
  --out "$OUT_PPTX"

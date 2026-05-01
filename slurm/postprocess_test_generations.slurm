#!/bin/bash
#SBATCH -A kumarlab
#SBATCH --job-name=edm2-test-postproc
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/postproc_test_%j.out
#SBATCH --error=logs/postproc_test_%j.err

set -euo pipefail

# --- Paths (edit these before submitting) ---
PROJECT_DIR="/scratch/tc2fh/gam_ai/edm2_gam_ai"
MANIFEST="${PROJECT_DIR}/test_generation_manifest.json"
GENERATION_ROOT="/scratch/tc2fh/gam_ai/test_generations"

cd "${PROJECT_DIR}"
mkdir -p logs

module purge
module load miniforge
conda activate gam_ai

echo "=== Post-processing test generations ==="
echo "Start time:       $(date)"
echo "Manifest:         ${MANIFEST}"
echo "Generation root:  ${GENERATION_ROOT}"
echo

python3 postprocess_test_generations.py \
    --manifest "${MANIFEST}" \
    --generation-root "${GENERATION_ROOT}"

echo
echo "=== Done at $(date) ==="
echo "Probability maps and overlays saved under: ${GENERATION_ROOT}/<scan_id>/"

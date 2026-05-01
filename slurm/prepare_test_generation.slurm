#!/bin/bash
#SBATCH -A kumarlab
#SBATCH --job-name=edm2-test-manifest
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prep_test_manifest_%j.out
#SBATCH --error=logs/prep_test_manifest_%j.err

set -euo pipefail

# --- Paths (edit these before submitting) ---
PROJECT_DIR="/scratch/tc2fh/gam_ai/edm2_gam_ai"
TEST_EMB_DIR="/scratch/tc2fh/gam_ai/GrowthNet/slurm/test_data_embeddings"
UVA_SPLIT_ROOT="/scratch/tc2fh/gam_ai/uva_vs_v0003/uva_vs_v0003/split"
MANIFEST_OUT="${PROJECT_DIR}/test_generation_manifest.json"

cd "${PROJECT_DIR}"
mkdir -p logs

module purge
module load miniforge
conda activate gam_ai

echo "=== Building test generation manifest ==="
echo "Start time:       $(date)"
echo "Test embeddings:  ${TEST_EMB_DIR}"
echo "UVA split root:   ${UVA_SPLIT_ROOT}"
echo "Manifest output:  ${MANIFEST_OUT}"
echo

python3 prepare_test_generation.py \
    --test-embeddings-dir "${TEST_EMB_DIR}" \
    --uva-root "${UVA_SPLIT_ROOT}" \
    --manifest-out "${MANIFEST_OUT}"

echo
echo "=== Done at $(date) ==="
echo "Next: edit slurm/generate_test_conditional.slurm to set --array=0-N"
echo "      (N is printed above), then sbatch it."

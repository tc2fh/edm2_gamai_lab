#!/bin/bash
#SBATCH -A kumarlab
#SBATCH --job-name=edm2-test-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C gpupod
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gen_test_%A_%a.out
#SBATCH --error=logs/gen_test_%A_%a.err
#
# --- IMPORTANT: Set --array before submitting ---
# Run prepare_test_generation.slurm first. It will print the correct value of N.
# Then set: #SBATCH --array=0-N%20
# (the %20 limits concurrent tasks to 20; adjust as needed for your cluster quota)
#SBATCH --array=0-0%20

set -euo pipefail

# --- Paths (edit these before submitting) ---
PROJECT_DIR="/scratch/tc2fh/gam_ai/edm2_gam_ai"
MANIFEST="${PROJECT_DIR}/test_generation_manifest.json"
GENERATION_ROOT="/scratch/tc2fh/gam_ai/test_generations"
MODEL_PKL="/scratch/tc2fh/gam_ai/edm2_gam_ai/training-runs/<YOUR_RUN>/network-snapshot-<XXXXXX>.pkl"

cd "${PROJECT_DIR}"
mkdir -p logs "${GENERATION_ROOT}"

module purge
module load miniforge
conda activate gam_ai

echo "=== Test conditional generation ==="
echo "Start time:   $(date)"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Extract per-scan metadata from the manifest using Python
read -r SCAN_ID EMB_PATH D H W < <(python3 - <<EOF
import json, sys
manifest = json.load(open("${MANIFEST}"))
entry = manifest[${SLURM_ARRAY_TASK_ID}]
shape = entry["original_shape"]
print(entry["scan_id"], entry["emb_path"], shape[0], shape[1], shape[2])
EOF
)

OUTDIR="${GENERATION_ROOT}/${SCAN_ID}"
mkdir -p "${OUTDIR}"

echo "Scan ID:      ${SCAN_ID}"
echo "Embedding:    ${EMB_PATH}"
echo "Orig shape:   ${D},${H},${W}"
echo "Output dir:   ${OUTDIR}"
echo

python3 generate_images.py \
    --net "${MODEL_PKL}" \
    --embeddings "${EMB_PATH}" \
    --original-shape "${D},${H},${W}" \
    --outdir "${OUTDIR}" \
    --seeds 0-99 \
    --batch 10

echo
echo "=== Done at $(date) ==="
echo "Generated $(ls "${OUTDIR}"/*.nii.gz 2>/dev/null | wc -l) volumes in ${OUTDIR}"

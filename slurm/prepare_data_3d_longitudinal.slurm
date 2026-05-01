#!/bin/bash
#SBATCH -A kumarlab
#SBATCH -p standard
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=edm2-prep-long
#SBATCH --output=logs/prep_long_%j.out
#SBATCH --error=logs/prep_long_%j.err

set -euo pipefail

# --- Paths (edit these before submitting) ---
PROJECT_DIR="/scratch/tc2fh/gam_ai/edm2_gam_ai"
TRAIN_EMB_DIR="/scratch/tc2fh/gam_ai/GrowthNet/train_data_embeddings"
VAL_EMB_DIR="/scratch/tc2fh/gam_ai/GrowthNet/val_data_embeddings"
UVA_ROOT="/scratch/tc2fh/uva_vs_v0003"
OUTDIR="/scratch/tc2fh/gam_ai/edm2_gam_ai/datasets/uva_longitudinal_trainval"
RESOLUTION=128

cd "${PROJECT_DIR}"
mkdir -p logs

# Activate conda environment
module load miniforge
conda activate gam_ai

echo "=== Longitudinal dataset prep (train + val combined) ==="
echo "Start time:        $(date)"
echo "Project dir:       ${PROJECT_DIR}"
echo "Train embeddings:  ${TRAIN_EMB_DIR}"
echo "Val embeddings:    ${VAL_EMB_DIR}"
echo "UVA root:          ${UVA_ROOT}"
echo "Combined outdir:   ${OUTDIR}"
echo "Resolution:        ${RESOLUTION}"
echo

python3 prepare_data_3d_longitudinal.py \
    --train-embeddings-dir "${TRAIN_EMB_DIR}" \
    --val-embeddings-dir "${VAL_EMB_DIR}" \
    --uva-root "${UVA_ROOT}" \
    --outdir "${OUTDIR}" \
    --resolution "${RESOLUTION}"

echo
echo "=== Done ==="
echo "End time: $(date)"

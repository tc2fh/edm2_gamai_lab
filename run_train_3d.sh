#!/bin/bash
#SBATCH -A kumarlab
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH -C a100_80gb
#SBATCH --time=24:00:00

set -euo pipefail

# --- Paths ---
PROJECT_DIR="/scratch/tc2fh/gam_ai/edm2_gam_ai"
PREPARED_DATASET="/scratch/tc2fh/gam_ai/edm2_gam_ai/datasets/uva_longitudinal_trainval"
OUTDIR="/scratch/tc2fh/gam_ai/edm2_gam_ai/training-runs/edm2-vol128-xxs-longitudinal"

# --- Config ---
PRESET="edm2-vol128-xxs"
BATCH_GPU=1
SNAPSHOT="32Ki"
CHECKPOINT="32Ki"
STATUS="32Ki"

cd "${PROJECT_DIR}"

# Activate conda environment
module load miniforge
conda activate gam_ai

# --- Step 2: Train ---

start=$(date +%s)

echo "=== Starting training ==="
torchrun --standalone --nproc_per_node=1 train_edm2.py \
    --outdir="${OUTDIR}" \
    --data="${PREPARED_DATASET}" \
    --preset="${PRESET}" \
    --batch-gpu="${BATCH_GPU}" \
    --snapshot="${SNAPSHOT}" \
    --checkpoint="${CHECKPOINT}" \
    --status="${STATUS}" \
    --cond=True

echo "=== Training complete ==="

end=$(date +%s)
echo "Training took $((end - start)) seconds"
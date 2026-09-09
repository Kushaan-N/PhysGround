#!/bin/bash
#SBATCH --job-name=pg-generate
#SBATCH --array=0-19
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/generate_%A_%a.out
#SBATCH --error=slurm/logs/generate_%A_%a.err
#
# Corpus generation (spec 5, 13).
#
#   sbatch slurm/generate.sbatch base 3000 0
#   sbatch slurm/generate.sbatch occluded 1000 0
#
# --time is set explicitly. Omitting it kills jobs at the partition default with
# no informative error (spec 14), which on a preemptible array looks exactly
# like a code bug.
#
# CPU only, deliberately. Rendering is a minority of per-scene cost and a GPU
# second costs ~13x a core second for a ~2x speedup, so a GPU here is slower per
# dollar (spec 17.2). The GPU is for feature extraction and nothing else.
set -euo pipefail

CONDITION="${1:-base}"
N_SCENES="${2:-3000}"
SEED="${3:-0}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-20}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p slurm/logs

# MUST be exported before python starts: MuJoCo reads it once at import and
# setting it later silently does nothing (spec 14, 17.6).
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
# MuJoCo's internal threading fights array-level parallelism (spec 17.4.4).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

echo "[slurm] node=$(hostname) shard=${SHARD}/${N_SHARDS} condition=${CONDITION}"

# Fails loudly if this node silently fell back to software rendering (spec 17.3).
python scripts/preflight_render.py

python scripts/generate_corpus.py \
    --condition "${CONDITION}" \
    --n "${N_SCENES}" \
    --seed "${SEED}" \
    --shard "${SHARD}" \
    --n-shards "${N_SHARDS}" \
    --skip-preflight

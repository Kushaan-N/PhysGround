#!/bin/bash
#SBATCH --job-name=pg-extract
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/extract_%A_%a.out
#SBATCH --error=slurm/logs/extract_%A_%a.err
#
# Feature extraction (spec 7).
#
#   sbatch slurm/extract.sbatch dinov2_b base
#   sbatch slurm/extract.sbatch random_b  base --no-patches
#
# 16G because patch tokens for dinov2_b are the memory driver; the pooled-only
# encoders fit in far less (spec 7.3, 17.6).
set -euo pipefail

ENCODER="${1:-dinov2_b}"
CONDITION="${2:-base}"
shift 2 || true

N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-8}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p slurm/logs
export MUJOCO_GL=egl
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

echo "[slurm] node=$(hostname) shard=${SHARD}/${N_SHARDS} encoder=${ENCODER}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/extract_features.py \
    --encoder "${ENCODER}" \
    --condition "${CONDITION}" \
    --shard "${SHARD}" \
    --n-shards "${N_SHARDS}" \
    "$@"

#!/bin/bash
#SBATCH --job-name=pg-extract-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/extract_all_%j.out
#SBATCH --error=slurm/logs/extract_all_%j.err
#
# Every encoder, sequentially, on ONE GPU allocation.
#
#   sbatch slurm/extract_all.sbatch
#
# The per-encoder array (extract.sbatch) queues N GPU requests; on a busy
# cluster each re-enters the queue, and with per-shard runtimes measured in
# minutes the waits dominate the work. One allocation running shards
# back-to-back trades a little wall-clock for a single wait.
#
# Sharding is kept (8 frame-encoder shards, 4 video) rather than collapsed to
# one, for two reasons: the dinov2_b patch cache buffers ~3 GB per shard --
# one giant shard would need ~24 GB resident -- and the on-disk layout stays
# identical to the array path, so a preempted job resumes per shard instead of
# redoing an encoder. Do not re-run with different shard counts on a populated
# features directory (stale-shard trap, HANDOFF 5.7).
#
# raw_pixel needs no GPU but takes ~2 min; a separate CPU submission would
# wait in queue longer than it runs, so it rides along here.
#
# 16G: the patch-token shards are the memory driver (extract.sbatch's note).
set -euo pipefail

mkdir -p slurm/logs
export MUJOCO_GL=egl
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_encoder () {  # encoder condition n_shards extra-args...
    local enc="$1" cond="$2" n="$3"; shift 3
    for (( s=0; s<n; s++ )); do
        echo "[extract_all] ${enc}/${cond} shard ${s}/${n}"
        python scripts/extract_features.py \
            --encoder "${enc}" --condition "${cond}" \
            --shard "${s}" --n-shards "${n}" "$@"
    done
}

# dinov2_b base keeps patch tokens: Experiment S reads them (spec 8.3).
run_encoder dinov2_b   base     8
run_encoder dinov2_b   occluded 8 --no-patches
run_encoder random_b   base     8 --no-patches
run_encoder random_b   occluded 8 --no-patches
run_encoder raw_pixel  base     8 --no-patches
run_encoder raw_pixel  occluded 8 --no-patches
# Video encoders are base-only: Exp F is scene-pooled and the occlusion
# analysis cannot use a one-embedding-per-scene encoder (HANDOFF 4.4).
run_encoder videomae_b base     4 --no-patches
run_encoder vjepa2     base     4 --no-patches

echo "[extract_all] done"

#!/bin/bash
#SBATCH --job-name=pg-probes
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/probes_%A_%a.out
#SBATCH --error=slurm/logs/probes_%A_%a.err
#
# Probe grid, one array task per probe seed (spec 3.5 asks for 5).
#
#   sbatch slurm/probes.sbatch D
#   sbatch slurm/probes.sbatch A,B,C,D,E
#
# CPU only: the probes are linear algebra on cached features and never touch a
# model. 32G because the spatial contact probe holds patch tokens resident
# (spec 17.6); a pooled-only run needs a fraction of that.
set -euo pipefail

EXPS="${1:-D}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p slurm/logs
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

echo "[slurm] node=$(hostname) seed=${SEED} exps=${EXPS}"
python scripts/run_probes.py --exp "${EXPS}" --seed "${SEED}"

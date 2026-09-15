#!/bin/bash
#SBATCH --job-name=pg-figures
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/figures_%j.out
#SBATCH --error=slurm/logs/figures_%j.err
#
# Tables, hypothesis tests, and figures 1-4, recomputed from the raw
# per-item archives (spec 0.4). No model, no GPU.
#
#   sbatch --dependency=afterok:$GRID slurm/figures.sbatch
set -euo pipefail

mkdir -p slurm/logs
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID}"
python scripts/make_figures.py --seeds 0,1,2,3,4

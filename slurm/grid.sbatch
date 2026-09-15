#!/bin/bash
#SBATCH --job-name=pg-grid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/grid_%j.out
#SBATCH --error=slurm/logs/grid_%j.err
#
# Experiments C,D,E,F, all five probe seeds, in ONE job:
#
#   sbatch --dependency=afterok:$KILLSWITCH slurm/grid.sbatch
#
# --jobs 5 runs the seeds as worker processes that pin themselves to one BLAS
# thread each (HANDOFF 5.10), so five seeds cost the wall-clock of roughly one
# and six cores total -- against 40 for the five-task array of probes.sbatch,
# whose per-task BLAS cannot saturate 8 cores on these probe sizes anyway.
# OMP_NUM_THREADS=1 here is for the parent; keep probes.sbatch for the case
# where seeds should fail independently.
set -euo pipefail

mkdir -p slurm/logs
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID}"
python scripts/run_probes.py --exp C,D,E,F --seed 0,1,2,3,4 --jobs 5

#!/bin/bash
#SBATCH --job-name=pg-grid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=21
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
# 21 cores = 5 workers x 4 BLAS threads + parent. The logistic solver is
# newton-cholesky, whose Newton steps are dense gemm at ~2nd^2 flops -- that
# DOES scale to 4 threads, unlike the probe-sized gemv that made a 1-thread
# pin correct for lbfgs. OMP_NUM_THREADS=4 propagates to the workers, whose
# pin uses setdefault. Keep probes.sbatch for the case where seeds should
# fail independently.
set -euo pipefail

mkdir -p slurm/logs
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=4
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID}"
python scripts/run_probes.py --exp C,D,E,F --seed 0,1,2,3,4 --jobs 5

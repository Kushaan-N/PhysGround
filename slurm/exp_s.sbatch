#!/bin/bash
#SBATCH --job-name=pg-exp-s
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/exp_s_%j.out
#SBATCH --error=slurm/logs/exp_s_%j.err
#
# Experiment S, the spatial contact probe (spec 8.3), five seeds SERIALLY:
#
#   sbatch --dependency=afterok:$KILLSWITCH slurm/exp_s.sbatch
#
# --jobs stays 1 because each worker holds its own copy of the patch-token
# matrix, which is gigabytes -- parallel seeds here trade a wall-clock win for
# an OOM (HANDOFF 5.10). 64G, not the handoff's 32G estimate: the 23G fp16
# patch cache upcasts to fp32 in the solver (~46G resident), and 32G was
# OOM-killed during the initial load. Seeds resume individually if interrupted.
set -euo pipefail

mkdir -p slurm/logs
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID}"
python scripts/run_probes.py --exp S --seed 0,1,2,3,4 --jobs 1

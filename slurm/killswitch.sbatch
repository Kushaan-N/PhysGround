#!/bin/bash
#SBATCH --job-name=pg-killswitch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/killswitch_%j.out
#SBATCH --error=slurm/logs/killswitch_%j.err
#
# Gate G4 plus Experiments A and B, the kill-switches (spec 10): everything
# after them is spent compute if any fails, so they run alone, first, and a
# non-zero exit here leaves the dependent probe grid pending forever.
#
#   J1=$(sbatch --parsable slurm/killswitch.sbatch)
#   sbatch --dependency=afterok:$J1 slurm/probes.sbatch C,D,E,F
#
# G4 re-embeds 200 scenes with DINOv2 at fp16 and fp32 in memory -- real
# encoder inference, which is why this is a batch job and not a login-node
# command (it OOM-killed a login node).
#
# Exp A's 0.90 threshold applies at this full 3,000-scene corpus
# (prereg.md 5): a miss here is a real stop condition, not pilot noise.
set -euo pipefail

mkdir -p slurm/logs
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID}"

python scripts/run_gates.py --corpus-scenes 3000 --seed 0 --gates 4
python scripts/run_probes.py --exp A,B --seed 0

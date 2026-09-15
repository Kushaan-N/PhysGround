#!/bin/bash
#SBATCH --job-name=pg-check
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/check_%j.out
#SBATCH --error=slurm/logs/check_%j.err
#
# Corpus completeness + gates G2/G3, as the bridge between generation and
# extraction:
#
#   J2=$(sbatch --parsable --dependency=afterok:$J1 slurm/check.sbatch)
#   sbatch --dependency=afterok:$J2 slurm/extract_all.sbatch
#
# Exits non-zero on an incomplete or unpaired corpus, which leaves the
# downstream extraction job pending forever instead of spending GPU time on a
# bad corpus. A file rather than --wrap because sbatch runs wrapped commands
# under sh, where bashisms like `set -o pipefail` die before the first command.
set -euo pipefail

mkdir -p slurm/logs
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${PWD}:${PWD}/scripts:${PYTHONPATH:-}"

python scripts/check_corpus.py --condition base --expected 3000
python scripts/check_corpus.py --condition occluded --expected 1000 --paired-with base
python scripts/run_gates.py --corpus-scenes 3000 --seed 0 --gates 2,3

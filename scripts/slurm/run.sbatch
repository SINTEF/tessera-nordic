#!/bin/bash
# Run the whole pipeline for one config as a Slurm job.
#
#   sbatch scripts/slurm/run.sbatch configs/full.yaml [WORKERS]
#
# Submit from the repository root (paths come from .env there). The job is resumable:
# if it is cancelled or hits the time limit, submit it again and finished tiles are
# skipped. Each submission writes a new run directory; aggregate reuses the cache.
#SBATCH --job-name=tessera-nordic
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
config=${1:?usage: sbatch scripts/slurm/run.sbatch CONFIG [WORKERS]}
workers=${2:-${SLURM_CPUS_PER_TASK:-12}}
cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="$HOME/.local/bin:$PATH"
# Debian/Ubuntu thread defaults oversubscribe with many worker processes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "job ${SLURM_JOB_ID:-local} on $(hostname), config $config, $workers workers, $(date -Is)"
uv run --frozen tessera-nordic run "$config" --workers "$workers"

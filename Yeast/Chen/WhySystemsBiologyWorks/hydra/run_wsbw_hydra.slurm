#!/bin/bash
#SBATCH --job-name=wsbw-freqcomp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=long
#SBATCH --output=logs/wsbw_%j.out
#SBATCH --error=logs/wsbw_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs .mplconfig

# Adjust this block to match the module/conda setup on Hydra.
# If needed in a graphical/x2go session, run `source startmodules` first.
if [[ -n "${WSBW_PYTHON:-}" ]]; then
  PYTHON="${WSBW_PYTHON}"
elif [[ -x "${PROJECT_DIR}/bioevo/bin/python" ]]; then
  PYTHON="${PROJECT_DIR}/bioevo/bin/python"
elif [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_DIR}/.venv/bin/python"
elif [[ -x "${HOME}/.venvs/bioevo/bin/python" ]]; then
  PYTHON="${HOME}/.venvs/bioevo/bin/python"
elif [[ -x "${HOME}/bioevo/bin/python" ]]; then
  PYTHON="${HOME}/bioevo/bin/python"
elif [[ -n "${WSBW_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${WSBW_ENV}"
  PYTHON="python"
else
  PYTHON="python3"
fi

export MPLCONFIGDIR="${PROJECT_DIR}/.mplconfig"
if [[ -n "${WSBW_WORKERS:-}" ]]; then
  WORKERS="${WSBW_WORKERS}"
elif [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" != "1" ]]; then
  WORKERS="${SLURM_CPUS_PER_TASK}"
elif [[ -n "${SLURM_NTASKS:-}" ]]; then
  WORKERS="${SLURM_NTASKS}"
else
  WORKERS="16"
fi

echo "Using ${WORKERS} GP-map workers"

# Optional: set this to the manuscript checkout on the cluster, if present.
# export WSBW_PAPER_DIR=/path/to/PNAS-WhySystemsBiologyWorks-git

"${PYTHON}" hydra/wsbw_hydra_chunk.py \
  --samples-per-model "${WSBW_SAMPLES_PER_MODEL:-10000}" \
  --seed "${WSBW_SEED:-42}" \
  --workers "${WORKERS}" \
  --chunk-id "${WSBW_CHUNK_ID:-${SLURM_JOB_ID:-0}}" \
  --tag "${WSBW_TAG:-hydra_run}"

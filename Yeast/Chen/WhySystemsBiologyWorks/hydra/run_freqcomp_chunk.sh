#!/bin/bash
# Hydra launcher for one FreqComp-only brute-force chunk.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs .mplconfig

if [[ -n "${WSBW_PYTHON:-}" ]]; then
  PYTHON="${WSBW_PYTHON}"
elif [[ -x "${PROJECT_DIR}/bioevo/bin/python" ]]; then
  PYTHON="${PROJECT_DIR}/bioevo/bin/python"
elif [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_DIR}/.venv/bin/python"
elif [[ -x "${HOME}/bioevo/bin/python" ]]; then
  PYTHON="${HOME}/bioevo/bin/python"
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

echo "Using ${WORKERS} FreqComp-only workers"

ARGS=(
  --model "${WSBW_FC_MODEL:-chen2004}" \
  --samples "${WSBW_FC_SAMPLES:-1000000}" \
  --workers "${WORKERS}" \
  --batch-size "${WSBW_FC_BATCH_SIZE:-250}" \
  --tag "${WSBW_TAG:-freqcomp}" \
  --seed "${WSBW_SEED:-42}"
)

if [[ -n "${WSBW_FC_CHUNK_ID:-}" ]]; then
  ARGS+=(--chunk-id "${WSBW_FC_CHUNK_ID}")
fi

"${PYTHON}" hydra/wsbw_freqcomp_chunk.py "${ARGS[@]}"

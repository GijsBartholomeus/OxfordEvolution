#!/bin/bash
# Hydra launcher for the parallel NNSE random-candidate initialisation benchmark.

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

echo "Using ${WORKERS} NNSE batch workers"

ARGS=(
  --model "${WSBW_NNSE_MODEL:-chen2004}" \
  --candidates "${WSBW_NNSE_CANDIDATES:-10000}" \
  --workers "${WORKERS}" \
  --batch-size "${WSBW_NNSE_BATCH_SIZE:-250}" \
  --keep-per-bin "${WSBW_NNSE_KEEP_PER_BIN:-32}" \
  --tag "${WSBW_TAG:-nnse_batch_init}" \
  --seed "${WSBW_SEED:-42}" \
  --n-bins "${WSBW_NNSE_BINS:-50}" \
  --bin-min "${WSBW_NNSE_BIN_MIN:-1e-2}" \
  --bin-max "${WSBW_NNSE_BIN_MAX:-250.0}" \
  --bin-top "${WSBW_NNSE_BIN_TOP:-1000.0}" \
  --spacing "${WSBW_NNSE_SPACING:-log}"
)

if [[ -n "${WSBW_NNSE_CHUNK_ID:-}" ]]; then
  ARGS+=(--chunk-id "${WSBW_NNSE_CHUNK_ID}")
fi

"${PYTHON}" hydra/wsbw_nnse_batch_init.py "${ARGS[@]}"

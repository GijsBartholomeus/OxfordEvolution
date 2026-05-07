#!/bin/bash
# Hydra launcher for large-sample brute-force radius accessibility.

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

if [[ -n "${WSBW_WORKERS:-}" ]]; then
  WORKERS="${WSBW_WORKERS}"
elif [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" != "1" ]]; then
  WORKERS="${SLURM_CPUS_PER_TASK}"
elif [[ -n "${SLURM_NTASKS:-}" ]]; then
  WORKERS="${SLURM_NTASKS}"
else
  WORKERS="8"
fi

export MPLCONFIGDIR="${PROJECT_DIR}/.mplconfig"
export OMP_NUM_THREADS="${WORKERS}"
export OPENBLAS_NUM_THREADS="${WORKERS}"
export MKL_NUM_THREADS="${WORKERS}"
export NUMEXPR_NUM_THREADS="${WORKERS}"

echo "Using ${WORKERS} threads for radius accessibility"

ARGS=(
  --model "${WSBW_MODEL:-chen2004}" \
  --source-tag "${WSBW_SOURCE_TAG:-chen_bfc_1e8}" \
  --output-tag "${WSBW_OUTPUT_TAG:-chen_radius_accessibility_1e6}" \
  --max-points "${WSBW_MAX_POINTS:-1000000}" \
  --centers "${WSBW_CENTERS:-100}" \
  --center-block "${WSBW_CENTER_BLOCK:-4}" \
  --min-radius "${WSBW_MIN_RADIUS:-0.03}" \
  --n-radii "${WSBW_N_RADII:-24}" \
  --seed "${WSBW_SEED:-42}"
)

if [[ -n "${WSBW_MAX_RADIUS:-}" ]]; then
  ARGS+=(--max-radius "${WSBW_MAX_RADIUS}")
fi

if [[ "${WSBW_X_LOG:-0}" == "1" ]]; then
  ARGS+=(--x-log)
fi

"${PYTHON}" bruteforce_radius_accessibility_large.py "${ARGS[@]}"

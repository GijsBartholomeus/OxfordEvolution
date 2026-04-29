#!/bin/bash
# Hydra launcher for one parallel NNSE chain from a merged initial-population npz.

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

if [[ -z "${WSBW_NNSE_INIT_NPZ:-}" ]]; then
  echo "Set WSBW_NNSE_INIT_NPZ to the merged initial-population .npz" >&2
  exit 2
fi

echo "Using ${WORKERS} parallel NNSE workers"

"${PYTHON}" wsbw_nnse_parallel.py \
  --model "${WSBW_NNSE_MODEL:-chen2004}" \
  --init-npz "${WSBW_NNSE_INIT_NPZ}" \
  --steps "${WSBW_NNSE_STEPS:-6000}" \
  --workers "${WORKERS}" \
  --sigma "${WSBW_NNSE_SIGMA:-0.01}" \
  --seed "${WSBW_SEED:-42}" \
  --chain-id "${WSBW_NNSE_CHAIN_ID:-0}" \
  --tag "${WSBW_TAG:-nnse_parallel}" \
  --checkpoint-every "${WSBW_NNSE_CHECKPOINT_EVERY:-250}" \
  --target-empty "${WSBW_NNSE_TARGET_EMPTY:-1}" \
  --extra-steps "${WSBW_NNSE_EXTRA_STEPS:-1000}" \
  --refill-attempts "${WSBW_NNSE_REFILL_ATTEMPTS:-0}" \
  --neutral-threshold "${WSBW_NNSE_NEUTRAL_THRESHOLD:-15.0}" \
  --n-bins "${WSBW_NNSE_BINS:-50}" \
  --bin-min "${WSBW_NNSE_BIN_MIN:-1e-2}" \
  --bin-max "${WSBW_NNSE_BIN_MAX:-250.0}" \
  --bin-top "${WSBW_NNSE_BIN_TOP:-1000.0}" \
  --spacing "${WSBW_NNSE_SPACING:-log}"

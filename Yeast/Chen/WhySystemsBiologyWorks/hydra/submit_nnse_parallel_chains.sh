#!/bin/bash
set -euo pipefail

TAG="${1:-chen_nnse_parallel}"
INIT_NPZ="${2:?Usage: submit_nnse_parallel_chains.sh TAG INIT_NPZ [CHAINS] [STEPS] [CORES] [MEM_GB] [QUEUE]}"
CHAINS="${3:-25}"
STEPS="${4:-6000}"
CORES="${5:-16}"
MEM_GB="${6:-2}"
QUEUE="${7:-long}"
MODEL="${WSBW_NNSE_MODEL:-chen2004}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs results/nnse_parallel/"${TAG}"
LAUNCH_DIR="results/nnse_parallel/${TAG}/launchers"
mkdir -p "${LAUNCH_DIR}"

for chain in $(seq 0 $((CHAINS - 1))); do
  LAUNCHER="${LAUNCH_DIR}/chain_${chain}.sh"
  SEED=$((42 + chain * 100003))
  cat > "${LAUNCHER}" <<EOF
#!/bin/bash
cd "${PROJECT_DIR}"
export WSBW_TAG="${TAG}"
export WSBW_NNSE_MODEL="${MODEL}"
export WSBW_NNSE_INIT_NPZ="${INIT_NPZ}"
export WSBW_NNSE_CHAIN_ID="${chain}"
export WSBW_NNSE_STEPS="${STEPS}"
export WSBW_SEED="${SEED}"
export WSBW_WORKERS="${CORES}"
exec ./hydra/run_nnse_parallel.sh
EOF
  chmod u+x "${LAUNCHER}"
  addqueue -q "${QUEUE}" -s -c "${TAG} chain ${chain}" -m "${MEM_GB}" -n "1x${CORES}" "./${LAUNCHER}"
done

echo "Submitted ${CHAINS} NNSE chains for ${TAG}."
echo "Outputs will appear in results/nnse_parallel/${TAG}/"

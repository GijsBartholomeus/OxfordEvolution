#!/bin/bash
set -euo pipefail

TAG="${1:-chen_nnse_batch_1e6_chunked}"
MODEL="${2:-chen2004}"
CHUNKS="${3:-25}"
CANDIDATES_PER_CHUNK="${4:-40000}"
CORES="${5:-16}"
MEM_GB="${6:-2}"
QUEUE="${7:-long}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs results/nnse_batch_init/"${TAG}"
LAUNCH_DIR="results/nnse_batch_init/${TAG}/launchers"
mkdir -p "${LAUNCH_DIR}"

for chunk in $(seq 0 $((CHUNKS - 1))); do
  LAUNCHER="${LAUNCH_DIR}/chunk_${chunk}.sh"
  cat > "${LAUNCHER}" <<EOF
#!/bin/bash
cd "${PROJECT_DIR}"
export WSBW_TAG="${TAG}"
export WSBW_NNSE_MODEL="${MODEL}"
export WSBW_NNSE_CHUNK_ID="${chunk}"
export WSBW_NNSE_CANDIDATES="${CANDIDATES_PER_CHUNK}"
export WSBW_WORKERS="${CORES}"
exec ./hydra/run_nnse_batch_init.sh
EOF
  chmod u+x "${LAUNCHER}"
  addqueue -q "${QUEUE}" -s -c "${TAG} chunk ${chunk}" -m "${MEM_GB}" -n "1x${CORES}" "./${LAUNCHER}"
done

TOTAL=$((CHUNKS * CANDIDATES_PER_CHUNK))
echo "Submitted ${CHUNKS} NNSE batch chunks for ${TAG} (${TOTAL} candidates total)."
echo "After all jobs finish, run:"
echo "  python hydra/wsbw_merge_nnse_batch_chunks.py --tag ${TAG} --model ${MODEL}"

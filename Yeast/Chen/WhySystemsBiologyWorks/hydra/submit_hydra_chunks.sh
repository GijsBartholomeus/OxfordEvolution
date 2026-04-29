#!/bin/bash
set -euo pipefail

TAG="${1:-hydra_1e5}"
CHUNKS="${2:-10}"
SAMPLES_PER_MODEL="${3:-10000}"
CORES="${4:-16}"
MEM_GB="${5:-32}"
QUEUE="${6:-long}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs results/hydra_chunks/"${TAG}"
LAUNCH_DIR="results/hydra_chunks/${TAG}/launchers"
mkdir -p "${LAUNCH_DIR}"

for chunk in $(seq 0 $((CHUNKS - 1))); do
  LAUNCHER="${LAUNCH_DIR}/chunk_${chunk}.sh"
  cat > "${LAUNCHER}" <<EOF
#!/bin/bash
cd "${PROJECT_DIR}"
export WSBW_TAG="${TAG}"
export WSBW_CHUNK_ID="${chunk}"
export WSBW_SAMPLES_PER_MODEL="${SAMPLES_PER_MODEL}"
export WSBW_WORKERS="${CORES}"
exec ./hydra/run_wsbw_hydra.slurm
EOF
  chmod u+x "${LAUNCHER}"
  addqueue -q "${QUEUE}" -s -c "${TAG} chunk ${chunk}" -m "${MEM_GB}" -n "1x${CORES}" "./${LAUNCHER}"
done

echo "Submitted ${CHUNKS} chunks for ${TAG}."
echo "After all jobs finish, run:"
echo "  python hydra/wsbw_merge_hydra_chunks.py --tag ${TAG}"

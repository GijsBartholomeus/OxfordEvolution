#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 7 ]]; then
  echo "Usage: submit_bruteforce_cloud_chunks.sh TAG MODEL CHUNKS SAMPLES_PER_CHUNK CORES MEM_GB QUEUE" >&2
  exit 2
fi

TAG="$1"
MODEL="$2"
CHUNKS="$3"
SAMPLES_PER_CHUNK="$4"
CORES="$5"
MEM_GB="$6"
QUEUE="$7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs results/bruteforce_cloud/"${TAG}"
LAUNCH_DIR="results/bruteforce_cloud/${TAG}/launchers"
mkdir -p "${LAUNCH_DIR}"

for chunk in $(seq 0 $((CHUNKS - 1))); do
  LAUNCHER="${LAUNCH_DIR}/chunk_${chunk}.sh"
  cat > "${LAUNCHER}" <<EOF
#!/bin/bash
cd "${PROJECT_DIR}"
export WSBW_TAG="${TAG}"
export WSBW_BFC_MODEL="${MODEL}"
export WSBW_BFC_CHUNK_ID="${chunk}"
export WSBW_BFC_SAMPLES="${SAMPLES_PER_CHUNK}"
export WSBW_WORKERS="${CORES}"
exec ./hydra/run_bruteforce_cloud_chunk.sh
EOF
  chmod u+x "${LAUNCHER}"
  addqueue -q "${QUEUE}" -s -c "${TAG} chunk ${chunk}" -m "${MEM_GB}" -n "1x${CORES}" "./${LAUNCHER}"
done

TOTAL=$((CHUNKS * SAMPLES_PER_CHUNK))
echo "Submitted ${CHUNKS} brute-force cloud chunks for ${TAG} (${TOTAL} attempted samples total)."
echo "After all jobs finish, run:"
echo "  python hydra/wsbw_merge_bruteforce_cloud.py --tag ${TAG} --model ${MODEL}"

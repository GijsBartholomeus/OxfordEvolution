#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 7 ]]; then
  echo "Usage: submit_freqcomp_chunks.sh TAG MODEL CHUNKS SAMPLES_PER_CHUNK CORES MEM_GB QUEUE" >&2
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

mkdir -p logs results/freqcomp_chunks/"${TAG}"
LAUNCH_DIR="results/freqcomp_chunks/${TAG}/launchers"
mkdir -p "${LAUNCH_DIR}"

for chunk in $(seq 0 $((CHUNKS - 1))); do
  LAUNCHER="${LAUNCH_DIR}/chunk_${chunk}.sh"
  cat > "${LAUNCHER}" <<EOF
#!/bin/bash
cd "${PROJECT_DIR}"
export WSBW_TAG="${TAG}"
export WSBW_FC_MODEL="${MODEL}"
export WSBW_FC_CHUNK_ID="${chunk}"
export WSBW_FC_SAMPLES="${SAMPLES_PER_CHUNK}"
export WSBW_WORKERS="${CORES}"
exec ./hydra/run_freqcomp_chunk.sh
EOF
  chmod u+x "${LAUNCHER}"
  addqueue -q "${QUEUE}" -s -c "${TAG} chunk ${chunk}" -m "${MEM_GB}" -n "1x${CORES}" "./${LAUNCHER}"
done

TOTAL=$((CHUNKS * SAMPLES_PER_CHUNK))
echo "Submitted ${CHUNKS} FreqComp-only chunks for ${TAG} (${TOTAL} attempted samples total)."

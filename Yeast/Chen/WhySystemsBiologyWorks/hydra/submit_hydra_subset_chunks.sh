#!/bin/bash
set -euo pipefail

TAG="${1:?Usage: submit_hydra_subset_chunks.sh TAG MODELS_CSV [CHUNKS] [SAMPLES_PER_MODEL] [CORES] [MEM_GB] [QUEUE]}"
MODELS_CSV="${2:?Usage: submit_hydra_subset_chunks.sh TAG MODELS_CSV [CHUNKS] [SAMPLES_PER_MODEL] [CORES] [MEM_GB] [QUEUE]}"
CHUNKS="${3:-100}"
SAMPLES_PER_MODEL="${4:-1000000}"
CORES="${5:-16}"
MEM_GB="${6:-32}"
QUEUE="${7:-long}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"

mkdir -p logs results/hydra_chunks/"${TAG}"
LAUNCH_DIR="results/hydra_chunks/${TAG}/launchers"
mkdir -p "${LAUNCH_DIR}"

for chunk in $(seq 0 $((CHUNKS - 1))); do
  LAUNCHER="${LAUNCH_DIR}/chunk_${chunk}.sh"
  {
    echo "#!/bin/bash"
    echo "set -euo pipefail"
    echo "cd \"${PROJECT_DIR}\""
    echo "export MPLCONFIGDIR=\"${PROJECT_DIR}/.mplconfig\""
    echo "PYTHON=\"${PROJECT_DIR}/bioevo/bin/python\""
    echo 'if [[ ! -x "${PYTHON}" ]]; then PYTHON="python3"; fi'
    echo "\"\${PYTHON}\" hydra/wsbw_hydra_chunk.py \\"
    echo "  --samples-per-model \"${SAMPLES_PER_MODEL}\" \\"
    echo "  --seed 42 \\"
    echo "  --workers \"${CORES}\" \\"
    echo "  --chunk-id \"${chunk}\" \\"
    echo "  --tag \"${TAG}\" \\"
    echo "  --models ${MODELS[*]}"
  } > "${LAUNCHER}"
  chmod u+x "${LAUNCHER}"
  addqueue -q "${QUEUE}" -s -c "${TAG} chunk ${chunk}" -m "${MEM_GB}" -n "1x${CORES}" "./${LAUNCHER}"
done

echo "Submitted ${CHUNKS} subset chunks for ${TAG}."
echo "Models: ${MODELS[*]}"
echo "After all jobs finish, run:"
echo "  python hydra/wsbw_merge_hydra_chunks.py --tag ${TAG} --models ${MODELS[*]}"

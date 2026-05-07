#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 8 ]]; then
  echo "Usage: submit_radius_accessibility_large.sh OUTPUT_TAG MODEL SOURCE_TAG MAX_POINTS CENTERS CORES MEM_GB_PER_CORE QUEUE [X_LOG=0]" >&2
  exit 2
fi

OUTPUT_TAG="$1"
MODEL="$2"
SOURCE_TAG="$3"
MAX_POINTS="$4"
CENTERS="$5"
CORES="$6"
MEM_GB_PER_CORE="$7"
QUEUE="$8"
X_LOG="${9:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs results_summaries/bruteforce_cloud/"${OUTPUT_TAG}"/launchers
LAUNCHER="results_summaries/bruteforce_cloud/${OUTPUT_TAG}/launchers/radius_accessibility.sh"

cat > "${LAUNCHER}" <<EOF
#!/bin/bash
cd "${PROJECT_DIR}"
export WSBW_OUTPUT_TAG="${OUTPUT_TAG}"
export WSBW_MODEL="${MODEL}"
export WSBW_SOURCE_TAG="${SOURCE_TAG}"
export WSBW_MAX_POINTS="${MAX_POINTS}"
export WSBW_CENTERS="${CENTERS}"
export WSBW_WORKERS="${CORES}"
export WSBW_X_LOG="${X_LOG}"
exec ./hydra/run_radius_accessibility_large.sh
EOF

chmod u+x "${LAUNCHER}"
addqueue -q "${QUEUE}" -s -c "${OUTPUT_TAG}" -m "${MEM_GB_PER_CORE}" -n "1x${CORES}" "./${LAUNCHER}"

echo "Submitted radius accessibility job ${OUTPUT_TAG}."
echo "Outputs will appear in:"
echo "  figures/bruteforce_cloud/${OUTPUT_TAG}/radius_accessibility_large/"
echo "  results_summaries/bruteforce_cloud/${OUTPUT_TAG}/radius_accessibility_large/"

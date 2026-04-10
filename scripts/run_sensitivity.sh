#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

MODEL="${1:-instanovo}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/results/sensitivity/${MODEL}}"
CONFIG="${3:-}"
DEVICE="${4:-cuda}"
MAX_SAMPLES="${5:-}"
EXTRA_ARGS=()

if [[ -n "${MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

case "${MODEL}" in
  instanovo)
    DEFAULT_CONFIG="${PROJECT_ROOT}/configs/baseline_instanovo.yaml"
    DATA_PATH="${WORKSPACE_ROOT}/dataset/hc_pt/test.parquet"
    SCALE_FACTORS="0.990 0.992 0.994 0.996 0.998 0.999 1.000 1.001 1.002 1.004 1.006 1.008 1.010"
    ;;
  casanovo)
    DEFAULT_CONFIG="${PROJECT_ROOT}/configs/baseline_casanovo.yaml"
    DATA_PATH="${WORKSPACE_ROOT}/dataset/novobench/test.parquet"
    SCALE_FACTORS="0.1 0.2 0.5 1.0 1.5 2.0 3.0 5.0 10.0"
    ;;
  *)
    echo "Unsupported model: ${MODEL}" >&2
    exit 1
    ;;
esac

if [[ -z "${CONFIG}" ]]; then
  CONFIG="${DEFAULT_CONFIG}"
fi

mkdir -p "${OUTPUT_DIR}"
RESULTS_JSON="${OUTPUT_DIR}/${MODEL}_sensitivity_results.json"
ANALYSIS_JSON="${OUTPUT_DIR}/${MODEL}_analysis.json"
CURVES_PDF="${OUTPUT_DIR}/${MODEL}_sensitivity_curves.pdf"

echo "Running sensitivity scaling experiment"
echo "Model: ${MODEL}"
echo "Config: ${CONFIG}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"

python "${PROJECT_ROOT}/sensitivity_scaling/experiment.py" \
  --model "${MODEL}" \
  --config "${CONFIG}" \
  --data "${DATA_PATH}" \
  --output "${RESULTS_JSON}" \
  --modality both \
  --scale-factors ${SCALE_FACTORS} \
  "${EXTRA_ARGS[@]}" \
  --device "${DEVICE}"

python "${PROJECT_ROOT}/sensitivity_scaling/analyze.py" \
  --input "${RESULTS_JSON}" \
  --output "${ANALYSIS_JSON}"

python "${PROJECT_ROOT}/sensitivity_scaling/visualize.py" \
  --input "${RESULTS_JSON}" \
  --output "${CURVES_PDF}" \
  --metric aa_precision

echo "Done. Results saved to ${OUTPUT_DIR}"

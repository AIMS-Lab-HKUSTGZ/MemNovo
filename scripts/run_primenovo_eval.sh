#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

PEAK_PATH="${1:?peak_path required}"
OUTPUT_BASE="${2:-${PROJECT_ROOT}/results/primenovo/primenovo_eval}"
CONFIG_PATH="${3:-${PROJECT_ROOT}/external/primenovo/config.yaml}"
MODEL_PATH="${4:-${WORKSPACE_ROOT}/weights/model_massive.ckpt}"

mkdir -p "$(dirname "${OUTPUT_BASE}")"

PYTHONPATH="${PROJECT_ROOT}/external${PYTHONPATH:+:${PYTHONPATH}}" \
python -m primenovo.PrimeNovo \
  --mode=eval \
  --peak_path="${PEAK_PATH}" \
  --model="${MODEL_PATH}" \
  --config="${CONFIG_PATH}" \
  --output="${OUTPUT_BASE}"

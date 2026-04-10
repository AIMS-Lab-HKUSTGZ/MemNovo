#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

SUITE_NAME="${1:?suite name required}"
CONFIG_PATH="${2:?config path required}"
RESULTS_DIR="${3:?results dir required}"
DEVICE="${4:-cuda}"
MAX_SAMPLES="${MEMNOVO_MAX_SAMPLES:-}"
FORCE_REDO="${MEMNOVO_FORCE_REDO:-0}"
EXTRA_ARGS=()

if [[ -n "${MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

mkdir -p "${RESULTS_DIR}"

SPECIES=(
  "Bacillus-subtilis:${WORKSPACE_ROOT}/dataset/NS2/Bacillus-subtilis.mgf"
  "Saccharomyces-cerevisiae:${WORKSPACE_ROOT}/dataset/NS1/Saccharomyces-cerevisiae.mgf"
  "Methanosarcina-mazei:${WORKSPACE_ROOT}/dataset/NS1/Methanosarcina-mazei.mgf"
  "Apis-mellifera:${WORKSPACE_ROOT}/dataset/NS3/Apis-mellifera.mgf"
  "Solanum-lycopersicum:${WORKSPACE_ROOT}/dataset/NS3/Solanum-lycopersicum.mgf"
  "Vigna-mungo:${WORKSPACE_ROOT}/dataset/NS3/Vigna-mungo.mgf"
  "Candidatus-endoloripes:${WORKSPACE_ROOT}/dataset/NS3/Candidatus-endoloripes.mgf"
  "H.-sapiens:${WORKSPACE_ROOT}/dataset/NS3/H.-sapiens.mgf"
  "Mus-musculus:${WORKSPACE_ROOT}/dataset/NS3/Mus-musculus.mgf"
)

echo "Running suite: ${SUITE_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Results dir: ${RESULTS_DIR}"
echo "Device: ${DEVICE}"

for entry in "${SPECIES[@]}"; do
  species="${entry%%:*}"
  input_path="${entry#*:}"
  metrics_path="${RESULTS_DIR}/${species}.metrics.json"
  output_path="${RESULTS_DIR}/${species}.jsonl"

  if [[ "${FORCE_REDO}" != "1" && -s "${metrics_path}" && -s "${output_path}" ]]; then
    echo
    echo "[${SUITE_NAME}] ${species} (skip: existing results)"
    continue
  fi

  echo
  echo "[${SUITE_NAME}] ${species}"
  python "${PROJECT_ROOT}/scripts/run_inference.py" \
    --config "${PROJECT_ROOT}/${CONFIG_PATH}" \
    --input "${input_path}" \
    --output "${output_path}" \
    --metrics-output "${metrics_path}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}" \
    --evaluate
done

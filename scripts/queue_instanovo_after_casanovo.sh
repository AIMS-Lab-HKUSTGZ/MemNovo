#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

CASANOVO_BASELINE_PID="${1:?casa baseline pid required}"
CASANOVO_MEMNOVO_PID="${2:?casa memnovo pid required}"
POLL_SECONDS="${3:-60}"

RESULTS_ROOT="${PROJECT_ROOT}/results/paper_repro/main"
LOG_ROOT="${PROJECT_ROOT}/results/paper_repro/logs"

wait_for_pid() {
  local pid="$1"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
}

echo "Waiting for Casanovo jobs to finish: ${CASANOVO_BASELINE_PID}, ${CASANOVO_MEMNOVO_PID}"
wait_for_pid "${CASANOVO_BASELINE_PID}"
wait_for_pid "${CASANOVO_MEMNOVO_PID}"

echo "Casanovo jobs finished. Starting queued InstaNovo reruns."

CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL bash "${PROJECT_ROOT}/scripts/run_nine_species_suite.sh" \
  instanovo_baseline \
  configs/baseline_instanovo.yaml \
  "${RESULTS_ROOT}/instanovo_baseline" \
  cuda 2>&1 | tee "${LOG_ROOT}/instanovo_baseline_rerun.log"

CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL bash "${PROJECT_ROOT}/scripts/run_nine_species_suite.sh" \
  instanovo_memnovo \
  configs/memnovo_instanovo.yaml \
  "${RESULTS_ROOT}/instanovo_memnovo" \
  cuda 2>&1 | tee "${LOG_ROOT}/instanovo_memnovo_rerun.log"

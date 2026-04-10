#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

INPUT_PATH="${1:?subset mgf required}"
RESULTS_DIR="${2:?results dir required}"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/pids"

launch_run() {
  local run_name="$1"
  local gpu_id="$2"
  local config_rel="$3"
  local log_path="${RESULTS_DIR}/logs/${run_name}.log"
  local output_path="${RESULTS_DIR}/${run_name}.jsonl"
  local metrics_path="${RESULTS_DIR}/${run_name}.metrics.json"
  local pid_path="${RESULTS_DIR}/pids/${run_name}.pid"

  if [[ -s "${metrics_path}" && -s "${output_path}" ]]; then
    echo "[skip] ${run_name}: existing outputs found"
    return
  fi

  echo "[launch] ${run_name} on GPU ${gpu_id}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python "${PROJECT_ROOT}/scripts/run_inference.py" \
      --config "${PROJECT_ROOT}/${config_rel}" \
      --input "${INPUT_PATH}" \
      --output "${output_path}" \
      --metrics-output "${metrics_path}" \
      --device cuda \
      --evaluate \
      >"${log_path}" 2>&1 &
  echo $! >"${pid_path}"
  echo "  pid=$(cat "${pid_path}")"
  echo "  log=${log_path}"
}

launch_run "casanovo_baseline" "0" "configs/baseline_casanovo.yaml"
launch_run "casanovo_memnovo" "1" "configs/memnovo_casanovo.yaml"
launch_run "instanovo_baseline" "2" "configs/baseline_instanovo.yaml"
launch_run "instanovo_memnovo" "3" "configs/memnovo_instanovo.yaml"

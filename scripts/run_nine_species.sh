#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RESULTS_DIR="${1:-${PROJECT_ROOT}/results/nine_species}"
MAX_SAMPLES="${MEMNOVO_MAX_SAMPLES:-}"
EXTRA_ARGS=()

if [[ -n "${MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

mkdir -p "${RESULTS_DIR}"

declare -A SPECIES_FILES=(
  ["Bacillus-subtilis"]="${WORKSPACE_ROOT}/dataset/NS2/Bacillus-subtilis.mgf"
  ["Saccharomyces-cerevisiae"]="${WORKSPACE_ROOT}/dataset/NS1/Saccharomyces-cerevisiae.mgf"
  ["Methanosarcina-mazei"]="${WORKSPACE_ROOT}/dataset/NS1/Methanosarcina-mazei.mgf"
  ["Apis-mellifera"]="${WORKSPACE_ROOT}/dataset/NS3/Apis-mellifera.mgf"
  ["Solanum-lycopersicum"]="${WORKSPACE_ROOT}/dataset/NS3/Solanum-lycopersicum.mgf"
  ["Vigna-mungo"]="${WORKSPACE_ROOT}/dataset/NS3/Vigna-mungo.mgf"
  ["Candidatus-endoloripes"]="${WORKSPACE_ROOT}/dataset/NS3/Candidatus-endoloripes.mgf"
  ["H.-sapiens"]="${WORKSPACE_ROOT}/dataset/NS3/H.-sapiens.mgf"
  ["Mus-musculus"]="${WORKSPACE_ROOT}/dataset/NS3/Mus-musculus.mgf"
)

run_suite() {
  local suite_name="$1"
  local config_path="$2"
  local suite_dir="${RESULTS_DIR}/${suite_name}"
  mkdir -p "${suite_dir}"

  echo "========================================="
  echo "Running ${suite_name}"
  echo "Config: ${config_path}"
  echo "========================================="

  for species in "${!SPECIES_FILES[@]}"; do
    local input_path="${SPECIES_FILES[$species]}"
    echo
    echo "[${suite_name}] ${species}"
    python "${PROJECT_ROOT}/scripts/run_inference.py" \
      --config "${PROJECT_ROOT}/${config_path}" \
      --input "${input_path}" \
      --output "${suite_dir}/${species}.jsonl" \
      --metrics-output "${suite_dir}/${species}.metrics.json" \
      "${EXTRA_ARGS[@]}" \
      --evaluate
  done
}

run_suite "instanovo_baseline" "configs/baseline_instanovo.yaml"
run_suite "instanovo_memnovo" "configs/memnovo_instanovo.yaml"
run_suite "casanovo_baseline" "configs/baseline_casanovo.yaml"
run_suite "casanovo_memnovo" "configs/memnovo_casanovo.yaml"

python - <<'PY' "${RESULTS_DIR}"
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
suites = [
    "instanovo_baseline",
    "instanovo_memnovo",
    "casanovo_baseline",
    "casanovo_memnovo",
]

summary = {}
for suite in suites:
    metrics = []
    for path in sorted((results_dir / suite).glob("*.metrics.json")):
        metrics.append(json.loads(path.read_text()))
    if metrics:
        summary[suite] = {
            "aa_precision_avg": sum(item["aa_precision"] for item in metrics) / len(metrics),
            "pep_precision_avg": sum(item["pep_precision"] for item in metrics) / len(metrics),
            "species": len(metrics),
        }

summary_path = results_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Saved summary to {summary_path}")
PY

#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/data/private/instanovo"
PROJECT="$ROOT/MemNovo"
RUN_ROOT="$PROJECT/results/primenovo_full_nine_species"
LOG_MD="$ROOT/primenovo_log.md"
SPECTRA_PER_SHARD="${MEMNOVO_SPECTRA_PER_SHARD:-50000}"

SPECIES=(
  "Bacillus-subtilis:${ROOT}/dataset/NS2/Bacillus-subtilis.mgf"
  "Saccharomyces-cerevisiae:${ROOT}/dataset/NS1/Saccharomyces-cerevisiae.mgf"
  "Methanosarcina-mazei:${ROOT}/dataset/NS1/Methanosarcina-mazei.mgf"
  "Apis-mellifera:${ROOT}/dataset/NS3/Apis-mellifera.mgf"
  "Solanum-lycopersicum:${ROOT}/dataset/NS3/Solanum-lycopersicum.mgf"
  "Vigna-mungo:${ROOT}/dataset/NS3/Vigna-mungo.mgf"
  "Candidatus-endoloripes:${ROOT}/dataset/NS3/Candidatus-endoloripes.mgf"
  "H.-sapiens:${ROOT}/dataset/NS3/H.-sapiens.mgf"
  "Mus-musculus:${ROOT}/dataset/NS3/Mus-musculus.mgf"
)

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

append_md() {
  printf '%s\n' "$1" >> "$LOG_MD"
}

run_track() {
  local name="$1"
  local config_path="$2"
  local results_dir="$3"
  local gpu_id="$4"
  local track_log="$RUN_ROOT/logs/${name}.stable.log"

  mkdir -p "$results_dir" "$results_dir/logs" "$RUN_ROOT/logs"
  : > "$track_log"

  {
    echo "[$(timestamp)] start ${name} gpu=${gpu_id}"

    for entry in "${SPECIES[@]}"; do
      local species="${entry%%:*}"
      local input_path="${entry#*:}"
      local output_path="${results_dir}/${species}.jsonl"
      local metrics_path="${results_dir}/${species}.metrics.json"

      if [[ -s "$output_path" && -s "$metrics_path" ]]; then
        echo "[$(timestamp)] skip ${name} ${species}"
        append_md "- ${name} skip \`${species}\` at \`$(timestamp)\`"
        continue
      fi

      echo "[$(timestamp)] run ${name} ${species}"
      append_md "- ${name} start \`${species}\` at \`$(timestamp)\`"

      CUDA_VISIBLE_DEVICES="$gpu_id" python "$PROJECT/scripts/run_species_sharded.py" \
        --config "$PROJECT/$config_path" \
        --input "$input_path" \
        --output "$output_path" \
        --metrics-output "$metrics_path" \
        --gpus 0 \
        --device cuda \
        --spectra-per-shard "$SPECTRA_PER_SHARD" \
        --evaluate \
        --log-dir "$results_dir/logs"

      python - "$metrics_path" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
metrics = json.loads(path.read_text())
print(
    f"[metrics] aa_precision={metrics['aa_precision']:.6f} "
    f"aa_recall={metrics['aa_recall']:.6f} "
    f"pep_precision={metrics['pep_precision']:.6f} "
    f"pep_recall={metrics['pep_recall']:.6f}"
)
PY

      append_md "- ${name} finish \`${species}\` at \`$(timestamp)\`"
    done

    echo "[$(timestamp)] done ${name}"
  } >> "$track_log" 2>&1
}

mkdir -p "$RUN_ROOT/logs"

append_md ""
append_md "## PrimeNovo Stable Resume Queue Start"
append_md ""
append_md "- start: \`$(timestamp)\`"
append_md "- gpu policy: \`GPU2 baseline / GPU3 memnovo\`"
append_md "- spectra per shard: \`${SPECTRA_PER_SHARD}\`"
append_md "- orchestration: \`species-by-species stable resume\`"

run_track "primenovo_baseline" "configs/baseline_primenovo.yaml" "$RUN_ROOT/baseline" 2 &
PID_BASE=$!
run_track "primenovo_memnovo" "configs/memnovo_primenovo.yaml" "$RUN_ROOT/memnovo" 3 &
PID_MEM=$!

append_md "- pid baseline: \`${PID_BASE}\`"
append_md "- pid memnovo: \`${PID_MEM}\`"
append_md "- baseline log: \`$RUN_ROOT/logs/primenovo_baseline.stable.log\`"
append_md "- memnovo log: \`$RUN_ROOT/logs/primenovo_memnovo.stable.log\`"

STATUS_BASE=0
STATUS_MEM=0
wait "$PID_BASE" || STATUS_BASE=$?
wait "$PID_MEM" || STATUS_MEM=$?

append_md "- finish: \`$(timestamp)\`"
append_md "- exit baseline: \`${STATUS_BASE}\`"
append_md "- exit memnovo: \`${STATUS_MEM}\`"

if [[ "$STATUS_BASE" -ne 0 || "$STATUS_MEM" -ne 0 ]]; then
  append_md "- queue status: \`failed\`"
  exit 1
fi

append_md "- queue status: \`ok\`"

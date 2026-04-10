#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/data/private/instanovo"
PROJECT="$ROOT/MemNovo"
LOG_MD="$ROOT/primenovo_log.md"
RUN_ROOT="$PROJECT/results/primenovo_full_hybrid_best"
BASE_DIR="$RUN_ROOT/baseline_beams"
CONFIRM_DIR="$RUN_ROOT/hybrid_confirm"
LOG_DIR="$RUN_ROOT/logs"

mkdir -p "$BASE_DIR" "$CONFIRM_DIR" "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

append_md() {
  printf '%s\n' "$1" >> "$LOG_MD"
}

species_path() {
  case "$1" in
    "Bacillus-subtilis") echo "$ROOT/dataset/NS2/Bacillus-subtilis.mgf" ;;
    "Saccharomyces-cerevisiae") echo "$ROOT/dataset/NS1/Saccharomyces-cerevisiae.mgf" ;;
    "Methanosarcina-mazei") echo "$ROOT/dataset/NS1/Methanosarcina-mazei.mgf" ;;
    "Apis-mellifera") echo "$ROOT/dataset/NS3/Apis-mellifera.mgf" ;;
    "Solanum-lycopersicum") echo "$ROOT/dataset/NS3/Solanum-lycopersicum.mgf" ;;
    "Vigna-mungo") echo "$ROOT/dataset/NS3/Vigna-mungo.mgf" ;;
    "Candidatus-endoloripes") echo "$ROOT/dataset/NS3/Candidatus-endoloripes.mgf" ;;
    "H.-sapiens") echo "$ROOT/dataset/NS3/H.-sapiens.mgf" ;;
    "Mus-musculus") echo "$ROOT/dataset/NS3/Mus-musculus.mgf" ;;
    *)
      echo "unknown species: $1" >&2
      return 1
      ;;
  esac
}

run_one() {
  local gpu="$1"
  local species="$2"
  local spectra
  spectra="$(species_path "$species")"
  local base_json="$BASE_DIR/${species}.jsonl"
  local base_metrics="$BASE_DIR/${species}.metrics.json"
  local confirm_json="$CONFIRM_DIR/${species}.hybrid_bestconfirm.json"

  echo "[$(timestamp)] GPU${gpu} start ${species}"

  python "$PROJECT/scripts/run_inference.py" \
    --config "$PROJECT/configs/baseline_primenovo_beams.yaml" \
    --input "$spectra" \
    --output "$base_json" \
    --device "cuda:${gpu}" \
    --evaluate \
    --metrics-output "$base_metrics"

  python "$PROJECT/scripts/sweep_hybrid_rerank.py" \
    --model primenovo \
    --baseline "$base_json" \
    --spectra "$spectra" \
    --output "$confirm_json" \
    --alphas "0.75" \
    --spec-gaps "0.1" \
    --decoder-margins "0.7" \
    --confidence-thresholds "0.8" \
    --ion-mode "both" \
    --gate-requires-disagreement

  echo "[$(timestamp)] GPU${gpu} done ${species}"
}

worker() {
  local gpu="$1"
  shift
  local worker_log="$LOG_DIR/gpu${gpu}.log"
  : > "$worker_log"
  for species in "$@"; do
    run_one "$gpu" "$species" >> "$worker_log" 2>&1
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker "$@"
  exit 0
fi

append_md ""
append_md "## 2026-04-08 PrimeNovo Full Hybrid Best Queue"
append_md ""
append_md "- start: \`$(timestamp)\`"
append_md "- policy: \`GPU2/GPU3 full-species baseline-beam export + fixed hybrid confirm\`"
append_md "- best config:"
append_md "  - \`alpha=0.75\`"
append_md "  - \`spec_gap_threshold=0.1\`"
append_md "  - \`decoder_margin_threshold=0.7\`"
append_md "  - \`confidence_threshold=0.8\`"
append_md "  - \`ion_mode=both\`"
append_md "- baseline beam dir: \`$BASE_DIR\`"
append_md "- confirm dir: \`$CONFIRM_DIR\`"

bash "$0" --worker 2 "Bacillus-subtilis" "Vigna-mungo" "Candidatus-endoloripes" "H.-sapiens" > "$LOG_DIR/worker_gpu2.nohup.log" 2>&1 &
PID2=$!
bash "$0" --worker 3 "Saccharomyces-cerevisiae" "Methanosarcina-mazei" "Apis-mellifera" "Solanum-lycopersicum" "Mus-musculus" > "$LOG_DIR/worker_gpu3.nohup.log" 2>&1 &
PID3=$!

append_md "- pid gpu2 worker: \`$PID2\`"
append_md "- pid gpu3 worker: \`$PID3\`"
append_md "- gpu2 log: \`$LOG_DIR/worker_gpu2.nohup.log\`"
append_md "- gpu3 log: \`$LOG_DIR/worker_gpu3.nohup.log\`"

wait "$PID2"
STATUS2=$?
wait "$PID3"
STATUS3=$?

append_md "- finish: \`$(timestamp)\`"
append_md "- exit gpu2 worker: \`$STATUS2\`"
append_md "- exit gpu3 worker: \`$STATUS3\`"

if [[ "$STATUS2" -ne 0 || "$STATUS3" -ne 0 ]]; then
  append_md "- queue status: \`failed\`"
  exit 1
fi

append_md "- queue status: \`ok\`"

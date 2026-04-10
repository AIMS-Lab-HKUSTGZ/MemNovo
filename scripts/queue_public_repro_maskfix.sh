#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/data/private/instanovo"
PROJECT="$ROOT/MemNovo"
LOG_MD="$ROOT/public_repro_maskfix_log.md"
RUN_ROOT="$PROJECT/results/public_repro_maskfix"
LOG_DIR="$RUN_ROOT/logs"

CAS_INPUT="$PROJECT/results/diagnostic_subset/weighted_50k_seed20260405.mgf"
INST_INPUT="$PROJECT/results/weighted_subset_150k/nine_species_weighted_150000.mgf"
INST_SUBSET="0.0333333333"

mkdir -p "$LOG_DIR" "$RUN_ROOT/casanovo" "$RUN_ROOT/instanovo"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

append_md() {
  printf '%s\n' "$1" >> "$LOG_MD"
}

append_md ""
append_md "## Public Repro Maskfix Queue Start"
append_md ""
append_md "- start: $(timestamp)"
append_md "- gpu policy: GPU0/1 only"
append_md "- goal: Casanovo 50k baseline/memnovo + InstaNovo official beam5 5k baseline/memnovo"
append_md "- casanovo input: $CAS_INPUT"
append_md "- instanovo input: $INST_INPUT"
append_md "- instanovo subset: $INST_SUBSET"
append_md "- outputs root: $RUN_ROOT"

run_pair() {
  local stage_name="$1"
  local cmd_a="$2"
  local log_a="$3"
  local out_a="$4"
  local metrics_a="$5"
  local cmd_b="$6"
  local log_b="$7"
  local out_b="$8"
  local metrics_b="$9"

  append_md ""
  append_md "### $stage_name"
  append_md ""
  append_md "- started: $(timestamp)"
  append_md "- cmd gpu0: $cmd_a"
  append_md "- cmd gpu1: $cmd_b"

  bash -lc "$cmd_a" > "$log_a" 2>&1 &
  local pid_a=$!
  bash -lc "$cmd_b" > "$log_b" 2>&1 &
  local pid_b=$!

  append_md "- pid gpu0: $pid_a"
  append_md "- pid gpu1: $pid_b"
  append_md "- log gpu0: $log_a"
  append_md "- log gpu1: $log_b"
  append_md "- output gpu0: $out_a"
  append_md "- output gpu1: $out_b"
  append_md "- metrics gpu0: $metrics_a"
  append_md "- metrics gpu1: $metrics_b"

  local status_a=0
  local status_b=0
  wait "$pid_a" || status_a=$?
  wait "$pid_b" || status_b=$?

  append_md "- finished: $(timestamp)"
  append_md "- exit gpu0: $status_a"
  append_md "- exit gpu1: $status_b"
  if [[ "$status_a" -ne 0 || "$status_b" -ne 0 ]]; then
    append_md "- stage status: failed"
    return 1
  fi

  append_md "- stage status: ok"
}

CAS_BASE_OUT="$RUN_ROOT/casanovo/baseline_50k.jsonl"
CAS_MEM_OUT="$RUN_ROOT/casanovo/memnovo_50k.jsonl"
CAS_BASE_METRICS="$RUN_ROOT/casanovo/baseline_50k.metrics.json"
CAS_MEM_METRICS="$RUN_ROOT/casanovo/memnovo_50k.metrics.json"
run_pair \
  "Casanovo Maskfix 50k" \
  "CUDA_VISIBLE_DEVICES=0 python $PROJECT/scripts/run_inference.py --config $PROJECT/configs/baseline_casanovo.yaml --input $CAS_INPUT --output $CAS_BASE_OUT --metrics-output $CAS_BASE_METRICS --device cuda --evaluate --log-level INFO" \
  "$LOG_DIR/casanovo_baseline.log" \
  "$CAS_BASE_OUT" \
  "$CAS_BASE_METRICS" \
  "CUDA_VISIBLE_DEVICES=1 python $PROJECT/scripts/run_inference.py --config $PROJECT/configs/memnovo_casanovo.yaml --input $CAS_INPUT --output $CAS_MEM_OUT --metrics-output $CAS_MEM_METRICS --device cuda --evaluate --log-level INFO" \
  "$LOG_DIR/casanovo_memnovo.log" \
  "$CAS_MEM_OUT" \
  "$CAS_MEM_METRICS"

python - <<'PY' >> "$LOG_MD"
import json
from pathlib import Path
root = Path("/opt/data/private/instanovo/MemNovo/results/public_repro_maskfix/casanovo")
for name in ["baseline_50k", "memnovo_50k"]:
    metrics = json.loads((root / f"{name}.metrics.json").read_text())
    print("")
    print(f"- {name} first metrics:")
    print(f"  - aa_precision: {metrics['aa_precision']:.6f}")
    print(f"  - aa_recall: {metrics['aa_recall']:.6f}")
    print(f"  - pep_precision: {metrics['pep_precision']:.6f}")
    print(f"  - pep_recall: {metrics['pep_recall']:.6f}")
PY

INST_BASE_OUT="$RUN_ROOT/instanovo/baseline_beam5_5k.csv"
INST_MEM_OUT="$RUN_ROOT/instanovo/memnovo_beam5_5k.csv"
INST_BASE_METRICS="$RUN_ROOT/instanovo/baseline_beam5_5k.metrics.json"
INST_MEM_METRICS="$RUN_ROOT/instanovo/memnovo_beam5_5k.metrics.json"
run_pair \
  "InstaNovo Official Beam5 5k Maskfix" \
  "CUDA_VISIBLE_DEVICES=0 python $PROJECT/scripts/run_instanovo_official.py --config $PROJECT/configs/baseline_instanovo.yaml --input $INST_INPUT --output $INST_BASE_OUT --metrics-output $INST_BASE_METRICS --device cuda --subset $INST_SUBSET --batch-size 128 --beam-size 5 --fp16 --no-use-knapsack --save-beams --log-interval 2 --log-level INFO" \
  "$LOG_DIR/instanovo_baseline.log" \
  "$INST_BASE_OUT" \
  "$INST_BASE_METRICS" \
  "CUDA_VISIBLE_DEVICES=1 python $PROJECT/scripts/run_instanovo_official.py --config $PROJECT/configs/memnovo_instanovo.yaml --input $INST_INPUT --output $INST_MEM_OUT --metrics-output $INST_MEM_METRICS --device cuda --subset $INST_SUBSET --batch-size 128 --beam-size 5 --fp16 --no-use-knapsack --save-beams --log-interval 2 --log-level INFO" \
  "$LOG_DIR/instanovo_memnovo.log" \
  "$INST_MEM_OUT" \
  "$INST_MEM_METRICS"

python - <<'PY' >> "$LOG_MD"
import json
from pathlib import Path
root = Path("/opt/data/private/instanovo/MemNovo/results/public_repro_maskfix/instanovo")
for name in ["baseline_beam5_5k", "memnovo_beam5_5k"]:
    metrics = json.loads((root / f"{name}.metrics.json").read_text())
    print("")
    print(f"- {name} first metrics:")
    print(f"  - aa_precision: {metrics['aa_precision']:.6f}")
    print(f"  - aa_recall: {metrics['aa_recall']:.6f}")
    print(f"  - pep_precision: {metrics['pep_precision']:.6f}")
    print(f"  - pep_recall: {metrics['pep_recall']:.6f}")
PY

append_md ""
append_md "## Public Repro Maskfix Queue Finish"
append_md ""
append_md "- finish: $(timestamp)"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/data/private/instanovo"
PROJECT="$ROOT/MemNovo"
LOG_MD="$ROOT/casanovo_instanovo_log.md"
RUN_ROOT="$PROJECT/results/casanovo_instanovo_queue"
LOG_DIR="$RUN_ROOT/logs"
SUBSET_DIR="$PROJECT/results/diagnostic_subset"
SUBSET_MGF="$SUBSET_DIR/weighted_50k_seed20260405.mgf"

mkdir -p "$LOG_DIR" "$SUBSET_DIR" "$RUN_ROOT/casanovo" "$RUN_ROOT/instanovo"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

append_md() {
  printf '%s\n' "$1" >> "$LOG_MD"
}

append_block() {
  cat >> "$LOG_MD"
}

append_md ""
append_md "## Queue Start"
append_md ""
append_md "- start: \`$(timestamp)\`"
append_md "- gpu policy: \`GPU0/1 only\`"
append_md "- queue order: \`Casanovo baseline + MemNovo\` -> \`InstaNovo official baseline + MemNovo\`"

if [[ ! -f "$SUBSET_MGF" ]]; then
  append_md "- building diagnostic subset: \`$SUBSET_MGF\`"
  python "$PROJECT/scripts/sample_mgf_subset.py" \
    --input "$PROJECT/results/weighted_subset_150k/nine_species_weighted_150000.mgf" \
    --output "$SUBSET_MGF" \
    --num-spectra 50000 \
    --seed 20260405 >> "$LOG_MD" 2>&1
else
  append_md "- reusing diagnostic subset: \`$SUBSET_MGF\`"
fi

run_pair() {
  local stage_name="$1"
  local cmd_a="$2"
  local log_a="$3"
  local cmd_b="$4"
  local log_b="$5"

  append_md ""
  append_md "### $stage_name"
  append_md ""
  append_md "- started: \`$(timestamp)\`"

  bash -lc "$cmd_a" > "$log_a" 2>&1 &
  local pid_a=$!
  bash -lc "$cmd_b" > "$log_b" 2>&1 &
  local pid_b=$!

  append_md "- pid gpu0: \`$pid_a\`"
  append_md "- pid gpu1: \`$pid_b\`"
  append_md "- log gpu0: \`$log_a\`"
  append_md "- log gpu1: \`$log_b\`"

  local status_a=0
  local status_b=0
  wait "$pid_a" || status_a=$?
  wait "$pid_b" || status_b=$?

  append_md "- finished: \`$(timestamp)\`"
  append_md "- exit gpu0: \`$status_a\`"
  append_md "- exit gpu1: \`$status_b\`"

  if [[ "$status_a" -ne 0 || "$status_b" -ne 0 ]]; then
    append_md "- stage status: \`failed\`"
    return 1
  fi

  append_md "- stage status: \`ok\`"
  return 0
}

CAS_BASE_OUT="$RUN_ROOT/casanovo/baseline_50k.jsonl"
CAS_MEM_OUT="$RUN_ROOT/casanovo/memnovo_50k.jsonl"
run_pair \
  "Casanovo Weighted50k" \
  "CUDA_VISIBLE_DEVICES=0 python $PROJECT/scripts/run_inference.py --config $PROJECT/configs/baseline_casanovo.yaml --input $SUBSET_MGF --output $CAS_BASE_OUT --device cuda --evaluate --log-level INFO" \
  "$LOG_DIR/casanovo_baseline.log" \
  "CUDA_VISIBLE_DEVICES=1 python $PROJECT/scripts/run_inference.py --config $PROJECT/configs/memnovo_casanovo.yaml --input $SUBSET_MGF --output $CAS_MEM_OUT --device cuda --evaluate --log-level INFO" \
  "$LOG_DIR/casanovo_memnovo.log"

python - <<'PY' >> "$LOG_MD"
import json
from pathlib import Path
import pandas as pd

root = Path("/opt/data/private/instanovo/MemNovo/results/casanovo_instanovo_queue/casanovo")
b_metrics = json.loads((root / "baseline_50k.metrics.json").read_text())
m_metrics = json.loads((root / "memnovo_50k.metrics.json").read_text())
b = pd.read_json(root / "baseline_50k.jsonl", lines=True)
m = pd.read_json(root / "memnovo_50k.jsonl", lines=True)

print("")
print("- metrics baseline:")
print(f"  - aa_precision: {b_metrics['aa_precision']:.6f}")
print(f"  - aa_recall: {b_metrics['aa_recall']:.6f}")
print(f"  - pep_precision: {b_metrics['pep_precision']:.6f}")
print(f"  - pep_recall: {b_metrics['pep_recall']:.6f}")
print("- metrics memnovo:")
print(f"  - aa_precision: {m_metrics['aa_precision']:.6f}")
print(f"  - aa_recall: {m_metrics['aa_recall']:.6f}")
print(f"  - pep_precision: {m_metrics['pep_precision']:.6f}")
print(f"  - pep_recall: {m_metrics['pep_recall']:.6f}")
print(f"- pred_diff: {int((b['sequence'] != m['sequence']).sum())}")
print(f"- score_diff: {int((b['score'] != m['score']).sum())}")
print(f"- exact_match_baseline: {int((b['sequence'] == b['true_sequence']).sum())}")
print(f"- exact_match_memnovo: {int((m['sequence'] == m['true_sequence']).sum())}")
PY

INST_BASE_OUT="$RUN_ROOT/instanovo/baseline_paperfaithful_1k.csv"
INST_MEM_OUT="$RUN_ROOT/instanovo/memnovo_paperfaithful_1k.csv"
run_pair \
  "InstaNovo Official PaperFaithful Weighted1k" \
  "CUDA_VISIBLE_DEVICES=0 python $PROJECT/scripts/run_instanovo_official.py --config $PROJECT/configs/baseline_instanovo.yaml --input $SUBSET_MGF --output $INST_BASE_OUT --device cuda --subset 0.02 --log-interval 2 --log-level INFO" \
  "$LOG_DIR/instanovo_baseline.log" \
  "CUDA_VISIBLE_DEVICES=1 python $PROJECT/scripts/run_instanovo_official.py --config $PROJECT/configs/memnovo_instanovo.yaml --input $SUBSET_MGF --output $INST_MEM_OUT --device cuda --subset 0.02 --log-interval 2 --log-level INFO" \
  "$LOG_DIR/instanovo_memnovo.log"

python - <<'PY' >> "$LOG_MD"
import json
from pathlib import Path
import pandas as pd

root = Path("/opt/data/private/instanovo/MemNovo/results/casanovo_instanovo_queue/instanovo")
b_metrics = json.loads((root / "baseline_paperfaithful_1k.metrics.json").read_text())
m_metrics = json.loads((root / "memnovo_paperfaithful_1k.metrics.json").read_text())
b = pd.read_csv(root / "baseline_paperfaithful_1k.csv")
m = pd.read_csv(root / "memnovo_paperfaithful_1k.csv")

print("")
print("- metrics baseline:")
print(f"  - aa_precision: {b_metrics['aa_precision']:.6f}")
print(f"  - aa_recall: {b_metrics['aa_recall']:.6f}")
print(f"  - pep_precision: {b_metrics['pep_precision']:.6f}")
print(f"  - pep_recall: {b_metrics['pep_recall']:.6f}")
print("- metrics memnovo:")
print(f"  - aa_precision: {m_metrics['aa_precision']:.6f}")
print(f"  - aa_recall: {m_metrics['aa_recall']:.6f}")
print(f"  - pep_precision: {m_metrics['pep_precision']:.6f}")
print(f"  - pep_recall: {m_metrics['pep_recall']:.6f}")
print(f"- pred_diff: {int((b['predictions'].fillna('') != m['predictions'].fillna('')).sum())}")
print(f"- exact_match_baseline: {int((b['predictions'].fillna('') == b['targets'].fillna('')).sum())}")
print(f"- exact_match_memnovo: {int((m['predictions'].fillna('') == m['targets'].fillna('')).sum())}")
PY

append_md ""
append_md "## Queue Finish"
append_md ""
append_md "- finish: \`$(timestamp)\`"

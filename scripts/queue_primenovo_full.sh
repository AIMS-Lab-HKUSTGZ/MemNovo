#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/data/private/instanovo"
PROJECT="$ROOT/MemNovo"
LOG_MD="$ROOT/primenovo_log.md"
RUN_ROOT="$PROJECT/results/primenovo_full_nine_species"
BASE_DIR="$RUN_ROOT/baseline"
MEM_DIR="$RUN_ROOT/memnovo"
LOG_DIR="$RUN_ROOT/logs"
SPECTRA_PER_SHARD="${MEMNOVO_SPECTRA_PER_SHARD:-50000}"

mkdir -p "$BASE_DIR" "$MEM_DIR" "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

append_md() {
  printf '%s\n' "$1" >> "$LOG_MD"
}

append_md ""
append_md "## PrimeNovo Full Nine-Species Queue Start"
append_md ""
append_md "- start: \`$(timestamp)\`"
append_md "- gpu policy: \`GPU2 baseline / GPU3 memnovo\`"
append_md "- scale: \`full nine species\`"
append_md "- inference config: \`beam=5, batch=64, fp32\`"
append_md "- spectra per shard: \`${SPECTRA_PER_SHARD}\`"
append_md "- baseline dir: \`${BASE_DIR}\`"
append_md "- memnovo dir: \`${MEM_DIR}\`"

BASE_CMD="cd '$ROOT' && MEMNOVO_SPECTRA_PER_SHARD='$SPECTRA_PER_SHARD' bash '$PROJECT/scripts/run_nine_species_sharded_suite.sh' primenovo_baseline configs/baseline_primenovo.yaml '$BASE_DIR' 2"
MEM_CMD="cd '$ROOT' && MEMNOVO_SPECTRA_PER_SHARD='$SPECTRA_PER_SHARD' bash '$PROJECT/scripts/run_nine_species_sharded_suite.sh' primenovo_memnovo configs/memnovo_primenovo.yaml '$MEM_DIR' 3"

bash -lc "$BASE_CMD" > "$LOG_DIR/primenovo_baseline.log" 2>&1 &
PID_BASE=$!
bash -lc "$MEM_CMD" > "$LOG_DIR/primenovo_memnovo.log" 2>&1 &
PID_MEM=$!

append_md "- pid baseline gpu2: \`$PID_BASE\`"
append_md "- pid memnovo gpu3: \`$PID_MEM\`"
append_md "- log baseline: \`$LOG_DIR/primenovo_baseline.log\`"
append_md "- log memnovo: \`$LOG_DIR/primenovo_memnovo.log\`"

STATUS_BASE=0
STATUS_MEM=0
wait "$PID_BASE" || STATUS_BASE=$?
wait "$PID_MEM" || STATUS_MEM=$?

append_md "- finish: \`$(timestamp)\`"
append_md "- exit baseline: \`$STATUS_BASE\`"
append_md "- exit memnovo: \`$STATUS_MEM\`"

if [[ "$STATUS_BASE" -ne 0 || "$STATUS_MEM" -ne 0 ]]; then
  append_md "- queue status: \`failed\`"
  exit 1
fi

python - <<'PY' >> "$LOG_MD"
import json
from pathlib import Path

species = [
    "Bacillus-subtilis",
    "Saccharomyces-cerevisiae",
    "Methanosarcina-mazei",
    "Apis-mellifera",
    "Solanum-lycopersicum",
    "Vigna-mungo",
    "Candidatus-endoloripes",
    "H.-sapiens",
    "Mus-musculus",
]

root = Path("/opt/data/private/instanovo/MemNovo/results/primenovo_full_nine_species")

def load_suite(name: str):
    rows = []
    for sp in species:
        path = root / name / f"{sp}.metrics.json"
        rows.append((sp, json.loads(path.read_text())))
    return rows

def summarize(rows):
    keys = ["aa_precision", "aa_recall", "pep_precision", "pep_recall"]
    out = {}
    for key in keys:
        out[key] = sum(row[1][key] for row in rows) / len(rows)
    return out

base_rows = load_suite("baseline")
mem_rows = load_suite("memnovo")
base = summarize(base_rows)
mem = summarize(mem_rows)

print("")
print("### Species-Average Summary")
print("")
print("- baseline:")
for key, value in base.items():
    print(f"  - {key}: {value:.6f}")
print("- memnovo:")
for key, value in mem.items():
    print(f"  - {key}: {value:.6f}")
print("- delta:")
for key in base:
    print(f"  - {key}: {mem[key] - base[key]:+.6f}")
PY

append_md "- queue status: \`ok\`"

#!/bin/bash
#
# Run MemNovo evaluation on Nine Species benchmark
#
# This script runs both baseline (no MemNovo) and MemNovo-enhanced
# inference on all 9 species, then compares results.
#
# Usage:
#   bash scripts/run_nine_species.sh
#
# Prerequisites:
#   - Model checkpoints downloaded (bash scripts/download_models.sh)
#   - Nine Species data in data/ directory
#

set -e

# Configuration
DATA_DIR="data"
RESULTS_DIR="results/nine_species"
CONFIG_BASELINE="configs/baseline_instanovo.yaml"
CONFIG_MEMNOVO="configs/memnovo.yaml"

# Species list
SPECIES=(
    "Bacillus-subtilis"
    "Saccharomyces-cerevisiae"
    "Methanosarcina-mazei"
    "Apis-mellifera"
    "Solanum-lycopersicum"
    "Vigna-mungo"
    "Candidatus-endoloripes"
    "H.-sapiens"
    "Mus-musculus"
)

echo "========================================="
echo "MemNovo Nine Species Benchmark"
echo "========================================="
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR/baseline"
mkdir -p "$RESULTS_DIR/memnovo"

# Check for data files
echo "Checking data files..."
MISSING=0
for species in "${SPECIES[@]}"; do
    if [ ! -f "$DATA_DIR/$species.mgf" ]; then
        echo "[MISSING] $DATA_DIR/$species.mgf"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "[ERROR] $MISSING data files missing"
    echo "Please download the Nine Species dataset first."
    echo "See: bash scripts/download_data.sh"
    exit 1
fi

echo "[OK] All data files found"
echo ""

# Check for model checkpoints
if [ ! -f "weights/instanovo-v1.1.0.ckpt" ]; then
    echo "[ERROR] InstaNovo checkpoint not found"
    echo "Please run: bash scripts/download_models.sh"
    exit 1
fi

echo "========================================="
echo "Running Baseline (No MemNovo)"
echo "========================================="

for species in "${SPECIES[@]}"; do
    echo ""
    echo "Processing $species..."

    python scripts/run_inference.py \
        --config "$CONFIG_BASELINE" \
        --input "$DATA_DIR/$species.mgf" \
        --output "$RESULTS_DIR/baseline/$species.csv" \
        --evaluate \
        2>&1 | tee "$RESULTS_DIR/baseline/$species.log"
done

echo ""
echo "========================================="
echo "Running MemNovo"
echo "========================================="

for species in "${SPECIES[@]}"; do
    echo ""
    echo "Processing $species..."

    python scripts/run_inference.py \
        --config "$CONFIG_MEMNOVO" \
        --input "$DATA_DIR/$species.mgf" \
        --output "$RESULTS_DIR/memnovo/$species.csv" \
        --evaluate \
        2>&1 | tee "$RESULTS_DIR/memnovo/$species.log"
done

echo ""
echo "========================================="
echo "Generating Comparison Report"
echo "========================================="

# Create comparison script inline
python << 'EOF'
import os
import re
from pathlib import Path

results_dir = Path("results/nine_species")
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

def extract_metrics(log_file):
    """Extract metrics from log file."""
    if not log_file.exists():
        return {}

    content = log_file.read_text()
    metrics = {}

    patterns = {
        'aa_precision': r'AA Precision:\s*([\d.]+)',
        'aa_recall': r'AA Recall:\s*([\d.]+)',
        'pep_precision': r'Pep Precision:\s*([\d.]+)',
        'pep_recall': r'Pep Recall:\s*([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))

    return metrics

# Collect results
print("\nNine Species Benchmark Results")
print("=" * 80)
print(f"{'Species':<25} {'Baseline AA%':<15} {'MemNovo AA%':<15} {'Delta':<10}")
print("-" * 80)

baseline_sum = 0
memnovo_sum = 0
count = 0

for sp in species:
    baseline_log = results_dir / "baseline" / f"{sp}.log"
    memnovo_log = results_dir / "memnovo" / f"{sp}.log"

    baseline_metrics = extract_metrics(baseline_log)
    memnovo_metrics = extract_metrics(memnovo_log)

    baseline_aa = baseline_metrics.get('aa_precision', 0)
    memnovo_aa = memnovo_metrics.get('aa_precision', 0)
    delta = memnovo_aa - baseline_aa

    if baseline_aa > 0:
        baseline_sum += baseline_aa
        memnovo_sum += memnovo_aa
        count += 1

    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    print(f"{sp:<25} {baseline_aa:<15.4f} {memnovo_aa:<15.4f} {delta_str:<10}")

print("-" * 80)
if count > 0:
    avg_baseline = baseline_sum / count
    avg_memnovo = memnovo_sum / count
    avg_delta = avg_memnovo - avg_baseline
    delta_str = f"+{avg_delta:.4f}" if avg_delta >= 0 else f"{avg_delta:.4f}"
    print(f"{'Average':<25} {avg_baseline:<15.4f} {avg_memnovo:<15.4f} {delta_str:<10}")
print("=" * 80)

EOF

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "========================================="

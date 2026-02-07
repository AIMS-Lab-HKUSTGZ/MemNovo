#!/bin/bash
# Run sensitivity scaling experiments

set -e

# Configuration
MODEL=${1:-"instanovo"}
DATA_DIR=${2:-"data/nine_species"}
OUTPUT_DIR=${3:-"results/sensitivity"}
CHECKPOINT=${4:-"models/instanovo.ckpt"}

# Scale factors to test
SCALE_FACTORS="0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0"

echo "Running sensitivity scaling experiment"
echo "Model: $MODEL"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Run spectrum scaling
echo "=== Spectrum Scaling ==="
python -m sensitivity_scaling.experiment \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA_DIR" \
    --modality spectrum \
    --scale-factors $SCALE_FACTORS \
    --output "$OUTPUT_DIR/spectrum_scaling.json"

# Run peptide scaling
echo "=== Peptide Scaling ==="
python -m sensitivity_scaling.experiment \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA_DIR" \
    --modality peptide \
    --scale-factors $SCALE_FACTORS \
    --output "$OUTPUT_DIR/peptide_scaling.json"

# Analyze results
echo "=== Analysis ==="
python -m sensitivity_scaling.analyze \
    --spectrum-results "$OUTPUT_DIR/spectrum_scaling.json" \
    --peptide-results "$OUTPUT_DIR/peptide_scaling.json" \
    --output "$OUTPUT_DIR/analysis.json"

# Generate visualization
echo "=== Visualization ==="
python -m sensitivity_scaling.visualize \
    --spectrum-results "$OUTPUT_DIR/spectrum_scaling.json" \
    --peptide-results "$OUTPUT_DIR/peptide_scaling.json" \
    --output "$OUTPUT_DIR/sensitivity_curves.pdf"

echo "Done! Results saved to $OUTPUT_DIR"

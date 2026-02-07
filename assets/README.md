# Assets

This directory contains figures and visual assets for documentation.

## Contents

- `sensitivity_comparison.png` - Sensitivity scaling curves
- `architecture.png` - MemNovo architecture diagram
- `results_table.png` - Main results summary

## Generating Figures

Figures can be regenerated using the visualization module:

```python
from sensitivity_scaling.visualize import plot_sensitivity_curves, plot_comparison

# Load experiment results
spectrum_results = load_json("results/sensitivity/spectrum_scaling.json")
peptide_results = load_json("results/sensitivity/peptide_scaling.json")

# Generate sensitivity curves
plot_sensitivity_curves(
    spectrum_results,
    peptide_results,
    output_path="assets/sensitivity_comparison.png"
)
```

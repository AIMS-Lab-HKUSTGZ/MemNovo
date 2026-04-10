#!/usr/bin/env python3
"""Build reviewer WSMy summary tables from existing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cas_path = Path("/opt/data/private/instanovo/MemNovo/results/rerank_probe/casanovo_nine_species_gap_audit.json")
    ins_path = Path("/opt/data/private/instanovo/MemNovo/results/rerank_probe/instanovo_nine_species_gap_audit.json")
    prime_paths = [
        Path("/opt/data/private/instanovo/MemNovo/results/rerank_probe/primenovo_cendo_5000_hybrid_bestconfirm.json"),
        Path("/opt/data/private/instanovo/MemNovo/results/rebuttal_primenovo_second_species/Apis-mellifera_subset5000_hybrid_bestconfirm.json"),
        Path("/opt/data/private/instanovo/MemNovo/results/rerank_probe/primenovo_mus_5000_hybrid_bestconfirm.json"),
    ]

    cas = load_json(cas_path)
    ins = load_json(ins_path)

    def build_quality_table(payload: dict, model_name: str) -> str:
        rows = sorted(payload["species_results"], key=lambda row: row["baseline_pep_recall"])
        lines = [
            f"# {model_name} Species Quality vs Gain",
            "",
            "| Species | Baseline Pep. Recall | Best Pep. Recall | Relative Gain |",
            "|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['species']} | {row['baseline_pep_recall']:.6f} | {row['best']['pep_recall']:.6f} | {row['best']['pep_recall_rel_pct']:.6f} |"
            )
        return "\n".join(lines) + "\n"

    (output_dir / "casanovo_species_quality_vs_gain.md").write_text(
        build_quality_table(cas, "Casanovo"),
        encoding="utf-8",
    )
    (output_dir / "instanovo_species_quality_vs_gain.md").write_text(
        build_quality_table(ins, "InstaNovo"),
        encoding="utf-8",
    )

    lines = [
        "# PrimeNovo Confirmation",
        "",
        "| Species / subset | Baseline Pep. Recall | Best Pep. Recall | Relative Gain |",
        "|---|---:|---:|---:|",
    ]
    for path in prime_paths:
        payload = load_json(path)
        best = max(payload["probes"], key=lambda row: row["pep_recall"])
        base = payload["baseline"]["pep_recall"]
        gain = (best["pep_recall"] - base) / max(base, 1e-12) * 100.0
        lines.append(f"| {path.stem} | {base:.6f} | {best['pep_recall']:.6f} | {gain:.6f} |")
    (output_dir / "primenovo_confirmation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# WSMy Experiment Summary",
        "",
        f"- Casanovo nine-species average relative gain: `{cas['avg_best_rel_pct']:.6f}`",
        f"- InstaNovo nine-species average relative gain: `{ins['avg_best_rel_pct']:.6f}`",
        "- PrimeNovo confirmed positive points are summarized in `primenovo_confirmation.md`.",
        "",
        "Generated assets:",
        f"- `{output_dir / 'casanovo_species_quality_vs_gain.md'}`",
        f"- `{output_dir / 'instanovo_species_quality_vs_gain.md'}`",
        f"- `{output_dir / 'primenovo_confirmation.md'}`",
    ]
    (output_dir / "wsmy_table_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

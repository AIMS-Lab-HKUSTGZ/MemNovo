#!/usr/bin/env python3
"""Build a compact diagnostic-consistency summary for rebuttal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path("/opt/data/private/instanovo")

    cas = json.loads((root / "MemNovo/results/rerank_probe/casanovo_nine_species_gap_audit.json").read_text())
    ins = json.loads((root / "MemNovo/results/rerank_probe/instanovo_nine_species_gap_audit.json").read_text())
    prime = json.loads((root / "MemNovo/results/rerank_probe/primenovo_cendo_5000_hybrid_bestconfirm.json").read_text())

    rows = [
        {
            "model": "Casanovo",
            "paper_reference_relative_gain_percent": cas["avg_paper_rel_pct"],
            "current_best_average_relative_gain_percent": cas["avg_best_rel_pct"],
            "paper_fraction_recovered_percent": cas["avg_paper_fraction_recovered"] * 100.0,
            "scope": "9-species average",
        },
        {
            "model": "InstaNovo",
            "paper_reference_relative_gain_percent": ins["avg_paper_rel_pct"],
            "current_best_average_relative_gain_percent": ins["avg_best_rel_pct"],
            "paper_fraction_recovered_percent": ins["avg_paper_fraction_recovered"] * 100.0,
            "scope": "9-species average",
        },
        {
            "model": "PrimeNovo",
            "paper_reference_relative_gain_percent": None,
            "current_best_average_relative_gain_percent": None,
            "paper_fraction_recovered_percent": None,
            "scope": "Candidatus-endoloripes 5k subset",
            "best_subset_peptide_recall": prime["probes"][-1]["pep_recall"],
            "baseline_subset_peptide_recall": prime["baseline"]["pep_recall"],
            "best_subset_relative_gain_percent": (
                (prime["probes"][-1]["pep_recall"] - prime["baseline"]["pep_recall"])
                / prime["baseline"]["pep_recall"]
                * 100.0
            ),
        },
    ]

    payload = {"rows": rows}
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Diagnostic Consistency Summary",
        "",
        "| Model | Scope | Paper Ref. Rel. Gain (%) | Current Rel. Gain (%) | Fraction Recovered (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows[:2]:
        lines.append(
            f"| {row['model']} | {row['scope']} | "
            f"{row['paper_reference_relative_gain_percent']:.3f} | "
            f"{row['current_best_average_relative_gain_percent']:.3f} | "
            f"{row['paper_fraction_recovered_percent']:.2f} |"
        )
    prime_row = rows[2]
    lines.extend(
        [
            "",
            "PrimeNovo extension:",
            (
                f"- scope: `{prime_row['scope']}`\n"
                f"- peptide recall: `{prime_row['baseline_subset_peptide_recall']:.6f} -> "
                f"{prime_row['best_subset_peptide_recall']:.6f}`\n"
                f"- relative gain: `{prime_row['best_subset_relative_gain_percent']:.3f}%`"
            ),
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build Reviewer C1qX analysis tables from existing results.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/opt/data/private/instanovo")
RESULTS = ROOT / "MemNovo" / "results"
OUTDIR = RESULTS / "rebuttal_c1qx_tables"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_species_quality_table(model: str, src: Path) -> dict:
    payload = load_json(src)
    rows = []
    for row in payload["species_results"]:
        best = row["best"]
        rows.append(
            {
                "species": row["species"],
                "baseline_pep_recall": row["baseline_pep_recall"],
                "best_pep_recall": best["pep_recall"],
                "relative_gain_percent": best["pep_recall_rel_pct"],
                "paper_relative_gain_percent": row["paper_rel_pct"],
            }
        )
    rows.sort(key=lambda x: x["baseline_pep_recall"])

    md = [
        f"# {model} Species Quality vs Gain",
        "",
        f"- source: `{src}`",
        "",
        "| Rank | Species | Baseline Peptide Recall | Best Peptide Recall | MemNovo Relative Gain | Paper Relative Gain |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(rows, start=1):
        md.append(
            f"| {idx} | {row['species']} | {row['baseline_pep_recall']:.6f} | "
            f"{row['best_pep_recall']:.6f} | {row['relative_gain_percent']:.3f}% | "
            f"{row['paper_relative_gain_percent']:.3f}% |"
        )

    out_json = OUTDIR / f"{model.lower()}_species_quality_vs_gain.json"
    out_md = OUTDIR / f"{model.lower()}_species_quality_vs_gain.md"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, md)
    return {"json": str(out_json), "md": str(out_md), "rows": rows}


def build_primenovo_confirmation_table() -> dict:
    paths = [
        RESULTS / "rerank_probe" / "primenovo_cendo_5000_hybrid_bestconfirm.json",
        RESULTS / "rebuttal_primenovo_second_species" / "Apis-mellifera_subset5000_hybrid_bestconfirm.json",
        RESULTS / "rerank_probe" / "primenovo_mus_5000_hybrid_bestconfirm.json",
        RESULTS / "rerank_probe" / "primenovo_cendo_full_hybrid_bestconfirm.json",
        RESULTS / "rerank_probe" / "primenovo_mus_full_hybrid_bestconfirm.json",
    ]

    rows = []
    for path in paths:
        obj = load_json(path)
        baseline = obj["baseline"]
        probe = obj["probes"][0]
        name = path.stem
        species = None
        subset = None
        if "cendo" in name:
            species = "Candidatus-endoloripes"
        elif "mus" in name:
            species = "Mus-musculus"
        elif "Apis-mellifera" in name:
            species = "Apis-mellifera"
        if "5000" in name or "subset5000" in name:
            subset = "5k subset"
        elif "full" in name:
            subset = "full species"
        else:
            subset = "unknown"
        rows.append(
            {
                "path": str(path),
                "species": species or path.stem,
                "evaluation_scale": subset,
                "baseline_pep_recall": baseline["pep_recall"],
                "best_pep_recall": probe["pep_recall"],
                "relative_gain_percent": ((probe["pep_recall"] - baseline["pep_recall"]) / baseline["pep_recall"]) * 100.0,
                "baseline_n_match_pep": baseline["n_match_pep"],
                "best_n_match_pep": probe["n_match_pep"],
                "best_config": {k: v for k, v in probe.items() if k in {"alpha", "spec_gap_threshold", "decoder_margin_threshold", "confidence_threshold", "refined_rows", "gated_rows"}},
            }
        )

    rows.sort(key=lambda x: (x["evaluation_scale"] != "5k subset", -x["relative_gain_percent"]))
    md = [
        "# PrimeNovo Confirmation Table",
        "",
        "| Species | Scale | Baseline Peptide Recall | Best Peptide Recall | Relative Gain | Baseline Matches | Best Matches | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md.append(
            f"| {row['species']} | {row['evaluation_scale']} | {row['baseline_pep_recall']:.6f} | "
            f"{row['best_pep_recall']:.6f} | {row['relative_gain_percent']:.3f}% | "
            f"{row['baseline_n_match_pep']} | {row['best_n_match_pep']} | `{row['path']}` |"
        )

    out_json = OUTDIR / "primenovo_confirmation.json"
    out_md = OUTDIR / "primenovo_confirmation.md"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, md)
    return {"json": str(out_json), "md": str(out_md), "rows": rows}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cas = build_species_quality_table("Casanovo", RESULTS / "rerank_probe" / "casanovo_nine_species_gap_audit.json")
    ins = build_species_quality_table("InstaNovo", RESULTS / "rerank_probe" / "instanovo_nine_species_gap_audit.json")
    prime = build_primenovo_confirmation_table()
    summary = {
        "casanovo_species_quality_vs_gain": cas,
        "instanovo_species_quality_vs_gain": ins,
        "primenovo_confirmation": prime,
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Case study for beam-level reranking behavior across models."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from replay_rerank_probe import (
    advanced_spectrum_match_score,
    align_spectra,
    build_candidates,
    diversity,
    entropy,
    load_spectra,
    minmax,
    normalize_peptide,
    normalize_ptm_format,
    softmax,
)


HIGH_RISK_AA = set("DNEQKM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["instanovo", "primenovo", "casanovo"], required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--spec-gap-threshold", type=float, required=True)
    parser.add_argument("--decoder-margin-threshold", type=float, required=True)
    parser.add_argument("--diversity-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--ion-mode", default="both")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument("--example-count", type=int, default=5)
    return parser.parse_args()


def norm(seq: str, model: str) -> str:
    return normalize_peptide(normalize_ptm_format(seq or "", model=model))


def lev(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i, x in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (x != y))
            prev = cur
    return dp[m]


def mean(values: List[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median(values: List[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    truth_lens = [r["truth_len"] for r in rows]
    charges = [r["charge"] for r in rows]
    peaks = [r["peak_count"] for r in rows]
    entropies = [r["beam_entropy"] for r in rows]
    margins = [r["decoder_margin"] for r in rows]
    diversities = [r["beam_diversity"] for r in rows]
    top1_ed = [r["top1_edit_to_truth"] for r in rows]
    chosen_ed = [r["chosen_edit_to_truth"] for r in rows]
    truth_ranks = [r["truth_rank"] for r in rows if r["truth_rank"] is not None]
    highrisk = sum(1 for r in rows if r["truth_has_highrisk"])
    ptm = sum(1 for r in rows if r["truth_has_ptm"])
    long_pep = sum(1 for r in rows if r["truth_len"] >= 21)
    return {
        "count": len(rows),
        "mean_truth_len": mean(truth_lens),
        "median_truth_len": median(truth_lens),
        "mean_charge": mean(charges),
        "mean_peak_count": mean(peaks),
        "mean_beam_entropy": mean(entropies),
        "mean_decoder_margin": mean(margins),
        "mean_beam_diversity": mean(diversities),
        "mean_top1_edit_to_truth": mean(top1_ed),
        "mean_chosen_edit_to_truth": mean(chosen_ed),
        "mean_truth_rank_when_present": mean(truth_ranks),
        "highrisk_fraction": highrisk / len(rows),
        "ptm_fraction": ptm / len(rows),
        "long_peptide_fraction": long_pep / len(rows),
        "charge_hist": dict(Counter(charges).most_common(5)),
    }


def example_rows(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    picked = sorted(
        rows,
        key=lambda r: (
            r["truth_rank"] is None,
            r["truth_rank"] if r["truth_rank"] is not None else 99,
            -r["spec_gap"],
            r["decoder_margin"],
        ),
    )[:limit]
    return [
        {
            "sample_id": r["sample_id"],
            "truth": r["truth"],
            "baseline_top1": r["baseline_top1"],
            "chosen": r["chosen"],
            "truth_rank": r["truth_rank"],
            "decoder_margin": r["decoder_margin"],
            "spec_gap": r["spec_gap"],
            "beam_entropy": r["beam_entropy"],
            "charge": r["charge"],
            "truth_len": r["truth_len"],
            "peak_count": r["peak_count"],
            "beams": r["beam_sequences"][:5],
        }
        for r in picked
    ]


def main() -> None:
    args = parse_args()
    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))

    analyzed: List[Dict[str, Any]] = []
    baseline_correct = 0
    chosen_correct = 0
    oracle_correct = 0
    improved = 0
    harmed = 0
    changed = 0

    for idx, (row, spectrum) in enumerate(zip(rows, spectra)):
        beams = [dict(item) for item in row["candidates"][: args.top_k]]
        if not beams:
            continue

        raw_scores = [float(beam["decoder_score"]) for beam in beams]
        if args.model in {"casanovo", "primenovo"}:
            decoder_probs = minmax([max(score, 0.0) for score in raw_scores])
            top1_conf = float(raw_scores[0]) if raw_scores else 0.0
            beam_entropy = 0.0
            beam_div = 0.0
        else:
            decoder_probs = softmax(raw_scores)
            top1_conf = 0.0
            beam_entropy = entropy(decoder_probs)
            beam_div = diversity([beam["sequence"] for beam in beams])
        spec_scores = [
            advanced_spectrum_match_score(
                beam["sequence"],
                spectrum["mz_array"],
                spectrum["intensity_array"],
                precursor_mz=float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0))),
                precursor_charge=int(spectrum.get("precursor_charge", row.get("precursor_charge", 0))),
                model=args.model,
                tol_da=args.mass_tol_da,
                precursor_ppm=args.precursor_ppm,
                top_peak_frac=args.top_peak_frac,
                ion_mode=args.ion_mode,
            )
            for beam in beams
        ]
        spec_norm = minmax(spec_scores)
        spec_best_idx = int(max(range(len(spec_norm)), key=lambda i: spec_norm[i]))
        spec_gap = spec_norm[spec_best_idx] - spec_norm[0]
        decoder_margin = decoder_probs[0] - decoder_probs[spec_best_idx]

        truth = row["truth"]
        truth_norm = norm(truth, args.model)
        beam_sequences = [beam["sequence"] for beam in beams]
        beam_norms = [norm(seq, args.model) for seq in beam_sequences]
        baseline_top1 = row["top1"]
        baseline_norm = norm(baseline_top1, args.model)

        final_scores = [
            (1.0 - args.alpha) * dec + args.alpha * spec_n
            for dec, spec_n in zip(decoder_probs, spec_norm)
        ]
        if args.model in {"casanovo", "primenovo"}:
            gated = args.confidence_threshold is None or top1_conf < float(args.confidence_threshold)
        else:
            gated = args.entropy_threshold is None or (
                beam_entropy > float(args.entropy_threshold)
                and beam_div >= float(args.diversity_threshold)
            )
        if not gated:
            chosen_idx = 0
            chosen = beam_sequences[chosen_idx]
            chosen_norm = beam_norms[chosen_idx]
        else:
            if (
                spec_best_idx != 0
                and spec_gap >= args.spec_gap_threshold
                and decoder_margin <= args.decoder_margin_threshold
                and beam_div >= args.diversity_threshold
            ):
                final_scores[spec_best_idx] = max(final_scores[spec_best_idx], final_scores[0] + 1e-6)
            chosen_idx = int(max(range(len(final_scores)), key=lambda i: final_scores[i]))
            chosen = beam_sequences[chosen_idx]
            chosen_norm = beam_norms[chosen_idx]

        truth_rank = None
        for rank, cand in enumerate(beam_norms, 1):
            if cand == truth_norm:
                truth_rank = rank
                break

        base_ok = baseline_norm == truth_norm
        chosen_ok = chosen_norm == truth_norm
        oracle_ok = truth_rank is not None

        baseline_correct += int(base_ok)
        chosen_correct += int(chosen_ok)
        oracle_correct += int(oracle_ok)
        improved += int((not base_ok) and chosen_ok)
        harmed += int(base_ok and (not chosen_ok))
        changed += int(chosen_norm != baseline_norm)

        analyzed.append(
            {
                "row_index": idx,
                "sample_id": f"{args.species}:{idx}",
                "truth": truth,
                "baseline_top1": baseline_top1,
                "chosen": chosen,
                "truth_rank": truth_rank,
                "truth_in_beam": oracle_ok,
                "baseline_correct": base_ok,
                "chosen_correct": chosen_ok,
                "changed": chosen_norm != baseline_norm,
                "beam_entropy": beam_entropy,
                "beam_diversity": beam_div,
                "decoder_margin": decoder_margin,
                "spec_gap": spec_gap,
                "truth_len": len(truth_norm),
                "charge": int(spectrum.get("precursor_charge", 0)),
                "peak_count": int(len(spectrum["mz_array"])),
                "truth_has_highrisk": any(ch in HIGH_RISK_AA for ch in truth_norm),
                "truth_has_ptm": "[" in truth or "+" in truth,
                "top1_edit_to_truth": lev(baseline_norm, truth_norm),
                "chosen_edit_to_truth": lev(chosen_norm, truth_norm),
                "beam_sequences": beam_sequences,
            }
        )

    total = len(analyzed)
    baseline_recall = baseline_correct / total if total else 0.0
    chosen_recall = chosen_correct / total if total else 0.0
    oracle_recall = oracle_correct / total if total else 0.0
    rel_gain = ((chosen_recall - baseline_recall) / baseline_recall * 100.0) if baseline_recall else 0.0
    oracle_headroom_rel = ((oracle_recall - baseline_recall) / baseline_recall * 100.0) if baseline_recall else 0.0

    truth_in_beam_wrong = [r for r in analyzed if r["truth_in_beam"] and not r["baseline_correct"]]
    improved_rows = [r for r in analyzed if (not r["baseline_correct"]) and r["chosen_correct"]]
    harmed_rows = [r for r in analyzed if r["baseline_correct"] and (not r["chosen_correct"])]
    changed_wrong_rows = [
        r for r in analyzed if r["changed"] and (not r["chosen_correct"]) and (not r["baseline_correct"])
    ]
    generation_failure_rows = [r for r in analyzed if (not r["truth_in_beam"]) and (not r["baseline_correct"])]

    report = {
        "model": args.model,
        "species": args.species,
        "config": {
            "alpha": args.alpha,
            "spec_gap_threshold": args.spec_gap_threshold,
            "decoder_margin_threshold": args.decoder_margin_threshold,
            "diversity_threshold": args.diversity_threshold,
            "confidence_threshold": args.confidence_threshold,
            "entropy_threshold": args.entropy_threshold,
            "ion_mode": args.ion_mode,
        },
        "summary": {
            "total": total,
            "baseline_pep_recall": baseline_recall,
            "reranked_pep_recall": chosen_recall,
            "relative_gain_pct": rel_gain,
            "oracle_top5_pep_recall": oracle_recall,
            "oracle_headroom_rel_pct": oracle_headroom_rel,
            "changed_rows": changed,
            "improved_rows": improved,
            "harmed_rows": harmed,
            "changed_wrong_rows": len(changed_wrong_rows),
            "truth_in_beam_wrong_rows": len(truth_in_beam_wrong),
            "generation_failure_rows": len(generation_failure_rows),
            "rescue_rate_within_truth_in_beam_wrong": (
                improved / len(truth_in_beam_wrong) if truth_in_beam_wrong else 0.0
            ),
        },
        "groups": {
            "improved": summarize_group(improved_rows),
            "harmed": summarize_group(harmed_rows),
            "truth_in_beam_wrong": summarize_group(truth_in_beam_wrong),
            "generation_failure": summarize_group(generation_failure_rows),
            "changed_wrong": summarize_group(changed_wrong_rows),
        },
        "examples": {
            "improved": example_rows(improved_rows, args.example_count),
            "harmed": example_rows(harmed_rows, args.example_count),
            "changed_wrong": example_rows(changed_wrong_rows, args.example_count),
        },
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# {args.model.capitalize()} Case Study: {args.species}",
        "",
        "## Summary",
        f"- baseline peptide recall: `{baseline_recall:.6f}`",
        f"- reranked peptide recall: `{chosen_recall:.6f}`",
        f"- relative gain: `{rel_gain:.3f}%`",
        f"- top-{args.top_k} oracle peptide recall: `{oracle_recall:.6f}`",
        f"- oracle headroom vs baseline: `{oracle_headroom_rel:.3f}%`",
        f"- changed rows: `{changed}`",
        f"- improved rows: `{improved}`",
        f"- harmed rows: `{harmed}`",
        f"- changed-but-still-wrong rows: `{len(changed_wrong_rows)}`",
        f"- baseline-wrong but truth-in-beam rows: `{len(truth_in_beam_wrong)}`",
        f"- rescue rate within truth-in-beam wrong rows: `{(improved / len(truth_in_beam_wrong) * 100.0) if truth_in_beam_wrong else 0.0:.2f}%`",
        f"- baseline generation-failure rows (truth not in top-{args.top_k} beam): `{len(generation_failure_rows)}`",
        "",
        "## Group Statistics",
    ]
    for name, stats in report["groups"].items():
        lines.append(f"### {name}")
        for key, value in stats.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    for name, rows in report["examples"].items():
        lines.append(f"## Examples: {name}")
        for row in rows:
            lines.append(
                f"- sample `{row['sample_id']}` | truth `{row['truth']}` | baseline `{row['baseline_top1']}` | "
                f"chosen `{row['chosen']}` | truth_rank `{row['truth_rank']}` | "
                f"decoder_margin `{row['decoder_margin']:.4f}` | spec_gap `{row['spec_gap']:.4f}` | "
                f"entropy `{row['beam_entropy']:.4f}` | charge `{row['charge']}` | len `{row['truth_len']}`"
            )
            lines.append(f"  beams: `{row['beams']}`")
        lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

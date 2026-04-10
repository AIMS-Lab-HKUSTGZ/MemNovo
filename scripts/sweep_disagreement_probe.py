#!/usr/bin/env python3
"""Sweep disagreement-gated reranking settings on existing beam outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.evaluator import Evaluator
from replay_rerank_probe import (
    align_spectra,
    build_candidates,
    load_spectra,
    rerank_row,
)


def parse_list(values: str, cast=float) -> list[float]:
    return [cast(item) for item in values.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["casanovo", "instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alphas", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--spec-gaps", default="0.05,0.1,0.15,0.2")
    parser.add_argument("--decoder-margins", default="0.1,0.2,0.3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument("--scorer", choices=["basic", "advanced"], default="advanced")
    parser.add_argument("--refine-all", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--entropy-threshold", type=float, default=1.00)
    parser.add_argument("--diversity-threshold", type=float, default=0.80)
    return parser.parse_args()


def evaluate(model: str, sequences: list[str], truths: list[str]) -> dict:
    evaluator = Evaluator()
    return evaluator.evaluate(
        [{"sequence": seq} for seq in sequences],
        [{"sequence": seq} for seq in truths],
        model_name=model,
    )


def main() -> None:
    args = parse_args()
    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))
    truths = [row["truth"] for row in rows]
    baseline_sequences = [row["top1"] for row in rows]

    results = {
        "model": args.model,
        "baseline": evaluate(args.model, baseline_sequences, truths),
        "probes": [],
    }

    alphas = parse_list(args.alphas, float)
    spec_gaps = parse_list(args.spec_gaps, float)
    decoder_margins = parse_list(args.decoder_margins, float)

    for alpha in alphas:
        for spec_gap in spec_gaps:
            for decoder_margin in decoder_margins:
                args.decision_mode = "disagreement"
                args.spec_gap_threshold = spec_gap
                args.decoder_margin_threshold = decoder_margin
                reranked_sequences = []
                refined = 0
                for row, spectrum in zip(rows, spectra):
                    reranked = rerank_row(args.model, row, spectrum, alpha, args)
                    reranked_sequences.append(reranked["sequence"])
                    if reranked["beam_predictions"] and reranked["beam_predictions"][0]["sequence"] != row["top1"]:
                        refined += 1
                metrics = evaluate(args.model, reranked_sequences, truths)
                metrics["alpha"] = alpha
                metrics["spec_gap_threshold"] = spec_gap
                metrics["decoder_margin_threshold"] = decoder_margin
                metrics["refined_rows"] = refined
                results["probes"].append(metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

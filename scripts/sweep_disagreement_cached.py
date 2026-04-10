#!/usr/bin/env python3
"""Cached disagreement-gated reranking sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.evaluator import Evaluator
from replay_rerank_probe import (
    advanced_spectrum_match_score,
    align_spectra,
    build_candidates,
    load_spectra,
    minmax,
    softmax,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["casanovo", "instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alphas", default="0.0,0.05,0.1,0.2,0.3")
    parser.add_argument("--spec-gaps", default="0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--decoder-margins", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument(
        "--ion-mode",
        choices=["both", "y_only", "y_heavy", "b_only"],
        default="both",
    )
    return parser.parse_args()


def parse_list(values: str) -> list[float]:
    return [float(item) for item in values.split(",") if item]


def evaluate(model: str, sequences: list[str], truths: list[str]) -> dict:
    evaluator = Evaluator()
    return evaluator.evaluate(
        [{"sequence": seq} for seq in sequences],
        [{"sequence": seq} for seq in truths],
        model_name=model,
    )


def precompute(args: argparse.Namespace) -> tuple[list[dict], list[str], list[str]]:
    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))
    cached_rows = []
    baseline_sequences = []
    truths = []
    for row, spectrum in zip(rows, spectra):
        beams = [dict(item) for item in row["candidates"][: args.top_k]]
        if not beams:
            continue
        if args.model in {"casanovo", "primenovo"}:
            decoder_probs = minmax([max(float(beam["decoder_score"]), 0.0) for beam in beams])
        else:
            decoder_probs = softmax([beam["decoder_score"] for beam in beams])
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
        cached_rows.append(
            {
                "beams": beams,
                "decoder_probs": decoder_probs,
                "spec_scores": spec_scores,
                "spec_norm": spec_norm,
                "top1": row["top1"],
                "truth": row["truth"],
            }
        )
        baseline_sequences.append(row["top1"])
        truths.append(row["truth"])
    return cached_rows, baseline_sequences, truths


def main() -> None:
    args = parse_args()
    alphas = parse_list(args.alphas)
    spec_gaps = parse_list(args.spec_gaps)
    decoder_margins = parse_list(args.decoder_margins)

    cached_rows, baseline_sequences, truths = precompute(args)
    results = {
        "model": args.model,
        "ion_mode": args.ion_mode,
        "baseline": evaluate(args.model, baseline_sequences, truths),
        "probes": [],
    }

    for alpha in alphas:
        for spec_gap in spec_gaps:
            for decoder_margin in decoder_margins:
                reranked_sequences = []
                refined = 0
                for row in cached_rows:
                    beams = [dict(item) for item in row["beams"]]
                    decoder_probs = row["decoder_probs"]
                    spec_norm = row["spec_norm"]
                    for beam, dec, spec_n in zip(beams, decoder_probs, spec_norm):
                        beam["final_score"] = (1.0 - alpha) * dec + alpha * spec_n
                    spec_best_idx = int(np.argmax(spec_norm))
                    top_idx = 0
                    gap = spec_norm[spec_best_idx] - spec_norm[top_idx]
                    margin = decoder_probs[top_idx] - decoder_probs[spec_best_idx]
                    if (
                        spec_best_idx != top_idx
                        and gap >= spec_gap
                        and margin <= decoder_margin
                    ):
                        beams[spec_best_idx]["final_score"] = max(
                            beams[spec_best_idx]["final_score"],
                            beams[top_idx]["final_score"] + 1e-6,
                        )
                    beams.sort(key=lambda item: item["final_score"], reverse=True)
                    chosen = beams[0]["sequence"]
                    reranked_sequences.append(chosen)
                    if chosen != row["top1"]:
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

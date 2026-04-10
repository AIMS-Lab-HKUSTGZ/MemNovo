#!/usr/bin/env python3
"""Sweep hybrid offline reranking with local confidence gating plus disagreement gating."""

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
    diversity,
    entropy,
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
    parser.add_argument("--alphas", default="0.0,0.05,0.1,0.2,0.3,0.5,0.8")
    parser.add_argument("--spec-gaps", default="0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--decoder-margins", default="0.05,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--confidence-thresholds", default="none,0.45,0.5,0.6,0.7,0.8")
    parser.add_argument("--entropy-thresholds", default="none,0.0,0.5,1.0,1.5")
    parser.add_argument("--diversity-thresholds", default="0.0,0.5,0.8")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument(
        "--ion-mode",
        choices=["both", "y_only", "y_heavy", "b_only"],
        default="both",
    )
    parser.add_argument("--gate-requires-disagreement", action="store_true")
    return parser.parse_args()


def parse_float_list(values: str) -> list[float]:
    return [float(item) for item in values.split(",") if item]


def parse_optional_float_list(values: str) -> list[float | None]:
    parsed: list[float | None] = []
    for item in values.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"none", "null"}:
            parsed.append(None)
        else:
            parsed.append(float(item))
    return parsed


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

        raw_scores = [float(beam["decoder_score"]) for beam in beams]
        if args.model in {"casanovo", "primenovo"}:
            decoder_probs = minmax([max(score, 0.0) for score in raw_scores])
            top1_conf = float(raw_scores[0]) if raw_scores else 0.0
            row_entropy = 0.0
            row_diversity = 0.0
        else:
            decoder_probs = softmax(raw_scores)
            top1_conf = 0.0
            row_entropy = entropy(decoder_probs)
            row_diversity = diversity([beam["sequence"] for beam in beams])

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
        spec_best_idx = int(np.argmax(spec_norm))
        top_idx = 0
        spec_gap = spec_norm[spec_best_idx] - spec_norm[top_idx]
        decoder_margin = decoder_probs[top_idx] - decoder_probs[spec_best_idx]
        cached_rows.append(
            {
                "beams": beams,
                "decoder_probs": decoder_probs,
                "spec_norm": spec_norm,
                "spec_best_idx": spec_best_idx,
                "spec_gap": spec_gap,
                "decoder_margin": decoder_margin,
                "top1": row["top1"],
                "truth": row["truth"],
                "top1_conf": top1_conf,
                "entropy": row_entropy,
                "diversity": row_diversity,
            }
        )
        baseline_sequences.append(row["top1"])
        truths.append(row["truth"])

    return cached_rows, baseline_sequences, truths


def row_is_gated(row: dict, args: argparse.Namespace, confidence_threshold: float | None, entropy_threshold: float | None, diversity_threshold: float | None) -> bool:
    if args.model in {"casanovo", "primenovo"}:
        if confidence_threshold is None:
            return True
        return row["top1_conf"] < confidence_threshold
    if entropy_threshold is None:
        return True
    return row["entropy"] > entropy_threshold and row["diversity"] >= float(diversity_threshold or 0.0)


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.alphas)
    spec_gaps = parse_float_list(args.spec_gaps)
    decoder_margins = parse_float_list(args.decoder_margins)
    confidence_thresholds = parse_optional_float_list(args.confidence_thresholds)
    entropy_thresholds = parse_optional_float_list(args.entropy_thresholds)
    diversity_thresholds = parse_float_list(args.diversity_thresholds)

    cached_rows, baseline_sequences, truths = precompute(args)
    results = {
        "model": args.model,
        "ion_mode": args.ion_mode,
        "baseline": evaluate(args.model, baseline_sequences, truths),
        "probes": [],
    }

    for alpha in alphas:
        for spec_gap_threshold in spec_gaps:
            for decoder_margin_threshold in decoder_margins:
                if args.model in {"casanovo", "primenovo"}:
                    gate_grid = [(threshold, None, None) for threshold in confidence_thresholds]
                else:
                    gate_grid = [
                        (None, entropy_threshold, diversity_threshold)
                        for entropy_threshold in entropy_thresholds
                        for diversity_threshold in diversity_thresholds
                    ]

                for confidence_threshold, entropy_threshold, diversity_threshold in gate_grid:
                    reranked_sequences = []
                    gated_rows = 0
                    refined_rows = 0
                    changed_rows = 0

                    for row in cached_rows:
                        gated = row_is_gated(
                            row,
                            args,
                            confidence_threshold,
                            entropy_threshold,
                            diversity_threshold,
                        )
                        if gated:
                            gated_rows += 1

                        beams = [dict(item) for item in row["beams"]]
                        if not gated:
                            reranked_sequences.append(row["top1"])
                            continue

                        for beam, dec, spec_n in zip(beams, row["decoder_probs"], row["spec_norm"]):
                            beam["final_score"] = (1.0 - alpha) * dec + alpha * spec_n

                        disagreement_ok = (
                            row["spec_best_idx"] != 0
                            and row["spec_gap"] >= spec_gap_threshold
                            and row["decoder_margin"] <= decoder_margin_threshold
                        )
                        if disagreement_ok:
                            best_idx = row["spec_best_idx"]
                            beams[best_idx]["final_score"] = max(
                                beams[best_idx]["final_score"],
                                beams[0]["final_score"] + 1e-6,
                            )
                        elif args.gate_requires_disagreement:
                            reranked_sequences.append(row["top1"])
                            continue

                        beams.sort(key=lambda item: item["final_score"], reverse=True)
                        chosen = beams[0]["sequence"]
                        reranked_sequences.append(chosen)
                        refined_rows += 1
                        if chosen != row["top1"]:
                            changed_rows += 1

                    metrics = evaluate(args.model, reranked_sequences, truths)
                    metrics["alpha"] = alpha
                    metrics["spec_gap_threshold"] = spec_gap_threshold
                    metrics["decoder_margin_threshold"] = decoder_margin_threshold
                    metrics["gated_rows"] = gated_rows
                    metrics["refined_rows"] = refined_rows
                    metrics["changed_rows"] = changed_rows
                    if args.model in {"casanovo", "primenovo"}:
                        metrics["confidence_threshold"] = confidence_threshold
                    else:
                        metrics["entropy_threshold"] = entropy_threshold
                        metrics["diversity_threshold"] = diversity_threshold
                    results["probes"].append(metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

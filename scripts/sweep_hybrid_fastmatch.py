#!/usr/bin/env python3
"""Fast hybrid reranking sweep using normalized peptide exact-match only."""

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

from replay_rerank_probe import (
    advanced_spectrum_match_score,
    align_spectra,
    build_candidates,
    diversity,
    entropy,
    local_transition_features,
    local_transition_ok,
    load_spectra,
    minmax,
    softmax,
)
from evaluation.statistics_utils import normalize_peptide, normalize_ptm_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["casanovo", "instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alphas", default="0.0,0.05,0.1,0.2,0.3,0.5,0.8,1.0")
    parser.add_argument("--spec-gaps", default="0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--decoder-margins", default="0.05,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--confidence-thresholds", default="none,0.45,0.55,0.65,0.7,0.75,0.8")
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
    parser.add_argument("--max-local-diffs", type=int, default=None)
    parser.add_argument("--max-spec-rank", type=int, default=None)
    parser.add_argument("--require-suffix", action="store_true")
    parser.add_argument("--require-highrisk", action="store_true")
    parser.add_argument("--suffix-tail", type=int, default=3)
    parser.add_argument("--local-only", action="store_true")
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


def normalize_for_match(sequence: str, model: str) -> str:
    return normalize_peptide(normalize_ptm_format(sequence or "", model=model))


def precompute(args: argparse.Namespace) -> dict:
    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))
    cached_rows = []
    truths_norm = []
    baseline_norm = []
    oracle_topk_matches = 0

    for row, spectrum in zip(rows, spectra):
        beams = [dict(item) for item in row["candidates"][: args.top_k]]
        if not beams:
            continue

        raw_scores = [float(beam["decoder_score"]) for beam in beams]
        seqs = [beam["sequence"] for beam in beams]
        seqs_norm = [normalize_for_match(seq, args.model) for seq in seqs]
        truth_norm = normalize_for_match(row["truth"], args.model)
        top1_norm = normalize_for_match(row["top1"], args.model)
        if truth_norm in seqs_norm:
            oracle_topk_matches += 1

        if args.model in {"casanovo", "primenovo"}:
            decoder_probs = minmax([max(score, 0.0) for score in raw_scores])
            top1_conf = float(raw_scores[0]) if raw_scores else 0.0
            row_entropy = 0.0
            row_diversity = 0.0
        else:
            decoder_probs = softmax(raw_scores)
            top1_conf = 0.0
            row_entropy = entropy(decoder_probs)
            row_diversity = diversity(seqs)

        spec_scores = [
            advanced_spectrum_match_score(
                seq,
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
            for seq in seqs
        ]
        spec_norm = minmax(spec_scores)
        spec_best_idx = int(np.argmax(spec_norm))
        local_features = local_transition_features(seqs_norm[0], seqs_norm[spec_best_idx], args.suffix_tail)
        cached_rows.append(
            {
                "decoder_probs": decoder_probs,
                "spec_norm": spec_norm,
                "seqs_norm": seqs_norm,
                "top1_norm": top1_norm,
                "truth_norm": truth_norm,
                "top1_conf": top1_conf,
                "entropy": row_entropy,
                "diversity": row_diversity,
                "spec_best_idx": spec_best_idx,
                "spec_gap": spec_norm[spec_best_idx] - spec_norm[0],
                "decoder_margin": decoder_probs[0] - decoder_probs[spec_best_idx],
                "local_features": local_features,
            }
        )
        truths_norm.append(truth_norm)
        baseline_norm.append(top1_norm)

    baseline_matches = sum(int(pred == truth) for pred, truth in zip(baseline_norm, truths_norm))
    total = len(truths_norm)
    return {
        "rows": cached_rows,
        "baseline_matches": baseline_matches,
        "oracle_topk_matches": oracle_topk_matches,
        "total": total,
        "baseline_pep_recall": baseline_matches / total if total else 0.0,
        "oracle_topk_pep_recall": oracle_topk_matches / total if total else 0.0,
    }


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

    cached = precompute(args)
    rows = cached["rows"]
    baseline_matches = cached["baseline_matches"]
    total = cached["total"]
    results = {
        "model": args.model,
        "ion_mode": args.ion_mode,
        "baseline_matches": baseline_matches,
        "baseline_pep_recall": cached["baseline_pep_recall"],
        "oracle_topk_matches": cached["oracle_topk_matches"],
        "oracle_topk_pep_recall": cached["oracle_topk_pep_recall"],
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
                    matches = 0
                    gated_rows = 0
                    refined_rows = 0
                    changed_rows = 0
                    for row in rows:
                        gated = row_is_gated(
                            row,
                            args,
                            confidence_threshold,
                            entropy_threshold,
                            diversity_threshold,
                        )
                        if gated:
                            if not local_transition_ok(row["local_features"], row["spec_best_idx"], args):
                                gated = False
                        if gated:
                            gated_rows += 1

                        chosen_idx = 0
                        if gated:
                            disagreement_ok = (
                                row["spec_best_idx"] != 0
                                and row["spec_gap"] >= spec_gap_threshold
                                and row["decoder_margin"] <= decoder_margin_threshold
                            )
                            if disagreement_ok or not args.gate_requires_disagreement:
                                scores = [
                                    (1.0 - alpha) * dec + alpha * spec
                                    for dec, spec in zip(row["decoder_probs"], row["spec_norm"])
                                ]
                                if disagreement_ok:
                                    scores[row["spec_best_idx"]] = max(
                                        scores[row["spec_best_idx"]],
                                        scores[0] + 1e-6,
                                    )
                                if args.local_only and row["spec_best_idx"] != 0:
                                    top_idx = 0
                                    spec_idx = row["spec_best_idx"]
                                    chosen_idx = spec_idx if scores[spec_idx] > scores[top_idx] else top_idx
                                else:
                                    chosen_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
                                refined_rows += 1
                                if chosen_idx != 0:
                                    changed_rows += 1
                        if row["seqs_norm"][chosen_idx] == row["truth_norm"]:
                            matches += 1

                    pep_recall = matches / total if total else 0.0
                    results["probes"].append(
                        {
                            "alpha": alpha,
                            "spec_gap_threshold": spec_gap_threshold,
                            "decoder_margin_threshold": decoder_margin_threshold,
                            "confidence_threshold": confidence_threshold,
                            "entropy_threshold": entropy_threshold,
                            "diversity_threshold": diversity_threshold,
                            "gated_rows": gated_rows,
                            "refined_rows": refined_rows,
                            "changed_rows": changed_rows,
                            "matched_peptides": matches,
                            "pep_recall": pep_recall,
                            "pep_recall_delta": pep_recall - cached["baseline_pep_recall"],
                            "pep_recall_rel_pct": 0.0 if cached["baseline_pep_recall"] <= 0 else 100.0 * (pep_recall - cached["baseline_pep_recall"]) / cached["baseline_pep_recall"],
                            "n_match_pep_delta": matches - baseline_matches,
                        }
                    )

    best = max(results["probes"], key=lambda item: (item["pep_recall"], item["matched_peptides"]))
    results["best"] = best

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.model, "best": best}, indent=2))


if __name__ == "__main__":
    main()

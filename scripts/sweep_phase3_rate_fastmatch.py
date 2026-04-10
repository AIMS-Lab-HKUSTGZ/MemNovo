#!/usr/bin/env python3
"""Phase-3-style fast sweep with target refine-rate gating."""

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

from sweep_hybrid_fastmatch import parse_float_list, precompute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alphas", default="0.05,0.1,0.14,0.2,0.3")
    parser.add_argument("--spec-gaps", default="0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--decoder-margins", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--target-refine-rates", default="0.10,0.12,0.14,0.16,0.18,0.20")
    parser.add_argument(
        "--gate-score-modes",
        default="entropy,entropy_x_diversity,entropy_plus_diversity,phase3_combo",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument(
        "--ion-mode",
        choices=["both", "y_only", "y_heavy", "b_only"],
        default="y_heavy",
    )
    parser.add_argument("--gate-requires-disagreement", action="store_true")
    parser.add_argument("--suffix-tail", type=int, default=3)
    parser.add_argument("--require-suffix", action="store_true")
    parser.add_argument("--require-highrisk", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    return parser.parse_args()


def _normalize(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _phase3_gate_scores(rows: list[dict], mode: str) -> np.ndarray:
    entropy_vals = _normalize([float(row.get("entropy", 0.0)) for row in rows])
    diversity_vals = _normalize([float(row.get("diversity", 0.0)) for row in rows])
    conflict_vals = _normalize(
        [
            max(0.0, float(row.get("spec_gap", 0.0)) - max(float(row.get("decoder_margin", 0.0)), 0.0))
            for row in rows
        ]
    )
    if mode == "entropy":
        return entropy_vals
    if mode == "entropy_x_diversity":
        return entropy_vals * diversity_vals
    if mode == "entropy_plus_diversity":
        return 0.5 * entropy_vals + 0.5 * diversity_vals
    if mode == "phase3_combo":
        return 0.50 * entropy_vals + 0.30 * diversity_vals + 0.20 * conflict_vals
    raise ValueError(f"unknown gate score mode: {mode}")


def _passes_local_filters(row: dict, args: argparse.Namespace) -> bool:
    if args.require_suffix and not row["local_features"]["suffix_bias"]:
        return False
    if args.require_highrisk and not row["local_features"]["highrisk"]:
        return False
    return True


def main() -> None:
    args = parse_args()
    if args.model != "instanovo":
        raise SystemExit("phase3-style rate sweep is currently intended for instanovo only")

    alphas = parse_float_list(args.alphas)
    spec_gaps = parse_float_list(args.spec_gaps)
    decoder_margins = parse_float_list(args.decoder_margins)
    target_refine_rates = parse_float_list(args.target_refine_rates)
    gate_score_modes = [item.strip() for item in args.gate_score_modes.split(",") if item.strip()]

    cached = precompute(args)
    rows = cached["rows"]
    total = cached["total"]
    baseline_matches = cached["baseline_matches"]

    results = {
        "model": args.model,
        "ion_mode": args.ion_mode,
        "baseline_matches": baseline_matches,
        "baseline_pep_recall": cached["baseline_pep_recall"],
        "oracle_topk_matches": cached["oracle_topk_matches"],
        "oracle_topk_pep_recall": cached["oracle_topk_pep_recall"],
        "probes": [],
    }

    for gate_score_mode in gate_score_modes:
        gate_scores = _phase3_gate_scores(rows, gate_score_mode)
        order = np.argsort(-gate_scores)
        for target_refine_rate in target_refine_rates:
            n_target = max(1, int(round(total * target_refine_rate)))
            selected = set(int(idx) for idx in order[:n_target])

            for alpha in alphas:
                for spec_gap_threshold in spec_gaps:
                    for decoder_margin_threshold in decoder_margins:
                        matches = 0
                        refined_rows = 0
                        changed_rows = 0
                        for idx, row in enumerate(rows):
                            chosen_idx = 0
                            gated = idx in selected and _passes_local_filters(row, args)
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
                                        spec_idx = row["spec_best_idx"]
                                        chosen_idx = spec_idx if scores[spec_idx] > scores[0] else 0
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
                                "gate_score_mode": gate_score_mode,
                                "target_refine_rate": target_refine_rate,
                                "alpha": alpha,
                                "spec_gap_threshold": spec_gap_threshold,
                                "decoder_margin_threshold": decoder_margin_threshold,
                                "refined_rows": refined_rows,
                                "changed_rows": changed_rows,
                                "matched_peptides": matches,
                                "pep_recall": pep_recall,
                                "pep_recall_delta": pep_recall - cached["baseline_pep_recall"],
                                "pep_recall_rel_pct": 0.0
                                if cached["baseline_pep_recall"] <= 0
                                else 100.0 * (pep_recall - cached["baseline_pep_recall"]) / cached["baseline_pep_recall"],
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

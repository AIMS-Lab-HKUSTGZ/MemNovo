#!/usr/bin/env python3
"""Sweep InstaNovo local refinement variants with a single cached precompute pass."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from replay_rerank_probe import local_transition_ok
from sweep_hybrid_fastmatch import parse_float_list, parse_optional_float_list, precompute, row_is_gated


VARIANTS = [
    {"name": "d2_highrisk", "max_local_diffs": 2, "require_highrisk": True, "require_suffix": False, "max_spec_rank": 2, "local_only": True},
    {"name": "d2_suffix", "max_local_diffs": 2, "require_highrisk": False, "require_suffix": True, "max_spec_rank": 2, "local_only": True},
    {"name": "d2_highrisk_suffix", "max_local_diffs": 2, "require_highrisk": True, "require_suffix": True, "max_spec_rank": 2, "local_only": True},
    {"name": "d1_highrisk", "max_local_diffs": 1, "require_highrisk": True, "require_suffix": False, "max_spec_rank": 2, "local_only": True},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alphas", default="0.05,0.1,0.14,0.2,0.3")
    parser.add_argument("--spec-gaps", default="0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--decoder-margins", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--entropy-thresholds", default="none")
    parser.add_argument("--diversity-thresholds", default="0.0")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument("--ion-mode", choices=["both", "y_only", "y_heavy", "b_only"], default="y_heavy")
    parser.add_argument("--gate-requires-disagreement", action="store_true")
    parser.add_argument("--model", default="instanovo")
    parser.add_argument("--suffix-tail", type=int, default=3)
    return parser.parse_args()


def make_variant_args(base: argparse.Namespace, variant: dict) -> Namespace:
    data = vars(base).copy()
    data.update(variant)
    data["confidence_threshold"] = None
    return Namespace(**data)


def evaluate_variant(
    rows: list[dict],
    baseline_pep_recall: float,
    baseline_matches: int,
    total: int,
    base_args: argparse.Namespace,
    variant: dict,
    alpha: float,
    spec_gap_threshold: float,
    decoder_margin_threshold: float,
    entropy_threshold: float | None,
    diversity_threshold: float,
) -> dict:
    args = make_variant_args(base_args, variant)
    matches = 0
    gated_rows = 0
    refined_rows = 0
    changed_rows = 0
    for row in rows:
        gated = row_is_gated(
            row,
            args,
            None,
            entropy_threshold,
            diversity_threshold,
        )
        if gated and not local_transition_ok(row["local_features"], row["spec_best_idx"], args):
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
            if disagreement_ok or not base_args.gate_requires_disagreement:
                scores = [
                    (1.0 - alpha) * dec + alpha * spec
                    for dec, spec in zip(row["decoder_probs"], row["spec_norm"])
                ]
                if disagreement_ok:
                    scores[row["spec_best_idx"]] = max(scores[row["spec_best_idx"]], scores[0] + 1e-6)
                if args.local_only and row["spec_best_idx"] != 0:
                    chosen_idx = row["spec_best_idx"] if scores[row["spec_best_idx"]] > scores[0] else 0
                else:
                    chosen_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
                refined_rows += 1
                if chosen_idx != 0:
                    changed_rows += 1

        if row["seqs_norm"][chosen_idx] == row["truth_norm"]:
            matches += 1

    pep_recall = matches / total if total else 0.0
    return {
        "variant": variant["name"],
        "alpha": alpha,
        "spec_gap_threshold": spec_gap_threshold,
        "decoder_margin_threshold": decoder_margin_threshold,
        "entropy_threshold": entropy_threshold,
        "diversity_threshold": diversity_threshold,
        "gated_rows": gated_rows,
        "refined_rows": refined_rows,
        "changed_rows": changed_rows,
        "matched_peptides": matches,
        "pep_recall": pep_recall,
        "pep_recall_delta": pep_recall - baseline_pep_recall,
        "pep_recall_rel_pct": 0.0 if baseline_pep_recall <= 0 else 100.0 * (pep_recall - baseline_pep_recall) / baseline_pep_recall,
        "n_match_pep_delta": matches - baseline_matches,
        "max_local_diffs": variant["max_local_diffs"],
        "require_highrisk": variant["require_highrisk"],
        "require_suffix": variant["require_suffix"],
        "max_spec_rank": variant["max_spec_rank"],
        "local_only": variant["local_only"],
    }


def main() -> None:
    args = parse_args()
    if args.model != "instanovo":
        raise ValueError("This script is specialized for InstaNovo.")

    cached = precompute(args)
    rows = cached["rows"]
    baseline_matches = cached["baseline_matches"]
    total = cached["total"]
    baseline_pep_recall = cached["baseline_pep_recall"]
    alphas = parse_float_list(args.alphas)
    spec_gaps = parse_float_list(args.spec_gaps)
    decoder_margins = parse_float_list(args.decoder_margins)
    entropy_thresholds = parse_optional_float_list(args.entropy_thresholds)
    diversity_thresholds = parse_float_list(args.diversity_thresholds)

    results = {
        "model": args.model,
        "ion_mode": args.ion_mode,
        "baseline_matches": baseline_matches,
        "baseline_pep_recall": baseline_pep_recall,
        "oracle_topk_matches": cached["oracle_topk_matches"],
        "oracle_topk_pep_recall": cached["oracle_topk_pep_recall"],
        "probes": [],
    }

    for variant in VARIANTS:
        print(f"RUN {variant['name']}", flush=True)
        best = None
        for alpha in alphas:
            for spec_gap_threshold in spec_gaps:
                for decoder_margin_threshold in decoder_margins:
                    for entropy_threshold in entropy_thresholds:
                        for diversity_threshold in diversity_thresholds:
                            metrics = evaluate_variant(
                                rows,
                                baseline_pep_recall,
                                baseline_matches,
                                total,
                                args,
                                variant,
                                alpha,
                                spec_gap_threshold,
                                decoder_margin_threshold,
                                entropy_threshold,
                                diversity_threshold,
                            )
                            results["probes"].append(metrics)
                            if best is None or (metrics["pep_recall"], metrics["matched_peptides"]) > (best["pep_recall"], best["matched_peptides"]):
                                best = metrics
        print(json.dumps({"variant": variant["name"], "best_rel_pct": best["pep_recall_rel_pct"], "best_pep_recall": best["pep_recall"], "n_match_pep_delta": best["n_match_pep_delta"]}), flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

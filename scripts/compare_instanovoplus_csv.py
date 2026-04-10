#!/usr/bin/env python3
"""Compare official InstaNovo baseline and refined CSV outputs on the same MGF subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.data_handler import DataHandler
from replay_rerank_probe import normalize_peptide, normalize_ptm_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--refined", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def norm(sequence: str) -> str:
    return normalize_peptide(normalize_ptm_format(str(sequence or ""), model="instanovo"))


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline)
    refined = pd.read_csv(args.refined)
    truth = DataHandler({"path": args.spectra, "format": "auto"}).load_data()

    if not (len(baseline) == len(refined) == len(truth)):
        raise ValueError(
            f"Length mismatch: baseline={len(baseline)} refined={len(refined)} truth={len(truth)}"
        )

    baseline_col = "preds" if "preds" in baseline.columns else "final_prediction"
    if "final_prediction" in refined.columns:
        refined_col = "final_prediction"
    elif "preds" in refined.columns:
        refined_col = "preds"
    else:
        raise KeyError(
            "Refined CSV must contain either 'final_prediction' or 'preds' column"
        )

    baseline_pred = [norm(x) for x in baseline[baseline_col]]
    refined_pred = [norm(x) for x in refined[refined_col]]
    truth_seq = [norm(x) for x in truth["sequence"]]

    baseline_match = sum(int(pred == target) for pred, target in zip(baseline_pred, truth_seq))
    refined_match = sum(int(pred == target) for pred, target in zip(refined_pred, truth_seq))
    pred_diff = sum(1 for before, after in zip(baseline_pred, refined_pred) if before != after)
    if "selected_model" in refined.columns:
        selected_model_counts = refined["selected_model"].value_counts(dropna=False).to_dict()
    else:
        selected_model_counts = {
            "baseline_like": sum(
                int(before == after) for before, after in zip(baseline_pred, refined_pred)
            ),
            "changed_prediction": sum(
                int(before != after) for before, after in zip(baseline_pred, refined_pred)
            ),
        }

    total = len(truth_seq)
    result = {
        "n": total,
        "baseline_match": baseline_match,
        "baseline_pep_recall": baseline_match / total if total else 0.0,
        "refined_match": refined_match,
        "refined_pep_recall": refined_match / total if total else 0.0,
        "delta_match": refined_match - baseline_match,
        "delta_pep_recall": (
            refined_match / total - baseline_match / total if total else 0.0
        ),
        "delta_rel_pct": (
            (refined_match - baseline_match) / baseline_match * 100.0
            if baseline_match
            else None
        ),
        "pred_diff": pred_diff,
        "selected_model_counts": selected_model_counts,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

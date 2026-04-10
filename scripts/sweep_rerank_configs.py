#!/usr/bin/env python3
"""Grid search offline selective reranking configs on existing beam outputs."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from replay_rerank_probe import (
    align_spectra,
    build_candidates,
    diversity,
    entropy,
    evaluate_predictions,
    load_spectra,
    minmax,
    softmax,
    peptide_ions,
    spectrum_match_score,
)
from evaluation.statistics_utils import calculate_sequence_masses, normalize_ptm_format, parse_peptide_with_ptm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["casanovo", "instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument(
        "--score-mode",
        choices=["basic_norm", "improved_norm", "guided_raw"],
        default="basic_norm",
    )
    parser.add_argument("--mass-feature-weight", type=float, default=0.0)
    parser.add_argument("--guided-model-weight", type=float, default=0.967)
    parser.add_argument("--guided-spectrum-weight", type=float, default=0.034)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--refine-modes", nargs="+", default=["selective", "all"])
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
    )
    parser.add_argument(
        "--entropy-thresholds",
        type=float,
        nargs="+",
        default=[0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00],
    )
    parser.add_argument(
        "--diversity-thresholds",
        type=float,
        nargs="+",
        default=[0.20, 0.40, 0.60, 0.80, 1.00],
    )
    return parser.parse_args()


def _sequence_masses_full(sequence: str, model: str) -> List[float]:
    normalized = normalize_ptm_format(sequence, model=model)
    residues = parse_peptide_with_ptm(normalized)
    prefix_masses, _ = calculate_sequence_masses(residues)
    return prefix_masses


def peptide_total_mass(sequence: str, model: str) -> float | None:
    try:
        masses = _sequence_masses_full(sequence, model)
        if not masses:
            return None
        return float(masses[-1])
    except Exception:
        return None


def improved_match_components(
    sequence: str,
    spectrum: Dict[str, Any],
    model: str,
    tolerance: float = 0.5,
    exact_tolerance: float = 0.1,
) -> Dict[str, float]:
    if not sequence:
        return {
            "b_coverage": 0.0,
            "y_coverage": 0.0,
            "intensity_ratio": 0.0,
            "exact_matches": 0.0,
            "combined": 0.0,
        }

    mz = np.asarray(spectrum["mz_array"], dtype=np.float32)
    intensity = np.asarray(spectrum["intensity_array"], dtype=np.float32)
    if intensity.size == 0 or float(intensity.max()) <= 0:
        return {
            "b_coverage": 0.0,
            "y_coverage": 0.0,
            "intensity_ratio": 0.0,
            "exact_matches": 0.0,
            "combined": 0.0,
        }

    intens = intensity / float(intensity.max())
    prefix_masses = _sequence_masses_full(sequence, model=model)
    if len(prefix_masses) <= 1:
        return {
            "b_coverage": 0.0,
            "y_coverage": 0.0,
            "intensity_ratio": 0.0,
            "exact_matches": 0.0,
            "combined": 0.0,
        }

    b_ions = prefix_masses[1:]
    total_mass = prefix_masses[-1]
    y_ions = [total_mass - mass + 18.015 for mass in prefix_masses[:-1]]

    def _match(ions: List[float]) -> tuple[int, float, int]:
        matched = 0
        total_i = 0.0
        exact = 0
        for ion in ions:
            diffs = np.abs(mz - ion)
            idx = int(np.argmin(diffs))
            diff = float(diffs[idx])
            if diff <= tolerance:
                matched += 1
                total_i += float(intens[idx])
                if diff <= exact_tolerance:
                    exact += 1
        return matched, total_i, exact

    b_match_n, b_i, b_exact = _match(b_ions)
    y_match_n, y_i, y_exact = _match(y_ions)
    b_cov = b_match_n / len(b_ions) if b_ions else 0.0
    y_cov = y_match_n / len(y_ions) if y_ions else 0.0
    intensity_ratio = b_i + y_i
    exact_matches = b_exact + y_exact
    combined = (
        0.3 * b_cov
        + 0.3 * y_cov
        + 0.35 * intensity_ratio
        + 0.05 * min(exact_matches / max(len(sequence), 1), 1.0)
    )
    return {
        "b_coverage": float(b_cov),
        "y_coverage": float(y_cov),
        "intensity_ratio": float(intensity_ratio),
        "exact_matches": float(exact_matches),
        "combined": float(combined),
    }


def guided_spectrum_score(
    sequence: str,
    spectrum: Dict[str, Any],
    model: str,
    tolerance: float = 0.02,
) -> float:
    ions = peptide_ions(sequence, model=model)
    if not ions:
        return 0.0
    mz = np.asarray(spectrum["mz_array"], dtype=np.float32)
    intensity = np.asarray(spectrum["intensity_array"], dtype=np.float32)
    if intensity.size == 0 or float(intensity.max()) <= 0:
        return 0.0
    intens = intensity / float(intensity.max())
    total = 0.0
    for ion in ions:
        within = np.abs(mz - ion) < tolerance
        if np.any(within):
            total += float(np.sum(intens[within]))
    return min(total / len(ions), 1.0)


def precompute_features(args: argparse.Namespace) -> Dict[str, Any]:
    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))
    truths = [row["truth"] for row in rows]
    baseline_sequences = [row["top1"] for row in rows]
    prepared_rows: List[Dict[str, Any]] = []

    for row, spectrum in zip(rows, spectra):
        beams = [dict(item) for item in row["candidates"][: args.top_k]]
        raw_scores = [float(beam["decoder_score"]) for beam in beams]
        seqs = [beam["sequence"] for beam in beams]
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

        precursor_mass = float(spectrum["precursor_mz"]) * max(int(spectrum.get("precursor_charge", 1)), 1)
        mass_errors = []
        for seq in seqs:
            total_mass = peptide_total_mass(seq, args.model)
            mass_errors.append(float("inf") if total_mass is None else abs(total_mass - precursor_mass))

        finite_mass_errors = [err for err in mass_errors if np.isfinite(err)]
        if finite_mass_errors:
            inv = [0.0 if not np.isfinite(err) else 1.0 / (1.0 + err) for err in mass_errors]
            mass_features = minmax(inv)
        else:
            mass_features = [0.0 for _ in seqs]

        if args.score_mode == "basic_norm":
            spec_scores = [
                spectrum_match_score(
                    seq,
                    spectrum["mz_array"],
                    spectrum["intensity_array"],
                    model=args.model,
                    tol_da=args.mass_tol_da,
                )
                for seq in seqs
            ]
            spec_features = minmax(spec_scores)
        elif args.score_mode == "improved_norm":
            spec_scores = [
                improved_match_components(
                    seq,
                    spectrum,
                    model=args.model,
                    tolerance=args.mass_tol_da,
                    exact_tolerance=0.1,
                )["combined"]
                for seq in seqs
            ]
            spec_features = minmax(spec_scores)
        else:
            spec_scores = [
                guided_spectrum_score(
                    seq,
                    spectrum,
                    model=args.model,
                    tolerance=0.02,
                )
                for seq in seqs
            ]
            spec_features = spec_scores

        if args.mass_feature_weight > 0:
            spec_features = [
                (1.0 - args.mass_feature_weight) * spec + args.mass_feature_weight * mass_f
                for spec, mass_f in zip(spec_features, mass_features)
            ]

        prepared_rows.append(
            {
                "top1": row["top1"],
                "sequences": seqs,
                "raw_scores": raw_scores,
                "decoder_probs": decoder_probs,
                "spec_features": spec_features,
                "mass_features": mass_features,
                "top1_conf": top1_conf,
                "entropy": row_entropy,
                "diversity": row_diversity,
            }
        )

    baseline_metrics = evaluate_predictions(args.model, baseline_sequences, truths)
    return {
        "truths": truths,
        "baseline_sequences": baseline_sequences,
        "baseline_metrics": baseline_metrics,
        "rows": prepared_rows,
    }


def should_refine_row(model: str, row: Dict[str, Any], config: Dict[str, Any]) -> bool:
    if config["refine_all"]:
        return True
    if model in {"casanovo", "primenovo"}:
        return row["top1_conf"] < config["confidence_threshold"]
    return row["entropy"] > config["entropy_threshold"] and row["diversity"] >= config["diversity_threshold"]


def choose_sequence(row: Dict[str, Any], alpha: float, args: argparse.Namespace) -> str:
    if not row["sequences"]:
        return ""
    if len(row["sequences"]) == 1:
        return row["sequences"][0]

    if args.score_mode == "guided_raw":
        scores = [
            args.guided_model_weight * raw + args.guided_spectrum_weight * spec * 100.0
            for raw, spec in zip(row["raw_scores"], row["spec_features"])
        ]
    else:
        scores = [
            (1.0 - alpha) * dec + alpha * spec
            for dec, spec in zip(row["decoder_probs"], row["spec_features"])
        ]
    best_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return row["sequences"][best_idx]


def evaluate_config(
    model: str,
    args: argparse.Namespace,
    truths: List[str],
    baseline_sequences: List[str],
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    predictions: List[str] = []
    refined_rows = 0
    changed_rows = 0

    for baseline_seq, row in zip(baseline_sequences, rows):
        if should_refine_row(model, row, config):
            seq = choose_sequence(row, config["alpha"], args)
            if seq != baseline_seq:
                changed_rows += 1
            refined_rows += 1
        else:
            seq = baseline_seq
        predictions.append(seq)

    metrics = evaluate_predictions(model, predictions, truths)
    metrics.update(
        {
            "alpha": config["alpha"],
            "refine_all": config["refine_all"],
            "refined_rows": refined_rows,
            "changed_rows": changed_rows,
        }
    )
    if model in {"casanovo", "primenovo"}:
        metrics["confidence_threshold"] = config["confidence_threshold"]
    else:
        metrics["entropy_threshold"] = config["entropy_threshold"]
        metrics["diversity_threshold"] = config["diversity_threshold"]
    return metrics


def build_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    refine_modes = [mode.lower() for mode in args.refine_modes]
    grid: List[Dict[str, Any]] = []

    if args.model in {"casanovo", "primenovo"}:
        for alpha, refine_mode in product(args.alpha, refine_modes):
            if refine_mode == "all":
                grid.append({"alpha": alpha, "refine_all": True, "confidence_threshold": None})
                continue
            for threshold in args.confidence_thresholds:
                grid.append(
                    {
                        "alpha": alpha,
                        "refine_all": False,
                        "confidence_threshold": threshold,
                    }
                )
        return grid

    for alpha, refine_mode in product(args.alpha, refine_modes):
        if refine_mode == "all":
            grid.append(
                {
                    "alpha": alpha,
                    "refine_all": True,
                    "entropy_threshold": None,
                    "diversity_threshold": None,
                }
            )
            continue
        for entropy_threshold, diversity_threshold in product(
            args.entropy_thresholds,
            args.diversity_thresholds,
        ):
            grid.append(
                {
                    "alpha": alpha,
                    "refine_all": False,
                    "entropy_threshold": entropy_threshold,
                    "diversity_threshold": diversity_threshold,
                }
            )
    return grid


def main() -> None:
    args = parse_args()
    cached = precompute_features(args)
    baseline_metrics = cached["baseline_metrics"]
    truths = cached["truths"]
    baseline_sequences = cached["baseline_sequences"]
    rows = cached["rows"]

    results = []
    for config in build_grid(args):
        metrics = evaluate_config(args.model, args, truths, baseline_sequences, rows, config)
        base_pep = float(baseline_metrics.get("pep_recall", 0.0))
        pep_recall = float(metrics.get("pep_recall", 0.0))
        metrics["pep_recall_delta"] = pep_recall - base_pep
        metrics["pep_recall_rel_pct"] = 0.0 if base_pep <= 0 else 100.0 * (pep_recall - base_pep) / base_pep
        metrics["n_match_pep_delta"] = int(metrics.get("n_match_pep", 0)) - int(baseline_metrics.get("n_match_pep", 0))
        results.append(metrics)

    best = max(
        results,
        key=lambda item: (
            float(item.get("pep_recall", -1.0)),
            float(item.get("aa_recall", -1.0)),
            int(item.get("n_match_pep", -1)),
        ),
    )
    output = {
        "model": args.model,
        "score_mode": args.score_mode,
        "mass_feature_weight": args.mass_feature_weight,
        "baseline": baseline_metrics,
        "best": best,
        "probes": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.model, "best": best}, indent=2))


if __name__ == "__main__":
    main()

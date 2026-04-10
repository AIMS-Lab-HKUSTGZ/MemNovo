#!/usr/bin/env python3
"""Fast sweep using advanced scorer + archive SpectrumMatcher as a blended spectral signal."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ARCHIVE_PYC_ROOT = WORKSPACE_ROOT / "archieved" / "legacy_workspace" / "src" / "__pycache__"

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
    load_spectra,
    minmax,
    softmax,
)
from statistics_utils import normalize_peptide, normalize_ptm_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alphas", default="0.05,0.1,0.2,0.3,0.5,0.8")
    parser.add_argument("--betas", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--confidence-thresholds", default="none,0.70,0.75,0.8,0.85")
    parser.add_argument("--entropy-thresholds", default="none,0.0,0.5,1.0,1.5")
    parser.add_argument("--diversity-thresholds", default="0.0,0.5,0.8")
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument("--ion-mode", choices=["both", "y_only", "y_heavy", "b_only"], default="both")
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


def load_archive_matcher_module():
    path = ARCHIVE_PYC_ROOT / "spectrum_matcher.cpython-312.pyc"
    name = "archive_spectrum_matcher_dual"
    loader = importlib.machinery.SourcelessFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def normalize_for_match(sequence: str, model: str) -> str:
    return normalize_peptide(normalize_ptm_format(sequence or "", model=model))


def row_is_gated(model: str, row: dict, confidence_threshold: float | None, entropy_threshold: float | None, diversity_threshold: float | None) -> bool:
    if model == "primenovo":
        if confidence_threshold is None:
            return True
        return row["top1_conf"] < confidence_threshold
    if entropy_threshold is None:
        return True
    return row["entropy"] > entropy_threshold and row["diversity"] >= float(diversity_threshold or 0.0)


def precompute(args: argparse.Namespace) -> dict:
    module = load_archive_matcher_module()
    matcher = module.SpectrumMatcher(tolerance_ppm=20.0, use_intensity=True, fragment_types=["b", "y"])
    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))
    cached_rows = []
    baseline_matches = 0
    for row, spectrum in zip(rows, spectra):
        beams = [dict(item) for item in row["candidates"][: args.top_k]]
        if not beams:
            continue
        seqs = [beam["sequence"] for beam in beams]
        seqs_norm = [normalize_for_match(seq, args.model) for seq in seqs]
        truth_norm = normalize_for_match(row["truth"], args.model)
        top1_norm = normalize_for_match(row["top1"], args.model)
        if top1_norm == truth_norm:
            baseline_matches += 1

        raw_scores = [float(beam["decoder_score"]) for beam in beams]
        if args.model == "primenovo":
            decoder_scores = minmax([max(score, 0.0) for score in raw_scores])
            row_entropy = 0.0
            row_diversity = 0.0
            top1_conf = float(raw_scores[0]) if raw_scores else 0.0
        else:
            decoder_scores = softmax(raw_scores)
            row_entropy = entropy(decoder_scores)
            row_diversity = diversity(seqs)
            top1_conf = 0.0

        mz_array_np = np.asarray(spectrum["mz_array"], dtype=np.float32)
        intensity_array_np = np.asarray(spectrum["intensity_array"], dtype=np.float32)
        mz_array_t = torch.tensor(mz_array_np)
        intensity_array_t = torch.tensor(intensity_array_np)
        precursor_mz = float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0)))
        precursor_charge = int(spectrum.get("precursor_charge", row.get("precursor_charge", 0)))

        adv_scores = []
        matcher_scores = []
        for seq in seqs:
            adv_scores.append(
                advanced_spectrum_match_score(
                    seq,
                    mz_array_np,
                    intensity_array_np,
                    precursor_mz=precursor_mz,
                    precursor_charge=precursor_charge,
                    model=args.model,
                    tol_da=args.mass_tol_da,
                    precursor_ppm=args.precursor_ppm,
                    top_peak_frac=args.top_peak_frac,
                    ion_mode=args.ion_mode,
                )
            )
            matcher_scores.append(
                float(
                    matcher.compute_matching_score(
                        sequence=seq,
                        mz_array=mz_array_t,
                        intensity_array=intensity_array_t,
                        precursor_mz=precursor_mz,
                        precursor_charge=precursor_charge,
                    )
                )
            )

        cached_rows.append(
            {
                "seqs_norm": seqs_norm,
                "truth_norm": truth_norm,
                "decoder_scores": decoder_scores,
                "adv_norm": minmax(adv_scores),
                "matcher_norm": minmax(matcher_scores),
                "entropy": row_entropy,
                "diversity": row_diversity,
                "top1_conf": top1_conf,
            }
        )

    total = len(cached_rows)
    return {
        "rows": cached_rows,
        "baseline_matches": baseline_matches,
        "baseline_pep_recall": baseline_matches / total if total else 0.0,
        "total": total,
    }


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    confidence_thresholds = parse_optional_float_list(args.confidence_thresholds)
    entropy_thresholds = parse_optional_float_list(args.entropy_thresholds)
    diversity_thresholds = parse_float_list(args.diversity_thresholds)

    cached = precompute(args)
    rows = cached["rows"]
    results = {
        "model": args.model,
        "baseline_matches": cached["baseline_matches"],
        "baseline_pep_recall": cached["baseline_pep_recall"],
        "probes": [],
    }

    for alpha in alphas:
        for beta in betas:
            gate_grid = (
                [(None, et, dt) for et in entropy_thresholds for dt in diversity_thresholds]
                if args.model == "instanovo"
                else [(ct, None, None) for ct in confidence_thresholds]
            )
            for confidence_threshold, entropy_threshold, diversity_threshold in gate_grid:
                matches = 0
                refined_rows = 0
                changed_rows = 0
                for row in rows:
                    gated = row_is_gated(args.model, row, confidence_threshold, entropy_threshold, diversity_threshold)
                    chosen_idx = 0
                    if gated:
                        blended_spec = [
                            beta * adv + (1.0 - beta) * matcher
                            for adv, matcher in zip(row["adv_norm"], row["matcher_norm"])
                        ]
                        scores = [
                            (1.0 - alpha) * dec + alpha * spec
                            for dec, spec in zip(row["decoder_scores"], blended_spec)
                        ]
                        chosen_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
                        refined_rows += 1
                        if chosen_idx != 0:
                            changed_rows += 1
                    if row["seqs_norm"][chosen_idx] == row["truth_norm"]:
                        matches += 1
                pep_recall = matches / cached["total"] if cached["total"] else 0.0
                results["probes"].append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "confidence_threshold": confidence_threshold,
                        "entropy_threshold": entropy_threshold,
                        "diversity_threshold": diversity_threshold,
                        "refined_rows": refined_rows,
                        "changed_rows": changed_rows,
                        "matched_peptides": matches,
                        "pep_recall": pep_recall,
                        "pep_recall_delta": pep_recall - cached["baseline_pep_recall"],
                        "pep_recall_rel_pct": 0.0 if cached["baseline_pep_recall"] <= 0 else 100.0 * (pep_recall - cached["baseline_pep_recall"]) / cached["baseline_pep_recall"],
                    }
                )

    best = max(results["probes"], key=lambda item: item["pep_recall_rel_pct"], default=None)
    results["best"] = best
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "best": best}, indent=2))


if __name__ == "__main__":
    main()

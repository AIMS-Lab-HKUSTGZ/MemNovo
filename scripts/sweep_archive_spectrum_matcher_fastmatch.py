#!/usr/bin/env python3
"""Fast sweep using archive spectrum_matcher + uncertainty detector on saved beam outputs."""

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

from replay_rerank_probe import align_spectra, build_candidates, load_spectra
from evaluation.statistics_utils import normalize_peptide, normalize_ptm_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-weights", default="0.5,0.7,0.9,1.0")
    parser.add_argument("--spectrum-weights", default="1,5,10,20,50,100")
    parser.add_argument("--uncertainty-thresholds", default="none,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--confidence-thresholds", default="none,0.70,0.75,0.8,0.85")
    parser.add_argument("--refine-all", action="store_true")
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


def load_archive_module():
    path = ARCHIVE_PYC_ROOT / "spectrum_matcher.cpython-312.pyc"
    name = "archive_spectrum_matcher_fast"
    loader = importlib.machinery.SourcelessFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def normalize_for_match(sequence: str, model: str) -> str:
    return normalize_peptide(normalize_ptm_format(sequence or "", model=model))


def precompute(args: argparse.Namespace) -> dict:
    module = load_archive_module()
    matcher = module.SpectrumMatcher(tolerance_ppm=20.0, use_intensity=True, fragment_types=["b", "y"])
    uncertainty_detector = module.UncertaintyDetector(uncertainty_threshold=0.3)
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
        if args.model == "instanovo":
            score_tensor = torch.tensor(raw_scores, dtype=torch.float32).unsqueeze(0)
            _, uncertainty_scores = uncertainty_detector.detect_uncertainty(score_tensor)
            uncertainty = float(uncertainty_scores[0].item())
            top1_conf = 0.0
        else:
            uncertainty = 0.0
            top1_conf = float(raw_scores[0]) if raw_scores else 0.0

        mz_array = torch.tensor(np.asarray(spectrum["mz_array"], dtype=np.float32))
        intensity_array = torch.tensor(np.asarray(spectrum["intensity_array"], dtype=np.float32))
        precursor_mz = float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0)))
        precursor_charge = int(spectrum.get("precursor_charge", row.get("precursor_charge", 0)))
        match_scores = []
        for seq in seqs:
            match_scores.append(
                float(
                    matcher.compute_matching_score(
                        sequence=seq,
                        mz_array=mz_array,
                        intensity_array=intensity_array,
                        precursor_mz=precursor_mz,
                        precursor_charge=precursor_charge,
                    )
                )
            )

        cached_rows.append(
            {
                "seqs_norm": seqs_norm,
                "truth_norm": truth_norm,
                "raw_scores": raw_scores,
                "match_scores": match_scores,
                "uncertainty": uncertainty,
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


def row_is_gated(model: str, row: dict, uncertainty_threshold: float | None, confidence_threshold: float | None) -> bool:
    if model == "instanovo":
        if uncertainty_threshold is None:
            return True
        return row["uncertainty"] > uncertainty_threshold
    if confidence_threshold is None:
        return True
    return row["top1_conf"] < confidence_threshold


def main() -> None:
    args = parse_args()
    model_weights = parse_float_list(args.model_weights)
    spectrum_weights = parse_float_list(args.spectrum_weights)
    uncertainty_thresholds = parse_optional_float_list(args.uncertainty_thresholds)
    confidence_thresholds = parse_optional_float_list(args.confidence_thresholds)

    cached = precompute(args)
    rows = cached["rows"]
    results = {
        "model": args.model,
        "baseline_matches": cached["baseline_matches"],
        "baseline_pep_recall": cached["baseline_pep_recall"],
        "probes": [],
    }

    for model_weight in model_weights:
        for spectrum_weight in spectrum_weights:
            gate_grid = (
                [(u, None) for u in uncertainty_thresholds]
                if args.model == "instanovo"
                else [(None, c) for c in confidence_thresholds]
            )
            for uncertainty_threshold, confidence_threshold in gate_grid:
                matches = 0
                refined_rows = 0
                changed_rows = 0
                for row in rows:
                    gated = args.refine_all or row_is_gated(args.model, row, uncertainty_threshold, confidence_threshold)
                    chosen_idx = 0
                    if gated:
                        scores = [
                            model_weight * raw + spectrum_weight * ms
                            for raw, ms in zip(row["raw_scores"], row["match_scores"])
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
                        "model_weight": model_weight,
                        "spectrum_weight": spectrum_weight,
                        "uncertainty_threshold": uncertainty_threshold,
                        "confidence_threshold": confidence_threshold,
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

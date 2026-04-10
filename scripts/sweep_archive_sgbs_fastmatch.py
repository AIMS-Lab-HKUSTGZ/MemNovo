#!/usr/bin/env python3
"""Fast sweep using archive spectrum_guided_beam_search rescoring on saved beam outputs."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ARCHIVE_PYC_ROOT = WORKSPACE_ROOT / "archieved" / "legacy_workspace" / "src" / "__pycache__"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from replay_rerank_probe import align_spectra, build_candidates, diversity, entropy, load_spectra
from statistics_utils import normalize_peptide, normalize_ptm_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-weights", default="0.94,0.95,0.96,0.967,0.975,0.98")
    parser.add_argument("--spectrum-weights", default="0.01,0.02,0.03,0.034,0.04,0.05")
    parser.add_argument("--entropy-thresholds", default="none,0.0,0.5,1.0,1.5")
    parser.add_argument("--diversity-thresholds", default="0.0,0.5,0.8")
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
    path = ARCHIVE_PYC_ROOT / "spectrum_guided_beam_search.cpython-312.pyc"
    name = "archive_spectrum_guided_beam_search_fast"
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


def build_observed_matrix(spectrum: dict) -> np.ndarray:
    mz = np.asarray(spectrum["mz_array"], dtype=np.float32)
    intensity = np.asarray(spectrum["intensity_array"], dtype=np.float32)
    return np.stack([mz, intensity], axis=1)


def precompute(args: argparse.Namespace) -> dict:
    module = load_archive_module()
    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))
    residue_masses = None
    matcher = module.SpectrumMatcher(residue_masses=residue_masses, tolerance_da=0.02, use_b_ions=True, use_y_ions=True)

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
            top1_conf = float(raw_scores[0]) if raw_scores else 0.0
            row_entropy = 0.0
            row_diversity = 0.0
        else:
            probs = np.asarray(raw_scores, dtype=np.float64)
            probs = np.exp(probs - probs.max())
            probs = probs / max(float(probs.sum()), 1e-12)
            row_entropy = entropy(probs.tolist())
            row_diversity = diversity(seqs)
            top1_conf = 0.0

        observed = build_observed_matrix(spectrum)
        charge = int(spectrum.get("precursor_charge", row.get("precursor_charge", 0)))
        spectrum_scores = []
        for seq in seqs:
            frags = matcher.predict_spectrum(seq, charge)
            spectrum_scores.append(float(matcher.match_spectrum(observed, frags)))

        cached_rows.append(
            {
                "seqs_norm": seqs_norm,
                "truth_norm": truth_norm,
                "raw_scores": raw_scores,
                "spectrum_scores": spectrum_scores,
                "top1_conf": top1_conf,
                "entropy": row_entropy,
                "diversity": row_diversity,
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
    model_weights = parse_float_list(args.model_weights)
    spectrum_weights = parse_float_list(args.spectrum_weights)
    entropy_thresholds = parse_optional_float_list(args.entropy_thresholds)
    diversity_thresholds = parse_float_list(args.diversity_thresholds)
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
            if args.model == "primenovo":
                gate_grid = [(threshold, None, None) for threshold in confidence_thresholds]
            else:
                gate_grid = [
                    (None, entropy_threshold, diversity_threshold)
                    for entropy_threshold in entropy_thresholds
                    for diversity_threshold in diversity_thresholds
                ]

            for confidence_threshold, entropy_threshold, diversity_threshold in gate_grid:
                matches = 0
                refined_rows = 0
                changed_rows = 0
                for row in rows:
                    gated = args.refine_all or row_is_gated(
                        args.model,
                        row,
                        confidence_threshold,
                        entropy_threshold,
                        diversity_threshold,
                    )
                    chosen_idx = 0
                    if gated:
                        candidates = []
                        for seq_norm, raw_score, spectrum_score in zip(
                            row["seqs_norm"], row["raw_scores"], row["spectrum_scores"]
                        ):
                            candidates.append(
                                SimpleNamespace(
                                    sequence=seq_norm,
                                    sequence_log_probability=float(raw_score),
                                    score=float(raw_score),
                                    spectrum_score=float(spectrum_score),
                                )
                            )
                        for cand in candidates:
                            cand.original_score = cand.sequence_log_probability
                            cand.combined_score = model_weight * cand.original_score + spectrum_weight * cand.spectrum_score * 100.0
                        candidates.sort(key=lambda x: x.combined_score, reverse=True)
                        best_seq = str(candidates[0].sequence)
                        if best_seq != row["seqs_norm"][0]:
                            changed_rows += 1
                        chosen_idx = row["seqs_norm"].index(best_seq) if best_seq in row["seqs_norm"] else 0
                        refined_rows += 1
                    if row["seqs_norm"][chosen_idx] == row["truth_norm"]:
                        matches += 1

                pep_recall = matches / cached["total"] if cached["total"] else 0.0
                results["probes"].append(
                    {
                        "model_weight": model_weight,
                        "spectrum_weight": spectrum_weight,
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

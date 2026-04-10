#!/usr/bin/env python3
"""Replay a simple selective beam reranking probe on existing baseline outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.data_handler import DataHandler
from evaluation.evaluator import Evaluator
from evaluation.statistics_utils import (
    calculate_sequence_masses,
    normalize_peptide,
    normalize_ptm_format,
    parse_peptide_with_ptm,
)


PROTON = 1.007276
WATER = 18.010565
HIGH_RISK_AA = set("DNEQKM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["casanovo", "instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument("--scorer", choices=["basic", "advanced"], default="advanced")
    parser.add_argument(
        "--ion-mode",
        choices=["both", "y_only", "y_heavy", "b_only"],
        default="both",
    )
    parser.add_argument("--decision-mode", choices=["blend", "disagreement"], default="blend")
    parser.add_argument("--spec-gap-threshold", type=float, default=0.15)
    parser.add_argument("--decoder-margin-threshold", type=float, default=0.25)
    parser.add_argument("--refine-all", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--entropy-threshold", type=float, default=1.00)
    parser.add_argument("--diversity-threshold", type=float, default=0.80)
    parser.add_argument("--max-local-diffs", type=int, default=None)
    parser.add_argument("--max-spec-rank", type=int, default=None)
    parser.add_argument("--require-suffix", action="store_true")
    parser.add_argument("--require-highrisk", action="store_true")
    parser.add_argument("--suffix-tail", type=int, default=3)
    parser.add_argument("--local-only", action="store_true")
    return parser.parse_args()


def load_spectra(path: str) -> List[Dict[str, Any]]:
    handler = DataHandler({"path": path, "format": "auto"})
    df = handler.load_data()
    spectra: List[Dict[str, Any]] = []
    for index, row in df.iterrows():
        spectra.append(
            {
                "spectrum_id": row.get("spectrum_id", f"spectrum_{index}"),
                "mz_array": np.asarray(row["mz_array"], dtype=np.float32),
                "intensity_array": np.asarray(row["intensity_array"], dtype=np.float32),
                "precursor_mz": float(row.get("precursor_mz", 0.0)),
                "precursor_charge": int(row.get("precursor_charge", 0)),
                "sequence": row.get("sequence", ""),
            }
        )
    return spectra


def softmax(values: List[float]) -> List[float]:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        return [0.0 for _ in values]
    safe = array.copy()
    safe[~finite] = -1e9
    safe -= safe.max()
    exp = np.exp(safe)
    total = float(exp.sum())
    if total <= 0:
        return [0.0 for _ in values]
    return (exp / total).tolist()


def minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return [0.0 for _ in values]
    return ((arr - lo) / (hi - lo)).tolist()


def entropy(probs: List[float]) -> float:
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0))


def diversity(sequences: List[str]) -> float:
    if not sequences:
        return 0.0
    normalized = [normalize_peptide(seq) for seq in sequences]
    return len(set(normalized)) / len(normalized)


def diff_positions(seq_a: str, seq_b: str) -> List[int]:
    n = min(len(seq_a), len(seq_b))
    diffs = [i for i in range(n) if seq_a[i] != seq_b[i]]
    diffs.extend(range(n, max(len(seq_a), len(seq_b))))
    return diffs


def local_transition_features(src_seq: str, dst_seq: str, suffix_tail: int) -> Dict[str, Any]:
    diffs = diff_positions(src_seq, dst_seq)
    chars = {src_seq[i] for i in diffs if i < len(src_seq)} | {dst_seq[i] for i in diffs if i < len(dst_seq)}
    tail_start = max(0, max(len(src_seq), len(dst_seq)) - suffix_tail)
    return {
        "diff_count": len(diffs),
        "suffix_bias": bool(diffs) and max(diffs) >= tail_start,
        "highrisk": bool(chars & HIGH_RISK_AA),
    }


def local_transition_ok(features: Dict[str, Any], spec_best_idx: int, args: argparse.Namespace) -> bool:
    if args.max_spec_rank is not None and spec_best_idx > args.max_spec_rank:
        return False
    if args.max_local_diffs is not None and int(features["diff_count"]) > args.max_local_diffs:
        return False
    if args.require_suffix and not bool(features["suffix_bias"]):
        return False
    if args.require_highrisk and not bool(features["highrisk"]):
        return False
    return True


def build_candidates(model: str, baseline_path: str) -> List[Dict[str, Any]]:
    baseline_file = Path(baseline_path)
    if baseline_file.suffix == ".jsonl":
        items: List[Dict[str, Any]] = []
        with open(baseline_file, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                beams = []
                for beam in row.get("beam_predictions", []):
                    beams.append(
                        {
                            "sequence": beam.get("pred_peptide", ""),
                            "decoder_score": float(beam.get("confidence", 0.0)),
                        }
                    )
                items.append(
                    {
                        "truth": row.get("true_sequence", "") or row.get("true_peptide", ""),
                        "top1": row.get("sequence", "") or (beams[0]["sequence"] if beams else ""),
                        "candidates": beams,
                        "precursor_mz": float(row.get("precursor_mz", 0.0)),
                        "precursor_charge": int(row.get("precursor_charge", 0)),
                    }
                )
        return items

    df = pd.read_csv(baseline_file)
    items = []
    for _, row in df.iterrows():
        beams = []
        idx = 0
        while f"preds_beam_{idx}" in df.columns:
            beams.append(
                {
                    "sequence": str(row.get(f"preds_beam_{idx}", "")),
                    "decoder_score": float(row.get(f"log_probs_beam_{idx}", float("-inf"))),
                }
            )
            idx += 1
        items.append(
            {
                "truth": str(row.get("targets", "")),
                "top1": str(row.get("predictions", "")),
                "candidates": beams,
                "precursor_mz": float(row.get("precursor_mz", 0.0)),
                "precursor_charge": int(row.get("precursor_charge", 0)),
            }
        )
    return items


def align_spectra(rows: List[Dict[str, Any]], spectra: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(rows) == len(spectra):
        return spectra
    aligned: List[Dict[str, Any]] = []
    spec_idx = 0
    for row in rows:
        target_mz = float(row.get("precursor_mz", 0.0))
        target_charge = int(row.get("precursor_charge", 0))
        found = None
        while spec_idx < len(spectra):
            spectrum = spectra[spec_idx]
            spec_idx += 1
            mz_ok = abs(float(spectrum.get("precursor_mz", 0.0)) - target_mz) < 1e-3
            charge_ok = int(spectrum.get("precursor_charge", 0)) == target_charge
            if mz_ok and charge_ok:
                found = spectrum
                break
        if found is None:
            raise ValueError(
                f"Unable to align row with precursor_mz={target_mz} charge={target_charge}"
            )
        aligned.append(found)
    return aligned


def peptide_ions(sequence: str, model: str) -> List[float]:
    normalized = normalize_ptm_format(sequence, model=model)
    residues = parse_peptide_with_ptm(normalized)
    prefix_masses, total_mass = calculate_sequence_masses(residues)
    if len(prefix_masses) <= 2 or total_mass <= 0:
        return []
    b_ions = [mass + PROTON for mass in prefix_masses[1:-1]]
    y_ions = [total_mass - mass + WATER + PROTON for mass in prefix_masses[1:-1]]
    return b_ions + y_ions


def peptide_ion_series(sequence: str, model: str) -> tuple[List[float], List[float], float]:
    normalized = normalize_ptm_format(sequence, model=model)
    residues = parse_peptide_with_ptm(normalized)
    prefix_masses, total_mass = calculate_sequence_masses(residues)
    if len(prefix_masses) <= 2 or total_mass <= 0:
        return [], [], 0.0
    b_ions = [mass + PROTON for mass in prefix_masses[1:-1]]
    y_ions = [total_mass - mass + WATER + PROTON for mass in prefix_masses[1:-1]]
    return b_ions, y_ions, total_mass


def longest_true_run(mask: List[bool]) -> int:
    best = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _match_series(
    ions: List[float],
    mz: np.ndarray,
    intens: np.ndarray,
    tol_da: float,
    top_mask: np.ndarray,
) -> tuple[float, int, int, int]:
    if not ions:
        return 0.0, 0, 0, 0
    matched = []
    intensity_sum = 0.0
    top_hits = 0
    for ion in ions:
        diffs = np.abs(mz - ion)
        min_idx = int(np.argmin(diffs))
        is_match = float(diffs[min_idx]) <= tol_da
        matched.append(is_match)
        if is_match:
            intensity_sum += float(intens[min_idx])
            if bool(top_mask[min_idx]):
                top_hits += 1
    return intensity_sum, sum(matched), top_hits, longest_true_run(matched)


def spectrum_match_score(sequence: str, mz_array: np.ndarray, intensity_array: np.ndarray, model: str, tol_da: float) -> float:
    ions = peptide_ions(sequence, model=model)
    if not ions or len(mz_array) == 0:
        return 0.0
    intens = np.asarray(intensity_array, dtype=np.float32)
    if intens.size == 0 or float(intens.max()) <= 0:
        return 0.0
    intens = intens / float(intens.max())
    mz = np.asarray(mz_array, dtype=np.float32)
    total = 0.0
    for ion in ions:
        diffs = np.abs(mz - ion)
        min_idx = int(np.argmin(diffs))
        if float(diffs[min_idx]) <= tol_da:
            total += float(intens[min_idx])
    return total / len(ions)


def advanced_spectrum_match_score(
    sequence: str,
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
    model: str,
    tol_da: float,
    precursor_ppm: float,
    top_peak_frac: float,
    ion_mode: str,
) -> float:
    b_ions, y_ions, seq_mass = peptide_ion_series(sequence, model=model)
    if (not b_ions and not y_ions) or len(mz_array) == 0 or precursor_mz <= 0 or precursor_charge <= 0:
        return 0.0

    mz = np.asarray(mz_array, dtype=np.float32)
    intens = np.asarray(intensity_array, dtype=np.float32)
    if intens.size == 0 or float(intens.max()) <= 0:
        return 0.0
    intens = intens / float(intens.max())

    top_count = max(1, min(len(intens), int(math.ceil(len(intens) * top_peak_frac))))
    top_idx = np.argpartition(intens, -top_count)[-top_count:]
    top_mask = np.zeros(len(intens), dtype=bool)
    top_mask[top_idx] = True

    b_intensity, b_hits, b_top_hits, b_run = _match_series(b_ions, mz, intens, tol_da, top_mask)
    y_intensity, y_hits, y_top_hits, y_run = _match_series(y_ions, mz, intens, tol_da, top_mask)
    ion_count = max(1, len(b_ions) + len(y_ions))
    hit_count = b_hits + y_hits

    if ion_mode == "y_only":
        intensity_score = y_intensity / max(1, len(y_ions))
        coverage_score = y_hits / max(1, len(y_ions))
        top_peak_score = y_top_hits / max(1, len(y_ions))
        ladder_score = y_run / max(1, len(y_ions))
    elif ion_mode == "b_only":
        intensity_score = b_intensity / max(1, len(b_ions))
        coverage_score = b_hits / max(1, len(b_ions))
        top_peak_score = b_top_hits / max(1, len(b_ions))
        ladder_score = b_run / max(1, len(b_ions))
    else:
        intensity_score = (b_intensity + y_intensity) / ion_count
        coverage_score = hit_count / ion_count
        top_peak_score = (b_top_hits + y_top_hits) / ion_count
        ladder_score = 0.0
        if b_ions:
            ladder_score += b_run / len(b_ions)
        if y_ions:
            ladder_score += y_run / len(y_ions)
        ladder_score /= 2 if b_ions and y_ions else 1

    precursor_neutral_mass = max(0.0, (precursor_mz - PROTON) * precursor_charge)
    ppm_error = abs(seq_mass - precursor_neutral_mass) / max(precursor_neutral_mass, 1e-6) * 1e6
    mass_score = max(0.0, 1.0 - (ppm_error / max(precursor_ppm, 1e-6)))

    if ion_mode == "y_heavy":
        y_intensity_score = y_intensity / max(1, len(y_ions))
        y_coverage_score = y_hits / max(1, len(y_ions))
        y_top_score = y_top_hits / max(1, len(y_ions))
        y_ladder_score = y_run / max(1, len(y_ions))
        b_coverage_score = b_hits / max(1, len(b_ions))
        return (
            0.28 * y_intensity_score
            + 0.24 * y_coverage_score
            + 0.18 * y_ladder_score
            + 0.10 * y_top_score
            + 0.10 * b_coverage_score
            + 0.10 * mass_score
        )

    return 0.30 * intensity_score + 0.25 * coverage_score + 0.20 * ladder_score + 0.15 * top_peak_score + 0.10 * mass_score


def should_refine(model: str, candidates: List[Dict[str, Any]], args: argparse.Namespace) -> bool:
    if args.refine_all:
        return True
    if model in {"casanovo", "primenovo"}:
        top1_conf = float(candidates[0]["decoder_score"]) if candidates else 0.0
        return top1_conf < args.confidence_threshold
    scores = [cand["decoder_score"] for cand in candidates[: args.top_k]]
    probs = softmax(scores)
    seqs = [cand["sequence"] for cand in candidates[: args.top_k]]
    return entropy(probs) > args.entropy_threshold and diversity(seqs) >= args.diversity_threshold


def rerank_row(
    model: str,
    row: Dict[str, Any],
    spectrum: Dict[str, Any],
    alpha: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    beams = [dict(item) for item in row["candidates"][: args.top_k]]
    if not beams:
        return {"sequence": "", "beam_predictions": []}

    if not should_refine(model, beams, args):
        return {
            "sequence": beams[0]["sequence"],
            "beam_predictions": beams,
        }

    if model in {"casanovo", "primenovo"}:
        decoder_probs = minmax([max(float(beam["decoder_score"]), 0.0) for beam in beams])
    else:
        decoder_probs = softmax([beam["decoder_score"] for beam in beams])
    spec_scores = []
    for beam in beams:
        if args.scorer == "advanced":
            score = advanced_spectrum_match_score(
                beam["sequence"],
                spectrum["mz_array"],
                spectrum["intensity_array"],
                precursor_mz=float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0))),
                precursor_charge=int(spectrum.get("precursor_charge", row.get("precursor_charge", 0))),
                model=model,
                tol_da=args.mass_tol_da,
                precursor_ppm=args.precursor_ppm,
                top_peak_frac=args.top_peak_frac,
                ion_mode=args.ion_mode,
            )
        else:
            score = spectrum_match_score(
                beam["sequence"],
                spectrum["mz_array"],
                spectrum["intensity_array"],
                model=model,
                tol_da=args.mass_tol_da,
            )
        spec_scores.append(score)
    spec_norm = minmax(spec_scores)
    spec_best_idx = int(np.argmax(spec_norm))
    top1_norm = normalize_peptide(normalize_ptm_format(beams[0]["sequence"], model=model))
    spec_best_norm = normalize_peptide(normalize_ptm_format(beams[spec_best_idx]["sequence"], model=model))
    local_features = local_transition_features(top1_norm, spec_best_norm, args.suffix_tail)
    if not local_transition_ok(local_features, spec_best_idx, args):
        return {
            "sequence": beams[0]["sequence"],
            "beam_predictions": beams,
        }

    reranked = []
    for beam, dec, spec, spec_n in zip(beams, decoder_probs, spec_scores, spec_norm):
        beam["decoder_prob"] = dec
        beam["spectrum_score"] = spec
        beam["final_score"] = (1.0 - alpha) * dec + alpha * spec_n
        reranked.append(beam)
    if args.decision_mode == "disagreement":
        top_idx = 0
        spec_gap = spec_norm[spec_best_idx] - spec_norm[top_idx]
        decoder_margin = decoder_probs[top_idx] - decoder_probs[spec_best_idx]
        if (
            spec_best_idx != top_idx
            and spec_gap >= args.spec_gap_threshold
            and decoder_margin <= args.decoder_margin_threshold
        ):
            reranked[spec_best_idx]["final_score"] = max(
                reranked[spec_best_idx]["final_score"],
                reranked[top_idx]["final_score"] + 1e-6,
            )
    if args.local_only and spec_best_idx != 0:
        keep = {0, spec_best_idx}
        reranked = [beam for idx, beam in enumerate(reranked) if idx in keep]
    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    return {
        "sequence": reranked[0]["sequence"],
        "beam_predictions": reranked,
    }


def evaluate_predictions(model: str, sequences: List[str], truths: List[str]) -> Dict[str, float]:
    evaluator = Evaluator()
    preds = [{"sequence": seq} for seq in sequences]
    targets = [{"sequence": seq} for seq in truths]
    return evaluator.evaluate(preds, targets, model_name=model)


def main() -> None:
    args = parse_args()
    spectra = load_spectra(args.spectra)
    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, spectra)

    truths = [row["truth"] for row in rows]
    baseline_sequences = [row["top1"] for row in rows]
    results = {
        "model": args.model,
        "scorer": args.scorer,
        "baseline": evaluate_predictions(args.model, baseline_sequences, truths),
        "probes": [],
    }

    for alpha in args.alpha:
        reranked_sequences = []
        refined = 0
        for row, spectrum in zip(rows, spectra):
            reranked = rerank_row(args.model, row, spectrum, alpha, args)
            reranked_sequences.append(reranked["sequence"])
            if reranked["beam_predictions"] and reranked["beam_predictions"][0]["sequence"] != row["top1"]:
                refined += 1
        metrics = evaluate_predictions(args.model, reranked_sequences, truths)
        metrics["alpha"] = alpha
        metrics["refined_rows"] = refined
        results["probes"].append(metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

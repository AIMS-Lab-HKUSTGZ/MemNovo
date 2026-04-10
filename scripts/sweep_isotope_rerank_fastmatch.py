#!/usr/bin/env python3
"""Fast sweep for archive isotope-aware reranking on saved beam outputs."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ARCHIVE_ROOT = WORKSPACE_ROOT / "archieved" / "legacy_workspace"
ARCHIVE_PYC_ROOT = ARCHIVE_ROOT / "src" / "__pycache__"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from replay_rerank_probe import PROTON, WATER, align_spectra, build_candidates, diversity, entropy, load_spectra, softmax
from evaluation.statistics_utils import (
    calculate_sequence_masses,
    normalize_peptide,
    normalize_ptm_format,
    parse_peptide_with_ptm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--isotope-weights", default="0.05,0.1,0.2,0.3,0.5,0.8,1.0")
    parser.add_argument("--entropy-thresholds", default="none,0.5,1.0,1.5")
    parser.add_argument("--diversity-thresholds", default="0.0,0.5,0.8")
    parser.add_argument("--confidence-thresholds", default="none,0.45,0.55,0.65,0.75,0.8")
    parser.add_argument("--tolerance-ppms", default="20,50,100")
    parser.add_argument("--min-peptide-mass", type=float, default=500.0)
    parser.add_argument("--require-state-match", action="store_true")
    parser.add_argument("--refine-all", action="store_true")
    return parser.parse_args()


def parse_float_list(values: str) -> list[float]:
    return [float(item) for item in values.split(",") if item.strip()]


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


@dataclass
class ScoredSequence:
    sequence: str
    sequence_log_probability: float
    mass_error: float = 0.0


def _ensure_parent_modules(module_name: str) -> None:
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def load_archive_isotope_modules() -> tuple[types.ModuleType, types.ModuleType]:
    scorer_name = "isotope_scorer"
    scorer_path = ARCHIVE_PYC_ROOT / "isotope_scorer.cpython-312.pyc"
    loader = importlib.machinery.SourcelessFileLoader(scorer_name, str(scorer_path))
    spec = importlib.util.spec_from_loader(scorer_name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[scorer_name] = module
    loader.exec_module(module)

    interfaces_name = "instanovo_src.instanovo.inference.interfaces"
    _ensure_parent_modules(interfaces_name)
    fake_interfaces = types.ModuleType(interfaces_name)
    fake_interfaces.ScoredSequence = ScoredSequence
    sys.modules[interfaces_name] = fake_interfaces

    reranker_name = "isotope_aware_reranker"
    reranker_path = ARCHIVE_PYC_ROOT / "isotope_aware_reranker.cpython-312.pyc"
    reranker_loader = importlib.machinery.SourcelessFileLoader(reranker_name, str(reranker_path))
    reranker_spec = importlib.util.spec_from_loader(reranker_name, reranker_loader)
    reranker_module = importlib.util.module_from_spec(reranker_spec)
    sys.modules[reranker_name] = reranker_module
    reranker_loader.exec_module(reranker_module)
    return module, reranker_module


def peptide_mass(sequence: str, model: str) -> float:
    normalized = normalize_ptm_format(sequence or "", model=model)
    residues = parse_peptide_with_ptm(normalized)
    _, total_mass = calculate_sequence_masses(residues)
    if total_mass <= 0:
        return 0.0
    return total_mass + WATER


def precursor_neutral_mass(precursor_mz: float, precursor_charge: int) -> float:
    if precursor_mz <= 0 or precursor_charge <= 0:
        return 0.0
    return (precursor_mz - PROTON) * precursor_charge


def normalize_for_match(sequence: str, model: str) -> str:
    return normalize_peptide(normalize_ptm_format(sequence or "", model=model))


def build_cached_rows(args: argparse.Namespace) -> dict:
    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))
    cached_rows = []
    baseline_matches = 0
    total = 0

    for row, spectrum in zip(rows, spectra):
        beams = [dict(item) for item in row["candidates"][: args.top_k]]
        if not beams:
            continue
        precursor_mass = precursor_neutral_mass(
            float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0))),
            int(spectrum.get("precursor_charge", row.get("precursor_charge", 0))),
        )
        if precursor_mass <= 0:
            continue
        seqs = [beam["sequence"] for beam in beams]
        seqs_norm = [normalize_for_match(seq, args.model) for seq in seqs]
        truth_norm = normalize_for_match(row["truth"], args.model)
        raw_scores = [float(beam["decoder_score"]) for beam in beams]
        decoder_probs = softmax(raw_scores)
        top1_norm = seqs_norm[0]
        baseline_matches += int(top1_norm == truth_norm)
        total += 1

        mass_errors = []
        for seq in seqs:
            seq_mass = peptide_mass(seq, args.model)
            mass_errors.append(0.0 if seq_mass <= 0 else precursor_mass - seq_mass)

        cached_rows.append(
            {
                "truth_norm": truth_norm,
                "seqs": seqs,
                "seqs_norm": seqs_norm,
                "raw_scores": raw_scores,
                "decoder_probs": decoder_probs,
                "entropy": entropy(decoder_probs),
                "diversity": diversity(seqs),
                "top1_conf": raw_scores[0],
                "precursor_mass": precursor_mass,
                "mass_errors": mass_errors,
            }
        )

    return {
        "rows": cached_rows,
        "baseline_matches": baseline_matches,
        "baseline_pep_recall": baseline_matches / total if total else 0.0,
        "total": total,
    }


def row_is_gated(row: dict, args: argparse.Namespace, confidence_threshold: float | None, entropy_threshold: float | None, diversity_threshold: float | None) -> bool:
    if args.refine_all:
        return True
    if args.model == "primenovo":
        if confidence_threshold is None:
            return True
        return row["top1_conf"] < confidence_threshold
    if entropy_threshold is None:
        return True
    return row["entropy"] > entropy_threshold and row["diversity"] >= float(diversity_threshold or 0.0)


def main() -> None:
    args = parse_args()
    isotope_weights = parse_float_list(args.isotope_weights)
    entropy_thresholds = parse_optional_float_list(args.entropy_thresholds)
    diversity_thresholds = parse_float_list(args.diversity_thresholds)
    confidence_thresholds = parse_optional_float_list(args.confidence_thresholds)
    tolerance_ppms = parse_float_list(args.tolerance_ppms)

    _, reranker_module = load_archive_isotope_modules()
    Reranker = reranker_module.IsotopeAwareReranker
    cached = build_cached_rows(args)
    rows = cached["rows"]
    baseline_matches = cached["baseline_matches"]
    total = cached["total"]

    results = {
        "model": args.model,
        "baseline_matches": baseline_matches,
        "baseline_pep_recall": cached["baseline_pep_recall"],
        "probes": [],
    }

    for isotope_weight in isotope_weights:
        for tolerance_ppm in tolerance_ppms:
            reranker = Reranker(
                residue_set={},
                isotope_weight=isotope_weight,
                use_isotope=True,
                tolerance_ppm=tolerance_ppm,
                min_peptide_mass=args.min_peptide_mass,
            )

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
                valid_state_rows = 0

                for row in rows:
                    chosen_idx = 0
                    gated = row_is_gated(row, args, confidence_threshold, entropy_threshold, diversity_threshold)
                    if gated:
                        candidates = [
                            ScoredSequence(
                                sequence=seq,
                                sequence_log_probability=score,
                                mass_error=mass_error,
                            )
                            for seq, score, mass_error in zip(row["seqs"], row["raw_scores"], row["mass_errors"])
                        ]
                        reranked = reranker.rerank_results(candidates, row["precursor_mass"], verbose=False)
                        if reranked:
                            best_seq = reranked[0]
                            state, _ = reranker.scorer.find_isotope_state(
                                getattr(best_seq, "mass_error", 0.0),
                                row["precursor_mass"],
                                tolerance_ppm,
                            )
                            if state != -1:
                                valid_state_rows += 1
                            if (not args.require_state_match) or state != -1:
                                best_norm = normalize_for_match(best_seq.sequence, args.model)
                                try:
                                    chosen_idx = row["seqs_norm"].index(best_norm)
                                except ValueError:
                                    chosen_idx = 0
                                refined_rows += 1
                                if chosen_idx != 0:
                                    changed_rows += 1

                    if row["seqs_norm"][chosen_idx] == row["truth_norm"]:
                        matches += 1

                pep_recall = matches / total if total else 0.0
                results["probes"].append(
                    {
                        "isotope_weight": isotope_weight,
                        "tolerance_ppm": tolerance_ppm,
                        "confidence_threshold": confidence_threshold,
                        "entropy_threshold": entropy_threshold,
                        "diversity_threshold": diversity_threshold,
                        "refined_rows": refined_rows,
                        "changed_rows": changed_rows,
                        "valid_state_rows": valid_state_rows,
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

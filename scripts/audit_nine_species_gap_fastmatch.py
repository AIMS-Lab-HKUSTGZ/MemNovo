#!/usr/bin/env python3
"""Audit nine-species reproduction gap with narrowed fast-match rerank grids."""

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
    load_spectra,
    minmax,
    softmax,
)
from evaluation.statistics_utils import normalize_peptide, normalize_ptm_format


PAPER_REL_GAINS = {
    "casanovo": {
        "Mus-musculus": 40.3,
        "H.-sapiens": 46.2,
        "Saccharomyces-cerevisiae": 21.5,
        "Methanosarcina-mazei": 32.0,
        "Apis-mellifera": 81.8,
        "Solanum-lycopersicum": 35.6,
        "Vigna-mungo": 6.5,
        "Bacillus-subtilis": 48.1,
        "Candidatus-endoloripes": 55.2,
    },
    "instanovo": {
        "Mus-musculus": 1.1,
        "H.-sapiens": 2.5,
        "Saccharomyces-cerevisiae": 7.6,
        "Methanosarcina-mazei": 2.5,
        "Apis-mellifera": 2.3,
        "Solanum-lycopersicum": 3.7,
        "Vigna-mungo": 5.9,
        "Bacillus-subtilis": 2.2,
        "Candidatus-endoloripes": 6.8,
    },
}


SPECIES_FILES = {
    "Methanosarcina-mazei": "dataset/NS1/Methanosarcina-mazei.mgf",
    "Saccharomyces-cerevisiae": "dataset/NS1/Saccharomyces-cerevisiae.mgf",
    "Bacillus-subtilis": "dataset/NS2/Bacillus-subtilis.mgf",
    "Apis-mellifera": "dataset/NS3/Apis-mellifera.mgf",
    "Candidatus-endoloripes": "dataset/NS3/Candidatus-endoloripes.mgf",
    "H.-sapiens": "dataset/NS3/H.-sapiens.mgf",
    "Mus-musculus": "dataset/NS3/Mus-musculus.mgf",
    "Solanum-lycopersicum": "dataset/NS3/Solanum-lycopersicum.mgf",
    "Vigna-mungo": "dataset/NS3/Vigna-mungo.mgf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["casanovo", "instanovo"], required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def normalize_for_match(sequence: str, model: str) -> str:
    return normalize_peptide(normalize_ptm_format(sequence or "", model=model))


def casanovo_grid() -> list[dict]:
    grid = []
    for alpha in [0.5, 0.8, 1.0]:
        for spec_gap in [0.0, 0.05, 0.1]:
            for decoder_margin in [0.4, 0.5, 0.6]:
                for confidence_threshold in [0.7, 0.8]:
                    grid.append(
                        {
                            "alpha": alpha,
                            "spec_gap_threshold": spec_gap,
                            "decoder_margin_threshold": decoder_margin,
                            "confidence_threshold": confidence_threshold,
                        }
                    )
    return grid


def instanovo_grid() -> list[dict]:
    grid = []
    for alpha in [0.1, 0.14, 0.2]:
        for spec_gap in [0.1, 0.15, 0.2]:
            for decoder_margin in [0.2, 0.3, 0.4]:
                grid.append(
                    {
                        "alpha": alpha,
                        "spec_gap_threshold": spec_gap,
                        "decoder_margin_threshold": decoder_margin,
                        "entropy_threshold": None,
                        "diversity_threshold": 0.0,
                    }
                )
    return grid


def precompute_species(model: str, baseline_path: str, spectra_path: str, ion_mode: str) -> dict:
    rows = build_candidates(model, baseline_path)
    spectra = align_spectra(rows, load_spectra(spectra_path))
    cached_rows = []
    baseline_matches = 0
    total = 0

    for row, spectrum in zip(rows, spectra):
        beams = [dict(item) for item in row["candidates"][:5]]
        if not beams:
            continue
        raw_scores = [float(beam["decoder_score"]) for beam in beams]
        seqs_norm = [normalize_for_match(beam["sequence"], model) for beam in beams]
        truth_norm = normalize_for_match(row["truth"], model)
        top1_norm = normalize_for_match(row["top1"], model)
        if top1_norm == truth_norm:
            baseline_matches += 1
        total += 1

        if model == "casanovo":
            decoder_probs = minmax([max(score, 0.0) for score in raw_scores])
            gate_stats = {
                "top1_conf": float(raw_scores[0]) if raw_scores else 0.0,
                "entropy": 0.0,
                "diversity": 0.0,
            }
        else:
            decoder_probs = softmax(raw_scores)
            gate_stats = {
                "top1_conf": 0.0,
                "entropy": entropy(decoder_probs),
                "diversity": diversity([beam["sequence"] for beam in beams]),
            }

        spec_scores = [
            advanced_spectrum_match_score(
                beam["sequence"],
                spectrum["mz_array"],
                spectrum["intensity_array"],
                precursor_mz=float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0))),
                precursor_charge=int(spectrum.get("precursor_charge", row.get("precursor_charge", 0))),
                model=model,
                tol_da=0.5,
                precursor_ppm=20.0,
                top_peak_frac=0.2,
                ion_mode=ion_mode,
            )
            for beam in beams
        ]
        spec_norm = minmax(spec_scores)
        spec_best_idx = int(np.argmax(spec_norm))
        cached_rows.append(
            {
                "seqs_norm": seqs_norm,
                "truth_norm": truth_norm,
                "decoder_probs": decoder_probs,
                "spec_norm": spec_norm,
                "spec_best_idx": spec_best_idx,
                "spec_gap": spec_norm[spec_best_idx] - spec_norm[0],
                "decoder_margin": decoder_probs[0] - decoder_probs[spec_best_idx],
                **gate_stats,
            }
        )

    return {
        "rows": cached_rows,
        "baseline_matches": baseline_matches,
        "baseline_pep_recall": baseline_matches / total if total else 0.0,
        "total": total,
    }


def evaluate_config(model: str, cached: dict, config: dict) -> dict:
    matches = 0
    gated_rows = 0
    refined_rows = 0
    changed_rows = 0

    for row in cached["rows"]:
        if model == "casanovo":
            gated = row["top1_conf"] < config["confidence_threshold"]
        else:
            gated = True if config["entropy_threshold"] is None else (
                row["entropy"] > config["entropy_threshold"] and row["diversity"] >= config["diversity_threshold"]
            )
        if gated:
            gated_rows += 1

        chosen_idx = 0
        if gated:
            scores = [
                (1.0 - config["alpha"]) * dec + config["alpha"] * spec
                for dec, spec in zip(row["decoder_probs"], row["spec_norm"])
            ]
            disagreement_ok = (
                row["spec_best_idx"] != 0
                and row["spec_gap"] >= config["spec_gap_threshold"]
                and row["decoder_margin"] <= config["decoder_margin_threshold"]
            )
            if disagreement_ok:
                scores[row["spec_best_idx"]] = max(scores[row["spec_best_idx"]], scores[0] + 1e-6)
            elif model != "casanovo":
                scores = None

            if scores is not None:
                chosen_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
                refined_rows += 1
                if chosen_idx != 0:
                    changed_rows += 1

        if row["seqs_norm"][chosen_idx] == row["truth_norm"]:
            matches += 1

    pep_recall = matches / cached["total"] if cached["total"] else 0.0
    rel = 0.0 if cached["baseline_pep_recall"] <= 0 else 100.0 * (pep_recall - cached["baseline_pep_recall"]) / cached["baseline_pep_recall"]
    return {
        **config,
        "gated_rows": gated_rows,
        "refined_rows": refined_rows,
        "changed_rows": changed_rows,
        "matched_peptides": matches,
        "pep_recall": pep_recall,
        "pep_recall_delta": pep_recall - cached["baseline_pep_recall"],
        "pep_recall_rel_pct": rel,
        "n_match_pep_delta": matches - cached["baseline_matches"],
    }


def main() -> None:
    args = parse_args()
    model = args.model
    if model == "casanovo":
        baseline_dir = Path("archieved/legacy_workspace/results/casanovo_beam5_ns")
        grid = casanovo_grid()
        ion_mode = "both"
    else:
        baseline_dir = Path("archieved/legacy_workspace/results/instanovo_beam5_ns")
        grid = instanovo_grid()
        ion_mode = "y_heavy"

    species_results = []
    for species, spectra_path in SPECIES_FILES.items():
        baseline_path = baseline_dir / f"{species}.jsonl"
        print(f"[{model}] processing {species}", flush=True)
        cached = precompute_species(model, str(baseline_path), spectra_path, ion_mode)
        best = None
        for config in grid:
            result = evaluate_config(model, cached, config)
            if best is None or (result["pep_recall"], result["matched_peptides"]) > (best["pep_recall"], best["matched_peptides"]):
                best = result
        paper_rel = PAPER_REL_GAINS[model][species]
        species_results.append(
            {
                "species": species,
                "baseline_pep_recall": cached["baseline_pep_recall"],
                "baseline_matches": cached["baseline_matches"],
                "total": cached["total"],
                "paper_rel_pct": paper_rel,
                "best": best,
                "gap_to_paper_pct_points": paper_rel - best["pep_recall_rel_pct"],
                "paper_fraction_recovered": 0.0 if paper_rel <= 0 else best["pep_recall_rel_pct"] / paper_rel,
            }
        )
        print(
            json.dumps(
                {
                    "species": species,
                    "paper_rel_pct": paper_rel,
                    "best_rel_pct": best["pep_recall_rel_pct"],
                    "gap_to_paper_pct_points": paper_rel - best["pep_recall_rel_pct"],
                }
            ),
            flush=True,
        )

    avg_best_rel = float(np.mean([item["best"]["pep_recall_rel_pct"] for item in species_results]))
    avg_paper_rel = float(np.mean([item["paper_rel_pct"] for item in species_results]))
    output = {
        "model": model,
        "species_results": species_results,
        "avg_best_rel_pct": avg_best_rel,
        "avg_paper_rel_pct": avg_paper_rel,
        "avg_gap_pct_points": avg_paper_rel - avg_best_rel,
        "avg_paper_fraction_recovered": 0.0 if avg_paper_rel <= 0 else avg_best_rel / avg_paper_rel,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"model": model, "avg_best_rel_pct": avg_best_rel, "avg_gap_pct_points": avg_paper_rel - avg_best_rel}, indent=2))


if __name__ == "__main__":
    main()

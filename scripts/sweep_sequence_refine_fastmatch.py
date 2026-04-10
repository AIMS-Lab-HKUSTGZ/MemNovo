#!/usr/bin/env python3
"""Fast sequence-level local-search refinement sweep for beam predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_rerank_probe import (
    advanced_spectrum_match_score,
    align_spectra,
    build_candidates,
    diversity,
    entropy,
    load_spectra,
    minmax,
    normalize_peptide,
    normalize_ptm_format,
    parse_peptide_with_ptm,
    softmax,
)
from sweep_candidate_rewrite_fastmatch import generate_rewrite_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="instanovo", choices=["instanovo", "primenovo"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--source-beams", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-edit-span", type=int, default=3)
    parser.add_argument("--max-len-delta", type=int, default=2)
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--rewrite-penalties", default="0.0,0.02,0.05,0.1")
    parser.add_argument("--spec-gap-thresholds", default="0.0,0.05,0.1")
    parser.add_argument("--decoder-margin-thresholds", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--entropy-thresholds", default="1.0")
    parser.add_argument("--diversity-thresholds", default="0.8")
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--ion-mode", choices=["both", "y_only", "y_heavy", "b_only"], default="y_heavy")
    parser.add_argument("--require-improvement", action="store_true")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x]


def normalize_for_match(sequence: str, model: str) -> str:
    return normalize_peptide(normalize_ptm_format(sequence or "", model=model))


def local_search_candidates(
    top_seq: str,
    alt_sequences: list[str],
    alt_priors: list[float],
    spectrum: dict,
    args: argparse.Namespace,
) -> list[dict]:
    """Search a small local rewrite space rooted at top_seq and top beam alternatives."""
    top_res = parse_peptide_with_ptm(normalize_ptm_format(top_seq, model=args.model))
    if not top_res:
        return []

    spec_cache: dict[str, float] = {}

    def spec_score(seq: str) -> float:
        if seq not in spec_cache:
            spec_cache[seq] = advanced_spectrum_match_score(
                seq,
                spectrum["mz_array"],
                spectrum["intensity_array"],
                precursor_mz=float(spectrum.get("precursor_mz", 0.0)),
                precursor_charge=int(spectrum.get("precursor_charge", 0)),
                model=args.model,
                tol_da=args.mass_tol_da,
                precursor_ppm=args.precursor_ppm,
                top_peak_frac=args.top_peak_frac,
                ion_mode=args.ion_mode,
            )
        return spec_cache[seq]

    seen: dict[str, dict] = {}
    frontier: list[dict] = []

    top_norm = normalize_for_match(top_seq, args.model)
    top_entry = {
        "sequence": top_seq,
        "norm": top_norm,
        "prior": 1.0,
        "depth": 0,
        "edit_span": 0,
        "spec_score": spec_score(top_seq),
    }
    seen[top_norm] = top_entry
    frontier.append(top_entry)

    for alt_seq, alt_prior in zip(alt_sequences, alt_priors):
        alt_norm = normalize_for_match(alt_seq, args.model)
        entry = {
            "sequence": alt_seq,
            "norm": alt_norm,
            "prior": float(alt_prior),
            "depth": 0,
            "edit_span": 0,
            "spec_score": spec_score(alt_seq),
        }
        prev = seen.get(alt_norm)
        if prev is None or entry["prior"] > prev["prior"] or entry["spec_score"] > prev["spec_score"]:
            seen[alt_norm] = entry
        frontier.append(entry)

    alt_residues = []
    for alt_seq, alt_prior in zip(alt_sequences, alt_priors):
        alt_res = parse_peptide_with_ptm(normalize_ptm_format(alt_seq, model=args.model))
        if alt_res:
            alt_residues.append((alt_seq, alt_res, float(alt_prior)))

    for depth in range(1, args.max_depth + 1):
        ranked = sorted(
            seen.values(),
            key=lambda x: (x["spec_score"], x["prior"]),
            reverse=True,
        )[: args.beam_width]
        new_entries: list[dict] = []
        for state in ranked:
            base_res = parse_peptide_with_ptm(normalize_ptm_format(state["sequence"], model=args.model))
            if not base_res:
                continue
            for _, alt_res, alt_prior in alt_residues:
                rewrites = generate_rewrite_candidates(
                    base_res,
                    alt_res,
                    args.max_edit_span,
                    args.max_len_delta,
                )
                for seq, span in rewrites:
                    norm = normalize_for_match(seq, args.model)
                    prior = max(float(state["prior"]), float(alt_prior))
                    entry = {
                        "sequence": seq,
                        "norm": norm,
                        "prior": prior,
                        "depth": depth,
                        "edit_span": int(state["edit_span"]) + int(span),
                        "spec_score": spec_score(seq),
                    }
                    prev = seen.get(norm)
                    replace = (
                        prev is None
                        or entry["spec_score"] > prev["spec_score"]
                        or (
                            entry["spec_score"] == prev["spec_score"]
                            and (entry["prior"], -entry["edit_span"], -entry["depth"])
                            > (prev["prior"], -prev["edit_span"], -prev["depth"])
                        )
                    )
                    if replace:
                        seen[norm] = entry
                        new_entries.append(entry)
        frontier = new_entries
        if not frontier:
            break

    return list(seen.values())


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.alphas)
    rewrite_penalties = parse_float_list(args.rewrite_penalties)
    spec_gap_thresholds = parse_float_list(args.spec_gap_thresholds)
    decoder_margin_thresholds = parse_float_list(args.decoder_margin_thresholds)
    entropy_thresholds = parse_float_list(args.entropy_thresholds)
    diversity_thresholds = parse_float_list(args.diversity_thresholds)

    rows = build_candidates(args.model, args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))
    cached_rows = []
    baseline_matches = 0
    total = 0

    for row, spectrum in zip(rows, spectra):
        beams = [dict(item) for item in row["candidates"][: args.top_k]]
        if not beams:
            continue
        raw_scores = [float(beam["decoder_score"]) for beam in beams]
        decoder_probs = softmax(raw_scores) if args.model == "instanovo" else minmax([max(score, 0.0) for score in raw_scores])
        beam_entropy = entropy(decoder_probs) if args.model == "instanovo" else 0.0
        beam_div = diversity([beam["sequence"] for beam in beams]) if args.model == "instanovo" else 0.0

        truth_norm = normalize_for_match(row["truth"], args.model)
        top_norm = normalize_for_match(row["top1"], args.model)
        if top_norm == truth_norm:
            baseline_matches += 1
        total += 1

        alt_sequences = [beam["sequence"] for beam in beams[1 : min(len(beams), args.source_beams + 1)]]
        alt_priors = [decoder_probs[i] for i in range(1, min(len(beams), args.source_beams + 1))]
        candidates = local_search_candidates(row["top1"], alt_sequences, alt_priors, spectrum, args)
        spec_scores = [c["spec_score"] for c in candidates]
        spec_norm = minmax(spec_scores)
        for cand, sn in zip(candidates, spec_norm):
            cand["spec_norm"] = sn
        top_spec_norm = 0.0
        top_candidates = [c for c in candidates if c["norm"] == top_norm]
        if top_candidates:
            top_spec_norm = max(c["spec_norm"] for c in top_candidates)

        cached_rows.append(
            {
                "truth_norm": truth_norm,
                "top_norm": top_norm,
                "entropy": beam_entropy,
                "diversity": beam_div,
                "decoder_margin": decoder_probs[0] - (decoder_probs[1] if len(decoder_probs) > 1 else 0.0),
                "candidates": candidates,
                "top_spec_norm": top_spec_norm,
            }
        )

    baseline_pep_recall = baseline_matches / total if total else 0.0
    results = {
        "model": args.model,
        "baseline_pep_recall": baseline_pep_recall,
        "baseline_matches": baseline_matches,
        "total": total,
        "probes": [],
    }

    for alpha in alphas:
        for rewrite_penalty in rewrite_penalties:
            for spec_gap_threshold in spec_gap_thresholds:
                for decoder_margin_threshold in decoder_margin_thresholds:
                    for entropy_threshold in entropy_thresholds:
                        for diversity_threshold in diversity_thresholds:
                            matches = 0
                            gated_rows = 0
                            refined_rows = 0
                            changed_rows = 0
                            for row in cached_rows:
                                chosen_norm = row["top_norm"]
                                gated = (
                                    row["entropy"] > entropy_threshold
                                    and row["diversity"] >= diversity_threshold
                                    and row["decoder_margin"] <= decoder_margin_threshold
                                )
                                if gated:
                                    gated_rows += 1
                                else:
                                    if chosen_norm == row["truth_norm"]:
                                        matches += 1
                                    continue

                                best_score = None
                                best_norm = None
                                for cand in row["candidates"]:
                                    spec_gap = cand["spec_norm"] - row["top_spec_norm"]
                                    if spec_gap < spec_gap_threshold:
                                        continue
                                    final_score = (
                                        (1.0 - alpha)
                                        * max(0.0, cand["prior"] - rewrite_penalty * cand["edit_span"] - 0.05 * cand["depth"])
                                        + alpha * cand["spec_norm"]
                                    )
                                    if best_score is None or final_score > best_score:
                                        best_score = final_score
                                        best_norm = cand["norm"]

                                if best_norm is not None:
                                    if args.require_improvement and best_norm == row["top_norm"]:
                                        pass
                                    else:
                                        chosen_norm = best_norm
                                        refined_rows += 1
                                        if chosen_norm != row["top_norm"]:
                                            changed_rows += 1

                                if chosen_norm == row["truth_norm"]:
                                    matches += 1

                            pep_recall = matches / total if total else 0.0
                            results["probes"].append(
                                {
                                    "alpha": alpha,
                                    "rewrite_penalty": rewrite_penalty,
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
                                    "pep_recall_rel_pct": ((pep_recall - baseline_pep_recall) / baseline_pep_recall * 100.0)
                                    if baseline_pep_recall > 0
                                    else 0.0,
                                    "n_match_pep_delta": matches - baseline_matches,
                                }
                            )

    best = max(results["probes"], key=lambda item: item["pep_recall_rel_pct"], default=None)
    results["best"] = best

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.model, "best": best}, indent=2))


if __name__ == "__main__":
    main()

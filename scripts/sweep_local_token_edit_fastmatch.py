#!/usr/bin/env python3
"""Local token-edit refinement sweep on beam candidates, still training-free and plug-and-play."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

from replay_rerank_probe import (
    advanced_spectrum_match_score,
    align_spectra,
    build_candidates,
    load_spectra,
    minmax,
    normalize_peptide,
    normalize_ptm_format,
    parse_peptide_with_ptm,
    softmax,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["casanovo", "instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-alt-rank", type=int, default=2)
    parser.add_argument("--max-local-diffs", type=int, default=2)
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.5,0.8")
    parser.add_argument("--edit-penalties", default="0.0,0.02,0.05,0.1")
    parser.add_argument("--spec-gap-thresholds", default="0.0,0.05,0.1,0.15")
    parser.add_argument("--decoder-margin-thresholds", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--ion-mode", choices=["both", "y_only", "y_heavy", "b_only"], default="both")
    parser.add_argument("--require-highrisk", action="store_true")
    parser.add_argument("--require-suffix", action="store_true")
    parser.add_argument("--suffix-tail", type=int, default=3)
    return parser.parse_args()


HIGH_RISK = set("DNEQKM")


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x]


def residue_key(res: dict[str, Any]) -> tuple[str, str]:
    return (str(res.get("aa", "")), str(res.get("ptm") or ""))


def residues_to_string(residues: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for res in residues:
        aa = res.get("aa", "")
        ptm = res.get("ptm")
        if aa == "*":
            if ptm:
                ptm_str = str(ptm)
                if ptm_str.startswith(("+", "-")):
                    parts.append(ptm_str)
                else:
                    parts.append(f"({ptm_str})")
            continue
        if ptm:
            parts.append(f"{aa}({ptm})")
        else:
            parts.append(str(aa))
    return "".join(parts)


def token_features(top_seq: str, alt_seq: str, model: str, suffix_tail: int) -> dict[str, Any] | None:
    top_norm = normalize_ptm_format(top_seq or "", model=model)
    alt_norm = normalize_ptm_format(alt_seq or "", model=model)
    top_res = parse_peptide_with_ptm(top_norm)
    alt_res = parse_peptide_with_ptm(alt_norm)
    if not top_res or not alt_res:
        return None
    if len(top_res) != len(alt_res):
        return None
    diffs = [i for i, (a, b) in enumerate(zip(top_res, alt_res)) if residue_key(a) != residue_key(b)]
    chars = {top_res[i]["aa"] for i in diffs if top_res[i]["aa"] != "*"} | {alt_res[i]["aa"] for i in diffs if alt_res[i]["aa"] != "*"}
    tail_start = max(0, len(top_res) - suffix_tail)
    return {
        "top_norm": top_norm,
        "alt_norm": alt_norm,
        "top_res": top_res,
        "alt_res": alt_res,
        "diffs": diffs,
        "diff_count": len(diffs),
        "suffix_bias": bool(diffs) and max(diffs) >= tail_start,
        "highrisk": bool(chars & HIGH_RISK),
    }


def generate_local_candidates(features: dict[str, Any], require_suffix: bool) -> list[tuple[str, int]]:
    diffs = list(features["diffs"])
    top_res = features["top_res"]
    alt_res = features["alt_res"]
    if not diffs:
        return []
    if require_suffix and not features["suffix_bias"]:
        return []
    candidates: dict[str, int] = {}

    # Full alternate beam candidate.
    candidates[features["alt_norm"]] = len(diffs)

    # Single-position substitutions.
    for idx in diffs:
        edited = [dict(x) for x in top_res]
        edited[idx] = dict(alt_res[idx])
        candidates[residues_to_string(edited)] = 1

    # All differing positions substituted.
    if len(diffs) > 1:
        edited = [dict(x) for x in top_res]
        for idx in diffs:
            edited[idx] = dict(alt_res[idx])
        candidates[residues_to_string(edited)] = len(diffs)

    # Suffix swaps from each differing position.
    for idx in diffs:
        edited = [dict(x) for x in top_res[:idx]] + [dict(x) for x in alt_res[idx:]]
        candidates[residues_to_string(edited)] = max(1, len(diffs))

    out = [(seq, edit_count) for seq, edit_count in candidates.items() if seq and seq != features["top_norm"]]
    return out


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.alphas)
    edit_penalties = parse_float_list(args.edit_penalties)
    spec_gap_thresholds = parse_float_list(args.spec_gap_thresholds)
    decoder_margin_thresholds = parse_float_list(args.decoder_margin_thresholds)

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
        if args.model in {"casanovo", "primenovo"}:
            decoder_probs = minmax([max(score, 0.0) for score in raw_scores])
        else:
            decoder_probs = softmax(raw_scores)

        seqs_norm = [normalize_peptide(normalize_ptm_format(beam["sequence"], model=args.model)) for beam in beams]
        truth_norm = normalize_peptide(normalize_ptm_format(row["truth"], model=args.model))
        if seqs_norm[0] == truth_norm:
            baseline_matches += 1
        total += 1

        spec_scores = [
            advanced_spectrum_match_score(
                beam["sequence"],
                spectrum["mz_array"],
                spectrum["intensity_array"],
                precursor_mz=float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0))),
                precursor_charge=int(spectrum.get("precursor_charge", row.get("precursor_charge", 0))),
                model=args.model,
                tol_da=args.mass_tol_da,
                precursor_ppm=args.precursor_ppm,
                top_peak_frac=args.top_peak_frac,
                ion_mode=args.ion_mode,
            )
            for beam in beams
        ]
        spec_norm = minmax(spec_scores)
        local_edits = []
        for alt_idx in range(1, min(len(beams), args.max_alt_rank + 1)):
            feats = token_features(beams[0]["sequence"], beams[alt_idx]["sequence"], args.model, args.suffix_tail)
            if not feats:
                continue
            if feats["diff_count"] == 0 or feats["diff_count"] > args.max_local_diffs:
                continue
            if args.require_highrisk and not feats["highrisk"]:
                continue
            candidates = generate_local_candidates(feats, args.require_suffix)
            for seq, edit_count in candidates:
                score = advanced_spectrum_match_score(
                    seq,
                    spectrum["mz_array"],
                    spectrum["intensity_array"],
                    precursor_mz=float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0))),
                    precursor_charge=int(spectrum.get("precursor_charge", row.get("precursor_charge", 0))),
                    model=args.model,
                    tol_da=args.mass_tol_da,
                    precursor_ppm=args.precursor_ppm,
                    top_peak_frac=args.top_peak_frac,
                    ion_mode=args.ion_mode,
                )
                local_edits.append(
                    {
                        "sequence": seq,
                        "norm": normalize_peptide(seq),
                        "edit_count": edit_count,
                        "alt_rank": alt_idx,
                        "spec_score": score,
                        "spec_gap_vs_top": score - spec_scores[0],
                        "decoder_margin_vs_alt": decoder_probs[0] - decoder_probs[alt_idx],
                        "base_prior": max(decoder_probs[0], decoder_probs[alt_idx]),
                    }
                )
        cached_rows.append(
            {
                "truth_norm": truth_norm,
                "top_norm": seqs_norm[0],
                "decoder_probs": decoder_probs,
                "spec_scores": spec_scores,
                "local_edits": local_edits,
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
        for edit_penalty in edit_penalties:
            for spec_gap_threshold in spec_gap_thresholds:
                for decoder_margin_threshold in decoder_margin_thresholds:
                    matches = 0
                    refined_rows = 0
                    changed_rows = 0
                    for row in cached_rows:
                        chosen_norm = row["top_norm"]
                        best_score = None
                        best_norm = None
                        for cand in row["local_edits"]:
                            if cand["spec_gap_vs_top"] < spec_gap_threshold:
                                continue
                            if cand["decoder_margin_vs_alt"] > decoder_margin_threshold:
                                continue
                            final_score = (1.0 - alpha) * max(0.0, cand["base_prior"] - edit_penalty * cand["edit_count"]) + alpha * cand["spec_score"]
                            if best_score is None or final_score > best_score:
                                best_score = final_score
                                best_norm = cand["norm"]
                        if best_norm is not None:
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
                            "edit_penalty": edit_penalty,
                            "spec_gap_threshold": spec_gap_threshold,
                            "decoder_margin_threshold": decoder_margin_threshold,
                            "refined_rows": refined_rows,
                            "changed_rows": changed_rows,
                            "matched_peptides": matches,
                            "pep_recall": pep_recall,
                            "pep_recall_delta": pep_recall - baseline_pep_recall,
                            "pep_recall_rel_pct": 0.0 if baseline_pep_recall <= 0 else 100.0 * (pep_recall - baseline_pep_recall) / baseline_pep_recall,
                            "n_match_pep_delta": matches - baseline_matches,
                        }
                    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    best = max(results["probes"], key=lambda x: (x["pep_recall"], x["matched_peptides"]))
    print(json.dumps({"model": args.model, "best": best}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Training-free local candidate rewrite sweep based on beam pairs and spectrum scoring."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


HIGH_RISK = set("DNEQKM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["casanovo", "instanovo", "primenovo"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-alt-rank", type=int, default=2)
    parser.add_argument("--max-edit-span", type=int, default=3)
    parser.add_argument("--max-len-delta", type=int, default=2)
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--rewrite-penalties", default="0.0,0.02,0.05,0.1")
    parser.add_argument("--spec-gap-thresholds", default="0.0,0.05,0.1,0.15")
    parser.add_argument("--decoder-margin-thresholds", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--ion-mode", choices=["both", "y_only", "y_heavy", "b_only"], default="both")
    parser.add_argument("--require-highrisk", action="store_true")
    parser.add_argument("--require-suffix", action="store_true")
    parser.add_argument("--suffix-tail", type=int, default=3)
    parser.add_argument("--include-original-beams", action="store_true")
    return parser.parse_args()


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
                parts.append(ptm_str if ptm_str.startswith(("+", "-")) else f"({ptm_str})")
            continue
        parts.append(f"{aa}({ptm})" if ptm else str(aa))
    return "".join(parts)


def candidate_meta(top_res: list[dict[str, Any]], cand_res: list[dict[str, Any]], suffix_tail: int) -> dict[str, Any]:
    top_norm = normalize_peptide(residues_to_string(top_res))
    cand_norm = normalize_peptide(residues_to_string(cand_res))
    sm = difflib.SequenceMatcher(a=[residue_key(x) for x in top_res], b=[residue_key(x) for x in cand_res])
    opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not opcodes:
        return {
            "top_norm": top_norm,
            "cand_norm": cand_norm,
            "diff_count": 0,
            "suffix_bias": False,
            "highrisk": False,
            "len_delta": 0,
        }
    touched_top = []
    touched_cand = []
    chars = set()
    for tag, i1, i2, j1, j2 in opcodes:
        touched_top.extend(range(i1, i2))
        touched_cand.extend(range(j1, j2))
        chars |= {top_res[i]["aa"] for i in range(i1, i2) if top_res[i]["aa"] != "*"}
        chars |= {cand_res[j]["aa"] for j in range(j1, j2) if cand_res[j]["aa"] != "*"}
    tail_start = max(0, max(len(top_res), len(cand_res)) - suffix_tail)
    suffix_bias = bool(touched_top or touched_cand) and max(touched_top + touched_cand) >= tail_start
    return {
        "top_norm": top_norm,
        "cand_norm": cand_norm,
        "diff_count": sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in opcodes),
        "suffix_bias": suffix_bias,
        "highrisk": bool(chars & HIGH_RISK),
        "len_delta": abs(len(top_res) - len(cand_res)),
    }


def generate_rewrite_candidates(
    top_res: list[dict[str, Any]],
    alt_res: list[dict[str, Any]],
    max_edit_span: int,
    max_len_delta: int,
) -> list[tuple[str, int]]:
    top_keys = [residue_key(x) for x in top_res]
    alt_keys = [residue_key(x) for x in alt_res]
    sm = difflib.SequenceMatcher(a=top_keys, b=alt_keys)
    opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not opcodes:
        return []

    candidates: dict[str, int] = {}

    # Full alternate candidate.
    full_alt = residues_to_string(alt_res)
    candidates[full_alt] = max(len(top_res), len(alt_res))

    # Per-op local replacements / insertions / deletions.
    for _, i1, i2, j1, j2 in opcodes:
        span = max(i2 - i1, j2 - j1)
        if span > max_edit_span or abs((j2 - j1) - (i2 - i1)) > max_len_delta:
            continue
        cand = top_res[:i1] + [dict(x) for x in alt_res[j1:j2]] + top_res[i2:]
        candidates[residues_to_string(cand)] = span

        # One-token context expansion on both sides when possible.
        li1 = max(0, i1 - 1)
        li2 = min(len(top_res), i2 + 1)
        lj1 = max(0, j1 - 1)
        lj2 = min(len(alt_res), j2 + 1)
        span2 = max(li2 - li1, lj2 - lj1)
        if span2 <= max_edit_span + 2 and abs((lj2 - lj1) - (li2 - li1)) <= max_len_delta + 1:
            cand2 = top_res[:li1] + [dict(x) for x in alt_res[lj1:lj2]] + top_res[li2:]
            candidates[residues_to_string(cand2)] = span2

        # Prefix/suffix splice variants.
        cand3 = top_res[:i1] + [dict(x) for x in alt_res[j1:]]
        cand4 = [dict(x) for x in alt_res[:j2]] + top_res[i2:]
        candidates[residues_to_string(cand3)] = max(span, len(alt_res) - j1)
        candidates[residues_to_string(cand4)] = max(span, j2)

    # Combined opcode rewrite if there are multiple local disagreements.
    if len(opcodes) > 1:
        i1 = opcodes[0][1]
        i2 = opcodes[-1][2]
        j1 = opcodes[0][3]
        j2 = opcodes[-1][4]
        span = max(i2 - i1, j2 - j1)
        if span <= max_edit_span + 2 and abs((j2 - j1) - (i2 - i1)) <= max_len_delta + 1:
            cand = top_res[:i1] + [dict(x) for x in alt_res[j1:j2]] + top_res[i2:]
            candidates[residues_to_string(cand)] = span

    cleaned = []
    top_norm = normalize_peptide(residues_to_string(top_res))
    for seq, span in candidates.items():
        if not seq:
            continue
        if normalize_peptide(seq) == top_norm:
            continue
        cleaned.append((seq, span))
    return cleaned


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.alphas)
    rewrite_penalties = parse_float_list(args.rewrite_penalties)
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
        decoder_probs = (
            minmax([max(score, 0.0) for score in raw_scores])
            if args.model in {"casanovo", "primenovo"}
            else softmax(raw_scores)
        )

        top_seq = beams[0]["sequence"]
        top_norm = normalize_peptide(normalize_ptm_format(top_seq, model=args.model))
        truth_norm = normalize_peptide(normalize_ptm_format(row["truth"], model=args.model))
        if top_norm == truth_norm:
            baseline_matches += 1
        total += 1

        top_res = parse_peptide_with_ptm(normalize_ptm_format(top_seq, model=args.model))
        if not top_res:
            continue

        top_spec = advanced_spectrum_match_score(
            top_seq,
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

        rewrites = []
        for alt_idx in range(1, min(len(beams), args.max_alt_rank + 1)):
            alt_seq = beams[alt_idx]["sequence"]
            alt_res = parse_peptide_with_ptm(normalize_ptm_format(alt_seq, model=args.model))
            if not alt_res:
                continue

            if args.include_original_beams:
                alt_norm = normalize_peptide(normalize_ptm_format(alt_seq, model=args.model))
                alt_spec = advanced_spectrum_match_score(
                    alt_seq,
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
                rewrites.append(
                    {
                        "norm": alt_norm,
                        "rewrite_span": 0,
                        "spec_gap_vs_top": alt_spec - top_spec,
                        "decoder_margin_vs_alt": decoder_probs[0] - decoder_probs[alt_idx],
                        "base_prior": decoder_probs[alt_idx],
                        "spec_score": alt_spec,
                    }
                )

            candidates = generate_rewrite_candidates(top_res, alt_res, args.max_edit_span, args.max_len_delta)
            for seq, rewrite_span in candidates:
                cand_res = parse_peptide_with_ptm(normalize_ptm_format(seq, model=args.model))
                if not cand_res:
                    continue
                meta = candidate_meta(top_res, cand_res, args.suffix_tail)
                if meta["len_delta"] > args.max_len_delta:
                    continue
                if args.require_suffix and not meta["suffix_bias"]:
                    continue
                if args.require_highrisk and not meta["highrisk"]:
                    continue

                spec_score = advanced_spectrum_match_score(
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
                rewrites.append(
                    {
                        "norm": meta["cand_norm"],
                        "rewrite_span": rewrite_span,
                        "spec_gap_vs_top": spec_score - top_spec,
                        "decoder_margin_vs_alt": decoder_probs[0] - decoder_probs[alt_idx],
                        "base_prior": max(decoder_probs[0], decoder_probs[alt_idx]),
                        "spec_score": spec_score,
                    }
                )

        cached_rows.append({"truth_norm": truth_norm, "top_norm": top_norm, "rewrites": rewrites})

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
                    matches = 0
                    refined_rows = 0
                    changed_rows = 0
                    for row in cached_rows:
                        chosen_norm = row["top_norm"]
                        best_score = None
                        best_norm = None
                        for cand in row["rewrites"]:
                            if cand["spec_gap_vs_top"] < spec_gap_threshold:
                                continue
                            if cand["decoder_margin_vs_alt"] > decoder_margin_threshold:
                                continue
                            final_score = (1.0 - alpha) * max(
                                0.0, cand["base_prior"] - rewrite_penalty * cand["rewrite_span"]
                            ) + alpha * cand["spec_score"]
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
                            "rewrite_penalty": rewrite_penalty,
                            "spec_gap_threshold": spec_gap_threshold,
                            "decoder_margin_threshold": decoder_margin_threshold,
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

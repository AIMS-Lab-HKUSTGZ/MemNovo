#!/usr/bin/env python3
"""Training-free local mass-confusion rewrite sweep for InstaNovo-like errors."""

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
    normalize_peptide,
    normalize_ptm_format,
    parse_peptide_with_ptm,
    softmax,
)


CONFUSION_MAP = {
    "D": ["N"],
    "N": ["D"],
    "E": ["Q"],
    "Q": ["E", "K"],
    "K": ["Q"],
    "M": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spectra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alphas", default="0.05,0.1,0.14,0.2")
    parser.add_argument("--rewrite-penalties", default="0.0,0.01,0.02,0.05")
    parser.add_argument("--target-refine-rates", default="0.10,0.12,0.14,0.16,0.18,0.20")
    parser.add_argument("--gate-score-modes", default="phase3_combo,entropy_x_diversity")
    parser.add_argument("--decoder-margin-thresholds", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mass-tol-da", type=float, default=0.5)
    parser.add_argument("--precursor-ppm", type=float, default=20.0)
    parser.add_argument("--top-peak-frac", type=float, default=0.2)
    parser.add_argument("--ion-mode", choices=["both", "y_only", "y_heavy", "b_only"], default="y_heavy")
    parser.add_argument("--require-suffix", action="store_true")
    parser.add_argument("--max-pos-from-end", type=int, default=4)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x]


def _normalize(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _gate_scores(rows: list[dict], mode: str) -> np.ndarray:
    entropy_vals = _normalize([float(row.get("entropy", 0.0)) for row in rows])
    diversity_vals = _normalize([float(row.get("diversity", 0.0)) for row in rows])
    conflict_vals = _normalize(
        [
            max(0.0, float(row.get("spec_gap", 0.0)) - max(float(row.get("decoder_margin", 0.0)), 0.0))
            for row in rows
        ]
    )
    if mode == "entropy_x_diversity":
        return entropy_vals * diversity_vals
    if mode == "phase3_combo":
        return 0.50 * entropy_vals + 0.30 * diversity_vals + 0.20 * conflict_vals
    raise ValueError(f"unknown gate score mode: {mode}")


def mutate_confusions(sequence: str, require_suffix: bool, max_pos_from_end: int) -> list[tuple[str, int]]:
    residues = parse_peptide_with_ptm(normalize_ptm_format(sequence, model="instanovo"))
    if not residues:
        return []
    out: list[tuple[str, int]] = []
    length = len(residues)
    for idx, res in enumerate(residues):
        aa = str(res.get("aa", ""))
        if aa not in CONFUSION_MAP or not CONFUSION_MAP[aa]:
            continue
        if require_suffix and idx < max(0, length - max_pos_from_end):
            continue
        for repl in CONFUSION_MAP[aa]:
            new_res = [dict(x) for x in residues]
            new_res[idx]["aa"] = repl
            seq = "".join(
                f"{r['aa']}({r['ptm']})" if r.get("ptm") else str(r.get("aa", ""))
                for r in new_res
            )
            out.append((seq, idx))
    return out


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.alphas)
    rewrite_penalties = parse_float_list(args.rewrite_penalties)
    target_refine_rates = parse_float_list(args.target_refine_rates)
    decoder_margin_thresholds = parse_float_list(args.decoder_margin_thresholds)
    gate_score_modes = [item.strip() for item in args.gate_score_modes.split(",") if item.strip()]

    rows = build_candidates("instanovo", args.baseline)
    spectra = align_spectra(rows, load_spectra(args.spectra))

    cached_rows = []
    baseline_matches = 0
    total = 0
    for row, spectrum in zip(rows, spectra):
        beams = [dict(item) for item in row["candidates"][: args.top_k]]
        if not beams:
            continue
        raw_scores = [float(beam["decoder_score"]) for beam in beams]
        decoder_probs = softmax(raw_scores)
        seqs = [beam["sequence"] for beam in beams]
        truth_norm = normalize_peptide(normalize_ptm_format(row["truth"], model="instanovo"))
        top_seq = seqs[0]
        top_norm = normalize_peptide(normalize_ptm_format(top_seq, model="instanovo"))
        if top_norm == truth_norm:
            baseline_matches += 1
        total += 1

        row_entropy = entropy(decoder_probs)
        row_diversity = diversity(seqs)

        # Best alternative spectrum candidate from existing beam, used only for conflict signal.
        spec_scores = [
            advanced_spectrum_match_score(
                seq,
                spectrum["mz_array"],
                spectrum["intensity_array"],
                precursor_mz=float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0))),
                precursor_charge=int(spectrum.get("precursor_charge", row.get("precursor_charge", 0))),
                model="instanovo",
                tol_da=args.mass_tol_da,
                precursor_ppm=args.precursor_ppm,
                top_peak_frac=args.top_peak_frac,
                ion_mode=args.ion_mode,
            )
            for seq in seqs
        ]
        spec_best_idx = int(np.argmax(np.asarray(spec_scores, dtype=np.float64)))
        spec_gap = float(spec_scores[spec_best_idx] - spec_scores[0])
        decoder_margin = float(decoder_probs[0] - decoder_probs[spec_best_idx])

        rewrites = []
        for seq, pos in mutate_confusions(top_seq, args.require_suffix, args.max_pos_from_end):
            rewrite_norm = normalize_peptide(normalize_ptm_format(seq, model="instanovo"))
            score = advanced_spectrum_match_score(
                seq,
                spectrum["mz_array"],
                spectrum["intensity_array"],
                precursor_mz=float(spectrum.get("precursor_mz", row.get("precursor_mz", 0.0))),
                precursor_charge=int(spectrum.get("precursor_charge", row.get("precursor_charge", 0))),
                model="instanovo",
                tol_da=args.mass_tol_da,
                precursor_ppm=args.precursor_ppm,
                top_peak_frac=args.top_peak_frac,
                ion_mode=args.ion_mode,
            )
            rewrites.append({"sequence": seq, "norm": rewrite_norm, "score": float(score), "pos": pos})

        cached_rows.append(
            {
                "truth_norm": truth_norm,
                "top_norm": top_norm,
                "entropy": row_entropy,
                "diversity": row_diversity,
                "spec_gap": spec_gap,
                "decoder_margin": decoder_margin,
                "rewrites": rewrites,
            }
        )

    results = {
        "model": "instanovo",
        "baseline_matches": baseline_matches,
        "baseline_pep_recall": baseline_matches / total if total else 0.0,
        "probes": [],
    }

    for gate_score_mode in gate_score_modes:
        gate_scores = _gate_scores(cached_rows, gate_score_mode)
        order = np.argsort(-gate_scores)
        for target_refine_rate in target_refine_rates:
            n_target = max(1, int(round(total * target_refine_rate)))
            selected = set(int(idx) for idx in order[:n_target])
            for alpha in alphas:
                for rewrite_penalty in rewrite_penalties:
                    for decoder_margin_threshold in decoder_margin_thresholds:
                        matches = 0
                        refined_rows = 0
                        changed_rows = 0
                        for idx, row in enumerate(cached_rows):
                            pred_norm = row["top_norm"]
                            if idx in selected and row["decoder_margin"] <= decoder_margin_threshold and row["rewrites"]:
                                refined_rows += 1
                                best_seq = pred_norm
                                best_score = None
                                for item in row["rewrites"]:
                                    score = alpha * item["score"] - rewrite_penalty
                                    if best_score is None or score > best_score:
                                        best_score = score
                                        best_seq = item["norm"]
                                if best_seq != pred_norm:
                                    pred_norm = best_seq
                                    changed_rows += 1
                            if pred_norm == row["truth_norm"]:
                                matches += 1

                        pep_recall = matches / total if total else 0.0
                        results["probes"].append(
                            {
                                "gate_score_mode": gate_score_mode,
                                "target_refine_rate": target_refine_rate,
                                "alpha": alpha,
                                "rewrite_penalty": rewrite_penalty,
                                "decoder_margin_threshold": decoder_margin_threshold,
                                "refined_rows": refined_rows,
                                "changed_rows": changed_rows,
                                "matched_peptides": matches,
                                "pep_recall": pep_recall,
                                "pep_recall_delta": pep_recall - results["baseline_pep_recall"],
                                "pep_recall_rel_pct": 0.0
                                if results["baseline_pep_recall"] <= 0
                                else 100.0 * (pep_recall - results["baseline_pep_recall"]) / results["baseline_pep_recall"],
                                "n_match_pep_delta": matches - baseline_matches,
                            }
                        )

    best = max(results["probes"], key=lambda item: (item["pep_recall"], item["matched_peptides"]))
    results["best"] = best

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"model": "instanovo", "best": best}, indent=2))


if __name__ == "__main__":
    main()

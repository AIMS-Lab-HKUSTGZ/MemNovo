#!/usr/bin/env python3
"""
Build a selectively scaled MGF using archive selective_spectrum_scaling.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ARCHIVE_PYC_ROOT = WORKSPACE_ROOT / "archieved" / "legacy_workspace" / "src" / "__pycache__"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.data_handler import DataHandler
from replay_rerank_probe import build_candidates, softmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["instanovo", "primenovo"], required=True)
    parser.add_argument("--input", required=True, help="Input MGF")
    parser.add_argument("--baseline", required=True, help="Baseline CSV/JSONL with beam outputs")
    parser.add_argument("--output", required=True, help="Output scaled MGF")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--threshold-percentile", type=float, default=67.0)
    parser.add_argument("--use-history", action="store_true")
    return parser.parse_args()


def load_archive_scaler():
    path = next(ARCHIVE_PYC_ROOT.glob("selective_spectrum_scaling.cpython-*.pyc"))
    name = "archive_selective_spectrum_scaling_runtime"
    loader = importlib.machinery.SourcelessFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module.SelectiveSpectrumScaler


def sequence_uncertainties(rows: list[dict]) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        candidates = row.get("candidates", [])
        scores = [float(item.get("decoder_score", float("-inf"))) for item in candidates]
        finite = [score for score in scores if np.isfinite(score)]
        if finite and min(finite) >= 0.0 and max(finite) <= 1.0:
            total = sum(finite)
            probs = [score / total for score in finite] if total > 0 else []
        else:
            probs = softmax(scores)
        if not probs:
            values.append(0.0)
            continue
        ent = float(-sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0.0))
        max_ent = math.log(len(probs)) if len(probs) > 1 else 1.0
        values.append(ent / max_ent if max_ent > 0 else 0.0)
    return np.asarray(values, dtype=np.float32)


def write_mgf(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in df.iterrows():
            handle.write("BEGIN IONS\n")
            handle.write(f"TITLE={row.get('spectrum_id', f'spectrum_{index}')}\n")
            handle.write(f"PEPMASS={float(row.get('precursor_mz', 0.0))}\n")
            charge = int(row.get("precursor_charge", 0) or 0)
            if charge:
                handle.write(f"CHARGE={charge}+\n")
            sequence = row.get("sequence", "")
            if isinstance(sequence, str) and sequence:
                handle.write(f"SEQ={sequence}\n")
            mz_array = np.asarray(row["mz_array"], dtype=np.float32)
            intensity_array = np.asarray(row["intensity_array"], dtype=np.float32)
            for mz, intensity in zip(mz_array, intensity_array):
                handle.write(f"{float(mz):.6f} {float(intensity):.6f}\n")
            handle.write("END IONS\n\n")


def main() -> None:
    args = parse_args()
    handler = DataHandler({"path": args.input, "format": "mgf"})
    df = handler.load_data().reset_index(drop=True)
    rows = build_candidates(args.model, args.baseline)
    if len(rows) != len(df):
        raise ValueError(f"Baseline rows ({len(rows)}) != spectra ({len(df)})")

    uncertainties = sequence_uncertainties(rows)
    lengths = [len(np.asarray(x, dtype=np.float32)) for x in df["intensity_array"]]
    max_len = max(lengths) if lengths else 0
    padded = np.zeros((len(df), max_len), dtype=np.float32)
    for idx, intensity in enumerate(df["intensity_array"]):
        arr = np.asarray(intensity, dtype=np.float32)
        padded[idx, : len(arr)] = arr

    Scaler = load_archive_scaler()
    scaler = Scaler(
        beta=float(args.beta),
        threshold_percentile=float(args.threshold_percentile),
        use_history=bool(args.use_history),
    )
    scaled_matrix, scaling_mask = scaler.scale_intensity(padded, uncertainties)

    out_df = df.copy()
    for idx, length in enumerate(lengths):
        out_df.at[idx, "intensity_array"] = np.asarray(scaled_matrix[idx, :length], dtype=np.float32)

    output_path = Path(args.output)
    write_mgf(out_df, output_path)

    threshold = float(scaler.get_threshold(uncertainties))
    summary = {
        "input": args.input,
        "baseline": args.baseline,
        "output": str(output_path),
        "beta": float(args.beta),
        "threshold_percentile": float(args.threshold_percentile),
        "threshold": threshold,
        "scaled_rows": int(np.asarray(scaling_mask).sum()),
        "total_rows": int(len(df)),
        "mean_uncertainty": float(np.asarray(uncertainties).mean()) if len(df) else 0.0,
        "max_uncertainty": float(np.asarray(uncertainties).max()) if len(df) else 0.0,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

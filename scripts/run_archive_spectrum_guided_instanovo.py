#!/usr/bin/env python3
"""Run archive spectrum-guided InstaNovo inference and evaluate the outputs."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ARCHIVE_ROOT = WORKSPACE_ROOT / "archieved" / "legacy_workspace"
ARCHIVE_PYC_ROOT = ARCHIVE_ROOT / "src" / "__pycache__"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.data_handler import DataHandler
from evaluation.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--model-path", default=str(WORKSPACE_ROOT / "weights" / "instanovo-v1.1.0.ckpt"))
    parser.add_argument("--knapsack-path", default=str(WORKSPACE_ROOT / "knapsack_cache" / "instanovo_knapsack"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--precursor-mass-tolerance", type=float, default=50.0)
    parser.add_argument("--guidance-enabled", action="store_true")
    parser.add_argument("--guidance-weight", type=float, default=0.3)
    parser.add_argument("--guidance-mass-tolerance", type=float, default=0.5)
    return parser.parse_args()


def load_pyc(name: str, path: Path):
    loader = importlib.machinery.SourcelessFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def load_archive_interface():
    for p in [
        str(ARCHIVE_ROOT / "refer" / "InstaNovo"),
        str(PROJECT_ROOT / "external" / "instanovo"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)
    load_pyc("spectrum_guided_decoder", ARCHIVE_PYC_ROOT / "spectrum_guided_decoder.cpython-312.pyc")
    module = load_pyc(
        "instanovo_spectrum_guided_interface",
        ARCHIVE_PYC_ROOT / "instanovo_spectrum_guided_interface.cpython-312.pyc",
    )
    return module.SpectrumGuidedInstanovoInterface


def load_ground_truth(path: str) -> list[dict]:
    handler = DataHandler({"path": path, "format": "auto"})
    df = handler.load_data()
    truths: list[dict] = []
    for _, row in df.iterrows():
        truths.append(
            {
                "sequence": row.get("sequence", ""),
                "precursor_mz": float(row.get("precursor_mz", 0.0)),
                "precursor_charge": int(row.get("precursor_charge", 0)),
            }
        )
    return truths


def evaluate_predictions(df: pd.DataFrame, input_path: str) -> dict:
    predictions = [{"sequence": str(x) if x == x else ""} for x in df["predictions"]]
    truths = load_ground_truth(input_path)
    evaluator = Evaluator({"model_name": "instanovo"})
    if hasattr(evaluator, "evaluate_with_instanovo_metrics"):
        return evaluator.evaluate_with_instanovo_metrics(predictions, truths)
    return evaluator.evaluate_instanovo(predictions, truths)


def main() -> None:
    args = parse_args()
    Interface = load_archive_interface()
    config = {
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "max_length": args.max_length,
        "precursor_mass_tolerance": args.precursor_mass_tolerance,
        "use_knapsack": True,
        "knapsack_path": args.knapsack_path,
        "spectrum_guidance": {
            "enabled": bool(args.guidance_enabled),
            "weight": args.guidance_weight,
            "mass_tolerance": args.guidance_mass_tolerance,
        },
    }

    interface = Interface(config)
    interface.load_model()
    results_df = interface.predict_from_file(args.input)
    output_csv = Path(args.output_csv)
    metrics_json = Path(args.metrics_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    metrics = evaluate_predictions(results_df, args.input)
    metrics.update(
        {
            "guidance_enabled": bool(args.guidance_enabled),
            "guidance_weight": args.guidance_weight,
            "guidance_mass_tolerance": args.guidance_mass_tolerance,
            "input": args.input,
            "output_csv": str(output_csv),
        }
    )
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

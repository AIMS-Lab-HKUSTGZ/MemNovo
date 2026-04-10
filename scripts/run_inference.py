#!/usr/bin/env python3
"""
Run baseline or MemNovo-enhanced inference on a single dataset file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import DataHandler, Evaluator, load_mgf_file, save_predictions
from memnovo import MemNovoModel
from memnovo.backends import resolve_path
from memnovo.utils import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MemNovo inference")
    parser.add_argument("--config", "-c", required=True, help="YAML configuration file")
    parser.add_argument("--input", "-i", required=True, help="Input MGF/parquet/csv file")
    parser.add_argument("--output", "-o", required=True, help="Output prediction file")
    parser.add_argument("--checkpoint", default=None, help="Override model checkpoint")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--beam-size", type=int, default=None, help="Override beam size")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions against SEQ labels")
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional JSON file for evaluation metrics",
    )
    return parser.parse_args()


def infer_output_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".tsv":
        return "tsv"
    return "csv"


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    if args.checkpoint:
        config.setdefault("model", {})["checkpoint"] = args.checkpoint
    if args.batch_size:
        config.setdefault("inference", {})["batch_size"] = args.batch_size
    if args.beam_size:
        config.setdefault("inference", {})["beam_size"] = args.beam_size

    model_name = config.get("model", {}).get("name", "instanovo")
    checkpoint = config.get("model", {}).get("checkpoint")
    input_path = Path(resolve_path(args.input) or args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s from %s", model_name, checkpoint)
    model = MemNovoModel.from_pretrained(
        model_name=model_name,
        checkpoint_path=checkpoint,
        config=config,
        device=args.device,
    )

    spectra_input = str(input_path)
    ground_truth = None
    if args.max_samples is not None:
        handler = DataHandler({"path": str(input_path), "format": "auto", "max_samples": args.max_samples})
        ground_truth = handler.load_data().to_dict("records")
        spectra_input = []
        for index, row in enumerate(ground_truth):
            spectra_input.append(
                {
                    "spectrum_id": row.get("spectrum_id", row.get("experiment_name", f"spectrum_{index}")),
                    "mz_array": row["mz_array"],
                    "intensity_array": row["intensity_array"],
                    "precursor_mz": float(row.get("precursor_mz", 0.0)),
                    "precursor_charge": int(row.get("precursor_charge", 0)),
                    "sequence": row.get("sequence", ""),
                    "modified_sequence": row.get("modified_sequence", ""),
                }
            )

    predictions = model.predict(
        spectra_input,
        batch_size=config.get("inference", {}).get("batch_size"),
        beam_size=config.get("inference", {}).get("beam_size"),
    )
    save_predictions(predictions, str(output_path), format=infer_output_format(output_path))
    logger.info("Saved %s predictions to %s", len(predictions), output_path)

    if args.evaluate:
        if ground_truth is None:
            handler = DataHandler({"path": str(input_path), "format": "auto"})
            ground_truth = handler.load_data().to_dict("records")
        evaluator = Evaluator(
            mass_tolerance=float(config.get("evaluation", {}).get("mass_tolerance", 50)),
            normalize_il=bool(config.get("evaluation", {}).get("normalize_il", True)),
        )
        metrics = evaluator.evaluate(predictions, ground_truth, model_name=model_name)
        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            logger.info("  %s: %s", key, value)

        metrics_path = Path(args.metrics_output) if args.metrics_output else output_path.with_suffix(".metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Saved metrics to %s", metrics_path)

    stats = model.get_stats()
    logger.info("MemNovo stats: %s", stats)
    model.cleanup()


if __name__ == "__main__":
    main()

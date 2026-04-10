#!/usr/bin/env python3
"""
Run InstaNovo inference through the vendored official predictor stack.

This keeps the upstream data loading / decoding path intact and only injects
MemNovo by registering hooks on the loaded model before calling
`instanovo.transformer.predict.get_preds`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import Evaluator
from memnovo.backends import ensure_external_imports, resolve_path
from memnovo.manager import MemNovoManager
from memnovo.utils import load_config, setup_logging

DEFAULT_RESIDUE_REMAPPING = {
    "M(ox)": "M[UNIMOD:35]",
    "M(+15.99)": "M[UNIMOD:35]",
    "M+15.995": "M[UNIMOD:35]",
    "S(p)": "S[UNIMOD:21]",
    "T(p)": "T[UNIMOD:21]",
    "Y(p)": "Y[UNIMOD:21]",
    "S(+79.97)": "S[UNIMOD:21]",
    "T(+79.97)": "T[UNIMOD:21]",
    "Y(+79.97)": "Y[UNIMOD:21]",
    "Q(+0.98)": "Q[UNIMOD:7]",
    "N(+0.98)": "N[UNIMOD:7]",
    "Q(+.98)": "Q[UNIMOD:7]",
    "N(+.98)": "N[UNIMOD:7]",
    "Q+0.984": "Q[UNIMOD:7]",
    "N+0.984": "N[UNIMOD:7]",
    "C(+57.02)": "C[UNIMOD:4]",
    "C+57.021": "C[UNIMOD:4]",
    "(+42.01)": "[UNIMOD:1]",
    "(+43.01)": "[UNIMOD:5]",
    "(-17.03)": "[UNIMOD:385]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InstaNovo via official predictor stack")
    parser.add_argument("--config", "-c", required=True, help="YAML configuration file")
    parser.add_argument("--input", "-i", required=True, help="Annotated input MGF/parquet/csv file")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--beam-size", type=int, default=None, help="Override beam size")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:2")
    parser.add_argument("--subset", type=float, default=None, help="Optional fraction in (0, 1]")
    parser.add_argument("--max-length", type=int, default=None, help="Override max decoding length")
    parser.add_argument("--log-interval", type=int, default=None, help="Override progress logging interval")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fp16 autocast setting",
    )
    parser.add_argument(
        "--use-knapsack",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override knapsack beam search usage",
    )
    parser.add_argument(
        "--save-beams",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override full beam saving",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional JSON path for summarized evaluation metrics",
    )
    return parser.parse_args()


def _build_predict_config(config: dict[str, Any], args: argparse.Namespace, output_path: Path) -> Any:
    model_cfg = dict(config.get("model", {}))
    inference_cfg = dict(config.get("inference", {}))
    residue_remapping = dict(DEFAULT_RESIDUE_REMAPPING)
    residue_remapping.update(config.get("data", {}).get("residue_remapping", {}))
    residue_remapping.update(inference_cfg.get("residue_remapping", {}))

    checkpoint = args.checkpoint or model_cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Missing InstaNovo checkpoint")

    resolved_checkpoint = resolve_path(checkpoint)
    resolved_input = resolve_path(args.input) or args.input

    predict_cfg = {
        "instanovo_model": resolved_checkpoint,
        "data_path": resolved_input,
        "output_path": str(output_path),
        "denovo": False,
        "device": args.device,
        "batch_size": int(args.batch_size or inference_cfg.get("batch_size", 64)),
        "num_beams": int(args.beam_size or inference_cfg.get("beam_size", 5)),
        "max_length": int(args.max_length or inference_cfg.get("max_length", 30)),
        "use_knapsack": bool(
            inference_cfg.get("use_knapsack", True) if args.use_knapsack is None else args.use_knapsack
        ),
        "knapsack_path": resolve_path(inference_cfg.get("knapsack_path", "knapsack_cache/instanovo_knapsack")),
        "fp16": bool(inference_cfg.get("fp16", False) if args.fp16 is None else args.fp16),
        "save_beams": bool(
            inference_cfg.get("save_beams", True) if args.save_beams is None else args.save_beams
        ),
        "log_interval": int(args.log_interval or inference_cfg.get("log_interval", 50)),
        "subset": float(args.subset if args.subset is not None else inference_cfg.get("subset", 1.0)),
        "use_basic_logging": True,
        "residue_remapping": residue_remapping,
    }
    return OmegaConf.create(predict_cfg)


def _evaluate_output(output_path: Path, metrics_output: Path) -> dict[str, Any]:
    df = pd.read_csv(output_path)
    if "predictions" not in df.columns or "targets" not in df.columns:
        raise ValueError(f"Expected official InstaNovo output columns in {output_path}")

    predictions = [
        {
            "sequence": row["predictions"] if isinstance(row["predictions"], str) else "",
            "score": float(row["log_probabilities"]) if "log_probabilities" in df.columns else 0.0,
        }
        for _, row in df.iterrows()
    ]
    targets = [{"sequence": row["targets"] if isinstance(row["targets"], str) else ""} for _, row in df.iterrows()]

    evaluator = Evaluator()
    try:
        metrics = evaluator.evaluate(predictions, targets, model_name="instanovo")
    except Exception as exc:
        ensure_external_imports()
        from instanovo.utils.metrics import Metrics
        from instanovo.utils.residues import ResidueSet

        residue_masses = {
            "A": 71.037114,
            "R": 156.101111,
            "N": 114.042927,
            "D": 115.026943,
            "C": 103.009185,
            "E": 129.042593,
            "Q": 128.058578,
            "G": 57.021464,
            "H": 137.058912,
            "I": 113.084064,
            "L": 113.084064,
            "K": 128.094963,
            "M": 131.040485,
            "F": 147.068414,
            "P": 97.052764,
            "S": 87.032028,
            "T": 101.047670,
            "W": 186.079313,
            "Y": 163.063329,
            "V": 99.068414,
            "M[UNIMOD:35]": 147.035400,
            "C[UNIMOD:4]": 160.030649,
            "N[UNIMOD:7]": 115.026943,
            "Q[UNIMOD:7]": 129.042594,
            "S[UNIMOD:21]": 166.998028,
            "T[UNIMOD:21]": 181.013670,
            "Y[UNIMOD:21]": 243.029329,
            "[UNIMOD:1]": 42.010565,
            "[UNIMOD:5]": 43.005814,
            "[UNIMOD:385]": -17.026549,
        }
        calc = Metrics(
            ResidueSet(residue_masses, residue_remapping=DEFAULT_RESIDUE_REMAPPING),
            isotope_error_range=[0, 1],
        )
        pred_tokens = []
        if "predictions_tokenised" in df.columns:
            for value in df["predictions_tokenised"]:
                if isinstance(value, str) and value.strip():
                    pred_tokens.append(value.split(", "))
                else:
                    pred_tokens.append([])
        else:
            pred_tokens = [item["sequence"] for item in predictions]
        target_strings = [item["sequence"] for item in targets]
        aa_precision, aa_recall, pep_recall, pep_precision = calc.compute_precision_recall(
            target_strings,
            pred_tokens,
        )
        metrics = {
            "aa_precision": float(aa_precision),
            "aa_recall": float(aa_recall),
            "pep_precision": float(pep_precision),
            "pep_recall": float(pep_recall),
            "total_samples": int(len(df)),
            "valid_predictions": int(len(df)),
            "method": "instanovo_official_fallback",
            "evaluation_warning": str(exc),
        }
    metrics["rows"] = int(len(df))
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    if config.get("model", {}).get("name") != "instanovo":
        raise ValueError("This runner only supports model.name=instanovo")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output = (
        Path(args.metrics_output)
        if args.metrics_output
        else output_path.with_suffix(".metrics.json")
    )

    ensure_external_imports()
    from instanovo.transformer.model import InstaNovo
    from instanovo.transformer.predict import get_preds
    from instanovo.utils.s3 import S3FileHandler

    predict_cfg = _build_predict_config(config, args, output_path)
    checkpoint = str(predict_cfg.instanovo_model)
    logger.info("Loading InstaNovo checkpoint %s", checkpoint)
    model, model_config = InstaNovo.load(checkpoint)

    memnovo_cfg = dict(config.get("memnovo", {}))
    manager = MemNovoManager(memnovo_cfg)
    if manager.is_enabled:
        logger.info("Registering MemNovo hooks on official InstaNovo predictor")
        manager.register(model)
    else:
        logger.info("Running official InstaNovo baseline without MemNovo")

    s3 = S3FileHandler()
    memory_supported = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if memory_supported:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    try:
        get_preds(predict_cfg, model, model_config, s3)
    finally:
        manager.reset()
        manager.unregister()
        s3.cleanup()

    total_s = time.time() - start
    if memory_supported:
        torch.cuda.synchronize()
        max_memory_allocated_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        max_memory_reserved_gb = float(torch.cuda.max_memory_reserved() / (1024 ** 3))
    else:
        max_memory_allocated_gb = None
        max_memory_reserved_gb = None
    metrics = _evaluate_output(output_path, metrics_output)
    rows = int(metrics.get("rows", metrics.get("total_samples", 0)))
    runtime_path = output_path.with_suffix(".runtime.json")
    runtime_payload = {
        "total_seconds": total_s,
        "rows": rows,
        "ms_per_spectrum": float((total_s * 1000.0 / rows) if rows else 0.0),
        "spectra_per_second": float((rows / total_s) if rows else 0.0),
        "device": args.device,
        "batch_size": int(predict_cfg.batch_size),
        "num_beams": int(predict_cfg.num_beams),
        "fp16": bool(predict_cfg.fp16),
        "use_knapsack": bool(predict_cfg.use_knapsack),
        "save_beams": bool(predict_cfg.save_beams),
        "subset": float(predict_cfg.subset),
        "max_memory_allocated_gb": max_memory_allocated_gb,
        "max_memory_reserved_gb": max_memory_reserved_gb,
        "memnovo_enabled": bool(memnovo_cfg.get("enabled", True)),
        "memnovo_stats": manager.get_stats(),
        "metrics_output": str(metrics_output),
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")

    logger.info("Official predictor finished in %.1fs", total_s)
    logger.info("Metrics saved to %s", metrics_output)
    logger.info("AA recall %.5f, peptide recall %.5f", metrics["aa_recall"], metrics["pep_recall"])


if __name__ == "__main__":
    main()

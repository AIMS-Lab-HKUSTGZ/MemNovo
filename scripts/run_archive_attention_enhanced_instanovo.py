#!/usr/bin/env python3
"""
Run InstaNovo through the official predictor stack with archive attention enhancer V3.
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
WORKSPACE_ROOT = PROJECT_ROOT.parent
ARCHIVE_SRC_ROOT = WORKSPACE_ROOT / "archieved" / "legacy_workspace" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ARCHIVE_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_SRC_ROOT))

from evaluation import Evaluator
from memnovo.backends import ensure_external_imports, resolve_path
from memnovo.utils import load_config, setup_logging
from spectrum_attention_enhancer_v3 import SpectrumAttentionEnhancerV3

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subset", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-knapsack", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-beams", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--scale-factor", type=float, default=1.05)
    parser.add_argument("--target-layers", default="6,7,8")
    parser.add_argument("--per-layer-scales", default=None, help='JSON dict like {"6":1.05,"7":1.08,"8":1.12}')
    return parser.parse_args()


def _parse_target_layers(value: str) -> list[int] | str:
    stripped = value.strip().lower()
    if stripped == "all":
        return "all"
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_per_layer_scales(value: str | None) -> dict[int, float] | None:
    if not value:
        return None
    raw = json.loads(value)
    return {int(k): float(v) for k, v in raw.items()}


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
    predictions = [
        {
            "sequence": row["predictions"] if isinstance(row["predictions"], str) else "",
            "score": float(row["log_probabilities"]) if "log_probabilities" in df.columns else 0.0,
        }
        for _, row in df.iterrows()
    ]
    targets = [{"sequence": row["targets"] if isinstance(row["targets"], str) else ""} for _, row in df.iterrows()]

    evaluator = Evaluator()
    metrics = evaluator.evaluate(predictions, targets, model_name="instanovo")
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
    metrics_output = Path(args.metrics_output) if args.metrics_output else output_path.with_suffix(".metrics.json")

    ensure_external_imports()
    from instanovo.transformer.model import InstaNovo
    from instanovo.transformer.predict import get_preds
    from instanovo.utils.s3 import S3FileHandler

    predict_cfg = _build_predict_config(config, args, output_path)
    checkpoint = str(predict_cfg.instanovo_model)
    logger.info("Loading InstaNovo checkpoint %s", checkpoint)
    model, model_config = InstaNovo.load(checkpoint)

    enhancer = SpectrumAttentionEnhancerV3(
        scale_factor=float(args.scale_factor),
        target_layers=_parse_target_layers(args.target_layers),
        enabled=True,
        per_layer_scales=_parse_per_layer_scales(args.per_layer_scales),
    )
    enhancer.patch_model(model)

    s3 = S3FileHandler()
    memory_supported = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if memory_supported:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    try:
        get_preds(predict_cfg, model, model_config, s3)
    finally:
        enhancer.unpatch_model()
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
    runtime = {
        "input": str(args.input),
        "output": str(output_path),
        "scale_factor": float(args.scale_factor),
        "target_layers": _parse_target_layers(args.target_layers),
        "per_layer_scales": _parse_per_layer_scales(args.per_layer_scales),
        "total_s": total_s,
        "rows": rows,
        "spectrum_per_s": (rows / total_s) if rows and total_s > 0 else None,
        "max_memory_allocated_gb": max_memory_allocated_gb,
        "max_memory_reserved_gb": max_memory_reserved_gb,
        "enhancer_stats": enhancer.get_stats(),
    }
    output_path.with_suffix(".runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()

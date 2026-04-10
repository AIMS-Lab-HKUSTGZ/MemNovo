#!/usr/bin/env python3
"""Run architecture-aware decomposed sensitivity scaling for official InstaNovo."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import Evaluator
from memnovo.backends import ensure_external_imports, resolve_path
from memnovo.utils import load_config, setup_logging

DEFAULT_SCALE_FACTORS = [
    0.990,
    0.992,
    0.994,
    0.996,
    0.998,
    0.999,
    1.000,
    1.001,
    1.002,
    1.004,
    1.006,
    1.008,
    1.010,
]

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

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline_instanovo.yaml"),
    )
    parser.add_argument(
        "--input",
        default=str(WORKSPACE_ROOT / "dataset" / "hc_pt" / "test.parquet"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset-rows", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-knapsack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-beams", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--modes",
        default="spectrum_full,spectrum_memory_only,precursor_only,history_only,peptide_full",
    )
    parser.add_argument("--metric", default="aa_precision")
    parser.add_argument("--scale-factors", nargs="*", type=float, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def ensure_subset(input_path: Path, output_path: Path, rows: int) -> Path:
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
        df.head(rows).to_parquet(output_path, index=False)
        return output_path
    raise ValueError(f"Unsupported subset source: {input_path}")


def _build_predict_config(config: dict[str, Any], args: argparse.Namespace, input_path: Path, output_path: Path) -> Any:
    model_cfg = dict(config.get("model", {}))
    inference_cfg = dict(config.get("inference", {}))
    residue_remapping = dict(DEFAULT_RESIDUE_REMAPPING)
    residue_remapping.update(config.get("data", {}).get("residue_remapping", {}))
    residue_remapping.update(inference_cfg.get("residue_remapping", {}))

    checkpoint = model_cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Missing InstaNovo checkpoint")

    return OmegaConf.create(
        {
            "instanovo_model": resolve_path(checkpoint),
            "data_path": str(input_path),
            "output_path": str(output_path),
            "denovo": False,
            "device": args.device,
            "batch_size": int(args.batch_size),
            "num_beams": int(args.beam_size),
            "max_length": int(args.max_length),
            "use_knapsack": bool(args.use_knapsack),
            "knapsack_path": resolve_path(inference_cfg.get("knapsack_path", "../knapsack_cache/instanovo_knapsack")),
            "fp16": bool(args.fp16),
            "save_beams": bool(args.save_beams),
            "log_interval": int(args.log_interval),
            "subset": 1.0,
            "use_basic_logging": True,
            "residue_remapping": residue_remapping,
        }
    )


def evaluate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    predictions = []
    targets = []
    for _, row in df.iterrows():
        predictions.append(
            {
                "sequence": row["predictions"] if isinstance(row.get("predictions"), str) else "",
                "score": float(row.get("log_probabilities", 0.0)),
            }
        )
        targets.append({"sequence": row["targets"] if isinstance(row.get("targets"), str) else ""})
    evaluator = Evaluator()
    metrics = evaluator.evaluate(predictions, targets, model_name="instanovo")
    metrics["rows"] = int(len(df))
    return metrics


class PatchRegistry:
    def __init__(self) -> None:
        self._restore: list[Callable[[], None]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def add_hook(self, module: torch.nn.Module, hook_fn: Callable[..., Any]) -> None:
        self._handles.append(module.register_forward_hook(hook_fn))

    def wrap_attr(self, obj: Any, attr_name: str, wrapper_fn: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
        original = getattr(obj, attr_name)
        setattr(obj, attr_name, wrapper_fn(original))
        self._restore.append(lambda: setattr(obj, attr_name, original))

    def clear(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for restore in reversed(self._restore):
            restore()
        self._restore.clear()


def _scale_tensor_region(tensor: torch.Tensor, scale: float, token_slice: slice | None = None) -> torch.Tensor:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return tensor
    updated = tensor.clone()
    if token_slice is None:
        updated = updated * scale
    else:
        updated[:, token_slice, :] = updated[:, token_slice, :] * scale
    return updated


def register_mode(model: Any, mode: str, scale: float) -> PatchRegistry:
    patches = PatchRegistry()

    if mode == "spectrum_full":
        patches.add_hook(
            model.encoder,
            lambda _module, _inputs, output: _scale_tensor_region(output, scale),
        )
        return patches

    if mode == "spectrum_memory_only":
        def _wrap_encoder(original: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
                memory, mask = original(*args, **kwargs)
                return _scale_tensor_region(memory, scale, slice(1, None)), mask

            return wrapped

        patches.wrap_attr(model, "_encoder", _wrap_encoder)
        return patches

    if mode == "precursor_only":
        def _wrap_encoder(original: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
                memory, mask = original(*args, **kwargs)
                return _scale_tensor_region(memory, scale, slice(0, 1)), mask

            return wrapped

        patches.wrap_attr(model, "_encoder", _wrap_encoder)
        return patches

    if mode == "history_only":
        patches.add_hook(
            model.aa_pos_embed,
            lambda _module, _inputs, output: _scale_tensor_region(output, scale),
        )
        return patches

    if mode == "peptide_full":
        precursor_patches = register_mode(model, "precursor_only", scale)
        history_patches = register_mode(model, "history_only", scale)
        patches._restore.extend(precursor_patches._restore)
        patches._restore.extend(history_patches._restore)
        patches._handles.extend(precursor_patches._handles)
        patches._handles.extend(history_patches._handles)
        precursor_patches._restore = []
        precursor_patches._handles = []
        history_patches._restore = []
        history_patches._handles = []
        return patches

    raise ValueError(f"Unsupported mode: {mode}")


def compute_subrange_ratios(results: dict[str, list[dict[str, Any]]], metric: str) -> list[dict[str, Any]]:
    subranges = [
        ("±0.1%", [0.999, 1.001]),
        ("±0.2%", [0.998, 1.002]),
        ("±0.4%", [0.996, 1.004]),
        ("±0.6%", [0.994, 1.006]),
        ("±0.8%", [0.992, 1.008]),
        ("±1.0%", [0.990, 1.010]),
    ]

    def _lookup(mode: str, scale: float) -> dict[str, Any]:
        for row in results[mode]:
            if abs(float(row["scale_factor"]) - scale) < 1e-9:
                return row
        raise KeyError((mode, scale))

    baseline_s = float(_lookup("spectrum_full", 1.0)[metric])
    baseline_p = float(_lookup("peptide_full", 1.0)[metric])
    rows: list[dict[str, Any]] = []
    for name, scales in subranges:
        delta_s = sum(abs(float(_lookup("spectrum_full", s)[metric]) - baseline_s) / max(abs(baseline_s), 1e-12) for s in scales) / len(scales)
        delta_p = sum(abs(float(_lookup("peptide_full", s)[metric]) - baseline_p) / max(abs(baseline_p), 1e-12) for s in scales) / len(scales)
        rows.append(
            {
                "subrange": name,
                "spectrum_sensitivity": delta_s,
                "peptide_sensitivity": delta_p,
                "sensitivity_ratio": (delta_p / max(delta_s, 1e-12)),
            }
        )
    return rows


def summarize_decomposition(results: dict[str, list[dict[str, Any]]], metric: str) -> dict[str, Any]:
    summary = {}
    for mode in ("precursor_only", "history_only", "peptide_full", "spectrum_full", "spectrum_memory_only"):
        rows = results.get(mode, [])
        if not rows:
            continue
        baseline = next(row for row in rows if abs(float(row["scale_factor"]) - 1.0) < 1e-9)
        baseline_value = float(baseline[metric])
        deviations = []
        for row in rows:
            if abs(float(row["scale_factor"]) - 1.0) < 1e-9:
                continue
            deviations.append(abs(float(row[metric]) - baseline_value) / max(abs(baseline_value), 1e-12))
        summary[mode] = {
            "baseline": baseline_value,
            "mean_abs_relative_change": sum(deviations) / len(deviations) if deviations else 0.0,
        }
    return summary


def write_markdown(output_path: Path, metric: str, decomposition: dict[str, Any], subranges: list[dict[str, Any]]) -> None:
    lines = [
        "# WSMy Q2/Q3 InstaNovo Decomposed Sensitivity",
        "",
        f"Metric used for summary: `{metric}`",
        "",
        "## Decomposed Sensitivity Summary",
        "",
        "| Mode | Baseline | Mean Abs. Relative Change |",
        "|---|---:|---:|",
    ]
    for mode, payload in decomposition.items():
        lines.append(
            f"| {mode} | {payload['baseline']:.6f} | {payload['mean_abs_relative_change']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Sub-range Sensitivity Ratio",
            "",
            "| Sub-range | Spectrum Sensitivity | Peptide Sensitivity | Sensitivity Ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in subranges:
        lines.append(
            f"| {row['subrange']} | {row['spectrum_sensitivity']:.6f} | {row['peptide_sensitivity']:.6f} | {row['sensitivity_ratio']:.3f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(resolve_path(args.input) or args.input)
    subset_path = output_dir / f"{input_path.stem}_head{args.subset_rows}.parquet"
    subset_path = ensure_subset(input_path, subset_path, args.subset_rows)

    config = load_config(args.config)
    predict_cfg = _build_predict_config(config, args, subset_path, runs_dir / "placeholder.csv")

    ensure_external_imports()
    from instanovo.transformer.model import InstaNovo
    from instanovo.transformer.predict import get_preds
    from instanovo.utils.s3 import S3FileHandler

    model, model_config = InstaNovo.load(str(predict_cfg.instanovo_model))
    s3 = S3FileHandler()

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    all_results: dict[str, list[dict[str, Any]]] = {}

    scale_factors = args.scale_factors or list(DEFAULT_SCALE_FACTORS)

    for mode in modes:
        mode_results: list[dict[str, Any]] = []
        logger.info("Running mode=%s", mode)
        for scale in scale_factors:
            output_csv = runs_dir / f"{mode}_scale_{scale:.3f}.csv"
            metrics_json = output_csv.with_suffix(".metrics.json")
            patches = register_mode(model, mode, scale)
            try:
                local_cfg = OmegaConf.create(OmegaConf.to_container(predict_cfg, resolve=True))
                local_cfg.output_path = str(output_csv)
                start = time.perf_counter()
                get_preds(local_cfg, model, model_config, s3)
                pred_df = pd.read_csv(output_csv)
                elapsed = time.perf_counter() - start
                metrics = evaluate_dataframe(pred_df)
                record = {
                    **metrics,
                    "mode": mode,
                    "scale_factor": float(scale),
                    "elapsed_seconds": elapsed,
                    "rows": int(len(pred_df)),
                }
                metrics_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
                mode_results.append(record)
                logger.info(
                    "mode=%s scale=%.3f %s=%.4f pep_precision=%.4f elapsed=%.1fs",
                    mode,
                    scale,
                    args.metric,
                    record.get(args.metric, 0.0),
                    record.get("pep_precision", 0.0),
                    elapsed,
                )
            finally:
                patches.clear()
                if str(args.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        all_results[mode] = mode_results

    results_path = output_dir / "decomposed_sensitivity_results.json"
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    decomposition = summarize_decomposition(all_results, args.metric)
    subranges = compute_subrange_ratios(all_results, args.metric) if set(scale_factors) >= {0.99, 0.992, 0.994, 0.996, 0.998, 0.999, 1.0, 1.001, 1.002, 1.004, 1.006, 1.008, 1.01} else []

    (output_dir / "decomposition_summary.json").write_text(
        json.dumps(decomposition, indent=2),
        encoding="utf-8",
    )
    (output_dir / "subrange_ratio_summary.json").write_text(json.dumps(subranges, indent=2), encoding="utf-8")
    write_markdown(output_dir / "summary.md", args.metric, decomposition, subranges)


if __name__ == "__main__":
    main()

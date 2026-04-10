"""
Runnable sensitivity scaling experiment runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import DataHandler, Evaluator
from memnovo import MemNovoModel
from memnovo.backends import resolve_path
from memnovo.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


class SensitivityScaler:
    """Apply inference-time scaling to spectrum or peptide pathways."""

    DEFAULT_SCALE_FACTORS = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    CASANOVO_SCALE_FACTORS = DEFAULT_SCALE_FACTORS
    INSTANOVO_SCALE_FACTORS = [
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

    def __init__(self, pipeline: MemNovoModel):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.model_type = pipeline.model_name
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    @classmethod
    def default_scale_factors(cls, model_type: str) -> List[float]:
        if model_type.lower() == "instanovo":
            return list(cls.INSTANOVO_SCALE_FACTORS)
        return list(cls.CASANOVO_SCALE_FACTORS)

    def run_experiment(
        self,
        spectra: List[Dict[str, Any]],
        scale_factors: Optional[List[float]] = None,
        modality: str = "spectrum",
        evaluator: Optional[Evaluator] = None,
        batch_size: Optional[int] = None,
        beam_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        if scale_factors is None:
            scale_factors = self.default_scale_factors(self.model_type)

        if evaluator is None:
            evaluator = Evaluator()

        logger.info("Running %s scaling on %s with %d spectra", modality, self.model_type, len(spectra))

        results: List[Dict[str, Any]] = []
        for scale in scale_factors:
            try:
                self._register_scaling_hooks(modality=modality, scale=scale)
                predictions = self.pipeline.predict(
                    spectra,
                    batch_size=batch_size,
                    beam_size=beam_size,
                )
                metrics = evaluator.evaluate(predictions, spectra, model_name=self.model_type)
                metrics["scale_factor"] = float(scale)
                metrics["modality"] = modality
                metrics["avg_pred_length"] = self._average_prediction_length(predictions)
                metrics["n_predictions"] = len(predictions)
                results.append(metrics)
                logger.info(
                    "%s scale=%s aa_precision=%.4f pep_precision=%.4f",
                    modality,
                    scale,
                    metrics.get("aa_precision", 0.0),
                    metrics.get("pep_precision", 0.0),
                )
            except Exception as exc:
                logger.exception("Sensitivity run failed for %s scale=%s", modality, scale)
                results.append(
                    {
                        "scale_factor": float(scale),
                        "modality": modality,
                        "error": str(exc),
                    }
                )
            finally:
                self._remove_scaling_hooks()
                self.pipeline.memnovo_manager.reset()

        return {
            "modality": modality,
            "results": results,
            "scale_factors": list(scale_factors),
        }

    def compute_sensitivity(
        self,
        results: Dict[str, Any],
        baseline_scale: float = 1.0,
        metric: str = "aa_precision",
    ) -> float:
        result_list = results.get("results", [])
        baseline_perf = None
        for result in result_list:
            if result.get("scale_factor") == baseline_scale and "error" not in result:
                baseline_perf = result.get(metric, 0.0)
                break

        if baseline_perf in (None, 0.0):
            return 0.0

        deviations = []
        for result in result_list:
            if result.get("scale_factor") == baseline_scale or "error" in result:
                continue
            perf = result.get(metric, 0.0)
            deviations.append(abs(perf - baseline_perf) / baseline_perf)

        return float(np.mean(deviations)) if deviations else 0.0

    def _register_scaling_hooks(self, modality: str, scale: float) -> None:
        if modality == "spectrum":
            module = self._find_spectrum_module()
        elif modality == "peptide":
            module = self._find_peptide_module()
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if module is None:
            raise RuntimeError(f"Could not find a module to scale for modality '{modality}'")

        hook = module.register_forward_hook(self._make_scale_hook(scale))
        self.hooks.append(hook)

    def _remove_scaling_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _find_spectrum_module(self) -> Optional[torch.nn.Module]:
        candidates = [
            ("encoder",),
            ("spectrum_encoder",),
        ]
        for path in candidates:
            module = self._resolve_attr_path(self.model, path)
            if module is not None:
                return module

        for name, module in self.model.named_modules():
            lowered = name.lower()
            if lowered.endswith("encoder") and "decoder" not in lowered:
                return module
        return None

    def _find_peptide_module(self) -> Optional[torch.nn.Module]:
        candidates = [
            ("aa_embed",),
            ("decoder", "aa_embed"),
            ("decoder", "token_encoder"),
        ]
        for path in candidates:
            module = self._resolve_attr_path(self.model, path)
            if module is not None:
                return module

        for name, module in self.model.named_modules():
            lowered = name.lower()
            if lowered.endswith("aa_embed") or lowered.endswith("token_encoder"):
                return module
        return None

    def _resolve_attr_path(self, obj: Any, path: Sequence[str]) -> Optional[Any]:
        current = obj
        for item in path:
            if not hasattr(current, item):
                return None
            current = getattr(current, item)
        return current

    def _make_scale_hook(self, scale: float):
        def hook(_module, _input_tuple, output):
            return self._scale_structure(output, scale)

        return hook

    def _scale_structure(self, value: Any, scale: float) -> Any:
        if torch.is_tensor(value):
            if value.is_floating_point():
                return value * scale
            return value
        if isinstance(value, tuple):
            return tuple(self._scale_structure(item, scale) for item in value)
        if isinstance(value, list):
            return [self._scale_structure(item, scale) for item in value]
        return value

    def _average_prediction_length(self, predictions: List[Dict[str, Any]]) -> float:
        if not predictions:
            return 0.0
        lengths = [len(item.get("sequence", "")) for item in predictions]
        return float(np.mean(lengths)) if lengths else 0.0


def load_spectra_records(data_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    handler = DataHandler({"path": data_path, "format": "auto", "max_samples": max_samples or -1})
    df = handler.load_data()

    records: List[Dict[str, Any]] = []
    for index, row in df.iterrows():
        records.append(
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
    return records


def build_default_config(model_type: str, checkpoint_path: str) -> Dict[str, Any]:
    base = {
        "model": {
            "name": model_type,
            "checkpoint": checkpoint_path,
        },
        "memnovo": {
            "enabled": False,
        },
        "inference": {
            "beam_size": 5,
            "batch_size": 64,
        },
        "evaluation": {
            "mass_tolerance": 50,
            "normalize_il": True,
        },
    }
    if model_type == "instanovo":
        base["inference"].update(
            {
                "max_length": 30,
                "use_knapsack": True,
                "knapsack_path": "../knapsack_cache/instanovo_knapsack",
                "fp16": True,
            }
        )
    else:
        base["inference"].update({"max_length": 100})
    return base


def run_sensitivity_experiment(
    data_path: str,
    output_path: str,
    model_type: Optional[str] = None,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    scale_factors: Optional[List[float]] = None,
    modality: str = "both",
    device: str = "cuda",
    batch_size: Optional[int] = None,
    beam_size: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    if config_path:
        config = load_config(config_path)
        model_type = model_type or config.get("model", {}).get("name")
    else:
        if model_type is None:
            raise ValueError("model_type is required when config_path is not provided")
        if checkpoint_path is None:
            defaults = {
                "instanovo": "../weights/instanovo-v1.1.0.ckpt",
                "casanovo": "../weights/casanovo_v5_0_0.ckpt",
            }
            checkpoint_path = defaults[model_type]
        config = build_default_config(model_type, checkpoint_path)

    if model_type is None:
        raise ValueError("Unable to infer model type")

    if checkpoint_path:
        config.setdefault("model", {})["checkpoint"] = checkpoint_path
    if batch_size is not None:
        config.setdefault("inference", {})["batch_size"] = batch_size
    if beam_size is not None:
        config.setdefault("inference", {})["beam_size"] = beam_size

    spectra = load_spectra_records(resolve_path(data_path) or data_path, max_samples=max_samples)
    evaluator = Evaluator(
        mass_tolerance=float(config.get("evaluation", {}).get("mass_tolerance", 50)),
        normalize_il=bool(config.get("evaluation", {}).get("normalize_il", True)),
    )

    pipeline = MemNovoModel.from_pretrained(
        model_name=model_type,
        checkpoint_path=config.get("model", {}).get("checkpoint"),
        config=config,
        device=device,
    )
    scaler = SensitivityScaler(pipeline)
    effective_scales = scale_factors or SensitivityScaler.default_scale_factors(model_type)

    results: Dict[str, Any] = {
        "model_type": model_type,
        "config_path": str(config_path) if config_path else None,
        "checkpoint_path": config.get("model", {}).get("checkpoint"),
        "data_path": resolve_path(data_path) or data_path,
        "scale_factors": list(effective_scales),
        "n_samples": len(spectra),
        "spectrum_results": [],
        "peptide_results": [],
    }

    try:
        if modality in ("spectrum", "both"):
            results["spectrum_results"] = scaler.run_experiment(
                spectra=spectra,
                scale_factors=effective_scales,
                modality="spectrum",
                evaluator=evaluator,
                batch_size=config.get("inference", {}).get("batch_size"),
                beam_size=config.get("inference", {}).get("beam_size"),
            )["results"]

        if modality in ("peptide", "both"):
            results["peptide_results"] = scaler.run_experiment(
                spectra=spectra,
                scale_factors=effective_scales,
                modality="peptide",
                evaluator=evaluator,
                batch_size=config.get("inference", {}).get("batch_size"),
                beam_size=config.get("inference", {}).get("beam_size"),
            )["results"]
    finally:
        pipeline.cleanup()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved sensitivity results to %s", output)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sensitivity scaling experiments")
    parser.add_argument("--config", default=None, help="Optional YAML config file")
    parser.add_argument("--model", default=None, choices=["instanovo", "casanovo"], help="Model type")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--data", required=True, help="Input dataset (MGF/parquet/csv)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument(
        "--modality",
        default="both",
        choices=["spectrum", "peptide", "both"],
        help="Which modality to perturb",
    )
    parser.add_argument("--scale-factors", nargs="*", type=float, default=None, help="Custom scale factors")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--beam-size", type=int, default=None, help="Override beam size")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for smoke tests")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    run_sensitivity_experiment(
        data_path=args.data,
        output_path=args.output,
        model_type=args.model,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        scale_factors=args.scale_factors,
        modality=args.modality,
        device=args.device,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

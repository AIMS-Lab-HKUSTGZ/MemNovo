#!/usr/bin/env python3
"""Compare MemNovo against naive skip and cross-attention reweight baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import pandas as pd
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import DataHandler, Evaluator, save_predictions
from memnovo import MemNovoModel
from memnovo.backends import ensure_external_imports, resolve_path
from memnovo.hooks import HookManager
from memnovo.manager import MemNovoManager
from memnovo.utils import load_config, setup_logging

from run_instanovo_official import DEFAULT_RESIDUE_REMAPPING

SPECIES_PATHS = {
    "Apis-mellifera": WORKSPACE_ROOT / "dataset" / "NS3" / "Apis-mellifera.mgf",
    "Saccharomyces-cerevisiae": WORKSPACE_ROOT / "dataset" / "NS1" / "Saccharomyces-cerevisiae.mgf",
    "Vigna-mungo": WORKSPACE_ROOT / "dataset" / "NS3" / "Vigna-mungo.mgf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--casanovo-species", default="Apis-mellifera,Saccharomyces-cerevisiae")
    parser.add_argument("--instanovo-species", default="Saccharomyces-cerevisiae,Vigna-mungo")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--skip-alpha", type=float, default=0.005)
    parser.add_argument("--xattn-gain", type=float, default=1.005)
    parser.add_argument("--xattn-gain-strong", type=float, default=1.01)
    parser.add_argument("--instanovo-batch-size", type=int, default=128)
    parser.add_argument("--instanovo-beam-size", type=int, default=1)
    parser.add_argument("--instanovo-use-knapsack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def ensure_subset(source_path: Path, output_path: Path, size: int, seed: int) -> Path:
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sample_mgf_subset.py"),
        "--input",
        str(source_path),
        "--output",
        str(output_path),
        "--num-spectra",
        str(size),
        "--seed",
        str(seed),
    ]
    import subprocess

    subprocess.run(cmd, check=True, cwd=str(WORKSPACE_ROOT))
    return output_path


def evaluate_df(df: pd.DataFrame) -> dict[str, Any]:
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


def make_skip_manager(alpha: float) -> HookManager:
    manager = HookManager(
        {
            "enabled": True,
            "residual_scale": alpha,
            "apply_to_last_n_layers": 1,
            "confidence_threshold": None,
            "use_softmax": True,
        }
    )

    def _apply_skip(self: HookManager, hidden_state: torch.Tensor, spectral_memory: torch.Tensor, spectral_mask: torch.Tensor | None = None) -> torch.Tensor:
        if spectral_mask is not None:
            if spectral_mask.dtype == torch.bool:
                valid = (~spectral_mask).float().unsqueeze(-1)
            else:
                valid = (spectral_mask > 0).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            pooled = (spectral_memory * valid).sum(dim=1) / denom
        else:
            pooled = spectral_memory.mean(dim=1)
        return hidden_state + self.residual_scale * pooled.unsqueeze(1)

    manager._apply_injection = MethodType(_apply_skip, manager)
    return manager


def find_last_cross_attn(model: Any) -> torch.nn.Module:
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return model.decoder.layers[-1].multihead_attn
    if hasattr(model, "decoder") and hasattr(model.decoder, "transformer_decoder"):
        return model.decoder.transformer_decoder.layers[-1].multihead_attn
    raise ValueError("Could not locate last cross-attention module")


def register_xattn_gain(model: Any, gain: float) -> torch.utils.hooks.RemovableHandle:
    module = find_last_cross_attn(model)

    def hook_fn(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            first = output[0]
            if torch.is_tensor(first):
                return (first * gain, *output[1:])
            return output
        if torch.is_tensor(output):
            return output * gain
        return output

    return module.register_forward_hook(hook_fn)


def run_casanovo_variant(
    variant: str,
    subset_path: Path,
    output_path: Path,
    metrics_path: Path,
    device: str,
    skip_alpha: float,
    xattn_gain: float,
) -> dict[str, Any]:
    config = load_config(str(PROJECT_ROOT / "configs" / "baseline_casanovo.yaml"))
    model = MemNovoModel.from_pretrained(
        model_name="casanovo",
        checkpoint_path=config["model"]["checkpoint"],
        config=config,
        device=device,
    )
    handle = None
    custom_manager = None
    if variant == "memnovo":
        model.memnovo_manager.unregister()
        custom_manager = MemNovoManager({"enabled": True, "residual_scale": 0.005, "apply_to_last_n_layers": 1})
        custom_manager.register(model.model)
    elif variant == "skip":
        custom_manager = make_skip_manager(skip_alpha)
        custom_manager.register_hooks(model.model)
    elif variant.startswith("xattn"):
        gain = xattn_gain
        if variant.endswith("strong"):
            gain = xattn_gain
        handle = register_xattn_gain(model.model, gain)

    try:
        handler = DataHandler({"path": str(subset_path), "format": "auto"})
        spectra = handler.load_data().to_dict("records")
        start = time.perf_counter()
        predictions = model.predict(spectra, batch_size=64, beam_size=5)
        elapsed = time.perf_counter() - start
        save_predictions(predictions, str(output_path), format="jsonl")
        evaluator = Evaluator()
        metrics = evaluator.evaluate(predictions, spectra, model_name="casanovo")
        metrics["elapsed_seconds"] = elapsed
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
    finally:
        if handle is not None:
            handle.remove()
        if custom_manager is not None:
            if isinstance(custom_manager, HookManager):
                custom_manager.remove_hooks()
            else:
                custom_manager.unregister()
        model.cleanup()


def build_predict_config(
    input_path: Path,
    output_path: Path,
    device: str,
    beam_size: int = 1,
    batch_size: int = 128,
    use_knapsack: bool = False,
) -> Any:
    config = load_config(str(PROJECT_ROOT / "configs" / "baseline_instanovo.yaml"))
    inference_cfg = dict(config.get("inference", {}))
    residue_remapping = dict(DEFAULT_RESIDUE_REMAPPING)
    residue_remapping.update(config.get("data", {}).get("residue_remapping", {}))
    residue_remapping.update(inference_cfg.get("residue_remapping", {}))
    return OmegaConf.create(
        {
            "instanovo_model": resolve_path(config["model"]["checkpoint"]),
            "data_path": str(input_path),
            "output_path": str(output_path),
            "denovo": False,
            "device": device,
            "batch_size": batch_size,
            "num_beams": beam_size,
            "max_length": 30,
            "use_knapsack": bool(use_knapsack),
            "knapsack_path": resolve_path(inference_cfg.get("knapsack_path", "../knapsack_cache/instanovo_knapsack")),
            "fp16": True,
            "save_beams": False,
            "log_interval": 20,
            "subset": 1.0,
            "use_basic_logging": True,
            "residue_remapping": residue_remapping,
        }
    )


def run_instanovo_variant(
    variant: str,
    subset_path: Path,
    output_path: Path,
    metrics_path: Path,
    device: str,
    skip_alpha: float,
    xattn_gain: float,
    batch_size: int,
    beam_size: int,
    use_knapsack: bool,
) -> dict[str, Any]:
    ensure_external_imports()
    from instanovo.transformer.model import InstaNovo
    from instanovo.transformer.predict import get_preds
    from instanovo.utils.s3 import S3FileHandler

    predict_cfg = build_predict_config(
        subset_path,
        output_path,
        device,
        beam_size=beam_size,
        batch_size=batch_size,
        use_knapsack=use_knapsack,
    )
    model, model_config = InstaNovo.load(str(predict_cfg.instanovo_model))
    s3 = S3FileHandler()
    handle = None
    custom_manager = None
    if variant == "memnovo":
        custom_manager = MemNovoManager({"enabled": True, "residual_scale": 0.005, "apply_to_last_n_layers": 1})
        custom_manager.register(model)
    elif variant == "skip":
        custom_manager = make_skip_manager(skip_alpha)
        custom_manager.register_hooks(model)
    elif variant.startswith("xattn"):
        handle = register_xattn_gain(model, xattn_gain)

    try:
        start = time.perf_counter()
        get_preds(predict_cfg, model, model_config, s3)
        pred_df = pd.read_csv(output_path)
        elapsed = time.perf_counter() - start
        metrics = evaluate_df(pred_df)
        metrics["elapsed_seconds"] = elapsed
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
    finally:
        if handle is not None:
            handle.remove()
        if custom_manager is not None:
            if isinstance(custom_manager, HookManager):
                custom_manager.remove_hooks()
            else:
                custom_manager.unregister()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    output_dir = Path(args.output_dir).resolve()
    subset_dir = output_dir / "subsets"
    runs_dir = output_dir / "runs"
    subset_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    cas_device = f"cuda:{devices[0]}"
    ins_device = f"cuda:{devices[1] if len(devices) > 1 else devices[0]}"

    summary: dict[str, list[dict[str, Any]]] = {"casanovo": [], "instanovo": []}

    cas_species = [item.strip() for item in args.casanovo_species.split(",") if item.strip()]
    for idx, species in enumerate(cas_species):
        subset = ensure_subset(
            SPECIES_PATHS[species],
            subset_dir / f"{species}_subset{args.subset_size}.mgf",
            args.subset_size,
            args.seed + idx,
        )
        for variant in ("baseline", "memnovo", "skip", "xattn", "xattn_strong"):
            stem = f"casanovo_{species}_{variant}"
            output_jsonl = runs_dir / f"{stem}.jsonl"
            metrics_json = runs_dir / f"{stem}.metrics.json"
            metrics = run_casanovo_variant(
                variant,
                subset,
                output_jsonl,
                metrics_json,
                cas_device,
                args.skip_alpha,
                args.xattn_gain if variant == "xattn" else args.xattn_gain_strong,
            )
            summary["casanovo"].append(
                {
                    "species": species,
                    "variant": variant,
                    "subset": str(subset),
                    "metrics_path": str(metrics_json),
                    "metrics": metrics,
                }
            )

    ins_species = [item.strip() for item in args.instanovo_species.split(",") if item.strip()]
    for idx, species in enumerate(ins_species):
        subset = ensure_subset(
            SPECIES_PATHS[species],
            subset_dir / f"{species}_subset{args.subset_size}.mgf",
            args.subset_size,
            args.seed + 100 + idx,
        )
        for variant in ("baseline", "memnovo", "skip", "xattn", "xattn_strong"):
            stem = f"instanovo_{species}_{variant}"
            output_csv = runs_dir / f"{stem}.csv"
            metrics_json = runs_dir / f"{stem}.metrics.json"
            metrics = run_instanovo_variant(
                variant,
                subset,
                output_csv,
                metrics_json,
                ins_device,
                args.skip_alpha,
                args.xattn_gain if variant == "xattn" else args.xattn_gain_strong,
                args.instanovo_batch_size,
                args.instanovo_beam_size,
                args.instanovo_use_knapsack,
            )
            summary["instanovo"].append(
                {
                    "species": species,
                    "variant": variant,
                    "subset": str(subset),
                    "metrics_path": str(metrics_json),
                    "metrics": metrics,
                }
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# WSMy Q1 Naive Alternative Comparison",
        "",
        "| Model | Species | Variant | AA Prec. | AA Recall | Pep. Prec. | Pep. Recall |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for model_name in ("casanovo", "instanovo"):
        for row in summary[model_name]:
            metrics = row["metrics"]
            lines.append(
                f"| {model_name} | {row['species']} | {row['variant']} | "
                f"{metrics.get('aa_precision', 0.0):.6f} | {metrics.get('aa_recall', 0.0):.6f} | "
                f"{metrics.get('pep_precision', 0.0):.6f} | {metrics.get('pep_recall', 0.0):.6f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

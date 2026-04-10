"""
Runtime helpers for loading official Casanovo and InstaNovo backends.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
EXTERNAL_ROOT = PROJECT_ROOT / "external"


def ensure_external_imports() -> None:
    """Expose vendored official backends on sys.path."""
    for path in (EXTERNAL_ROOT, EXTERNAL_ROOT / "casanovo", EXTERNAL_ROOT / "instanovo", EXTERNAL_ROOT / "primenovo"):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def resolve_path(path: str | os.PathLike[str] | None) -> str | None:
    """Resolve a path relative to the repo or workspace root."""
    if path is None:
        return None

    raw = Path(path)
    if raw.is_absolute():
        return str(raw)

    project_path = (PROJECT_ROOT / raw).resolve()
    if project_path.exists():
        return str(project_path)

    workspace_path = (WORKSPACE_ROOT / raw).resolve()
    return str(workspace_path)


def load_instanovo_backend(
    checkpoint_path: str,
    device: str,
    inference_config: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Load the official InstaNovo model and decoder."""
    ensure_external_imports()

    import torch
    from instanovo.constants import MASS_SCALE, MAX_MASS
    from instanovo.inference import BeamSearchDecoder, Knapsack, KnapsackBeamSearchDecoder
    from instanovo.transformer.model import InstaNovo

    resolved_checkpoint = resolve_path(checkpoint_path)
    if resolved_checkpoint is None or not Path(resolved_checkpoint).exists():
        raise FileNotFoundError(f"InstaNovo checkpoint not found: {checkpoint_path}")

    model, model_config = InstaNovo.load(resolved_checkpoint)
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(torch_device)
    model.eval()

    beam_size = int(inference_config.get("beam_size", 5))
    use_knapsack = bool(inference_config.get("use_knapsack", True))
    knapsack_path = resolve_path(inference_config.get("knapsack_path", "knapsack_cache/instanovo_knapsack"))

    if use_knapsack:
        if knapsack_path is not None and Path(knapsack_path).exists():
            knapsack = Knapsack.from_file(path=knapsack_path)
        else:
            residue_masses = dict(model.residue_set.residue_masses.copy())
            negative_residues = [k for k, v in residue_masses.items() if v < 0]
            if negative_residues:
                residue_masses.update(dict.fromkeys(negative_residues, MAX_MASS))

            for special_residue in list(model.residue_set.residue_to_index.keys())[:3]:
                residue_masses[special_residue] = 0

            knapsack = Knapsack.construct_knapsack(
                residue_masses=residue_masses,
                residue_indices=model.residue_set.residue_to_index,
                max_mass=MAX_MASS,
                mass_scale=MASS_SCALE,
            )

            if knapsack_path is not None:
                knapsack_dir = Path(knapsack_path).parent
                knapsack_dir.mkdir(parents=True, exist_ok=True)
                try:
                    knapsack.save(knapsack_path)
                except FileExistsError:
                    pass

        decoder = KnapsackBeamSearchDecoder(model=model, knapsack=knapsack)
    else:
        decoder = BeamSearchDecoder(model=model)

    return model, {
        "device": torch_device,
        "decoder": decoder,
        "beam_size": beam_size,
        "model_config": model_config,
        "residue_set": model.residue_set,
    }


def load_casanovo_backend(
    checkpoint_path: str,
    device: str,
    inference_config: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Load the official Casanovo model."""
    ensure_external_imports()

    import torch
    from casanovo.denovo.model import Spec2Pep

    resolved_checkpoint = resolve_path(checkpoint_path)
    if resolved_checkpoint is None or not Path(resolved_checkpoint).exists():
        raise FileNotFoundError(f"Casanovo checkpoint not found: {checkpoint_path}")

    model = Spec2Pep.load_from_checkpoint(resolved_checkpoint)
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(torch_device)
    model.eval()

    beam_size = int(inference_config.get("beam_size", 5))
    model.n_beams = beam_size
    model.top_match = beam_size

    max_charge = getattr(model, "max_charge", 5)
    if hasattr(model, "decoder") and hasattr(model.decoder, "charge_encoder"):
        try:
            max_charge = int(model.decoder.charge_encoder.num_embeddings)
        except Exception:
            pass

    return model, {
        "device": torch_device,
        "beam_size": beam_size,
        "max_charge": max_charge,
    }


def load_primenovo_backend(
    checkpoint_path: str,
    device: str,
    config: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Load the vendored PrimeNovo model."""
    ensure_external_imports()

    import torch
    from primenovo.denovo.model import Spec2Pep

    resolved_checkpoint = resolve_path(checkpoint_path)
    if resolved_checkpoint is None or not Path(resolved_checkpoint).exists():
        raise FileNotFoundError(f"PrimeNovo checkpoint not found: {checkpoint_path}")

    model_cfg = dict(config.get("model", {}))
    primenovo_cfg = dict(config.get("primenovo", {}))
    inference_cfg = dict(config.get("inference", {}))

    base_config_path = primenovo_cfg.get("base_config", "external/primenovo/config.yaml")
    resolved_base_config = resolve_path(base_config_path)
    if resolved_base_config is None or not Path(resolved_base_config).exists():
        raise FileNotFoundError(f"PrimeNovo base config not found: {base_config_path}")

    with open(resolved_base_config, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    # Keep the official PrimeNovo hyperparameters unless explicitly overridden.
    merged = dict(base_config)
    merged.update(primenovo_cfg)
    merged["load_file_name"] = resolved_checkpoint
    if "beam_size" in inference_cfg:
        merged["n_beams"] = int(inference_cfg["beam_size"])
    if "batch_size" in inference_cfg:
        merged["predict_batch_size"] = int(inference_cfg["batch_size"])

    load_kwargs = {
        "PMC_enable": merged["PMC_enable"],
        "mass_control_tol": merged["mass_control_tol"],
        "dim_model": merged["dim_model"],
        "n_head": merged["n_head"],
        "dim_feedforward": merged["dim_feedforward"],
        "n_layers": merged["n_layers"],
        "dropout": merged["dropout"],
        "dim_intensity": merged["dim_intensity"],
        "custom_encoder": merged.get("custom_encoder"),
        "max_length": merged["max_length"],
        "residues": merged["residues"],
        "max_charge": merged["max_charge"],
        "precursor_mass_tol": merged["precursor_mass_tol"],
        "isotope_error_range": tuple(merged["isotope_error_range"]),
        "n_beams": merged["n_beams"],
        "n_log": merged["n_log"],
        "out_writer": None,
    }

    model = Spec2Pep.load_from_checkpoint(resolved_checkpoint, **load_kwargs)
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(torch_device)
    model.eval()

    return model, {
        "device": torch_device,
        "beam_size": int(merged["n_beams"]),
        "base_config": merged,
    }

#!/usr/bin/env python3
"""Run activation-only Softmax vs ReLU MemNovo ablations on selected species subsets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

SPECIES_PATHS = {
    "Apis-mellifera": WORKSPACE_ROOT / "dataset" / "NS3" / "Apis-mellifera.mgf",
    "Saccharomyces-cerevisiae": WORKSPACE_ROOT / "dataset" / "NS1" / "Saccharomyces-cerevisiae.mgf",
    "Vigna-mungo": WORKSPACE_ROOT / "dataset" / "NS3" / "Vigna-mungo.mgf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--casanovo-species", default="Apis-mellifera,Saccharomyces-cerevisiae")
    parser.add_argument("--instanovo-species", default="Vigna-mungo,Saccharomyces-cerevisiae")
    parser.add_argument("--devices", default="0,1,2,3")
    return parser.parse_args()


def ensure_subset(source_path: Path, output_path: Path, size: int, seed: int) -> Path:
    if output_path.exists():
        return output_path
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
    subprocess.run(cmd, check=True, cwd=str(WORKSPACE_ROOT))
    return output_path


def run_cmd(cmd: list[str], log_path: Path, env: dict[str, str]) -> float:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(WORKSPACE_ROOT))
    return time.perf_counter() - start


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    subset_dir = output_dir / "subsets"
    runs_dir = output_dir / "runs"
    subset_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    cas_device = devices[0]
    ins_device = devices[1] if len(devices) > 1 else devices[0]

    summary = {"casanovo": [], "instanovo": []}

    cas_species = [item.strip() for item in args.casanovo_species.split(",") if item.strip()]
    for idx, species in enumerate(cas_species):
        subset = ensure_subset(SPECIES_PATHS[species], subset_dir / f"{species}_subset{args.subset_size}.mgf", args.subset_size, args.seed + idx)
        for variant, config_name in [
            ("baseline", "baseline_casanovo.yaml"),
            ("softmax", "memnovo_casanovo.yaml"),
            ("relu", "memnovo_casanovo_relu.yaml"),
        ]:
            stem = f"casanovo_{species}_{variant}"
            output_jsonl = runs_dir / f"{stem}.jsonl"
            metrics_json = runs_dir / f"{stem}.metrics.json"
            log_path = runs_dir / f"{stem}.log"
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_inference.py"),
                "--config",
                str(PROJECT_ROOT / "configs" / config_name),
                "--input",
                str(subset),
                "--output",
                str(output_jsonl),
                "--metrics-output",
                str(metrics_json),
                "--device",
                f"cuda:{cas_device}",
                "--evaluate",
                "--log-level",
                "WARNING",
            ]
            env = dict(os.environ)
            if metrics_json.exists():
                elapsed = 0.0
            else:
                elapsed = run_cmd(cmd, log_path, env)
            summary["casanovo"].append(
                {
                    "species": species,
                    "variant": variant,
                    "subset_path": str(subset),
                    "subset_size": args.subset_size,
                    "elapsed_seconds": elapsed,
                    "metrics": load_json(metrics_json),
                    "metrics_path": str(metrics_json),
                    "log_path": str(log_path),
                }
            )

    ins_species = [item.strip() for item in args.instanovo_species.split(",") if item.strip()]
    for idx, species in enumerate(ins_species):
        subset = ensure_subset(SPECIES_PATHS[species], subset_dir / f"{species}_subset{args.subset_size}.mgf", args.subset_size, args.seed + 100 + idx)
        for variant, config_name in [
            ("baseline", "baseline_instanovo.yaml"),
            ("softmax", "memnovo_instanovo.yaml"),
            ("relu", "memnovo_instanovo_relu.yaml"),
        ]:
            stem = f"instanovo_{species}_{variant}"
            output_csv = runs_dir / f"{stem}.csv"
            metrics_json = runs_dir / f"{stem}.metrics.json"
            log_path = runs_dir / f"{stem}.log"
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_instanovo_official.py"),
                "--config",
                str(PROJECT_ROOT / "configs" / config_name),
                "--input",
                str(subset),
                "--output",
                str(output_csv),
                "--metrics-output",
                str(metrics_json),
                "--device",
                f"cuda:{ins_device}",
                "--batch-size",
                "64",
                "--beam-size",
                "5",
                "--use-knapsack",
                "--log-level",
                "WARNING",
            ]
            env = dict(os.environ)
            if metrics_json.exists():
                elapsed = 0.0
            else:
                elapsed = run_cmd(cmd, log_path, env)
            runtime_json = output_csv.with_suffix(".runtime.json")
            summary["instanovo"].append(
                {
                    "species": species,
                    "variant": variant,
                    "subset_path": str(subset),
                    "subset_size": args.subset_size,
                    "elapsed_seconds": elapsed,
                    "metrics": load_json(metrics_json),
                    "runtime": load_json(runtime_json) if runtime_json.exists() else None,
                    "metrics_path": str(metrics_json),
                    "runtime_path": str(runtime_json),
                    "log_path": str(log_path),
                }
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

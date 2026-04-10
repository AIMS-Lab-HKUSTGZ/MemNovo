#!/usr/bin/env python3
"""Run sharded inference for one species across multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import aggregate_metrics
from memnovo.backends import resolve_path
from scripts.split_mgf import split_mgf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one species across multiple GPUs by sharding the input MGF")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--input", required=True, help="Input MGF file")
    parser.add_argument("--output", required=True, help="Final merged JSONL output path")
    parser.add_argument("--metrics-output", required=True, help="Final merged metrics JSON path")
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--device", default="cuda", help="Device argument passed to run_inference.py")
    parser.add_argument("--spectra-per-shard", type=int, default=50000, help="Number of spectra per shard")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for each shard job")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation for each shard and aggregate metrics")
    parser.add_argument("--force-redo", action="store_true", help="Re-run completed shard jobs and overwrite merged outputs")
    parser.add_argument("--log-dir", default=None, help="Directory for shard logs")
    return parser.parse_args()


def load_manifest(shard_dir: Path, input_path: Path, spectra_per_shard: int, force_redo: bool) -> dict[str, Any]:
    if force_redo and shard_dir.exists():
        # Re-split from scratch only when explicitly requested.
        for path in sorted(shard_dir.glob("*")):
            if path.is_file():
                path.unlink()

    manifest = split_mgf(input_path, shard_dir, spectra_per_shard)
    return manifest


def launch_shard(
    gpu_id: str,
    shard_path: Path,
    config_path: Path,
    shard_output: Path,
    shard_metrics: Path,
    log_path: Path,
    device: str,
    evaluate: bool,
    max_samples: int | None,
) -> tuple[subprocess.Popen[str], Any]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_inference.py"),
        "--config",
        str(config_path),
        "--input",
        str(shard_path),
        "--output",
        str(shard_output),
        "--metrics-output",
        str(shard_metrics),
        "--device",
        device,
    ]
    if evaluate:
        cmd.append("--evaluate")
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PROJECT_ROOT.parent),
        text=True,
    )
    return process, log_handle


def merge_jsonl(shard_outputs: list[Path], final_output: Path) -> None:
    final_output.parent.mkdir(parents=True, exist_ok=True)
    with final_output.open("w", encoding="utf-8") as destination:
        for shard_output in shard_outputs:
            if not shard_output.exists():
                raise FileNotFoundError(f"Missing shard output: {shard_output}")
            with shard_output.open("r", encoding="utf-8") as source:
                for line in source:
                    destination.write(line)


def main() -> None:
    args = parse_args()
    input_path = Path(resolve_path(args.input) or args.input).resolve()
    config_path = Path(resolve_path(args.config) or args.config).resolve()
    final_output = Path(args.output).resolve()
    final_metrics = Path(args.metrics_output).resolve()
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("Expected at least one GPU id")

    if not args.force_redo and final_output.exists() and final_metrics.exists():
        print(f"skip: existing outputs found for {final_output.name}")
        return

    species_name = input_path.stem
    shard_root = final_output.parent / "_shards" / species_name
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_dir = shard_root / "mgf"
    log_dir = Path(args.log_dir).resolve() if args.log_dir else shard_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    shard_output_dir = shard_root / "predictions"
    shard_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(shard_dir, input_path, args.spectra_per_shard, args.force_redo)
    shard_items = manifest["shards"]

    pending: list[dict[str, Any]] = []
    for index, shard in enumerate(shard_items):
        shard_path = Path(str(shard["path"])).resolve()
        shard_output = shard_output_dir / f"{species_name}.shard_{index:04d}.jsonl"
        shard_metrics = shard_output_dir / f"{species_name}.shard_{index:04d}.metrics.json"
        log_path = log_dir / f"{species_name}.shard_{index:04d}.log"

        if not args.force_redo and shard_output.exists() and (not args.evaluate or shard_metrics.exists()):
            continue

        pending.append(
            {
                "index": index,
                "shard_path": shard_path,
                "shard_output": shard_output,
                "shard_metrics": shard_metrics,
                "log_path": log_path,
            }
        )

    running: dict[int, dict[str, Any]] = {}
    free_gpus = list(gpu_ids)

    while pending or running:
        while pending and free_gpus:
            gpu_id = free_gpus.pop(0)
            task = pending.pop(0)
            process, log_handle = launch_shard(
                gpu_id=gpu_id,
                shard_path=task["shard_path"],
                config_path=config_path,
                shard_output=task["shard_output"],
                shard_metrics=task["shard_metrics"],
                log_path=task["log_path"],
                device=args.device,
                evaluate=args.evaluate,
                max_samples=args.max_samples,
            )
            task["gpu_id"] = gpu_id
            task["process"] = process
            task["log_handle"] = log_handle
            task["start_time"] = time.time()
            running[process.pid] = task
            print(f"launched shard {task['index']:04d} on GPU {gpu_id}")

        if not running:
            continue

        time.sleep(2)
        finished: list[int] = []
        for pid, task in running.items():
            process = task["process"]
            return_code = process.poll()
            if return_code is None:
                continue

            task["log_handle"].close()
            free_gpus.append(task["gpu_id"])
            finished.append(pid)
            duration = time.time() - task["start_time"]

            if return_code != 0:
                raise RuntimeError(
                    f"Shard {task['index']:04d} failed on GPU {task['gpu_id']} "
                    f"after {duration:.1f}s. See {task['log_path']}"
                )

            print(f"finished shard {task['index']:04d} on GPU {task['gpu_id']} in {duration:.1f}s")

        for pid in finished:
            running.pop(pid, None)

    ordered_outputs = [
        shard_output_dir / f"{species_name}.shard_{index:04d}.jsonl"
        for index in range(len(shard_items))
    ]
    merge_jsonl(ordered_outputs, final_output)
    print(f"merged JSONL -> {final_output}")

    if args.evaluate:
        metric_dicts = []
        for index in range(len(shard_items)):
            shard_metrics = shard_output_dir / f"{species_name}.shard_{index:04d}.metrics.json"
            metric_dicts.append(json.loads(shard_metrics.read_text(encoding="utf-8")))
        aggregated = aggregate_metrics(metric_dicts)
        final_metrics.parent.mkdir(parents=True, exist_ok=True)
        final_metrics.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
        print(f"aggregated metrics -> {final_metrics}")


if __name__ == "__main__":
    main()

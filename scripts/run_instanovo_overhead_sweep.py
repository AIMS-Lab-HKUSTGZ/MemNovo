#!/usr/bin/env python3
"""Run an extended InstaNovo overhead sweep across batch sizes and beam sizes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--batch-sizes", default="32,64,128,256")
    parser.add_argument("--beam-sizes", default="1,5,10")
    parser.add_argument("--light-subset-size", type=int, default=4999)
    parser.add_argument("--heavy-subset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260407)
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
    subprocess.run(cmd, check=True)
    return output_path


def load_runtime(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = output_dir / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    source = PROJECT_ROOT / "results" / "weighted_subset_150k" / "nine_species_weighted_4999_seed42_official.mgf"
    if not source.exists():
        raise FileNotFoundError(source)

    light_subset = ensure_subset(source, subset_dir / f"weighted_{args.light_subset_size}.mgf", args.light_subset_size, args.seed)
    heavy_subset = ensure_subset(source, subset_dir / f"weighted_{args.heavy_subset_size}.mgf", args.heavy_subset_size, args.seed + 1)

    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    beam_sizes = [int(item) for item in args.beam_sizes.split(",") if item.strip()]
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]

    tasks: list[dict] = []
    for beam_size in beam_sizes:
        subset_path = light_subset if beam_size == 1 else heavy_subset
        for batch_size in batch_sizes:
            for variant, config_name in [
                ("baseline", "baseline_instanovo.yaml"),
                ("memnovo", "memnovo_instanovo.yaml"),
            ]:
                tasks.append(
                    {
                        "variant": variant,
                        "config": str(PROJECT_ROOT / "configs" / config_name),
                        "batch_size": batch_size,
                        "beam_size": beam_size,
                        "subset_path": str(subset_path),
                        "use_knapsack": beam_size > 1,
                    }
                )

    def launch(task: dict, gpu: str):
        stem = f"{task['variant']}_b{task['batch_size']}_k{task['beam_size']}"
        output_csv = runs_dir / f"{stem}.csv"
        runtime_json = output_csv.with_suffix(".runtime.json")
        metrics_json = output_csv.with_suffix(".metrics.json")
        log_path = runs_dir / f"{stem}.log"
        if runtime_json.exists():
            return {
                "proc": None,
                "gpu": gpu,
                "task": task,
                "output_csv": output_csv,
                "runtime_json": runtime_json,
                "metrics_json": metrics_json,
                "log_path": log_path,
                "precomputed": True,
            }
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_instanovo_official.py"),
            "--config",
            task["config"],
            "--input",
            str(task["subset_path"]),
            "--output",
            str(output_csv),
            "--device",
            f"cuda:{gpu}",
            "--batch-size",
            str(task["batch_size"]),
            "--beam-size",
            str(task["beam_size"]),
            "--log-level",
            "WARNING",
            "--metrics-output",
            str(metrics_json),
            "--no-save-beams",
        ]
        if task["use_knapsack"]:
            cmd.append("--use-knapsack")
        else:
            cmd.append("--no-use-knapsack")
        env = dict(os.environ)
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT.parent))
        return {
            "proc": proc,
            "gpu": gpu,
            "task": task,
            "output_csv": output_csv,
            "runtime_json": runtime_json,
            "metrics_json": metrics_json,
            "log_path": log_path,
            "precomputed": False,
        }

    running = []
    pending = list(tasks)
    results = []
    available_devices = list(devices)

    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as _:
        while pending or running:
            while pending and available_devices:
                gpu = available_devices.pop(0)
                task = pending.pop(0)
                launched = launch(task, gpu)
                if launched["precomputed"]:
                    record = {
                        **task,
                        "gpu": gpu,
                        "returncode": 0,
                        "log_path": str(launched["log_path"]),
                        "output_csv": str(launched["output_csv"]),
                        "runtime_json": str(launched["runtime_json"]),
                        "metrics_json": str(launched["metrics_json"]),
                        "runtime": load_runtime(launched["runtime_json"]),
                    }
                    results.append(record)
                    available_devices.append(gpu)
                else:
                    running.append(launched)

            if not running:
                break

            time.sleep(2.0)
            still_running = []
            for item in running:
                ret = item["proc"].poll()
                if ret is None:
                    still_running.append(item)
                    continue
                available_devices.append(item["gpu"])
                record = {
                    **item["task"],
                    "gpu": item["gpu"],
                    "returncode": ret,
                    "log_path": str(item["log_path"]),
                    "output_csv": str(item["output_csv"]),
                    "runtime_json": str(item["runtime_json"]),
                    "metrics_json": str(item["metrics_json"]),
                }
                if ret == 0 and item["runtime_json"].exists():
                    runtime = load_runtime(item["runtime_json"])
                    record["runtime"] = runtime
                results.append(record)
            running = still_running

    summary = {
        "source_subset": str(source),
        "light_subset": str(light_subset),
        "heavy_subset": str(heavy_subset),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

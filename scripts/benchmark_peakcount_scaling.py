#!/usr/bin/env python3
"""Synthetic CUDA benchmark for MemNovo retrieval scaling with peak count."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memnovo.layers import CrossAttentionRetrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--peak-counts", default="200,500,1000,2000,5000,10000")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--use-softmax", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark is intended for CUDA devices")

    dtype = parse_dtype(args.dtype)
    peak_counts = [int(item) for item in args.peak_counts.split(",") if item.strip()]
    effective_batch = args.batch_size * args.beam_size

    results: list[dict[str, float | int | str | bool]] = []
    layer = CrossAttentionRetrieval(
        dim_model=args.dim,
        residual_scale=0.005,
        use_softmax=bool(args.use_softmax),
    ).to(device)
    layer.eval()

    for n_peaks in peak_counts:
        hidden = torch.randn(effective_batch, args.seq_len, args.dim, device=device, dtype=dtype)
        memory = torch.randn(effective_batch, n_peaks, args.dim, device=device, dtype=dtype)
        mask = torch.ones(effective_batch, n_peaks, device=device, dtype=torch.bool)

        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = layer(hidden, memory, mask)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for _ in range(args.iters):
                _ = layer(hidden, memory, mask)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

        per_forward_ms = elapsed * 1000.0 / args.iters
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        tensor_bytes = (
            hidden.numel() * bytes_per_elem
            + memory.numel() * bytes_per_elem
            + mask.numel() * mask.element_size()
        )

        results.append(
            {
                "peak_count": n_peaks,
                "effective_batch": effective_batch,
                "batch_size": args.batch_size,
                "beam_size": args.beam_size,
                "seq_len": args.seq_len,
                "dim": args.dim,
                "dtype": args.dtype,
                "use_softmax": bool(args.use_softmax),
                "mean_forward_ms": per_forward_ms,
                "peak_allocated_gb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
                "peak_reserved_gb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 3)),
                "input_tensor_gb": float(tensor_bytes / (1024 ** 3)),
            }
        )

    payload = {
        "device": args.device,
        "dtype": args.dtype,
        "use_softmax": bool(args.use_softmax),
        "iters": args.iters,
        "warmup": args.warmup,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

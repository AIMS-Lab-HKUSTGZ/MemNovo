#!/usr/bin/env python3
"""
Sample a reproducible random subset of spectra from an MGF file.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample spectra from an MGF file")
    parser.add_argument("--input", "-i", required=True, help="Input MGF file")
    parser.add_argument("--output", "-o", required=True, help="Output MGF file")
    parser.add_argument("--num-spectra", "-n", type=int, required=True, help="Number of spectra to sample")
    parser.add_argument("--seed", type=int, default=20260405, help="Random seed")
    return parser.parse_args()


def load_blocks(path: Path) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    inside = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("BEGIN IONS"):
                current = [line]
                inside = True
                continue

            if inside:
                current.append(line)
                if line.startswith("END IONS"):
                    blocks.append(current)
                    current = []
                    inside = False

    return blocks


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks(input_path)
    if not blocks:
        raise ValueError(f"No spectra found in {input_path}")

    n = min(args.num_spectra, len(blocks))
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(range(len(blocks)), n))

    with output_path.open("w", encoding="utf-8") as handle:
        for idx in chosen:
            handle.writelines(blocks[idx])
            if not blocks[idx][-1].endswith("\n"):
                handle.write("\n")

    print(
        f"Sampled {n} / {len(blocks)} spectra from {input_path} to {output_path} with seed {args.seed}"
    )


if __name__ == "__main__":
    main()

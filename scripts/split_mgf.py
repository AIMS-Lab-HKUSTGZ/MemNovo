#!/usr/bin/env python3
"""Split an MGF file into contiguous spectrum shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split an MGF file into contiguous shards")
    parser.add_argument("--input", required=True, help="Input MGF file")
    parser.add_argument("--output-dir", required=True, help="Directory to store shard files")
    parser.add_argument(
        "--spectra-per-shard",
        type=int,
        default=50000,
        help="Maximum number of spectra in each shard",
    )
    parser.add_argument("--force-redo", action="store_true", help="Rebuild shards even if manifest matches")
    return parser.parse_args()


def build_manifest(input_path: Path, spectra_per_shard: int, shards: list[dict[str, object]]) -> dict[str, object]:
    stat = input_path.stat()
    return {
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "spectra_per_shard": spectra_per_shard,
        "num_shards": len(shards),
        "total_spectra": sum(int(item["num_spectra"]) for item in shards),
        "shards": shards,
    }


def manifest_matches(manifest_path: Path, input_path: Path, spectra_per_shard: int) -> bool:
    if not manifest_path.exists():
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    stat = input_path.stat()
    return (
        manifest.get("input_path") == str(input_path.resolve())
        and int(manifest.get("input_size", -1)) == stat.st_size
        and int(manifest.get("input_mtime_ns", -1)) == stat.st_mtime_ns
        and int(manifest.get("spectra_per_shard", -1)) == spectra_per_shard
    )


def split_mgf(input_path: Path, output_dir: Path, spectra_per_shard: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []

    shard_index = -1
    spectra_in_shard = 0
    total_spectra = 0
    handle = None
    current_path: Path | None = None

    def open_next_shard() -> None:
        nonlocal shard_index, spectra_in_shard, handle, current_path
        if handle is not None:
            handle.close()
        shard_index += 1
        spectra_in_shard = 0
        current_path = output_dir / f"shard_{shard_index:04d}.mgf"
        handle = current_path.open("w", encoding="utf-8")
        shards.append({"path": str(current_path), "num_spectra": 0})

    open_next_shard()

    with input_path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            if line.startswith("BEGIN IONS") and spectra_in_shard >= spectra_per_shard:
                open_next_shard()

            handle.write(line)

            if line.startswith("END IONS"):
                spectra_in_shard += 1
                total_spectra += 1
                shards[-1]["num_spectra"] = spectra_in_shard

    if handle is not None:
        handle.close()

    manifest = build_manifest(input_path, spectra_per_shard, shards)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "manifest.json"

    if not args.force_redo and manifest_matches(manifest_path, input_path, args.spectra_per_shard):
        print(manifest_path.read_text(encoding="utf-8"))
        return

    manifest = split_mgf(input_path, output_dir, args.spectra_per_shard)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

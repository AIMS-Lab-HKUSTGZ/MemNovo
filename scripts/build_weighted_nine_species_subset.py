#!/usr/bin/env python3
"""Build a fixed weighted nine-species MGF subset for quick validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SPECIES = [
    {
        "name": "Bacillus-subtilis",
        "path": Path("/opt/data/private/instanovo/dataset/NS2/Bacillus-subtilis.mgf"),
        "paper_count": 1_355_019,
    },
    {
        "name": "Saccharomyces-cerevisiae",
        "path": Path("/opt/data/private/instanovo/dataset/NS1/Saccharomyces-cerevisiae.mgf"),
        "paper_count": 583_801,
    },
    {
        "name": "Methanosarcina-mazei",
        "path": Path("/opt/data/private/instanovo/dataset/NS1/Methanosarcina-mazei.mgf"),
        "paper_count": 266_983,
    },
    {
        "name": "Apis-mellifera",
        "path": Path("/opt/data/private/instanovo/dataset/NS3/Apis-mellifera.mgf"),
        "paper_count": 193_805,
    },
    {
        "name": "Solanum-lycopersicum",
        "path": Path("/opt/data/private/instanovo/dataset/NS3/Solanum-lycopersicum.mgf"),
        "paper_count": 176_403,
    },
    {
        "name": "Vigna-mungo",
        "path": Path("/opt/data/private/instanovo/dataset/NS3/Vigna-mungo.mgf"),
        "paper_count": 108_266,
    },
    {
        "name": "Candidatus-endoloripes",
        "path": Path("/opt/data/private/instanovo/dataset/NS3/Candidatus-endoloripes.mgf"),
        "paper_count": 81_626,
    },
    {
        "name": "H.-sapiens",
        "path": Path("/opt/data/private/instanovo/dataset/NS3/H.-sapiens.mgf"),
        "paper_count": 44_286,
    },
    {
        "name": "Mus-musculus",
        "path": Path("/opt/data/private/instanovo/dataset/NS3/Mus-musculus.mgf"),
        "paper_count": 25_175,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weighted nine-species MGF subset")
    parser.add_argument("--output-dir", required=True, help="Output directory for subset files")
    parser.add_argument("--total-spectra", type=int, default=150000, help="Total spectra across all species")
    parser.add_argument("--seed", type=int, default=20260405, help="Base random seed")
    parser.add_argument("--force-redo", action="store_true", help="Rebuild subset even if manifest matches")
    return parser.parse_args()


def allocate_samples(total_spectra: int) -> dict[str, int]:
    total_weight = sum(int(item["paper_count"]) for item in SPECIES)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    used = 0

    for item in SPECIES:
        name = str(item["name"])
        raw = total_spectra * int(item["paper_count"]) / total_weight
        count = int(raw)
        allocations[name] = count
        used += count
        remainders.append((raw - count, name))

    remainders.sort(reverse=True)
    for _, name in remainders[: total_spectra - used]:
        allocations[name] += 1
    return allocations


def manifest_matches(manifest_path: Path, total_spectra: int, seed: int) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        int(manifest.get("total_requested_spectra", -1)) == total_spectra
        and int(manifest.get("seed", -1)) == seed
    )


def sample_mgf(input_path: Path, output_path: Path, sample_size: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    reservoir: list[tuple[int, str]] = []
    current_block: list[str] = []
    inside_block = False
    n_seen = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            if line.startswith("BEGIN IONS"):
                current_block = [line]
                inside_block = True
                continue

            if not inside_block:
                continue

            current_block.append(line)
            if line.startswith("END IONS"):
                block_text = "".join(current_block)
                if len(reservoir) < sample_size:
                    reservoir.append((n_seen, block_text))
                else:
                    replacement_index = rng.randint(0, n_seen)
                    if replacement_index < sample_size:
                        reservoir[replacement_index] = (n_seen, block_text)
                n_seen += 1
                current_block = []
                inside_block = False

    reservoir.sort(key=lambda item: item[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        for _, block_text in reservoir:
            sink.write(block_text)

    return {"encountered_spectra": n_seen, "written_spectra": len(reservoir)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    subset_dir = output_dir / "species"
    subset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    if not args.force_redo and manifest_matches(manifest_path, args.total_spectra, args.seed):
        print(manifest_path.read_text(encoding="utf-8"))
        return

    allocations = allocate_samples(args.total_spectra)
    manifest: dict[str, object] = {
        "seed": args.seed,
        "total_requested_spectra": args.total_spectra,
        "allocation_basis": "paper_nine_species_counts",
        "species": [],
    }

    merged_path = output_dir / f"nine_species_weighted_{args.total_spectra}.mgf"
    with merged_path.open("w", encoding="utf-8") as merged_sink:
        for index, item in enumerate(SPECIES):
            name = str(item["name"])
            input_path = Path(item["path"])
            output_path = subset_dir / f"{name}.mgf"
            requested = allocations[name]
            stats = sample_mgf(
                input_path=input_path,
                output_path=output_path,
                sample_size=requested,
                seed=args.seed + index,
            )
            with output_path.open("r", encoding="utf-8") as source:
                for line in source:
                    merged_sink.write(line)

            manifest["species"].append(
                {
                    "name": name,
                    "source_path": str(input_path),
                    "subset_path": str(output_path),
                    "paper_count": int(item["paper_count"]),
                    "requested_spectra": requested,
                    "encountered_spectra": int(stats["encountered_spectra"]),
                    "written_spectra": int(stats["written_spectra"]),
                    "seed": args.seed + index,
                }
            )
            print(
                f"[subset] {name}: requested={requested} "
                f"encountered={stats['encountered_spectra']} written={stats['written_spectra']}"
            )

    manifest["merged_path"] = str(merged_path)
    manifest["merged_written_spectra"] = sum(
        int(item["written_spectra"]) for item in manifest["species"]  # type: ignore[index]
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

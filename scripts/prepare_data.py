"""Create train/val/test split indices grouped by compound identity."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing spectrum JSON files")
    parser.add_argument("--output", type=Path, default=Path("data/splits.json"), help="Output split JSON path")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group_by",
        choices=["auto", "parent", "metadata", "parent_metadata"],
        default="parent_metadata",
        help=(
            "Split by parent folder, metadata identity, or metadata identity within each parent folder. "
            "auto uses parent_metadata when subfolders exist."
        ),
    )
    parser.add_argument("--spectrum_length", type=int, default=460, help="Required spectrum.y length")
    return parser.parse_args()


def metadata_identity(payload: dict[str, Any], sample_id: str) -> str:
    """Return the best available chemical identity from one JSON payload."""

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return str(
        metadata.get("cas_number")
        or metadata.get("compound_name")
        or payload.get("object_name")
        or sample_id
    )


def source_identity(path: Path, data_dir: Path) -> str:
    """Return the source folder identity for one JSON path."""

    parent = path.parent.relative_to(data_dir).as_posix()
    return parent if parent != "." else "__root__"


def read_sample(
    path: Path,
    data_dir: Path,
    group_by: str,
    spectrum_length: int,
) -> dict[str, str] | None:
    """Read split metadata from one valid spectrum JSON file."""

    if path.name.startswith("_"):
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    spectrum = payload.get("spectrum") if isinstance(payload.get("spectrum"), dict) else {}
    y_values = spectrum.get("y")
    if not isinstance(y_values, list) or len(y_values) != spectrum_length:
        return None

    sample_id = str(payload.get("id") or path.stem)
    source_id = source_identity(path, data_dir)
    compound_id = metadata_identity(payload, sample_id)

    if group_by == "parent":
        group_id = source_id
        stratum_id = "__all__"
    elif group_by == "parent_metadata":
        group_id = f"{source_id}::{compound_id}"
        stratum_id = source_id
    else:
        group_id = compound_id
        stratum_id = "__all__"

    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "source_id": source_id,
        "compound_id": compound_id,
        "stratum_id": stratum_id,
    }


def detect_grouping_mode(data_dir: Path, paths: list[Path], requested: str) -> str:
    """Return the grouping mode, preferring per-source compound splits when available."""

    if requested != "auto":
        return requested
    return "parent_metadata" if any(path.parent != data_dir for path in paths) else "metadata"


def assign_splits(
    group_sizes: dict[str, int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign group ids to splits while balancing spectrum counts."""

    total = sum(group_sizes.values())
    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * test_ratio,
    }
    split_sizes = {"train": 0, "val": 0, "test": 0}
    assignment: dict[str, str] = {}
    ordered = list(group_sizes.items())
    random.Random(seed).shuffle(ordered)
    ordered = sorted(ordered, key=lambda item: -item[1])

    for group_id, size in ordered:
        split = min(
            ("train", "val", "test"),
            key=lambda name: (split_sizes[name] + size - targets[name]) / max(targets[name], 1.0),
        )
        assignment[group_id] = split
        split_sizes[split] += size
    return assignment


def assign_stratified_splits(
    groups_by_stratum: dict[str, dict[str, int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign groups to splits independently inside each source stratum."""

    assignment: dict[str, str] = {}
    for offset, stratum_id in enumerate(sorted(groups_by_stratum)):
        stratum_assignment = assign_splits(
            groups_by_stratum[stratum_id],
            train_ratio,
            val_ratio,
            test_ratio,
            seed + offset,
        )
        assignment.update(stratum_assignment)
    return assignment


def main() -> None:
    """Create grouped split index JSON."""

    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    paths = sorted(args.data_dir.rglob("*.json"))
    if not paths:
        raise ValueError(f"No JSON files found under {args.data_dir}")

    group_by = detect_grouping_mode(args.data_dir, paths, args.group_by)
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    group_to_stratum: dict[str, str] = {}
    skipped = 0
    for path in tqdm(paths, desc="scan"):
        sample = read_sample(path, args.data_dir, group_by, args.spectrum_length)
        if sample is None:
            skipped += 1
            continue
        group_id = sample["group_id"]
        group_to_stratum[group_id] = sample["stratum_id"]
        by_group[group_id].append(
            {
                "path": str(path.relative_to(args.data_dir)),
                "id": sample["sample_id"],
                "group_id": group_id,
                "source_id": sample["source_id"],
                "compound_id": sample["compound_id"],
            }
        )

    if not by_group:
        raise ValueError("No valid spectrum JSON files were found")

    group_sizes = {group_id: len(items) for group_id, items in by_group.items()}
    groups_by_stratum: dict[str, dict[str, int]] = defaultdict(dict)
    for group_id, size in group_sizes.items():
        groups_by_stratum[group_to_stratum[group_id]][group_id] = size
    assignment = assign_stratified_splits(
        groups_by_stratum,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for group_id, items in by_group.items():
        splits[assignment[group_id]].extend(items)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2, ensure_ascii=True)

    counts = {name: len(items) for name, items in splits.items()}
    group_counts = {
        name: len({item["group_id"] for item in items})
        for name, items in splits.items()
    }
    source_counts = {
        name: len({item["source_id"] for item in items})
        for name, items in splits.items()
    }
    print(json.dumps({"spectra": counts, "groups": group_counts}, indent=2))
    print(json.dumps({"sources": source_counts, "group_by": group_by, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()

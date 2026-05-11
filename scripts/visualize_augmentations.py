"""Visualize original spectra against independently augmented views."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augmentations import SpectrumAugmentor
from src.data.dataset import IRSpectrumDataset
from src.utils.config import apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split_index", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=Path, default=Path("outputs/augmentation_examples.png"))
    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=int, default=None, help="Visualize one fixed dataset index")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override dotted config key")
    return parser.parse_args()


def build_dataset(config: dict[str, Any], args: argparse.Namespace) -> IRSpectrumDataset:
    """Build the configured dataset split."""

    data_cfg = config.get("data", {})
    data_dir = args.data_dir or Path(str(data_cfg.get("data_dir", "data/raw")))
    split_index = args.split_index or Path(str(data_cfg.get("split_index", "data/splits.json")))
    return IRSpectrumDataset(
        data_dir=data_dir,
        split_index=split_index if split_index.exists() else None,
        split=args.split if split_index.exists() else None,
        spectrum_length=int(data_cfg.get("spectrum_length", config.get("model", {}).get("spectrum_length", 460))),
        cache=False,
    )


def choose_indices(dataset_size: int, num_examples: int, seed: int, fixed_index: int | None) -> list[int]:
    """Choose dataset indices to visualize."""

    if fixed_index is not None:
        if fixed_index < 0 or fixed_index >= dataset_size:
            raise IndexError(f"index must be in [0, {dataset_size - 1}], got {fixed_index}")
        return [fixed_index]
    rng = random.Random(seed)
    return rng.sample(range(dataset_size), k=min(num_examples, dataset_size))


def main() -> None:
    """Create and save augmentation comparison plots."""

    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    dataset = build_dataset(config, args)
    augmentor = SpectrumAugmentor.from_config(config.get("augmentation", {}))
    indices = choose_indices(len(dataset), args.num_examples, args.seed, args.index)

    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 2.6 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        sample = dataset[index]
        x = sample["x"].numpy()
        original = sample["spectrum"]
        view1 = augmentor(original)
        view2 = augmentor(original)
        ax = axes[row, 0]
        ax.plot(x, original.numpy(), color="black", linewidth=1.2, label="original")
        ax.plot(x, view1.numpy(), color="#d62728", linewidth=0.9, alpha=0.8, label="aug view 1")
        ax.plot(x, view2.numpy(), color="#1f77b4", linewidth=0.9, alpha=0.8, label="aug view 2")
        ax.set_title(f"{sample['id']} | group={sample['group_id']}")
        ax.set_xlabel("Wavenumber (cm^-1)")
        ax.set_ylabel("Absorbance")
        ax.grid(alpha=0.25)
        if row == 0:
            ax.legend(loc="upper right")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote augmentation comparison to {args.output}")


if __name__ == "__main__":
    main()


"""Evaluate a Lightning checkpoint with Trainer.test()."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lightning_datamodule import IRSpectrumDataModule
from src.data.augmentations import SpectrumAugmentor
from src.training.lightning_module import IRContrastiveModule
from src.utils.config import apply_named_overrides, apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split_index", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override dotted config key")
    return parser.parse_args()


def main() -> None:
    """Run Lightning test evaluation."""

    args = parse_args()
    model = IRContrastiveModule.load_from_checkpoint(args.checkpoint)
    config = load_config(args.config) if args.config else model.config
    config = apply_named_overrides(config, vars(args))
    config = apply_overrides(config, args.overrides)
    if args.batch_size is not None:
        config.setdefault("evaluation", {})["batch_size"] = args.batch_size
    model.config = config
    model.eval_augmentor = SpectrumAugmentor.from_config(config.get("augmentation", {}))
    datamodule = IRSpectrumDataModule(config)
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

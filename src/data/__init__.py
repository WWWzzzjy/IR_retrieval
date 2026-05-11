"""Data loading, augmentation, and collation utilities."""

from typing import Any

from src.data.augmentations import SpectrumAugmentor
from src.data.collate import ContrastiveCollator, eval_collate
from src.data.dataset import IRSpectrumDataset

__all__ = [
    "ContrastiveCollator",
    "IRSpectrumDataModule",
    "IRSpectrumDataset",
    "SpectrumAugmentor",
    "eval_collate",
]


def __getattr__(name: str) -> Any:
    """Lazily import Lightning-only data utilities."""

    if name == "IRSpectrumDataModule":
        from src.data.lightning_datamodule import IRSpectrumDataModule

        return IRSpectrumDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

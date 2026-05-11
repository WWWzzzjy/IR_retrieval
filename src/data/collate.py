"""Collate functions for training and evaluation dataloaders."""

from __future__ import annotations

from typing import Any

import torch

from src.data.augmentations import SpectrumAugmentor


class ContrastiveCollator:
    """Create two independently augmented views for each spectrum sample.

    Args:
        augmentor: Callable spectrum augmentor.
    """

    def __init__(self, augmentor: SpectrumAugmentor | None = None) -> None:
        self.augmentor = augmentor or SpectrumAugmentor({})

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate raw samples into contrastive training tensors."""

        spectra = torch.stack([item["spectrum"] for item in batch], dim=0)
        view1 = torch.stack([self.augmentor(item["spectrum"]) for item in batch], dim=0)
        view2 = torch.stack([self.augmentor(item["spectrum"]) for item in batch], dim=0)
        x_values = torch.stack([item["x"] for item in batch], dim=0)
        return {
            "view1": view1,
            "view2": view2,
            "original": spectra,
            "x": x_values,
            "ids": [item["id"] for item in batch],
            "group_ids": [item["group_id"] for item in batch],
            "paths": [item["path"] for item in batch],
        }


def eval_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate evaluation samples without stochastic augmentations."""

    return {
        "spectrum": torch.stack([item["spectrum"] for item in batch], dim=0),
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "ids": [item["id"] for item in batch],
        "group_ids": [item["group_id"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


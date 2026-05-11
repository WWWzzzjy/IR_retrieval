"""PyTorch Lightning DataModule for mid-IR spectrum data."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.data.augmentations import SpectrumAugmentor
from src.data.collate import ContrastiveCollator, eval_collate
from src.data.dataset import IRSpectrumDataset


class IRSpectrumDataModule(pl.LightningDataModule):
    """Create train/val/test dataloaders from JSON spectra and split indices.

    Args:
        config: Full experiment configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        data_cfg = config.get("data", {})
        self.data_dir = Path(str(data_cfg.get("data_dir", "data/raw")))
        self.split_index = Path(str(data_cfg.get("split_index", "data/splits.json")))
        self.spectrum_length = int(data_cfg.get("spectrum_length", config.get("model", {}).get("spectrum_length", 460)))
        self.cache = bool(data_cfg.get("cache", False))
        model_cfg = config.get("model", {})
        patch_size = int(model_cfg.get("patch_size", 10))
        stride = int(model_cfg.get("stride", patch_size))
        self.num_patches = (self.spectrum_length - patch_size) // stride + 1
        self.mask_ratio = float(config.get("loss", {}).get("reconstruction", {}).get("mask_ratio", 0.15))
        self.train_dataset: Optional[IRSpectrumDataset] = None
        self.val_dataset: Optional[IRSpectrumDataset] = None
        self.test_dataset: Optional[IRSpectrumDataset] = None
        self.augmentor = SpectrumAugmentor.from_config(config.get("augmentation", {}))
        self.train_collator = ContrastiveCollator(self.augmentor)

    def prepare_data(self) -> None:
        """Generate the grouped split index when raw data exists and no index is present."""

        if self.split_index.exists():
            return
        paths = sorted(self.data_dir.rglob("*.json"))
        if not paths:
            return
        splits = self._build_grouped_splits(paths)
        self.split_index.parent.mkdir(parents=True, exist_ok=True)
        with self.split_index.open("w", encoding="utf-8") as handle:
            json.dump(splits, handle, indent=2, ensure_ascii=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load datasets for the requested Lightning stage."""

        if not self.split_index.exists():
            raise FileNotFoundError(
                f"Split index not found: {self.split_index}. Run scripts/prepare_data.py or place JSON files in {self.data_dir}."
            )

        if stage in (None, "fit"):
            self.train_dataset = IRSpectrumDataset(self.data_dir, self.split_index, "train", self.spectrum_length, self.cache)
            self.val_dataset = IRSpectrumDataset(self.data_dir, self.split_index, "val", self.spectrum_length, self.cache)
        if stage in (None, "validate"):
            self.val_dataset = IRSpectrumDataset(self.data_dir, self.split_index, "val", self.spectrum_length, self.cache)
        if stage in (None, "test"):
            self.test_dataset = IRSpectrumDataset(self.data_dir, self.split_index, "test", self.spectrum_length, self.cache)
        if stage in (None, "predict") and self.test_dataset is None:
            self.test_dataset = IRSpectrumDataset(self.data_dir, self.split_index, "test", self.spectrum_length, self.cache)

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the contrastive training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must be called before train_dataloader")
        train_cfg = self.config.get("train", {})
        return DataLoader(
            self.train_dataset,
            batch_size=int(train_cfg.get("batch_size", 256)),
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            persistent_workers=bool(train_cfg.get("persistent_workers", False)) and int(train_cfg.get("num_workers", 4)) > 0,
            drop_last=bool(train_cfg.get("drop_last", True)),
            collate_fn=self._train_collate,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the validation dataloader."""

        if self.val_dataset is None:
            raise RuntimeError("setup('fit') or setup('validate') must be called before val_dataloader")
        return self._eval_loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the test dataloader."""

        if self.test_dataset is None:
            raise RuntimeError("setup('test') must be called before test_dataloader")
        return self._eval_loader(self.test_dataset)

    def _eval_loader(self, dataset: IRSpectrumDataset) -> DataLoader[dict[str, Any]]:
        """Build a deterministic evaluation dataloader."""

        eval_cfg = self.config.get("evaluation", {})
        train_cfg = self.config.get("train", {})
        return DataLoader(
            dataset,
            batch_size=int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 256))),
            shuffle=False,
            num_workers=int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 4))),
            pin_memory=bool(eval_cfg.get("pin_memory", train_cfg.get("pin_memory", True))),
            persistent_workers=bool(eval_cfg.get("persistent_workers", False))
            and int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 4))) > 0,
            collate_fn=eval_collate,
        )

    def _train_collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate contrastive views and attach a reconstruction patch mask."""

        collated = self.train_collator(batch)
        mask = torch.zeros(collated["view1"].shape[0] * 2, self.num_patches, dtype=torch.bool)
        if self.mask_ratio > 0.0:
            mask_count = max(1, min(self.num_patches, int(round(self.num_patches * self.mask_ratio))))
            scores = torch.rand(mask.shape)
            indices = scores.argsort(dim=1)[:, :mask_count]
            mask.scatter_(1, indices, True)
        collated["patch_mask"] = mask
        return collated

    def _build_grouped_splits(self, paths: list[Path]) -> dict[str, list[dict[str, str]]]:
        """Scan JSON metadata and create grouped train/val/test split entries."""

        data_cfg = self.config.get("data", {})
        train_ratio = float(data_cfg.get("train_ratio", 0.8))
        val_ratio = float(data_cfg.get("val_ratio", 0.1))
        test_ratio = float(data_cfg.get("test_ratio", 0.1))
        group_by = str(data_cfg.get("group_by", "parent_metadata"))
        group_by = self._detect_grouping_mode(paths, group_by)
        seed = int(self.config.get("train", {}).get("seed", 42))
        by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        group_to_stratum: dict[str, str] = {}
        for path in paths:
            sample = self._read_sample(path, group_by)
            if sample is None:
                continue
            group_id = sample["group_id"]
            group_to_stratum[group_id] = sample["stratum_id"]
            by_group[group_id].append(
                {
                    "path": str(path.relative_to(self.data_dir)),
                    "id": sample["sample_id"],
                    "group_id": group_id,
                    "source_id": sample["source_id"],
                    "compound_id": sample["compound_id"],
                }
            )

        if not by_group:
            raise ValueError(f"No valid spectrum JSON files found under {self.data_dir}")

        splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
        groups_by_stratum: dict[str, dict[str, int]] = defaultdict(dict)
        for group_id, items in by_group.items():
            groups_by_stratum[group_to_stratum[group_id]][group_id] = len(items)

        assignment: dict[str, str] = {}
        for offset, stratum_id in enumerate(sorted(groups_by_stratum)):
            assignment.update(
                self._assign_splits(
                    groups_by_stratum[stratum_id],
                    train_ratio,
                    val_ratio,
                    test_ratio,
                    seed + offset,
                )
            )

        for group_id, items in by_group.items():
            split = assignment[group_id]
            splits[split].extend(by_group[group_id])
        return splits

    def _detect_grouping_mode(self, paths: list[Path], requested: str) -> str:
        """Return per-source compound grouping when available in auto mode."""

        if requested != "auto":
            return requested
        return "parent_metadata" if any(path.parent != self.data_dir for path in paths) else "metadata"

    def _read_sample(self, path: Path, group_by: str) -> dict[str, str] | None:
        """Read split metadata from one valid spectrum JSON file."""

        if path.name.startswith("_"):
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
        spectrum = payload.get("spectrum") if isinstance(payload.get("spectrum"), dict) else {}
        y_values = spectrum.get("y")
        if not isinstance(y_values, list) or len(y_values) != self.spectrum_length:
            return None
        sample_id = str(payload.get("id") or path.stem)
        source_id = self._source_identity(path)
        compound_id = self._metadata_identity(payload, sample_id)

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

    def _source_identity(self, path: Path) -> str:
        """Return the source folder identity for one JSON path."""

        parent = path.parent.relative_to(self.data_dir).as_posix()
        return parent if parent != "." else "__root__"

    @staticmethod
    def _metadata_identity(payload: dict[str, Any], sample_id: str) -> str:
        """Return the best available chemical identity from one JSON payload."""

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return str(
            metadata.get("cas_number")
            or metadata.get("compound_name")
            or payload.get("object_name")
            or sample_id
        )

    @staticmethod
    def _assign_splits(
        group_sizes: dict[str, int],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> dict[str, str]:
        """Assign groups to splits while balancing spectrum counts."""

        total = sum(group_sizes.values())
        targets = {
            "train": total * train_ratio,
            "val": total * val_ratio,
            "test": total * test_ratio,
        }
        split_sizes = {"train": 0, "val": 0, "test": 0}
        assignment: dict[str, str] = {}
        group_items = list(group_sizes.items())
        random.Random(seed).shuffle(group_items)
        group_items = sorted(group_items, key=lambda item: -item[1])
        for group_id, size in group_items:
            split = min(
                ("train", "val", "test"),
                key=lambda name: (split_sizes[name] + size - targets[name]) / max(targets[name], 1.0),
            )
            assignment[group_id] = split
            split_sizes[split] += size
        return assignment

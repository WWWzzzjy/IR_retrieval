"""Tests for JSON spectrum loading and collation."""

from __future__ import annotations

import json
from pathlib import Path

from src.data import ContrastiveCollator, IRSpectrumDataset, SpectrumAugmentor


def write_sample(path: Path, sample_id: str, cas_number: str, length: int = 20) -> None:
    """Write one synthetic spectrum JSON file."""

    payload = {
        "id": sample_id,
        "object_name": f"compound-{cas_number}",
        "metadata": {"compound_name": f"compound-{cas_number}", "cas_number": cas_number},
        "spectrum": {
            "point_count": length,
            "y": [float(index) / float(length) for index in range(length)],
            "x": [400.0 + float(index) for index in range(length)],
            "x_unit": "cm^-1",
            "y_axis_type": "absorbance",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dataset_loads_split_and_collates_views(tmp_path: Path) -> None:
    """Dataset should read split entries and collate two augmented views."""

    write_sample(tmp_path / "a.json", "a", "11-11-1")
    write_sample(tmp_path / "b.json", "b", "22-22-2")
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "train": [
                    {"path": "a.json", "group_id": "11-11-1"},
                    {"path": "b.json", "group_id": "22-22-2"},
                ],
                "val": [],
                "test": [],
            }
        ),
        encoding="utf-8",
    )

    dataset = IRSpectrumDataset(tmp_path, split_path, "train", spectrum_length=20)
    assert len(dataset) == 2
    assert dataset[0]["spectrum"].shape == (20,)
    assert dataset[0]["group_id"] == "11-11-1"

    collator = ContrastiveCollator(SpectrumAugmentor({"clamp": True}))
    batch = collator([dataset[0], dataset[1]])
    assert batch["view1"].shape == (2, 20)
    assert batch["view2"].shape == (2, 20)
    assert batch["x"].shape == (2, 20)


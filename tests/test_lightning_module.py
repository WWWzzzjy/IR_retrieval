"""Smoke tests for the Lightning training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl

from src.data.lightning_datamodule import IRSpectrumDataModule
from src.training.lightning_module import IRContrastiveModule


def write_sample(path: Path, sample_id: str, group_id: str, length: int = 40) -> None:
    """Write one synthetic spectrum JSON file."""

    payload = {
        "id": sample_id,
        "object_name": f"compound-{group_id}",
        "metadata": {"compound_name": f"compound-{group_id}", "cas_number": group_id},
        "spectrum": {
            "point_count": length,
            "y": [float((index + int(sample_id[-1])) % length) / float(length) for index in range(length)],
            "x": [400.0 + index * 10.0 for index in range(length)],
            "x_unit": "cm^-1",
            "y_axis_type": "absorbance",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_config(tmp_path: Path) -> dict:
    """Build a tiny Lightning config for fast_dev_run."""

    return {
        "data": {
            "data_dir": str(tmp_path / "raw"),
            "split_index": str(tmp_path / "splits.json"),
            "spectrum_length": 40,
            "cache": False,
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
        },
        "model": {
            "spectrum_length": 40,
            "patch_size": 10,
            "stride": 10,
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_dim": 64,
            "dropout": 0.0,
            "embedding_dim": 16,
            "projection_hidden_dim": None,
            "pooling": "attention",
            "pos_encoding": "wavenumber_sinusoidal",
            "wavenumber_min": 400.0,
            "wavenumber_max": 790.0,
            "position_scale": 100.0,
            "normalize_embeddings": True,
        },
        "loss": {
            "temperature": 0.1,
            "alpha": 1.0,
            "beta": 0.3,
            "reconstruction": {
                "mask_ratio": 0.25,
                "fingerprint_weighting": True,
                "fingerprint_threshold": 600.0,
                "fingerprint_weight": 1.5,
                "default_weight": 1.0,
            },
        },
        "augmentation": {"clamp": True},
        "train": {
            "seed": 123,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": True,
            "num_epochs": 1,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "min_lr_ratio": 0.0,
            "precision": "32-true",
            "grad_accum": 1,
            "grad_clip": 0.0,
            "val_every_n_epoch": 1,
            "log_every_n_steps": 1,
            "deterministic": False,
            "output_dir": str(tmp_path / "outputs"),
            "resume_from": None,
        },
        "evaluation": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "top_k": [1, 5, 10],
            "max_items": None,
        },
        "wandb": {"enabled": False},
        "callbacks": {"early_stopping": {"enabled": False}, "checkpoint": {"save_top_k": 1, "save_last": True}},
    }


def test_lightning_fast_dev_run(tmp_path: Path) -> None:
    """Lightning Trainer should run one fit smoke step end to end."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for index in range(8):
        write_sample(raw_dir / f"sample_{index}.json", f"sample_{index}", f"group-{index // 2}")

    config = build_config(tmp_path)
    datamodule = IRSpectrumDataModule(config)
    model = IRContrastiveModule(config)
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=datamodule)


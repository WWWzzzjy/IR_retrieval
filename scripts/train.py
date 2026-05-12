"""Train the mid-IR encoder with PyTorch Lightning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lightning_datamodule import IRSpectrumDataModule
from src.training.lightning_module import IRContrastiveModule
from src.training.utils import make_run_name
from src.utils.config import apply_named_overrides, apply_overrides, load_config


def parse_int_or_float(value: str) -> int | float:
    """Parse Lightning batch-limit arguments as int counts or float fractions."""

    return float(value) if "." in value or "e" in value.lower() else int(value)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split_index", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--fast-dev-run", action="store_true", help="Run one train/val/test batch for debugging")
    parser.add_argument("--overfit-batches", type=parse_int_or_float, default=0.0, help="Lightning overfit_batches value")
    parser.add_argument(
        "--limit-train-batches",
        type=parse_int_or_float,
        default=1.0,
        help="Lightning limit_train_batches value",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override dotted config key")
    return parser.parse_args()


def is_nullish(value: Any) -> bool:
    """Return whether a config value should be treated as unset."""

    return value is None or str(value).strip().lower() in {"", "none", "null"}


def resolve_run_name(config: dict[str, Any]) -> str:
    """Resolve the run name used by loggers and checkpoint directories."""

    train_cfg = config.setdefault("train", {})
    explicit_name = train_cfg.get("run_name")
    if not is_nullish(explicit_name):
        return str(explicit_name)

    resume_from = train_cfg.get("resume_from")
    if not is_nullish(resume_from):
        checkpoint_parent = Path(str(resume_from)).expanduser().parent
        output_dir = Path(str(train_cfg.get("output_dir", "checkpoints"))).expanduser()
        if checkpoint_parent != Path(".") and checkpoint_parent.resolve() != output_dir.resolve():
            return checkpoint_parent.name

    return make_run_name(config)


def checkpoint_dir(config: dict[str, Any]) -> Path:
    """Return the per-run checkpoint directory."""

    train_cfg = config.get("train", {})
    output_dir = Path(str(train_cfg.get("output_dir", "checkpoints")))
    run_name = str(train_cfg.get("run_name") or resolve_run_name(config))
    return output_dir / run_name


def build_logger(config: dict[str, Any]) -> CSVLogger | WandbLogger | list[CSVLogger | WandbLogger]:
    """Build Lightning loggers from config.

    A CSV logger is always enabled so every run leaves a local metrics.csv file
    for later analysis, even when WandB is also active.
    """

    train_cfg = config.get("train", {})
    output_dir = str(train_cfg.get("output_dir", "checkpoints"))
    run_name = str(train_cfg.get("run_name") or resolve_run_name(config))
    wandb_cfg = config.get("wandb", {})
    csv_logger = CSVLogger(save_dir=output_dir, name="csv_logs", version=run_name)
    if bool(wandb_cfg.get("enabled", False)):
        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "ir-encoder"),
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags", []),
            name=run_name,
            save_dir=output_dir,
            offline=wandb_cfg.get("mode", "online") == "offline",
            log_model=bool(wandb_cfg.get("log_model", False)),
        )
        return [wandb_logger, csv_logger]
    return csv_logger


def build_callbacks(config: dict[str, Any]) -> list[Any]:
    """Build Lightning callbacks for checkpointing, LR logging, and early stopping."""

    train_cfg = config.get("train", {})
    callback_cfg = config.get("callbacks", {})
    checkpoint_cfg = callback_cfg.get("checkpoint", {})
    early_cfg = callback_cfg.get("early_stopping", {})
    ckpt_dir = checkpoint_dir(config)

    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/recall_at_1",
            mode="max",
            save_top_k=int(checkpoint_cfg.get("save_top_k", 3)),
            save_last=bool(checkpoint_cfg.get("save_last", True)),
            filename="best-{epoch:02d}-{val_recall_at_1:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if bool(early_cfg.get("enabled", True)):
        callbacks.append(
            EarlyStopping(
                monitor="val/recall_at_1",
                mode="max",
                patience=int(early_cfg.get("patience", 20)),
            )
        )
    return callbacks


def main() -> None:
    """Run Lightning training."""

    args = parse_args()
    config = load_config(args.config)
    config = apply_named_overrides(config, vars(args))
    config = apply_overrides(config, args.overrides)
    train_cfg = config.get("train", {})
    train_cfg["run_name"] = resolve_run_name(config)
    pl.seed_everything(int(train_cfg.get("seed", 42)), workers=True)
    print(f"Run name: {train_cfg['run_name']}")
    print(f"Checkpoint dir: {checkpoint_dir(config)}")

    datamodule = IRSpectrumDataModule(config)
    model = IRContrastiveModule(config)
    trainer = pl.Trainer(
        max_epochs=int(train_cfg.get("num_epochs", 100)),
        accelerator="auto",
        devices="auto",
        precision=train_cfg.get("precision", "16-mixed"),
        accumulate_grad_batches=int(train_cfg.get("grad_accum", 1)),
        check_val_every_n_epoch=int(train_cfg.get("val_every_n_epoch", 1)),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
        logger=build_logger(config),
        callbacks=build_callbacks(config),
        deterministic=bool(train_cfg.get("deterministic", False)),
        gradient_clip_val=float(train_cfg.get("grad_clip", 0.0)),
        default_root_dir=str(checkpoint_dir(config)),
        fast_dev_run=bool(args.fast_dev_run),
        overfit_batches=args.overfit_batches,
        limit_train_batches=args.limit_train_batches,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=train_cfg.get("resume_from"))


if __name__ == "__main__":
    main()

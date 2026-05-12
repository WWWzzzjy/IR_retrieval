"""Evaluate a checkpoint and export retrieval error analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augmentations import SpectrumAugmentor
from src.data.collate import eval_collate
from src.data.dataset import IRSpectrumDataset
from src.data.lightning_datamodule import IRSpectrumDataModule
from src.losses import semi_hard_negative_margin_loss
from src.training.lightning_module import IRContrastiveModule
from src.training.utils import get_device
from src.utils.config import apply_named_overrides, apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split_index", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--top_k", type=int, nargs="+", default=None)
    parser.add_argument("--error_top_k", type=int, default=10)
    parser.add_argument("--num_error_cases", type=int, default=200)
    parser.add_argument("--skip_trainer_test", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override dotted config key")
    return parser.parse_args()


def build_eval_loader(config: dict[str, Any], split: str, batch_size: int | None) -> DataLoader[dict[str, Any]]:
    """Build a deterministic loader for one split."""

    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    dataset = IRSpectrumDataset(
        data_cfg.get("data_dir", "data/raw"),
        data_cfg.get("split_index", "data/splits.json"),
        split,
        spectrum_length=int(data_cfg.get("spectrum_length", config.get("model", {}).get("spectrum_length", 460))),
        cache=bool(data_cfg.get("cache", False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or int(eval_cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        pin_memory=bool(eval_cfg.get("pin_memory", True)),
        persistent_workers=bool(eval_cfg.get("persistent_workers", False)) and int(eval_cfg.get("num_workers", 4)) > 0,
        collate_fn=eval_collate,
    )


def limit_batch(batch: dict[str, Any], keep: int) -> dict[str, Any]:
    """Keep the first ``keep`` items of an evaluation batch."""

    limited = dict(batch)
    limited["spectrum"] = batch["spectrum"][:keep]
    limited["x"] = batch["x"][:keep]
    limited["ids"] = batch["ids"][:keep]
    limited["group_ids"] = batch["group_ids"][:keep]
    limited["paths"] = batch["paths"][:keep]
    return limited


@torch.no_grad()
def collect_raw_and_augmented_embeddings(
    model: IRContrastiveModule,
    loader: DataLoader[dict[str, Any]],
    augmentor: SpectrumAugmentor,
    device: torch.device,
    max_items: int | None,
) -> dict[str, Any]:
    """Encode raw library spectra and one augmented query per sample."""

    model.eval()
    raw_embeddings: list[torch.Tensor] = []
    query_embeddings: list[torch.Tensor] = []
    ids: list[str] = []
    group_ids: list[str] = []
    paths: list[str] = []
    seen = 0

    for batch in loader:
        if max_items is not None and seen >= max_items:
            break
        if max_items is not None and seen + batch["spectrum"].shape[0] > max_items:
            batch = limit_batch(batch, max_items - seen)

        spectra = batch["spectrum"]
        wavenumbers = batch.get("x")
        spectra_device = spectra.to(device)
        wavenumbers_device = wavenumbers.to(device) if wavenumbers is not None else None
        augmented = torch.stack([augmentor(spectrum) for spectrum in spectra.detach().cpu()], dim=0).to(device)

        raw_embeddings.append(model(spectra_device, wavenumbers=wavenumbers_device).detach().cpu())
        query_embeddings.append(model(augmented, wavenumbers=wavenumbers_device).detach().cpu())
        ids.extend(str(item) for item in batch["ids"])
        group_ids.extend(str(item) for item in batch["group_ids"])
        paths.extend(str(item) for item in batch["paths"])
        seen += spectra.shape[0]

    if not raw_embeddings:
        raise ValueError("No evaluation samples were encoded")
    return {
        "raw_embeddings": torch.cat(raw_embeddings, dim=0),
        "query_embeddings": torch.cat(query_embeddings, dim=0),
        "ids": ids,
        "group_ids": group_ids,
        "paths": paths,
    }


def compute_self_retrieval(similarity: torch.Tensor, top_k: list[int], top_count: int) -> dict[str, Any]:
    """Compute self-augmentation retrieval metrics and ranking tensors."""

    num_items = similarity.shape[0]
    target = torch.arange(num_items)
    positive = similarity.diag()
    eye = torch.eye(num_items, dtype=torch.bool)
    negatives = similarity.masked_select(~eye)
    ranks = (similarity > positive.unsqueeze(1)).sum(dim=1) + 1

    metrics: dict[str, float] = {
        "num_items": float(num_items),
        "same_cosine_mean": float(positive.mean().item()),
        "different_cosine_mean": float(negatives.mean().item()) if negatives.numel() else 0.0,
    }
    metrics["cosine_gap"] = metrics["same_cosine_mean"] - metrics["different_cosine_mean"]
    for k in top_k:
        hits = ranks <= int(k)
        metrics[f"recall_at_{k}"] = float(hits.float().mean().item())

    top_values, top_indices = similarity.topk(k=min(top_count, num_items), dim=1)
    return {
        "metrics": metrics,
        "positive": positive,
        "ranks": ranks,
        "top_values": top_values,
        "top_indices": top_indices,
        "target": target,
    }


def add_loss_metrics(metrics: dict[str, float], similarity: torch.Tensor, config: dict[str, Any]) -> None:
    """Add full-set retrieval CE and margin loss to the metrics mapping."""

    loss_cfg = config.get("loss", {})
    margin_cfg = loss_cfg.get("hard_negative", {})
    temperature = float(loss_cfg.get("temperature", 0.1))
    alpha = float(loss_cfg.get("alpha", 1.0))
    margin_enabled = bool(margin_cfg.get("enabled", True))
    margin_weight = float(margin_cfg.get("weight", 0.1))
    margin = float(margin_cfg.get("margin", 0.2))
    labels = torch.arange(similarity.shape[0])
    logits = similarity / temperature
    retrieval_ce = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    margin_loss, _positive, _negative, _retrieval_margin = semi_hard_negative_margin_loss(similarity, margin=margin)
    if not margin_enabled or margin_weight <= 0.0:
        margin_loss = similarity.sum() * 0.0
    metrics["loss_retrieval_ce"] = float(retrieval_ce.item())
    metrics["loss_margin"] = float(margin_loss.item())
    metrics["loss_total"] = float((alpha * retrieval_ce + margin_weight * margin_loss).item())


def source_from_path(path: str) -> str:
    """Return the source folder name for a spectrum path."""

    parent = Path(path).parent.name
    return parent or ""


def write_error_files(
    analysis: dict[str, Any],
    metadata: dict[str, list[str]],
    output_dir: Path,
    error_top_k: int,
    num_error_cases: int,
) -> tuple[Path, Path]:
    """Write ranked error cases and their top-k candidates."""

    ids = metadata["ids"]
    group_ids = metadata["group_ids"]
    paths = metadata["paths"]
    positive = analysis["positive"]
    ranks = analysis["ranks"]
    top_indices = analysis["top_indices"]
    top_values = analysis["top_values"]
    target = analysis["target"]
    misses = torch.nonzero(top_indices[:, 0] != target, as_tuple=False).flatten()
    if misses.numel():
        wrong_margin = top_values[misses, 0] - positive[misses]
        ordered_misses = misses[torch.argsort(wrong_margin, descending=True)]
    else:
        ordered_misses = misses
    if num_error_cases > 0:
        ordered_misses = ordered_misses[:num_error_cases]

    errors_path = output_dir / "errors.csv"
    topk_path = output_dir / "error_topk.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_index",
                "query_id",
                "query_group_id",
                "query_source",
                "query_path",
                "true_rank",
                "positive_cosine",
                "top1_index",
                "top1_id",
                "top1_group_id",
                "top1_source",
                "top1_path",
                "top1_cosine",
                "top1_minus_positive",
                "top1_same_group",
            ],
        )
        writer.writeheader()
        for query_index in ordered_misses.tolist():
            top1_index = int(top_indices[query_index, 0].item())
            top1_score = float(top_values[query_index, 0].item())
            positive_score = float(positive[query_index].item())
            writer.writerow(
                {
                    "query_index": query_index,
                    "query_id": ids[query_index],
                    "query_group_id": group_ids[query_index],
                    "query_source": source_from_path(paths[query_index]),
                    "query_path": paths[query_index],
                    "true_rank": int(ranks[query_index].item()),
                    "positive_cosine": positive_score,
                    "top1_index": top1_index,
                    "top1_id": ids[top1_index],
                    "top1_group_id": group_ids[top1_index],
                    "top1_source": source_from_path(paths[top1_index]),
                    "top1_path": paths[top1_index],
                    "top1_cosine": top1_score,
                    "top1_minus_positive": top1_score - positive_score,
                    "top1_same_group": group_ids[top1_index] == group_ids[query_index],
                }
            )

    with topk_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_index",
                "query_id",
                "true_rank",
                "positive_cosine",
                "candidate_rank",
                "candidate_index",
                "candidate_id",
                "candidate_group_id",
                "candidate_source",
                "candidate_path",
                "candidate_cosine",
                "is_target",
                "is_same_group",
            ],
        )
        writer.writeheader()
        for query_index in ordered_misses.tolist():
            limit = min(error_top_k, top_indices.shape[1])
            for rank in range(limit):
                candidate_index = int(top_indices[query_index, rank].item())
                writer.writerow(
                    {
                        "query_index": query_index,
                        "query_id": ids[query_index],
                        "true_rank": int(ranks[query_index].item()),
                        "positive_cosine": float(positive[query_index].item()),
                        "candidate_rank": rank + 1,
                        "candidate_index": candidate_index,
                        "candidate_id": ids[candidate_index],
                        "candidate_group_id": group_ids[candidate_index],
                        "candidate_source": source_from_path(paths[candidate_index]),
                        "candidate_path": paths[candidate_index],
                        "candidate_cosine": float(top_values[query_index, rank].item()),
                        "is_target": candidate_index == query_index,
                        "is_same_group": group_ids[candidate_index] == group_ids[query_index],
                    }
                )
    return errors_path, topk_path


def write_source_summary(
    analysis: dict[str, Any],
    metadata: dict[str, list[str]],
    output_dir: Path,
) -> Path:
    """Write per-source recall and error summary."""

    paths = metadata["paths"]
    positive = analysis["positive"]
    ranks = analysis["ranks"]
    top_values = analysis["top_values"]
    buckets: dict[str, dict[str, float]] = {}
    for index, path in enumerate(paths):
        source = source_from_path(path)
        bucket = buckets.setdefault(
            source,
            {
                "num_items": 0.0,
                "num_errors": 0.0,
                "positive_cosine_sum": 0.0,
                "top1_minus_positive_sum": 0.0,
            },
        )
        bucket["num_items"] += 1.0
        is_error = float(ranks[index].item() > 1)
        bucket["num_errors"] += is_error
        bucket["positive_cosine_sum"] += float(positive[index].item())
        bucket["top1_minus_positive_sum"] += float(top_values[index, 0].item() - positive[index].item())

    summary_path = output_dir / "source_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "num_items",
                "num_errors",
                "recall_at_1",
                "error_rate",
                "positive_cosine_mean",
                "top1_minus_positive_mean",
            ],
        )
        writer.writeheader()
        for source, bucket in sorted(buckets.items(), key=lambda item: (-item[1]["num_errors"], item[0])):
            num_items = bucket["num_items"]
            num_errors = bucket["num_errors"]
            writer.writerow(
                {
                    "source": source,
                    "num_items": int(num_items),
                    "num_errors": int(num_errors),
                    "recall_at_1": 1.0 - num_errors / num_items if num_items else 0.0,
                    "error_rate": num_errors / num_items if num_items else 0.0,
                    "positive_cosine_mean": bucket["positive_cosine_sum"] / num_items if num_items else 0.0,
                    "top1_minus_positive_mean": bucket["top1_minus_positive_sum"] / num_items if num_items else 0.0,
                }
            )
    return summary_path


def make_output_dir(base_dir: Path, checkpoint: Path, split: str) -> Path:
    """Create an evaluation output directory name from checkpoint and split."""

    run_name = checkpoint.parent.name if checkpoint.parent.name else checkpoint.stem
    return base_dir / run_name / split


def main() -> None:
    """Run Lightning test evaluation and export retrieval analysis."""

    args = parse_args()
    model = IRContrastiveModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    config = load_config(args.config) if args.config else model.config
    named_overrides = vars(args).copy()
    named_overrides["output_dir"] = None
    config = apply_named_overrides(config, named_overrides)
    config = apply_overrides(config, args.overrides)
    if args.batch_size is not None:
        config.setdefault("evaluation", {})["batch_size"] = args.batch_size
    config.setdefault("evaluation", {})["retrieval_mode"] = "self_augmentation"

    seed = int(config.get("train", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)
    model.config = config
    model.eval_augmentor = SpectrumAugmentor.from_config(config.get("augmentation", {}))

    if args.split == "test" and not args.skip_trainer_test:
        datamodule = IRSpectrumDataModule(config)
        trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
        trainer.test(model, datamodule=datamodule)

    pl.seed_everything(seed, workers=True)
    device = get_device(args.device)
    model = model.to(device)
    augmentor = SpectrumAugmentor.from_config(config.get("augmentation", {}))
    loader = build_eval_loader(config, args.split, args.batch_size)
    encoded = collect_raw_and_augmented_embeddings(model, loader, augmentor, device, args.max_items)

    query = F.normalize(encoded["query_embeddings"], dim=-1)
    library = F.normalize(encoded["raw_embeddings"], dim=-1)
    similarity = query @ library.T
    top_k = args.top_k or [int(item) for item in config.get("evaluation", {}).get("top_k", [1, 5, 10])]
    top_k = sorted(set(top_k))
    analysis = compute_self_retrieval(similarity, top_k, max(max(top_k), args.error_top_k, 1))
    metrics = analysis["metrics"]
    add_loss_metrics(metrics, similarity, config)
    metrics.update(
        {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "max_items": args.max_items,
            "num_errors": int((analysis["ranks"] > 1).sum().item()),
            "error_rate": float((analysis["ranks"] > 1).float().mean().item()),
        }
    )

    output_dir = make_output_dir(args.output_dir, args.checkpoint, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
    errors_path, topk_path = write_error_files(
        analysis,
        {"ids": encoded["ids"], "group_ids": encoded["group_ids"], "paths": encoded["paths"]},
        output_dir,
        args.error_top_k,
        args.num_error_cases,
    )
    source_summary_path = write_source_summary(
        analysis,
        {"ids": encoded["ids"], "group_ids": encoded["group_ids"], "paths": encoded["paths"]},
        output_dir,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote error cases to {errors_path}")
    print(f"Wrote error top-k candidates to {topk_path}")
    print(f"Wrote source summary to {source_summary_path}")


if __name__ == "__main__":
    main()

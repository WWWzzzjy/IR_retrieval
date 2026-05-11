"""Retrieval metrics for validation embeddings."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.augmentations import SpectrumAugmentor
from src.evaluation.embedding_stats import aligned_pair_statistics


@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    max_items: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], list[str], list[str], list[str]]:
    """Encode spectra from a dataloader into a single embedding matrix."""

    model.eval()
    embeddings: list[torch.Tensor] = []
    spectra: list[torch.Tensor] = []
    wavenumber_values: list[torch.Tensor] = []
    ids: list[str] = []
    group_ids: list[str] = []
    paths: list[str] = []

    seen = 0
    for batch in loader:
        batch_spectra = batch.get("spectrum", batch.get("original"))
        if batch_spectra is None:
            raise KeyError("Evaluation batch must contain 'spectrum' or 'original'")
        if max_items is not None and seen >= max_items:
            break
        if max_items is not None and seen + batch_spectra.shape[0] > max_items:
            keep = max_items - seen
            batch_spectra = batch_spectra[:keep]
            if "x" in batch:
                batch["x"] = batch["x"][:keep]
            batch["ids"] = batch["ids"][:keep]
            batch["group_ids"] = batch["group_ids"][:keep]
            if "paths" in batch:
                batch["paths"] = batch["paths"][:keep]

        wavenumbers = batch.get("x")
        output = model(batch_spectra.to(device), wavenumbers=wavenumbers.to(device) if wavenumbers is not None else None)
        embeddings.append(output["embedding"].detach().cpu())
        spectra.append(batch_spectra.detach().cpu())
        if wavenumbers is not None:
            wavenumber_values.append(wavenumbers.detach().cpu())
        ids.extend(str(item) for item in batch["ids"])
        group_ids.extend(str(item) for item in batch["group_ids"])
        paths.extend(str(item) for item in batch.get("paths", [""] * batch_spectra.shape[0]))
        seen += batch_spectra.shape[0]

    if not embeddings:
        raise ValueError("No embeddings were collected")
    wavenumber_tensor = torch.cat(wavenumber_values, dim=0) if wavenumber_values else None
    return torch.cat(embeddings, dim=0), torch.cat(spectra, dim=0), wavenumber_tensor, ids, group_ids, paths


@torch.no_grad()
def encode_tensor_batches(
    model: torch.nn.Module,
    spectra: torch.Tensor,
    wavenumbers: Optional[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode an in-memory spectrum tensor in batches."""

    outputs: list[torch.Tensor] = []
    for start in range(0, spectra.shape[0], batch_size):
        stop = start + batch_size
        batch_wavenumbers = wavenumbers[start:stop] if wavenumbers is not None and wavenumbers.ndim == 2 else wavenumbers
        output = model(
            spectra[start:stop].to(device),
            wavenumbers=batch_wavenumbers.to(device) if batch_wavenumbers is not None else None,
        )
        outputs.append(output["embedding"].detach().cpu())
    return torch.cat(outputs, dim=0)


def recall_at_k_from_groups(
    embeddings: torch.Tensor,
    group_ids: list[str],
    top_k: Iterable[int],
) -> tuple[dict[str, float], int]:
    """Compute same-group Recall@K after removing self matches."""

    normalized = F.normalize(embeddings, dim=-1)
    if normalized.shape[0] <= 1:
        return {f"recall@{k}": float("nan") for k in top_k}, 0
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(torch.finfo(similarity.dtype).min)
    groups = torch.tensor([[a == b for b in group_ids] for a in group_ids], dtype=torch.bool)
    groups.fill_diagonal_(False)
    valid = groups.any(dim=1)
    valid_count = int(valid.sum().item())
    metrics: dict[str, float] = {}
    if valid_count == 0:
        for k in top_k:
            metrics[f"recall@{k}"] = float("nan")
        return metrics, valid_count

    for k in top_k:
        effective_k = max(1, min(int(k), similarity.shape[1] - 1))
        top_indices = similarity.topk(k=effective_k, dim=1).indices
        hits = groups.gather(1, top_indices).any(dim=1)
        metrics[f"recall@{k}"] = float(hits[valid].float().mean().item())
    return metrics, valid_count


def recall_at_k_aligned(
    query_embeddings: torch.Tensor,
    library_embeddings: torch.Tensor,
    top_k: Iterable[int],
) -> dict[str, float]:
    """Compute Recall@K where query i should retrieve library item i."""

    query = F.normalize(query_embeddings, dim=-1)
    library = F.normalize(library_embeddings, dim=-1)
    similarity = query @ library.T
    target = torch.arange(similarity.shape[0])
    metrics: dict[str, float] = {}
    for k in top_k:
        top_indices = similarity.topk(k=min(int(k), similarity.shape[1]), dim=1).indices
        hits = (top_indices == target.unsqueeze(1)).any(dim=1)
        metrics[f"recall@{k}"] = float(hits.float().mean().item())
    return metrics


def average_retrieval_time_ms(
    query_embeddings: torch.Tensor,
    library_embeddings: torch.Tensor,
    repeats: int = 20,
) -> float:
    """Estimate average brute-force cosine retrieval latency per query."""

    if query_embeddings.numel() == 0 or library_embeddings.numel() == 0:
        return 0.0
    query = F.normalize(query_embeddings[: min(repeats, query_embeddings.shape[0])], dim=-1)
    library = F.normalize(library_embeddings, dim=-1)
    start = time.perf_counter()
    _ = query @ library.T
    elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0 / query.shape[0])


@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    top_k: Iterable[int] = (1, 5, 10),
    augmentor: Optional[SpectrumAugmentor] = None,
    max_items: Optional[int] = None,
) -> dict[str, float]:
    """Evaluate validation retrieval with same-group or augmented-pair fallback."""

    library_embeddings, spectra, wavenumbers, _ids, group_ids, _paths = collect_embeddings(
        model,
        loader,
        device,
        max_items=max_items,
    )
    metrics, valid_group_queries = recall_at_k_from_groups(library_embeddings, group_ids, top_k)
    mode = "same_group"

    if valid_group_queries == 0:
        mode = "augmented_pair"
        augmentor = augmentor or SpectrumAugmentor({})
        augmented = torch.stack([augmentor(spectrum) for spectrum in spectra], dim=0)
        query_embeddings = encode_tensor_batches(
            model,
            augmented,
            wavenumbers=wavenumbers,
            device=device,
            batch_size=getattr(loader, "batch_size", 256) or 256,
        )
        metrics = recall_at_k_aligned(query_embeddings, library_embeddings, top_k)
        metrics.update(aligned_pair_statistics(query_embeddings, library_embeddings))
        metrics["retrieval_time_ms"] = average_retrieval_time_ms(query_embeddings, library_embeddings)
    else:
        metrics["positive_cosine_mean"] = float("nan")
        metrics["negative_cosine_mean"] = float("nan")
        metrics["cosine_gap"] = float("nan")
        metrics["retrieval_time_ms"] = average_retrieval_time_ms(library_embeddings, library_embeddings)

    metrics["valid_group_queries"] = float(valid_group_queries)
    metrics["retrieval_mode"] = 0.0 if mode == "same_group" else 1.0
    return metrics

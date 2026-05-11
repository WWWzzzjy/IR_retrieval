"""Embedding-space summary statistics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aligned_pair_statistics(
    query_embeddings: torch.Tensor,
    library_embeddings: torch.Tensor,
) -> dict[str, float]:
    """Compute aligned positive and sampled negative cosine statistics."""

    query = F.normalize(query_embeddings, dim=-1)
    library = F.normalize(library_embeddings, dim=-1)
    positives = (query * library).sum(dim=-1)
    similarity = query @ library.T
    eye = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    negatives = similarity.masked_select(~eye) if similarity.shape[0] > 1 else torch.empty(0)
    negative_mean = float(negatives.mean().item()) if negatives.numel() else 0.0
    positive_mean = float(positives.mean().item()) if positives.numel() else 0.0
    return {
        "positive_cosine_mean": positive_mean,
        "negative_cosine_mean": negative_mean,
        "cosine_gap": positive_mean - negative_mean,
    }


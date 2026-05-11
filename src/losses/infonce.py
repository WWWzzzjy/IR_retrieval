"""Contrastive retrieval losses for paired spectrum views."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(view1: torch.Tensor, view2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Compute SimCLR-style symmetric InfoNCE over two augmented views.

    Args:
        view1: Embeddings for the first augmented view with shape [batch, dim].
        view2: Embeddings for the second augmented view with shape [batch, dim].
        temperature: Softmax temperature.

    Returns:
        Scalar cross-entropy loss.
    """

    if view1.shape != view2.shape:
        raise ValueError(f"Expected matching embedding shapes, got {view1.shape} and {view2.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch_size = view1.shape[0]
    embeddings = F.normalize(torch.cat([view1, view2], dim=0), dim=-1)
    logits = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    labels = torch.arange(2 * batch_size, device=logits.device)
    labels = (labels + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, labels)


def retrieval_ce_loss(
    query: torch.Tensor,
    library: torch.Tensor,
    temperature: float = 0.1,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute augmented-to-raw retrieval cross entropy.

    Args:
        query: Query embeddings, usually augmented spectra, with shape [batch, dim].
        library: Library embeddings, usually raw spectra, with shape [batch, dim].
        temperature: Softmax temperature for retrieval logits.
        symmetric: Whether to also optimize library-to-query retrieval.

    Returns:
        A tuple of ``(loss, logits, cosine_similarity)``. ``logits`` are scaled
        by temperature for cross entropy; ``cosine_similarity`` is unscaled and
        easier to interpret for margin metrics.
    """

    if query.shape != library.shape:
        raise ValueError(f"Expected matching embedding shapes, got {query.shape} and {library.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch_size = query.shape[0]
    query = F.normalize(query, dim=-1)
    library = F.normalize(library, dim=-1)
    cosine = query @ library.T
    logits = cosine / temperature
    labels = torch.arange(batch_size, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = loss + F.cross_entropy(logits.T, labels)
    return loss, logits, cosine


def semi_hard_negative_margin_loss(
    cosine: torch.Tensor,
    margin: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize retrieval candidates that sit too close to the positive pair.

    The margin is applied on unscaled cosine similarities so its value remains
    comparable to the logged ``positive_cosine`` and ``hard_negative_cosine``.
    For each query, the preferred negative is semi-hard: below the positive
    similarity but within ``margin``. If none exists, the closest negative is
    used as a fallback so the loss remains finite in early training.

    Args:
        cosine: Unscaled query-library cosine matrix with shape [batch, batch].
        margin: Desired cosine margin between the aligned positive and selected
            negative.

    Returns:
        ``(loss, positive_cosine, selected_negative_cosine, retrieval_margin)``.
    """

    if cosine.ndim != 2 or cosine.shape[0] != cosine.shape[1]:
        raise ValueError(f"Expected a square cosine matrix, got {cosine.shape}")
    if margin <= 0:
        raise ValueError("margin must be positive")

    batch_size = cosine.shape[0]
    device = cosine.device
    neg_inf = torch.tensor(float("-inf"), dtype=cosine.dtype, device=device)
    positive = cosine.diag()
    if batch_size <= 1:
        zero = cosine.sum() * 0.0
        return zero, positive, positive, positive - positive
    neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    neg_cosine = cosine.masked_fill(~neg_mask, neg_inf)
    semi_hard_mask = (neg_cosine < positive.unsqueeze(1)) & (neg_cosine > positive.unsqueeze(1) - margin)
    semi_hard = neg_cosine.masked_fill(~semi_hard_mask, neg_inf).max(dim=1).values
    closest_negative = neg_cosine.max(dim=1).values
    selected_negative = torch.where(torch.isfinite(semi_hard), semi_hard, closest_negative)
    retrieval_margin = positive - selected_negative
    loss = F.relu(float(margin) - retrieval_margin).mean()
    return loss, positive, selected_negative, retrieval_margin

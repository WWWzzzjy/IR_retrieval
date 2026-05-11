"""InfoNCE loss for paired augmented spectrum views."""

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


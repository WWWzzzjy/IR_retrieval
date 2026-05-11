"""Projection head for contrastive embeddings."""

from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """Map pooled transformer features to retrieval embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        projection_hidden_dim: int | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if projection_hidden_dim is None or projection_hidden_dim <= 0:
            self.net = nn.Linear(hidden_dim, embedding_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, projection_hidden_dim),
                nn.GELU(),
                nn.Linear(projection_hidden_dim, embedding_dim),
            )
        self.normalize = normalize

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return projected embeddings."""

        embeddings = self.net(features)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings

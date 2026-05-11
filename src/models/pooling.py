"""Pooling layers for transformer patch tokens."""

from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Learn a soft attention-weighted average over patch tokens."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool tokens with learned attention weights."""

        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=-1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


class MeanPooling(nn.Module):
    """Average patch tokens along the sequence dimension."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the mean pooled token representation."""

        return tokens.mean(dim=1)


def build_pooling(mode: str, hidden_dim: int) -> nn.Module:
    """Build a pooling module for the requested mode."""

    if mode == "attention":
        return AttentionPooling(hidden_dim)
    if mode == "mean":
        return MeanPooling()
    if mode == "cls":
        return nn.Identity()
    raise ValueError(f"Unknown pooling mode: {mode}")


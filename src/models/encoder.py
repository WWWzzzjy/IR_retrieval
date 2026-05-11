"""Transformer-based mid-IR spectrum encoder."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from src.models.patch_embedding import PatchEmbedding
from src.models.pooling import build_pooling
from src.models.projection_head import ProjectionHead


class IRTransformerEncoder(nn.Module):
    """Encode fixed-length absorbance spectra into low-dimensional embeddings.

    Args:
        spectrum_length: Number of input absorbance points.
        patch_size: Number of points per patch.
        stride: Patch stride.
        hidden_dim: Transformer hidden size.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward dimension.
        dropout: Dropout probability.
        embedding_dim: Output embedding dimension.
        pooling: Pooling mode: attention, mean, or cls.
        pos_encoding: Position encoding mode.
        wavenumber_min: Minimum wavenumber for physical position normalization.
        wavenumber_max: Maximum wavenumber for physical position normalization.
        position_scale: Scale for sinusoidal physical positions.
        projection_hidden_dim: Hidden size inside the projection head.
        normalize_embeddings: Whether to L2-normalize embeddings.
    """

    def __init__(
        self,
        spectrum_length: int = 460,
        patch_size: int = 10,
        stride: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        embedding_dim: int = 128,
        pooling: str = "attention",
        pos_encoding: str = "wavenumber_sinusoidal",
        wavenumber_min: float = 455.126,
        wavenumber_max: float = 3996.0,
        position_scale: float = 1000.0,
        projection_hidden_dim: int | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if pooling not in {"attention", "mean", "cls"}:
            raise ValueError("pooling must be one of: attention, mean, cls")

        self.pooling_mode = pooling
        self.patch_embedding = PatchEmbedding(
            spectrum_length=spectrum_length,
            patch_size=patch_size,
            stride=stride,
            hidden_dim=hidden_dim,
            pos_encoding=pos_encoding,
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
            position_scale=position_scale,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pooling == "cls" else None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pooling = build_pooling(pooling, hidden_dim)
        self.projection_head = ProjectionHead(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            projection_hidden_dim=projection_hidden_dim,
            normalize=normalize_embeddings,
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_size),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "IRTransformerEncoder":
        """Build an encoder from a model configuration dictionary."""

        return cls(**config)

    @property
    def num_patches(self) -> int:
        """Return the number of spectrum patches."""

        return self.patch_embedding.num_patches

    def forward(
        self,
        spectra: torch.Tensor,
        wavenumbers: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        return_reconstruction: bool = False,
        return_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encode spectra and optionally reconstruct masked patches."""

        tokens = self.patch_embedding(spectra, wavenumbers=wavenumbers, patch_mask=patch_mask)
        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.final_norm(self.transformer(tokens))
        patch_tokens = encoded[:, 1:] if self.cls_token is not None else encoded

        if self.pooling_mode == "cls":
            pooled = encoded[:, 0]
        else:
            pooled = self.pooling(patch_tokens)

        output = {"embedding": self.projection_head(pooled)}
        if return_reconstruction:
            output["reconstruction"] = self.reconstruction_head(patch_tokens)
        if return_tokens:
            output["tokens"] = patch_tokens
        return output


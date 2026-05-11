"""Patch embedding and physical wavenumber-aware position encodings."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def sinusoidal_encoding(positions: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    """Create sinusoidal encodings for arbitrary scalar positions.

    Args:
        positions: Tensor of positions with shape [...]. Values may be patch indices
            or normalized physical wavenumber coordinates.
        hidden_dim: Output encoding dimension.

    Returns:
        Sinusoidal encodings with shape [*, hidden_dim].
    """

    positions = positions.float().unsqueeze(-1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / hidden_dim)
    )
    encoding = torch.zeros(*positions.shape[:-1], hidden_dim, device=positions.device)
    encoding[..., 0::2] = torch.sin(positions * div_term)
    if hidden_dim > 1:
        encoding[..., 1::2] = torch.cos(positions * div_term[: encoding[..., 1::2].shape[-1]])
    return encoding


class PatchEmbedding(nn.Module):
    """Turn fixed-size spectrum windows into transformer tokens.

    Args:
        spectrum_length: Number of input absorbance points.
        patch_size: Number of raw points per patch.
        stride: Patch stride in raw points.
        hidden_dim: Transformer hidden dimension.
        pos_encoding: Position encoding mode.
        wavenumber_min: Minimum physical wavenumber used for normalization.
        wavenumber_max: Maximum physical wavenumber used for normalization.
        position_scale: Scale applied to normalized wavenumber positions.
    """

    def __init__(
        self,
        spectrum_length: int,
        patch_size: int,
        stride: int,
        hidden_dim: int,
        pos_encoding: str = "wavenumber_sinusoidal",
        wavenumber_min: float = 455.126,
        wavenumber_max: float = 3996.0,
        position_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        if spectrum_length < patch_size:
            raise ValueError("spectrum_length must be at least patch_size")

        self.spectrum_length = spectrum_length
        self.patch_size = patch_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.pos_encoding = pos_encoding
        self.wavenumber_min = float(wavenumber_min)
        self.wavenumber_max = float(wavenumber_max)
        self.position_scale = float(position_scale)
        self.num_patches = (spectrum_length - patch_size) // stride + 1

        self.projection = nn.Linear(patch_size, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        default_wavenumbers = torch.linspace(self.wavenumber_min, self.wavenumber_max, spectrum_length)
        self.register_buffer("default_wavenumbers", default_wavenumbers, persistent=False)

        if pos_encoding == "learned":
            self.learned_pos = nn.Parameter(torch.zeros(self.num_patches, hidden_dim))
            nn.init.trunc_normal_(self.learned_pos, std=0.02)
        else:
            self.learned_pos = None

    def forward(
        self,
        spectra: torch.Tensor,
        wavenumbers: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed spectra into patch tokens with positional information."""

        if spectra.ndim != 2:
            raise ValueError(f"Expected spectra shape [batch, length], got {tuple(spectra.shape)}")
        if spectra.shape[1] != self.spectrum_length:
            raise ValueError(f"Expected length {self.spectrum_length}, got {spectra.shape[1]}")

        patches = spectra.unfold(dimension=1, size=self.patch_size, step=self.stride)
        tokens = self.projection(patches)

        if patch_mask is not None:
            if patch_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"patch_mask shape {tuple(patch_mask.shape)} does not match tokens {tuple(tokens.shape[:2])}"
                )
            mask_token = self.mask_token.view(1, 1, -1).to(dtype=tokens.dtype, device=tokens.device)
            tokens = torch.where(patch_mask.unsqueeze(-1), mask_token, tokens)

        positions = self._position_encoding(wavenumbers, tokens.device, tokens.dtype)
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        return tokens + positions.to(dtype=tokens.dtype)

    def get_patch_centers(self, wavenumbers: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the physical center wavenumber of each patch."""

        if wavenumbers is None:
            wavenumbers = self.default_wavenumbers
        if wavenumbers.ndim == 1:
            patches = wavenumbers.unfold(dimension=0, size=self.patch_size, step=self.stride)
        elif wavenumbers.ndim == 2:
            patches = wavenumbers.unfold(dimension=1, size=self.patch_size, step=self.stride)
        else:
            raise ValueError(f"Expected wavenumbers shape [length] or [batch, length], got {wavenumbers.shape}")
        return patches.mean(dim=-1)

    def _position_encoding(
        self,
        wavenumbers: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the configured position encoding."""

        if self.pos_encoding == "learned":
            if self.learned_pos is None:
                raise RuntimeError("learned_pos was not initialized")
            return self.learned_pos.to(device=device, dtype=dtype)

        if self.pos_encoding == "index_sinusoidal":
            indices = torch.arange(self.num_patches, device=device, dtype=torch.float32)
            return sinusoidal_encoding(indices, self.hidden_dim).to(dtype=dtype)

        if self.pos_encoding != "wavenumber_sinusoidal":
            raise ValueError(f"Unknown position encoding: {self.pos_encoding}")

        centers = self.get_patch_centers(wavenumbers.to(device) if wavenumbers is not None else None)
        centers = centers.to(device=device, dtype=torch.float32)
        span = max(abs(self.wavenumber_max - self.wavenumber_min), 1e-6)
        normalized = (centers - self.wavenumber_min) / span
        return sinusoidal_encoding(normalized * self.position_scale, self.hidden_dim).to(dtype=dtype)


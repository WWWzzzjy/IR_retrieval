"""Masked patch reconstruction loss with optional fingerprint weighting."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class MaskedReconstructionLoss(nn.Module):
    """MSE on masked patch predictions, with optional fingerprint weights.

    Args:
        patch_size: Number of raw points per patch.
        stride: Patch stride in raw points.
        fingerprint_weighting: Whether to upweight fingerprint-region patches.
        fingerprint_threshold: Patches below this wavenumber get extra weight.
        fingerprint_weight: Weight for fingerprint-region patches.
        default_weight: Weight for other patches.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        fingerprint_weighting: bool = False,
        fingerprint_threshold: float = 1500.0,
        fingerprint_weight: float = 1.5,
        default_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.fingerprint_weighting = fingerprint_weighting
        self.fingerprint_threshold = fingerprint_threshold
        self.fingerprint_weight = fingerprint_weight
        self.default_weight = default_weight

    def forward(
        self,
        reconstruction: torch.Tensor,
        target_spectra: torch.Tensor,
        patch_mask: torch.Tensor,
        patch_centers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return weighted MSE over masked patches only."""

        if reconstruction.ndim != 3:
            raise ValueError("reconstruction must have shape [batch, patches, patch_size]")
        if target_spectra.ndim != 2:
            raise ValueError("target_spectra must have shape [batch, length]")
        if patch_mask.shape != reconstruction.shape[:2]:
            raise ValueError("patch_mask must match reconstruction batch and patch dimensions")

        target_patches = target_spectra.unfold(dimension=1, size=self.patch_size, step=self.stride)
        if target_patches.shape != reconstruction.shape:
            raise ValueError(
                f"Target patches {tuple(target_patches.shape)} do not match reconstruction {tuple(reconstruction.shape)}"
            )

        per_patch_mse = (reconstruction - target_patches).pow(2).mean(dim=-1)
        weights = torch.ones_like(per_patch_mse) * self.default_weight
        if self.fingerprint_weighting and patch_centers is not None:
            centers = patch_centers.to(device=per_patch_mse.device)
            if centers.ndim == 1:
                centers = centers.unsqueeze(0)
            fingerprint = centers < self.fingerprint_threshold
            weights = torch.where(
                fingerprint,
                torch.as_tensor(self.fingerprint_weight, device=per_patch_mse.device),
                torch.as_tensor(self.default_weight, device=per_patch_mse.device),
            ).to(dtype=per_patch_mse.dtype)

        mask = patch_mask.to(dtype=per_patch_mse.dtype)
        denominator = (mask * weights).sum().clamp_min(1.0)
        if denominator.item() == 1.0 and mask.sum().item() == 0.0:
            return reconstruction.sum() * 0.0
        return (per_patch_mse * mask * weights).sum() / denominator


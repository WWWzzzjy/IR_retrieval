"""Tests for contrastive and reconstruction losses."""

from __future__ import annotations

import torch

from src.losses import MaskedReconstructionLoss, info_nce_loss


def test_info_nce_is_finite() -> None:
    """InfoNCE should produce a finite scalar for paired embeddings."""

    embeddings = torch.eye(4)
    loss = info_nce_loss(embeddings, embeddings, temperature=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_masked_reconstruction_loss_zero_for_exact_prediction() -> None:
    """Exact patch reconstruction should have zero masked MSE."""

    target = torch.arange(40, dtype=torch.float32).view(2, 20) / 40.0
    target_patches = target.unfold(dimension=1, size=5, step=5).clone()
    patch_mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    patch_centers = torch.tensor([500.0, 1000.0, 2000.0, 3000.0])
    criterion = MaskedReconstructionLoss(
        patch_size=5,
        stride=5,
        fingerprint_weighting=True,
        fingerprint_threshold=1500.0,
    )

    loss = criterion(target_patches, target, patch_mask, patch_centers)
    assert torch.isclose(loss, torch.tensor(0.0))

    bad_loss = criterion(target_patches + 1.0, target, patch_mask, patch_centers)
    assert float(bad_loss) > 0.0


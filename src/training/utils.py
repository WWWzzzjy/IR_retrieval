"""Small helpers shared by Lightning training and index-building scripts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Return an available torch device for direct inference scripts."""

    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def random_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample a boolean reconstruction mask over patches."""

    if mask_ratio <= 0.0:
        return torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    mask_count = max(1, min(num_patches, int(round(num_patches * mask_ratio))))
    scores = torch.rand(batch_size, num_patches, device=device)
    indices = scores.argsort(dim=1)[:, :mask_count]
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, indices, True)
    return mask


def make_run_name(config: dict[str, Any]) -> str:
    """Create a compact Lightning logger run name."""

    model = config.get("model", {})
    train = config.get("train", {})
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{timestamp}_h{model.get('hidden_dim', 'na')}"
        f"_e{model.get('embedding_dim', 'na')}"
        f"_bs{train.get('batch_size', 'na')}"
        f"_lr{train.get('lr', 'na')}"
    )


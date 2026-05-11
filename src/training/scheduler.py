"""Learning-rate schedulers used by the trainer."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Build a warmup plus cosine-decay scheduler."""

    warmup_steps = max(0, int(warmup_epochs) * max(1, steps_per_epoch))
    total_steps = max(1, int(total_epochs) * max(1, steps_per_epoch))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        """Return the multiplicative LR factor for a scheduler step."""

        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


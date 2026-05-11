"""Spectrum augmentations for contrastive positive-pair generation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _is_enabled(config: dict[str, Any], name: str) -> bool:
    """Return whether an augmentation block is enabled."""

    block = config.get(name, {})
    return bool(block.get("enabled", False))


def _sample_uniform(low: float, high: float, device: torch.device) -> float:
    """Sample one scalar from a uniform distribution."""

    return float(torch.empty((), device=device).uniform_(low, high).item())


def _sample_int(low: int, high: int, device: torch.device) -> int:
    """Sample one integer from the inclusive interval [low, high]."""

    return int(torch.randint(low, high + 1, (), device=device).item())


def add_gaussian_noise(spectrum: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise to one spectrum."""

    return spectrum + torch.randn_like(spectrum) * sigma


def scale_intensity(spectrum: torch.Tensor, min_scale: float, max_scale: float) -> torch.Tensor:
    """Multiply spectrum intensity by a random scalar."""

    scale = _sample_uniform(min_scale, max_scale, spectrum.device)
    return spectrum * scale


def add_baseline_drift(spectrum: torch.Tensor, max_degree: int, amplitude: float) -> torch.Tensor:
    """Add a low-frequency polynomial baseline perturbation."""

    degree = _sample_int(1, max(1, max_degree), spectrum.device)
    x = torch.linspace(-1.0, 1.0, spectrum.numel(), device=spectrum.device, dtype=spectrum.dtype)
    curve = torch.zeros_like(spectrum)
    for power in range(degree + 1):
        coeff = _sample_uniform(-amplitude, amplitude, spectrum.device) / float(power + 1)
        curve = curve + coeff * x.pow(power)
    curve = curve - curve.mean()
    return spectrum + curve


def shift_wavenumber(spectrum: torch.Tensor, max_shift_points: int) -> torch.Tensor:
    """Shift a spectrum by a small integer number of points."""

    if max_shift_points <= 0:
        return spectrum
    shift = _sample_int(-max_shift_points, max_shift_points, spectrum.device)
    if shift == 0:
        return spectrum

    shifted = torch.empty_like(spectrum)
    if shift > 0:
        shifted[:shift] = spectrum[0]
        shifted[shift:] = spectrum[:-shift]
    else:
        amount = abs(shift)
        shifted[-amount:] = spectrum[-1]
        shifted[:-amount] = spectrum[amount:]
    return shifted


def apply_local_mask(
    spectrum: torch.Tensor,
    min_fraction: float,
    max_fraction: float,
) -> torch.Tensor:
    """Zero a random continuous region of a spectrum."""

    length = spectrum.numel()
    fraction = _sample_uniform(min_fraction, max_fraction, spectrum.device)
    mask_length = max(1, min(length, int(round(length * fraction))))
    start = _sample_int(0, max(0, length - mask_length), spectrum.device)
    augmented = spectrum.clone()
    augmented[start : start + mask_length] = 0.0
    return augmented


def gaussian_smooth(
    spectrum: torch.Tensor,
    kernel_size: int,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Apply a light 1D Gaussian smoothing kernel."""

    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = _sample_uniform(sigma_min, sigma_max, spectrum.device)
    radius = kernel_size // 2
    positions = torch.arange(-radius, radius + 1, device=spectrum.device, dtype=spectrum.dtype)
    kernel = torch.exp(-0.5 * (positions / sigma).pow(2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    padded = F.pad(spectrum.view(1, 1, -1), (radius, radius), mode="reflect")
    return F.conv1d(padded, kernel).view(-1)


class SpectrumAugmentor:
    """Apply independently sampled augmentations to a spectrum.

    Args:
        config: Augmentation configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SpectrumAugmentor":
        """Build an augmentor from a YAML augmentation block."""

        return cls(config or {})

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Return one augmented copy of a 1D spectrum tensor."""

        augmented = spectrum.detach().clone().float()
        cfg = self.config

        if self._should_apply("gaussian_noise"):
            block = cfg["gaussian_noise"]
            augmented = add_gaussian_noise(augmented, float(block.get("sigma", 0.01)))

        if self._should_apply("intensity_scale"):
            block = cfg["intensity_scale"]
            augmented = scale_intensity(
                augmented,
                float(block.get("min_scale", 0.9)),
                float(block.get("max_scale", 1.1)),
            )

        if self._should_apply("baseline_drift"):
            block = cfg["baseline_drift"]
            augmented = add_baseline_drift(
                augmented,
                int(block.get("max_degree", 3)),
                float(block.get("amplitude", 0.02)),
            )

        if self._should_apply("wavenumber_shift"):
            block = cfg["wavenumber_shift"]
            augmented = shift_wavenumber(augmented, int(block.get("max_shift_points", 20)))

        if self._should_apply("local_mask"):
            block = cfg["local_mask"]
            augmented = apply_local_mask(
                augmented,
                float(block.get("min_fraction", 0.05)),
                float(block.get("max_fraction", 0.15)),
            )

        if self._should_apply("peak_width"):
            block = cfg["peak_width"]
            augmented = gaussian_smooth(
                augmented,
                int(block.get("kernel_size", 7)),
                float(block.get("sigma_min", 0.6)),
                float(block.get("sigma_max", 1.2)),
            )

        if bool(cfg.get("clamp", True)):
            augmented = augmented.clamp(0.0, 1.0)
        return augmented

    def _should_apply(self, name: str) -> bool:
        """Sample an augmentation probability gate."""

        if not _is_enabled(self.config, name):
            return False
        probability = float(self.config[name].get("probability", 1.0))
        return bool(torch.rand(()) < probability)


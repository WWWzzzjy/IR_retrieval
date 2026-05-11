"""Tests for spectrum augmentations."""

from __future__ import annotations

import torch

from src.data.augmentations import SpectrumAugmentor, shift_wavenumber


def test_augmentor_preserves_shape_and_clamps_range() -> None:
    """Full augmentor should keep spectrum length and clamp values."""

    torch.manual_seed(7)
    augmentor = SpectrumAugmentor(
        {
            "gaussian_noise": {"enabled": True, "probability": 1.0, "sigma": 0.01},
            "intensity_scale": {"enabled": True, "probability": 1.0, "min_scale": 0.9, "max_scale": 1.1},
            "baseline_drift": {"enabled": True, "probability": 1.0, "max_degree": 3, "amplitude": 0.02},
            "wavenumber_shift": {"enabled": True, "probability": 1.0, "max_shift_points": 3},
            "local_mask": {"enabled": True, "probability": 1.0, "min_fraction": 0.05, "max_fraction": 0.1},
            "peak_width": {"enabled": True, "probability": 1.0, "kernel_size": 5, "sigma_min": 0.8, "sigma_max": 1.0},
            "clamp": True,
        }
    )
    spectrum = torch.linspace(0.0, 1.0, 64)
    augmented = augmentor(spectrum)
    assert augmented.shape == spectrum.shape
    assert float(augmented.min()) >= 0.0
    assert float(augmented.max()) <= 1.0


def test_shift_wavenumber_keeps_shape() -> None:
    """Integer wavenumber shifts should preserve shape."""

    torch.manual_seed(3)
    spectrum = torch.arange(16, dtype=torch.float32)
    shifted = shift_wavenumber(spectrum, max_shift_points=4)
    assert shifted.shape == spectrum.shape


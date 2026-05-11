"""Tests for model forward passes."""

from __future__ import annotations

import torch

from src.models import IRTransformerEncoder


def build_small_model(pooling: str = "attention") -> IRTransformerEncoder:
    """Construct a small encoder for fast tests."""

    return IRTransformerEncoder(
        spectrum_length=40,
        patch_size=10,
        stride=10,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        embedding_dim=16,
        pooling=pooling,
        wavenumber_min=400.0,
        wavenumber_max=4000.0,
    )


def test_model_forward_shapes_with_reconstruction() -> None:
    """Encoder should return embeddings and patch reconstructions."""

    torch.manual_seed(5)
    model = build_small_model()
    spectra = torch.rand(2, 40)
    wavenumbers = torch.linspace(400.0, 4000.0, 40).repeat(2, 1)
    patch_mask = torch.zeros(2, 4, dtype=torch.bool)
    patch_mask[:, 1] = True

    output = model(
        spectra,
        wavenumbers=wavenumbers,
        patch_mask=patch_mask,
        return_reconstruction=True,
        return_tokens=True,
    )
    assert output["embedding"].shape == (2, 16)
    assert output["reconstruction"].shape == (2, 4, 10)
    assert output["tokens"].shape == (2, 4, 32)


def test_pooling_modes_forward() -> None:
    """All configured pooling modes should produce embeddings."""

    spectra = torch.rand(2, 40)
    for pooling in ("attention", "mean", "cls"):
        model = build_small_model(pooling)
        output = model(spectra)
        assert output["embedding"].shape == (2, 16)


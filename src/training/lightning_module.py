"""PyTorch Lightning module for contrastive IR spectrum encoder training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.data.augmentations import SpectrumAugmentor
from src.evaluation.embedding_stats import aligned_pair_statistics
from src.evaluation.retrieval import (
    average_retrieval_time_ms,
    recall_at_k_aligned,
    recall_at_k_from_groups,
)
from src.losses import MaskedReconstructionLoss, info_nce_loss
from src.models import IRTransformerEncoder
from src.training.scheduler import build_warmup_cosine_scheduler
from src.training.utils import random_patch_mask


class IRContrastiveModule(pl.LightningModule):
    """LightningModule wrapping the IR encoder, losses, optimizer, and metrics.

    Args:
        config: Full experiment configuration. When loading from a checkpoint,
            Lightning may pass saved hyperparameters as keyword arguments instead.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config or dict(kwargs)
        self.save_hyperparameters(self.config)

        model_cfg = self.config.get("model", {})
        loss_cfg = self.config.get("loss", {})
        recon_cfg = loss_cfg.get("reconstruction", {})

        self.encoder = IRTransformerEncoder.from_config(model_cfg)
        self.temperature = float(loss_cfg.get("temperature", 0.1))
        self.alpha = float(loss_cfg.get("alpha", 1.0))
        self.beta = float(loss_cfg.get("beta", 0.3))
        self.mask_ratio = float(recon_cfg.get("mask_ratio", 0.15))
        self.reconstruction_loss = MaskedReconstructionLoss(
            patch_size=int(model_cfg.get("patch_size", 10)),
            stride=int(model_cfg.get("stride", model_cfg.get("patch_size", 10))),
            fingerprint_weighting=bool(recon_cfg.get("fingerprint_weighting", False)),
            fingerprint_threshold=float(recon_cfg.get("fingerprint_threshold", 1500.0)),
            fingerprint_weight=float(recon_cfg.get("fingerprint_weight", 1.5)),
            default_weight=float(recon_cfg.get("default_weight", 1.0)),
        )
        self.eval_augmentor = SpectrumAugmentor.from_config(self.config.get("augmentation", {}))
        self._validation_outputs: list[dict[str, Any]] = []
        self._test_outputs: list[dict[str, Any]] = []

    def forward(self, x: torch.Tensor, wavenumbers: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return embeddings for inference."""

        return self.encoder(x, wavenumbers=wavenumbers)["embedding"]

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute contrastive and reconstruction losses for a training batch."""

        losses = self._compute_losses(batch)
        self.log("train/loss_total", losses["loss"], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["view1"].shape[0])
        self.log("train/loss_infonce", losses["info_nce"], on_step=True, on_epoch=True, batch_size=batch["view1"].shape[0])
        self.log("train/loss_recon", losses["reconstruction"], on_step=True, on_epoch=True, batch_size=batch["view1"].shape[0])
        return losses["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Monitor validation loss and collect embeddings for retrieval metrics."""

        contrastive_batch = self._make_contrastive_batch(batch)
        losses = self._compute_losses(contrastive_batch)
        batch_size = batch["spectrum"].shape[0]
        self.log("val/loss_total", losses["loss"], on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/loss_infonce", losses["info_nce"], on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val/loss_recon", losses["reconstruction"], on_epoch=True, batch_size=batch_size, sync_dist=True)
        self._collect_epoch_output(batch, self._validation_outputs)
        return losses["loss"]

    def on_validation_epoch_end(self) -> None:
        """Compute validation retrieval metrics from collected embeddings."""

        self._log_epoch_retrieval_metrics(self._validation_outputs, prefix="val")
        self._validation_outputs.clear()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Monitor test loss and collect embeddings for retrieval metrics."""

        contrastive_batch = self._make_contrastive_batch(batch)
        losses = self._compute_losses(contrastive_batch)
        batch_size = batch["spectrum"].shape[0]
        self.log("test/loss_total", losses["loss"], on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("test/loss_infonce", losses["info_nce"], on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("test/loss_recon", losses["reconstruction"], on_epoch=True, batch_size=batch_size, sync_dist=True)
        self._collect_epoch_output(batch, self._test_outputs)
        return losses["loss"]

    def on_test_epoch_end(self) -> None:
        """Compute test retrieval metrics from collected embeddings."""

        self._log_epoch_retrieval_metrics(self._test_outputs, prefix="test")
        self._test_outputs.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW and a step-wise warmup cosine scheduler."""

        train_cfg = self.config.get("train", {})
        optimizer = AdamW(
            self.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )
        max_epochs = max(1, int(train_cfg.get("num_epochs", 100)))
        estimated_steps = max(1, int(getattr(self.trainer, "estimated_stepping_batches", max_epochs)))
        steps_per_epoch = max(1, estimated_steps // max_epochs)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=int(train_cfg.get("warmup_epochs", 5)),
            total_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.05)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _compute_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute InfoNCE and masked reconstruction losses."""

        view1 = batch["view1"]
        view2 = batch["view2"]
        wavenumbers = batch.get("x")
        output1 = self.encoder(view1, wavenumbers=wavenumbers)
        output2 = self.encoder(view2, wavenumbers=wavenumbers)
        contrastive = info_nce_loss(output1["embedding"], output2["embedding"], temperature=self.temperature)

        reconstruction = torch.zeros((), device=self.device, dtype=contrastive.dtype)
        if self.beta > 0.0 and self.mask_ratio > 0.0:
            recon_input = torch.cat([view1, view2], dim=0)
            recon_wavenumbers = torch.cat([wavenumbers, wavenumbers], dim=0) if wavenumbers is not None else None
            patch_mask = batch.get("patch_mask")
            if patch_mask is None:
                patch_mask = random_patch_mask(
                    batch_size=recon_input.shape[0],
                    num_patches=self.encoder.num_patches,
                    mask_ratio=self.mask_ratio,
                    device=recon_input.device,
                )
            recon_output = self.encoder(
                recon_input,
                wavenumbers=recon_wavenumbers,
                patch_mask=patch_mask,
                return_reconstruction=True,
            )
            patch_centers = self.encoder.patch_embedding.get_patch_centers(recon_wavenumbers)
            reconstruction = self.reconstruction_loss(
                recon_output["reconstruction"],
                recon_input,
                patch_mask,
                patch_centers=patch_centers,
            )

        total = self.alpha * contrastive + self.beta * reconstruction
        return {"loss": total, "info_nce": contrastive, "reconstruction": reconstruction}

    def _make_contrastive_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Create two augmented validation/test views without changing data loaders."""

        spectra = batch["spectrum"]
        view1 = torch.stack([self.eval_augmentor(spectrum) for spectrum in spectra.detach().cpu()], dim=0).to(self.device)
        view2 = torch.stack([self.eval_augmentor(spectrum) for spectrum in spectra.detach().cpu()], dim=0).to(self.device)
        result = {
            "view1": view1,
            "view2": view2,
            "original": spectra,
            "x": batch.get("x"),
        }
        return result

    @torch.no_grad()
    def _collect_epoch_output(self, batch: dict[str, Any], store: list[dict[str, Any]]) -> None:
        """Collect raw embeddings and metadata for epoch-end retrieval metrics."""

        spectra = batch["spectrum"]
        wavenumbers = batch.get("x")
        embeddings = self.forward(spectra, wavenumbers=wavenumbers)
        store.append(
            {
                "embeddings": embeddings.detach().cpu(),
                "spectra": spectra.detach().cpu(),
                "wavenumbers": wavenumbers.detach().cpu() if wavenumbers is not None else None,
                "group_ids": [str(item) for item in batch["group_ids"]],
            }
        )

    @torch.no_grad()
    def _log_epoch_retrieval_metrics(self, outputs: list[dict[str, Any]], prefix: str) -> None:
        """Compute and log Recall@K plus embedding-space statistics."""

        if not outputs:
            return
        embeddings = torch.cat([item["embeddings"] for item in outputs], dim=0)
        spectra = torch.cat([item["spectra"] for item in outputs], dim=0)
        wavenumbers = self._cat_optional_tensors([item["wavenumbers"] for item in outputs])
        group_ids = [group_id for item in outputs for group_id in item["group_ids"]]
        top_k = tuple(int(item) for item in self.config.get("evaluation", {}).get("top_k", [1, 5, 10]))

        metrics, valid_group_queries = recall_at_k_from_groups(embeddings, group_ids, top_k)
        if valid_group_queries == 0:
            augmented = torch.stack([self.eval_augmentor(spectrum) for spectrum in spectra], dim=0)
            query_embeddings = self._encode_tensor_batches(augmented, wavenumbers)
            metrics = recall_at_k_aligned(query_embeddings, embeddings, top_k)
            metrics.update(aligned_pair_statistics(query_embeddings, embeddings))
            metrics["retrieval_time_ms"] = average_retrieval_time_ms(query_embeddings, embeddings)
            metrics["retrieval_mode"] = 1.0
        else:
            metrics.update(self._group_similarity_statistics(embeddings, group_ids))
            metrics["retrieval_time_ms"] = average_retrieval_time_ms(embeddings, embeddings)
            metrics["retrieval_mode"] = 0.0
        metrics["valid_group_queries"] = float(valid_group_queries)

        renamed = self._rename_metrics_for_lightning(metrics)
        for name, value in renamed.items():
            tensor_value = torch.tensor(value, dtype=torch.float32, device=self.device)
            self.log(
                f"{prefix}/{name}",
                tensor_value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "recall_at_1",
                sync_dist=True,
            )
            if prefix == "val" and name == "recall_at_1":
                self.log("val_recall_at_1", tensor_value, on_step=False, on_epoch=True, sync_dist=True)

    def _encode_tensor_batches(
        self,
        spectra: torch.Tensor,
        wavenumbers: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode CPU tensors in batches for augmented-pair retrieval fallback."""

        batch_size = int(self.config.get("evaluation", {}).get("batch_size", 256))
        embeddings: list[torch.Tensor] = []
        for start in range(0, spectra.shape[0], batch_size):
            stop = start + batch_size
            batch_wavenumbers = wavenumbers[start:stop] if wavenumbers is not None and wavenumbers.ndim == 2 else wavenumbers
            embeddings.append(
                self.forward(
                    spectra[start:stop].to(self.device),
                    wavenumbers=batch_wavenumbers.to(self.device) if batch_wavenumbers is not None else None,
                )
                .detach()
                .cpu()
            )
        return torch.cat(embeddings, dim=0)

    @staticmethod
    def _cat_optional_tensors(values: list[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        """Concatenate optional tensor batches when all are present."""

        present = [value for value in values if value is not None]
        if not present or len(present) != len(values):
            return None
        return torch.cat(present, dim=0)

    @staticmethod
    def _rename_metrics_for_lightning(metrics: dict[str, float]) -> dict[str, float]:
        """Convert metric names to logger-friendly keys."""

        return {
            key.replace("@", "_at_")
            .replace("positive_cosine_mean", "same_cosine_mean")
            .replace("negative_cosine_mean", "different_cosine_mean"): value
            for key, value in metrics.items()
        }

    @staticmethod
    def _group_similarity_statistics(embeddings: torch.Tensor, group_ids: list[str]) -> dict[str, float]:
        """Compute same-group and different-group cosine means."""

        normalized = F.normalize(embeddings, dim=-1)
        similarity = normalized @ normalized.T
        groups_by_id: dict[str, list[int]] = defaultdict(list)
        for index, group_id in enumerate(group_ids):
            groups_by_id[group_id].append(index)
        same_mask = torch.zeros_like(similarity, dtype=torch.bool)
        for indices in groups_by_id.values():
            if len(indices) > 1:
                index_tensor = torch.tensor(indices, dtype=torch.long)
                same_mask[index_tensor[:, None], index_tensor[None, :]] = True
        eye = torch.eye(similarity.shape[0], dtype=torch.bool)
        same_mask = same_mask & ~eye
        different_mask = ~same_mask & ~eye
        same = similarity.masked_select(same_mask)
        different = similarity.masked_select(different_mask)
        same_mean = float(same.mean().item()) if same.numel() else float("nan")
        different_mean = float(different.mean().item()) if different.numel() else float("nan")
        return {
            "positive_cosine_mean": same_mean,
            "negative_cosine_mean": different_mean,
            "cosine_gap": same_mean - different_mean,
        }

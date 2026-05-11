# Mid-IR Spectrum Encoder

PyTorch Lightning training repo for a transformer encoder that maps fixed-length mid-IR absorbance spectra to retrieval embeddings.

## What It Builds

- Patch-based transformer encoder for `[batch, 460]` spectra.
- Wavenumber-aware sinusoidal position encoding based on patch center cm^-1 values.
- Pooling options: `attention`, `mean`, and `cls`.
- Contrastive InfoNCE training from two stochastic augmentations of the same spectrum.
- Auxiliary masked patch reconstruction with optional fingerprint-region weighting.
- Lightning-managed training, validation, logging, checkpointing, mixed precision, gradient accumulation, and early stopping.
- Validation/test retrieval metrics with same-compound Recall@K when duplicate compounds exist, otherwise augmented-query vs original-library Recall@K.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Splits

The default split mode is `parent_metadata`: each source folder is split internally,
while each chemical identity inside that source stays in only one split.

```bash
python scripts/prepare_data.py \
  --data_dir data/raw \
  --output data/splits.json \
  --group_by parent_metadata \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

`IRSpectrumDataModule.prepare_data()` can also create `data/splits.json` automatically when raw JSON files exist and the split index is missing.

## Train

```bash
python scripts/train.py --config configs/baseline.yaml
```

Common overrides:

```bash
python scripts/train.py --config configs/full.yaml --batch_size 128 --lr 5e-5
python scripts/train.py --config configs/full.yaml --set model.pooling=cls --set loss.beta=0.1
python scripts/train.py --config configs/full.yaml --resume_from checkpoints/last.ckpt
python scripts/train.py --config configs/baseline.yaml --set wandb.enabled=false
```

## Lightning Debugging

```bash
python scripts/train.py --config configs/baseline.yaml --fast-dev-run --set wandb.enabled=false
python scripts/train.py --config configs/baseline.yaml --overfit-batches 1 --set wandb.enabled=false
python scripts/train.py --config configs/baseline.yaml --limit-train-batches 0.05 --set wandb.enabled=false
```

Use `--fast-dev-run` for a one-batch smoke test, `--overfit-batches` to verify the model can fit a tiny subset, and `--limit-train-batches` for quick partial-epoch checks.

## Evaluate

Lightning checkpoints are loaded through `IRContrastiveModule.load_from_checkpoint()` and evaluated with `Trainer.test()`:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/last.ckpt \
  --config configs/baseline.yaml
```

## Build Retrieval Index

Index building uses direct `model.forward()` inference rather than a Lightning `Trainer`:

```bash
python scripts/build_index.py \
  --checkpoint checkpoints/last.ckpt \
  --config configs/baseline.yaml \
  --split train \
  --output outputs/train_embeddings.npz
```

The `.npz` contains `embeddings`, `ids`, `group_ids`, and `paths`.

## Visualize

```bash
python scripts/visualize_embeddings.py \
  --index outputs/train_embeddings.npz \
  --output outputs/train_tsne.png
```

## Tests

```bash
pytest
```

The tests cover dataset loading, augmentations, model forward shapes, losses, and a Lightning `Trainer(fast_dev_run=True)` smoke path.

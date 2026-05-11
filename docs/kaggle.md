# Kaggle Workflow

This repo is designed so code lives in GitHub and data lives in a Kaggle Dataset.
Do not commit `data/raw`, checkpoints, `wandb/`, or generated indexes.

## 1. Push Code To GitHub

From your local machine:

```bash
git status
git add .gitignore README.md requirements*.txt configs src scripts tests docs
git commit -m "Prepare Kaggle training workflow"
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

If `origin` already exists, skip `git remote add ...`.

## 2. Attach Data In Kaggle

Create or attach a Kaggle Dataset that contains the grouped raw files, for example:

```text
/kaggle/input/ir-spectra/raw/SEA028/*.json
/kaggle/input/ir-spectra/raw/SEA200/*.json
```

If your dataset slug is different, override `data.data_dir` from the command line.

## 3. Clone Or Pull In A Kaggle Notebook

```bash
%cd /kaggle/working
!git clone https://github.com/<USER>/<REPO>.git encoder-train
%cd /kaggle/working/encoder-train
```

For later notebook sessions:

```bash
%cd /kaggle/working/encoder-train
!git pull
```

## 4. Install Runtime Dependencies

Kaggle usually ships with a CUDA-compatible PyTorch build. Install the Kaggle-specific requirements to avoid replacing `torch`:

```bash
!pip install -q -r requirements-kaggle.txt
```

Sanity check:

```bash
!python - <<'PY'
import torch
import pytorch_lightning as pl
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("lightning", pl.__version__)
PY
```

For V100 cloud machines, if PyTorch reports that the minimum supported CUDA capability is 7.5,
replace the preinstalled torch wheel with the V100-compatible requirements:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir -r requirements-v100.txt
```

## 5. Generate Splits

```bash
!python scripts/prepare_data.py \
  --data_dir /kaggle/input/ir-spectra/raw \
  --output /kaggle/working/data/splits.json \
  --group_by parent_metadata \
  --spectrum_length 460
```

`parent_metadata` means each source folder is split internally, but each `(source, compound)` group stays in only one split.

## 6. Fast Debug Run

```bash
!python scripts/train.py \
  --config configs/kaggle.yaml \
  --fast-dev-run \
  --set data.data_dir=/kaggle/input/ir-spectra/raw \
  --set wandb.enabled=false
```

## 7. Train

```bash
!python scripts/train.py \
  --config configs/kaggle.yaml \
  --set data.data_dir=/kaggle/input/ir-spectra/raw
```

Checkpoints are written to:

```text
/kaggle/working/checkpoints/
```

Kaggle persists files in `/kaggle/working` as notebook outputs after the run finishes.

## 8. Resume

```bash
!python scripts/train.py \
  --config configs/kaggle.yaml \
  --set train.resume_from=/kaggle/working/checkpoints/last.ckpt
```

## 9. Build Embedding Index

```bash
!python scripts/build_index.py \
  --checkpoint /kaggle/working/checkpoints/last.ckpt \
  --config configs/kaggle.yaml \
  --split train \
  --output /kaggle/working/train_embeddings.npz
```

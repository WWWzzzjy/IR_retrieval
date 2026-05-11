#!/usr/bin/env bash
set -euo pipefail

# Effect-oriented training preset for the mid-IR retrieval encoder.
# Override any value from the shell, for example:
#   BATCH_SIZE=768 EPOCHS=200 bash scripts/runs/train_effective.sh

CONFIG="${CONFIG:-configs/baseline.yaml}"
DATA_DIR="${DATA_DIR:-data/raw}"
SPLIT_INDEX="${SPLIT_INDEX:-data/splits.json}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/effective}"
RUN_LOG_DIR="${RUN_LOG_DIR:-logs}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
BATCH_SIZE="${BATCH_SIZE:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-150}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_EVERY="${VAL_EVERY:-3}"
LOG_EVERY="${LOG_EVERY:-100}"

EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
PROJECTION_HIDDEN_DIM="${PROJECTION_HIDDEN_DIM:-512}"
TEMPERATURE="${TEMPERATURE:-0.2}"
RECON_BETA="${RECON_BETA:-0.2}"
MASK_RATIO="${MASK_RATIO:-0.2}"
RESUME_FROM="${RESUME_FROM:-null}"

mkdir -p "${RUN_LOG_DIR}"
RUN_LOG="${RUN_LOG:-${RUN_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log}"

if [[ ! -f "${SPLIT_INDEX}" ]]; then
  python scripts/prepare_data.py \
    --data_dir "${DATA_DIR}" \
    --output "${SPLIT_INDEX}" \
    --group_by parent_metadata \
    --spectrum_length 460
fi

echo "Writing terminal log to ${RUN_LOG}"

python scripts/train.py \
  --config "${CONFIG}" \
  --set data.data_dir="${DATA_DIR}" \
  --set data.split_index="${SPLIT_INDEX}" \
  --set data.cache=true \
  --set wandb.enabled="${WANDB_ENABLED}" \
  --set train.output_dir="${OUTPUT_DIR}" \
  --set train.batch_size="${BATCH_SIZE}" \
  --set train.grad_accum="${GRAD_ACCUM}" \
  --set train.lr="${LR}" \
  --set train.num_epochs="${EPOCHS}" \
  --set train.warmup_epochs="${WARMUP_EPOCHS}" \
  --set train.num_workers="${NUM_WORKERS}" \
  --set evaluation.num_workers="${NUM_WORKERS}" \
  --set train.persistent_workers=true \
  --set evaluation.persistent_workers=true \
  --set train.val_every_n_epoch="${VAL_EVERY}" \
  --set train.log_every_n_steps="${LOG_EVERY}" \
  --set model.embedding_dim="${EMBEDDING_DIM}" \
  --set model.projection_hidden_dim="${PROJECTION_HIDDEN_DIM}" \
  --set loss.temperature="${TEMPERATURE}" \
  --set loss.beta="${RECON_BETA}" \
  --set loss.reconstruction.mask_ratio="${MASK_RATIO}" \
  --set augmentation.peak_width.enabled=true \
  --set train.resume_from="${RESUME_FROM}" \
  2>&1 | tee "${RUN_LOG}"

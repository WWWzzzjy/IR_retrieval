#!/usr/bin/env bash
set -euo pipefail

# Evaluate one checkpoint and export self-augmentation retrieval error cases.
# Examples:
#   RUN_NAME=20260512-120000_h256_e256_bs512_lr1e-4 bash scripts/runs/eval.sh
#   CHECKPOINT=checkpoints/effective/my-run/best-epoch=74-val_recall_at_1=0.9711.ckpt bash scripts/runs/eval.sh

CONFIG="${CONFIG:-configs/baseline.yaml}"
DATA_DIR="${DATA_DIR:-data/libs_samples_460}"
SPLIT_INDEX="${SPLIT_INDEX:-data/libs_samples_460_split.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/effective}"
RUN_NAME="20260512-081631_h256_e256_bs512_lr1e-4"
CHECKPOINT="${CHECKPOINT:-}"
ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/eval}"
RUN_LOG_DIR="${RUN_LOG_DIR:-logs}"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOP_K="${TOP_K:-1 5 10}"
NUM_ERROR_CASES="${NUM_ERROR_CASES:-200}"
MAX_ITEMS="${MAX_ITEMS:-null}"
DEVICE="${DEVICE:-auto}"
SKIP_TRAINER_TEST="${SKIP_TRAINER_TEST:-false}"
DATA_CACHE="${DATA_CACHE:-false}"

if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -z "${RUN_NAME}" ]]; then
    echo "Set CHECKPOINT=/path/to/file.ckpt or RUN_NAME=<checkpoint-subdir>." >&2
    exit 1
  fi
  CHECKPOINT="${CHECKPOINT_ROOT}/${RUN_NAME}/last.ckpt"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${SPLIT_INDEX}" ]]; then
  python scripts/prepare_data.py \
    --data_dir "${DATA_DIR}" \
    --output "${SPLIT_INDEX}" \
    --group_by parent_metadata \
    --spectrum_length 460
fi

mkdir -p "${RUN_LOG_DIR}"
CHECKPOINT_RUN_NAME="$(basename "$(dirname "${CHECKPOINT}")")"
RUN_LOG="${RUN_LOG:-${RUN_LOG_DIR}/eval_${CHECKPOINT_RUN_NAME}_${SPLIT}_$(date +%Y%m%d_%H%M%S).log}"

read -r -a TOP_K_ARGS <<< "${TOP_K}"
EXTRA_ARGS=()
if [[ "${MAX_ITEMS}" != "null" && -n "${MAX_ITEMS}" ]]; then
  EXTRA_ARGS+=(--max_items "${MAX_ITEMS}")
fi
if [[ "${DEVICE}" != "auto" && -n "${DEVICE}" ]]; then
  EXTRA_ARGS+=(--device "${DEVICE}")
fi
if [[ "${SKIP_TRAINER_TEST}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_trainer_test)
fi

echo "Evaluating checkpoint: ${CHECKPOINT}"
echo "Writing evaluation log to ${RUN_LOG}"

python scripts/evaluate.py \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --batch_size "${BATCH_SIZE}" \
  --output_dir "${ANALYSIS_DIR}" \
  --top_k "${TOP_K_ARGS[@]}" \
  --num_error_cases "${NUM_ERROR_CASES}" \
  --set data.data_dir="${DATA_DIR}" \
  --set data.split_index="${SPLIT_INDEX}" \
  --set data.cache="${DATA_CACHE}" \
  --set evaluation.num_workers="${NUM_WORKERS}" \
  --set evaluation.persistent_workers=true \
  --set evaluation.retrieval_mode=self_augmentation \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${RUN_LOG}"

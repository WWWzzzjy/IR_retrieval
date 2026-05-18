#!/usr/bin/env bash
set -euo pipefail

# Plot spectra for retrieval error top-k JSON files.
# Examples:
#   RUN_NAME=my-run bash scripts/runs/plot_error_topk.sh
#   RUN_NAME=my-run TOP_K=10 NUM_CASES=50 bash scripts/runs/plot_error_topk.sh
#   ERRORS_JSON=outputs/eval/my-run/test/errors_top5.json bash scripts/runs/plot_error_topk.sh

RUN_NAME="20260512-081631_h256_e256_bs512_lr1e-4"
SPLIT="${SPLIT:-test}"
TOP_K="${TOP_K:-5}"
ERRORS_JSON="${ERRORS_JSON:-}"
EVAL_DIR="${EVAL_DIR:-outputs/eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/error_plots}"
DATA_DIR="${DATA_DIR:-data/raw}"

NUM_CASES="${NUM_CASES:-20}"
START="${START:-0}"
DPI="${DPI:-180}"
FIG_WIDTH="${FIG_WIDTH:-12}"
ROW_HEIGHT="${ROW_HEIGHT:-2.0}"
INVERT_X="${INVERT_X:-false}"

if [[ -z "${ERRORS_JSON}" ]]; then
  if [[ -z "${RUN_NAME}" ]]; then
    echo "Set ERRORS_JSON=/path/to/errors_top5.json or RUN_NAME=<eval-run-name>." >&2
    exit 1
  fi
  ERRORS_JSON="${EVAL_DIR}/${RUN_NAME}/${SPLIT}/errors_top${TOP_K}.json"
fi

if [[ ! -f "${ERRORS_JSON}" ]]; then
  echo "Error JSON not found: ${ERRORS_JSON}" >&2
  exit 1
fi

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="$(basename "$(dirname "$(dirname "${ERRORS_JSON}")")")"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${SPLIT}/top${TOP_K}}"
EXTRA_ARGS=()
if [[ "${INVERT_X}" == "true" ]]; then
  EXTRA_ARGS+=(--invert_x)
fi

echo "Reading errors from ${ERRORS_JSON}"
echo "Writing plots to ${OUTPUT_DIR}"

python scripts/visualize_error_topk.py \
  --errors "${ERRORS_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --data_dir "${DATA_DIR}" \
  --num_cases "${NUM_CASES}" \
  --start "${START}" \
  --dpi "${DPI}" \
  --fig_width "${FIG_WIDTH}" \
  --row_height "${ROW_HEIGHT}" \
  "${EXTRA_ARGS[@]}"

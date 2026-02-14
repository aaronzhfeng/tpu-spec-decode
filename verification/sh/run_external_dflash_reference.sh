#!/usr/bin/env bash
set -euo pipefail

# Runs the external DFlash benchmark script with configurable defaults.
# This reproduces the reference workflow shape from external/dflash.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/verification/outputs/external"
mkdir -p "${LOG_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
DRAFT_NAME="${DRAFT_NAME:-z-lab/Qwen3-4B-DFlash-b16}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"

cd "${ROOT_DIR}/deps/dflash"

echo "Running external DFlash reference benchmark..."
echo "model=${MODEL_NAME} draft=${DRAFT_NAME} dataset=${DATASET} samples=${MAX_SAMPLES}"

python benchmark.py \
  --block-size "${BLOCK_SIZE}" \
  --dataset "${DATASET}" \
  --max-samples "${MAX_SAMPLES}" \
  --model-name-or-path "${MODEL_NAME}" \
  --draft-name-or-path "${DRAFT_NAME}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  2>&1 | tee "${LOG_DIR}/external_${DATASET}.log"


#!/usr/bin/env bash
set -euo pipefail

# Smoke test entrypoint for local TPU-inference DFlash integration.
# Runs a tiny synthetic benchmark pass to validate method wiring and runtime stability.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/verification/outputs/tpu_smoke"
mkdir -p "${OUT_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
DRAFT_NAME="${DRAFT_NAME:-z-lab/Qwen3-8B-DFlash-b16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
TP_SIZE="${TP_SIZE:-1}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/smoke_${TIMESTAMP}.json}"

# Prefer local tpu-inference source if needed by the environment.
export PYTHONPATH="${ROOT_DIR}/deps/tpu-inference:${PYTHONPATH:-}"

echo "Running TPU DFlash smoke"
echo "model=${MODEL_NAME} draft=${DRAFT_NAME} out_json=${OUT_JSON}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; exiting before execution."
  exit 0
fi

python "${ROOT_DIR}/verification/py/run_tpu_dflash_eval.py" \
  --model "${MODEL_NAME}" \
  --draft-model "${DRAFT_NAME}" \
  --datasets "gsm8k" \
  --prompt-source "synthetic" \
  --max-samples 1 \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --num-speculative-tokens "${NUM_SPEC_TOKENS}" \
  --warmup-prompts 1 \
  --inter-run-sleep-sec 0 \
  --skip-baseline \
  --out-json "${OUT_JSON}"

echo "Smoke result JSON: ${OUT_JSON}"

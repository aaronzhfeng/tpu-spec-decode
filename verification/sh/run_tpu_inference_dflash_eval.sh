#!/usr/bin/env bash
set -euo pipefail

# Evaluation entrypoint for local TPU-inference DFlash runs.
# Produces JSON for comparison with verification/py/compare_results_to_baseline.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/verification/outputs/tpu_eval"
mkdir -p "${OUT_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
DRAFT_NAME="${DRAFT_NAME:-z-lab/Qwen3-8B-DFlash-b16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CATEGORY="${CATEGORY:-math}"  # math|code|chat
MAX_SAMPLES="${MAX_SAMPLES:-16}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-16}"
DRAFT_TP_SIZE="${DRAFT_TP_SIZE:-1}"
PROMPT_SOURCE="${PROMPT_SOURCE:-external}"  # external|synthetic
WARMUP_PROMPTS="${WARMUP_PROMPTS:-1}"
INTER_RUN_SLEEP_SEC="${INTER_RUN_SLEEP_SEC:-5}"
MIN_RETENTION="${MIN_RETENTION:-0.70}"
MODEL_KEY="${MODEL_KEY:-Qwen3-8B-DFlash-b16}"
DATASETS="${DATASETS:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/results_${CATEGORY}_t${TEMPERATURE}_${TIMESTAMP}.json}"

# Prefer local tpu-inference source if needed by the environment.
export PYTHONPATH="${ROOT_DIR}/tpu-inference:${PYTHONPATH:-}"

echo "Running TPU DFlash eval"
echo "model=${MODEL_NAME} draft=${DRAFT_NAME} category=${CATEGORY} temp=${TEMPERATURE} prompt_source=${PROMPT_SOURCE}"
echo "out_json=${OUT_JSON}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; exiting before execution."
  exit 0
fi

python "${ROOT_DIR}/verification/py/run_tpu_dflash_eval.py" \
  --model "${MODEL_NAME}" \
  --draft-model "${DRAFT_NAME}" \
  --category "${CATEGORY}" \
  --datasets "${DATASETS}" \
  --prompt-source "${PROMPT_SOURCE}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --num-speculative-tokens "${NUM_SPEC_TOKENS}" \
  --draft-tensor-parallel-size "${DRAFT_TP_SIZE}" \
  --warmup-prompts "${WARMUP_PROMPTS}" \
  --inter-run-sleep-sec "${INTER_RUN_SLEEP_SEC}" \
  --out-json "${OUT_JSON}"

python "${ROOT_DIR}/verification/py/compare_results_to_baseline.py" \
  --results-json "${OUT_JSON}" \
  --category "${CATEGORY}" \
  --model-key "${MODEL_KEY}" \
  --temperature "${TEMPERATURE%.*}" \
  --metric speedup \
  --min-retention "${MIN_RETENTION}"

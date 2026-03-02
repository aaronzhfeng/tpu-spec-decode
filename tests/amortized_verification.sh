#!/usr/bin/env bash
# tests/amortized_verification.sh — Run amortized verification experiment.
#
# Part 1: Microbenchmark — target model verify latency vs query token count
# Part 2: Multi-block speculation — draft 2+ blocks, verify together
#
# Usage:
#   bash tests/amortized_verification.sh
#   MAX_SAMPLES=6 bash tests/amortized_verification.sh

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-1}"
MICRO_TRIALS="${MICRO_TRIALS:-10}"
MICRO_WARMUP="${MICRO_WARMUP:-3}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/amortized_verification_${DATASET}.json}"

echo "==========================================="
echo "  Amortized Verification Experiment"
echo "==========================================="
echo "  target:  ${TARGET_MODEL}"
echo "  draft:   ${DRAFT_MODEL}"
echo "  dataset: ${DATASET}"
echo "  samples: ${MAX_SAMPLES}"
echo "  tokens:  ${MAX_NEW_TOKENS}"
echo "  micro:   ${MICRO_TRIALS} trials, ${MICRO_WARMUP} warmup"
echo "==========================================="
echo ""

require_dir "tpu-inference" "${TPU_INFERENCE_DIR}"

docker_exec "
  python3 /workspace/tpu-spec-decode/benchmarks/amortized_verification.py \
    --target-model ${TARGET_MODEL} \
    --draft-model ${DRAFT_MODEL} \
    --dataset ${DATASET} \
    --max-samples ${MAX_SAMPLES} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max-model-len ${MAX_MODEL_LEN} \
    --warmup-samples ${WARMUP_SAMPLES} \
    --micro-trials ${MICRO_TRIALS} \
    --micro-warmup ${MICRO_WARMUP} \
    --output-json ${OUTPUT_JSON} \
    $*
"

echo ""
ok "Amortized verification experiment complete."

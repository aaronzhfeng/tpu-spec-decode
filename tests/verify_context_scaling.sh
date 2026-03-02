#!/usr/bin/env bash
# tests/verify_context_scaling.sh — Experiment B: TPU verify vs context length
#
# Measures verification latency at K=16,64,128 across context lengths
# 64, 256, 512, 1024. Answers whether the flat-K property holds at
# longer contexts where attention becomes a larger fraction.
#
# Usage:
#   bash tests/verify_context_scaling.sh
#   TRIALS=50 WARMUP=10 bash tests/verify_context_scaling.sh

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
TRIALS="${TRIALS:-20}"
WARMUP="${WARMUP:-5}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-64,256,512,1024}"
K_VALUES="${K_VALUES:-16,64,128}"

echo "==========================================="
echo "  Experiment B: Verify Context Scaling"
echo "==========================================="
echo "  target:   ${TARGET_MODEL}"
echo "  draft:    ${DRAFT_MODEL}"
echo "  contexts: ${CONTEXT_LENGTHS}"
echo "  K values: ${K_VALUES}"
echo "  trials:   ${TRIALS}"
echo "  warmup:   ${WARMUP}"
echo "==========================================="
echo ""

require_dir "tpu-inference" "${TPU_INFERENCE_DIR}"

docker_exec "
  python3 /workspace/tpu-spec-decode/benchmarks/verify_context_scaling.py \
    --target-model ${TARGET_MODEL} \
    --draft-model ${DRAFT_MODEL} \
    --context-lengths ${CONTEXT_LENGTHS} \
    --k-values ${K_VALUES} \
    --trials ${TRIALS} \
    --warmup ${WARMUP} \
    --output-json /workspace/tpu-spec-decode/results/verify_context_scaling.json
"

echo ""
ok "Verify context scaling benchmark complete."
echo "Results saved to results/verify_context_scaling.json"

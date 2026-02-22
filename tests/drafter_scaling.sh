#!/usr/bin/env bash
# tests/drafter_scaling.sh — Run drafter forward-pass scaling benchmark.
#
# Measures raw matmul latency at K=16,32,64,128 for DFlash-sized and
# target-model-sized FFN layers, plus attention Q×K^T scaling.
# Validates that MXU tile amortization applies to the drafter too.
#
# Usage:
#   bash tests/drafter_scaling.sh

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
TRIALS="${TRIALS:-20}"
WARMUP="${WARMUP:-5}"

echo "==========================================="
echo "  Drafter Forward-Pass Scaling Benchmark"
echo "==========================================="
echo "  target:  ${TARGET_MODEL}"
echo "  draft:   ${DRAFT_MODEL}"
echo "  trials:  ${TRIALS}"
echo "  warmup:  ${WARMUP}"
echo "==========================================="
echo ""

require_dir "tpu-inference" "${TPU_INFERENCE_DIR}"

docker_exec "
  python3 /workspace/tpu-spec-decode/benchmarks/drafter_scaling.py \
    --target-model ${TARGET_MODEL} \
    --draft-model ${DRAFT_MODEL} \
    --trials ${TRIALS} \
    --warmup ${WARMUP} \
    $*
"

echo ""
ok "Drafter scaling benchmark complete."

#!/usr/bin/env bash
# tests/fused_benchmark.sh — Compare fused vs unfused DFlash pipeline on TPU.
#
# Direction 1 optimization: fused JIT functions that eliminate host-device
# roundtrips between draft/verify/acceptance phases.
#
# Usage:
#   bash tests/fused_benchmark.sh                         # Default: gsm8k, 4 samples
#   DATASET=math500 bash tests/fused_benchmark.sh         # Different dataset

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/fused_${DATASET}.json}"

echo "==========================================="
echo "  Fused vs Unfused DFlash (Direction 1)"
echo "==========================================="
echo "  target:  ${TARGET_MODEL}"
echo "  draft:   ${DRAFT_MODEL}"
echo "  dataset: ${DATASET}"
echo "  samples: ${MAX_SAMPLES}"
echo "  tokens:  ${MAX_NEW_TOKENS}"
echo "==========================================="
echo ""

require_dir "tpu-inference" "${TPU_INFERENCE_DIR}"

docker_exec "
  python3 /workspace/tpu-spec-decode/benchmarks/fused_dflash.py \
    --target-model ${TARGET_MODEL} \
    --draft-model ${DRAFT_MODEL} \
    --dataset ${DATASET} \
    --max-samples ${MAX_SAMPLES} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max-model-len ${MAX_MODEL_LEN} \
    --output-json ${OUTPUT_JSON} \
    $*
"

echo ""
ok "Fused benchmark complete."

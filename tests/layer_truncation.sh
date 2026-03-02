#!/usr/bin/env bash
# tests/layer_truncation.sh — Run layer-truncated verification experiment.
#
# Captures hidden states at every target model layer during verification,
# computes logits at truncation points (6, 12, 18, 24, 30, 36), and
# measures simulated acceptance rate (tau) at each point.
#
# Usage:
#   bash tests/layer_truncation.sh
#   MAX_SAMPLES=8 bash tests/layer_truncation.sh

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
WARMUP="${WARMUP:-1}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/layer_truncation_${DATASET}.json}"

echo "==========================================="
echo "  Layer-Truncated Verification Experiment"
echo "==========================================="
echo "  target:  ${TARGET_MODEL}"
echo "  draft:   ${DRAFT_MODEL}"
echo "  dataset: ${DATASET}"
echo "  samples: ${MAX_SAMPLES}"
echo "  tokens:  ${MAX_NEW_TOKENS}"
echo "  warmup:  ${WARMUP}"
echo "==========================================="
echo ""

require_dir "tpu-inference" "${TPU_INFERENCE_DIR}"

docker_exec "
  python3 /workspace/tpu-spec-decode/benchmarks/layer_truncation.py \
    --target-model ${TARGET_MODEL} \
    --draft-model ${DRAFT_MODEL} \
    --dataset ${DATASET} \
    --max-samples ${MAX_SAMPLES} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max-model-len ${MAX_MODEL_LEN} \
    --warmup ${WARMUP} \
    --output-json ${OUTPUT_JSON} \
    $*
"

echo ""
ok "Layer truncation experiment complete."

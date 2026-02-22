#!/usr/bin/env bash
# tests/tree_speculation.sh — Run tree speculation experiment.
#
# Drafts K candidate blocks with different first tokens,
# verifies each, takes the best. Measures actual throughput
# and theoretical throughput with amortized (batched) verification.
#
# Usage:
#   bash tests/tree_speculation.sh
#   K_CANDIDATES=1,2,4,8 bash tests/tree_speculation.sh

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-1}"
K_CANDIDATES="${K_CANDIDATES:-1,2,4}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/tree_speculation_${DATASET}.json}"

echo "==========================================="
echo "  Tree Speculation Experiment"
echo "==========================================="
echo "  target:      ${TARGET_MODEL}"
echo "  draft:       ${DRAFT_MODEL}"
echo "  dataset:     ${DATASET}"
echo "  samples:     ${MAX_SAMPLES}"
echo "  tokens:      ${MAX_NEW_TOKENS}"
echo "  candidates:  ${K_CANDIDATES}"
echo "==========================================="
echo ""

require_dir "tpu-inference" "${TPU_INFERENCE_DIR}"

docker_exec "
  python3 /workspace/tpu-spec-decode/benchmarks/tree_speculation.py \
    --target-model ${TARGET_MODEL} \
    --draft-model ${DRAFT_MODEL} \
    --dataset ${DATASET} \
    --max-samples ${MAX_SAMPLES} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max-model-len ${MAX_MODEL_LEN} \
    --warmup-samples ${WARMUP_SAMPLES} \
    --k-candidates ${K_CANDIDATES} \
    --output-json ${OUTPUT_JSON} \
    $*
"

echo ""
ok "Tree speculation experiment complete."

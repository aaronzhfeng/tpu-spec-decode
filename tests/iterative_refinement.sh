#!/usr/bin/env bash
# tests/iterative_refinement.sh — Test iterative refinement drafting on TPU.
#
# Direction 2 experiment: run DFlash with k=0,1,2,3 refinement steps and
# compare tau, latency, and net throughput.
#
# Usage:
#   bash tests/iterative_refinement.sh                          # Default: gsm8k, k=0,1,2,3
#   bash tests/iterative_refinement.sh --refinement-steps 0 1   # Only test k=0 and k=1
#   DATASET=math500 bash tests/iterative_refinement.sh          # Different dataset
#   bash tests/iterative_refinement.sh --run-baseline           # Include AR baseline

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/refinement_${DATASET}.json}"

echo "==========================================="
echo "  Iterative Refinement (Direction 2)"
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
  python3 /workspace/tpu-spec-decode/benchmarks/iterative_refinement.py \
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
ok "Iterative refinement benchmark complete."

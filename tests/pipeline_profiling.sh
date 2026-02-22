#!/usr/bin/env bash
# tests/pipeline_profiling.sh — Profile the DFlash spec-decode pipeline on TPU.
#
# Direction 1 experiment: fine-grained timing of every phase in the
# draft-verify-accept loop to identify where pipeline overhead lives.
#
# Usage:
#   bash tests/pipeline_profiling.sh                        # Default: gsm8k, 4 samples
#   bash tests/pipeline_profiling.sh --max-samples 8        # More samples
#   DATASET=math500 bash tests/pipeline_profiling.sh        # Different dataset

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/profiling_${DATASET}.json}"

echo "==========================================="
echo "  Pipeline Profiling (Direction 1)"
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
  python3 /workspace/tpu-spec-decode/benchmarks/pipeline_profiling.py \
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
ok "Pipeline profiling complete."

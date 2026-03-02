#!/usr/bin/env bash
# tests/ablation_study.sh — Run ablation study to isolate optimization targets.
#
# Tests 4 hypotheses:
#   1. LM Head microbenchmark (matmul cost at various batch sizes)
#   2. Host loop overhead (Python/jnp/transfer costs)
#   3. DFlash with/without draft LM head (real impact measurement)
#   4. Component overlap analysis (natural JAX pipelining)
#
# Usage:
#   bash tests/ablation_study.sh
#   MAX_SAMPLES=5 bash tests/ablation_study.sh

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/ablation_${DATASET}.json}"

echo "==========================================="
echo "  Ablation Study (Direction 1)"
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
  python3 /workspace/tpu-spec-decode/benchmarks/ablation_study.py \
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
ok "Ablation study complete."

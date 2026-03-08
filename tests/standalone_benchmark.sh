#!/usr/bin/env bash
# tests/standalone_benchmark.sh — Run the standalone DFlash benchmark (no vLLM engine).
#
# Usage:
#   bash tests/standalone_benchmark.sh                  # Default: gsm8k, 3 samples, 128 tokens
#   bash tests/standalone_benchmark.sh --max-samples 8  # Override args
#
# All extra args are passed through to standalone_dflash.py.
#
# Flax: Docker image default is 0.11.1 (v4). For v5p (nnx.List), set before running:
#   FLAX_VERSION=0.12.4 bash tests/standalone_benchmark.sh

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

# Default model paths (HF cache inside Docker)
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

echo "==========================================="
echo "  Standalone DFlash Benchmark"
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
  python3 /workspace/tpu-spec-decode/benchmarks/standalone_dflash.py \
    --target-model ${TARGET_MODEL} \
    --draft-model ${DRAFT_MODEL} \
    --dataset ${DATASET} \
    --max-samples ${MAX_SAMPLES} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max-model-len ${MAX_MODEL_LEN} \
    $*
"

echo ""
ok "Standalone benchmark complete."

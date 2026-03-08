#!/usr/bin/env bash
# tests/run_standalone_all_v5p.sh — Run standalone_dflash.py on 8 datasets (v5p batch).
#
# Batch variant of standalone_benchmark.sh. Outputs to HOST_OUTPUT_DIR/v5p-pr/.
#
# Usage:
#   bash tests/run_standalone_all_v5p.sh
#
# Uses pr-ready/pr and pr-ready/vllm-lkg (set by verify_all_wrappers v5p).
# For direct runs, export: TPU_INFERENCE_DIR=.../pr-ready/pr VLLM_DIR=.../pr-ready/vllm-lkg FLAX_VERSION=0.12.4

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

DATASETS=(math500 aime24 aime25 humaneval mbpp swe-bench mt-bench alpaca)
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
TOTAL=${#DATASETS[@]}
START_TIME=$(date +%s)

# Ensure output dirs are writable (docker_run will use these)
ensure_writable_dir HOST_OUTPUT_DIR "/dev/shm/${USER:-user}-dflash-outputs"

echo "========================================"
echo "  V5P Standalone Benchmark Suite"
echo "  Started: $(date)"
echo "  Datasets: ${DATASETS[*]}"
echo "  Output:   ${HOST_OUTPUT_DIR}/v5p-pr/"
echo "========================================"

require_dir "tpu-inference" "${TPU_INFERENCE_DIR}"

# Build inline loop to run all datasets in one container (avoids 8x model load)
DS_LIST="${DATASETS[*]}"
docker_exec "
  pip install flax==0.12.4 --quiet 2>/dev/null || true
  mkdir -p /output/v5p-pr
  i=0
  for ds in ${DS_LIST}; do
    i=\$((i + 1))
    echo ''
    echo \"[\$i/${TOTAL}] Running \$ds ... (\$(date +%H:%M:%S))\"
    echo \"----------------------------------------\"
    python3 /workspace/tpu-spec-decode/benchmarks/standalone_dflash.py \
      --target-model ${TARGET_MODEL} \
      --draft-model ${DRAFT_MODEL} \
      --dataset \$ds \
      --max-samples ${MAX_SAMPLES} \
      --max-new-tokens ${MAX_NEW_TOKENS} \
      --max-model-len ${MAX_MODEL_LEN} \
      --temperature 0.0 \
      --warmup 1 \
      --output-json /output/v5p-pr/v5p_pr_\${ds}.json
    echo \"[\$i/${TOTAL}] \$ds DONE\"
  done
"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "========================================"
echo "  All ${TOTAL} benchmarks complete!"
echo "  Total time: ${MINS}m ${SECS}s"
echo "  Results in: ${HOST_OUTPUT_DIR}/v5p-pr/"
echo "========================================"
ls -la "${HOST_OUTPUT_DIR}/v5p-pr/" 2>/dev/null || true

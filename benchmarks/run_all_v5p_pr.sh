#!/bin/bash
# Run all 9 benchmarks using pr-ready/pr (tpu-inference pr/dflash branch)
# Output goes to /dev/shm/dflash-test-outputs/v5p-pr/

set -e

DATASETS=(math500 aime24 aime25 humaneval mbpp swe-bench mt-bench alpaca)
OUTPUT_DIR=/dev/shm/dflash-test-outputs/v5p-pr
TOTAL=${#DATASETS[@]}
START_TIME=$(date +%s)

echo "========================================"
echo "  V5P PR Benchmark Suite (pr/dflash)"
echo "  Started: $(date)"
echo "  Datasets: ${DATASETS[*]}"
echo "========================================"

for i in "${!DATASETS[@]}"; do
  ds=${DATASETS[$i]}
  num=$((i + 1))
  echo ""
  echo "[$num/$TOTAL] Running $ds ... ($(date +%H:%M:%S))"
  echo "----------------------------------------"

  sudo docker run --rm --privileged --network host --ipc host \
    --tmpfs "/tmp:rw,size=80g" \
    -e HF_HOME=/hf-cache \
    -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub \
    -e XDG_CACHE_HOME=/hf-cache/xdg \
    -e JAX_COMPILATION_CACHE_DIR=/hf-cache/jax \
    -e TMPDIR=/tmp \
    -e PYTHONNOUSERSITE=1 \
    -e PYTHONPATH=/workspace/tpu-spec-decode/pr-ready/vllm-lkg:/workspace/tpu-spec-decode/pr-ready/pr \
    -v /dev/shm/hf-cache:/hf-cache \
    -v /dev/shm/dflash-test-outputs:/output \
    -v /dev/shm/tpu-logs:/tmp/tpu_logs \
    -v /home/aaronfeng/tpu-spec-decode:/workspace/tpu-spec-decode \
    -w /workspace/tpu-spec-decode \
    vllm/vllm-tpu:latest \
    bash -c "pip install flax==0.12.4 --quiet 2>/dev/null && python3 benchmarks/standalone_dflash.py \
      --target-model Qwen/Qwen3-4B \
      --draft-model z-lab/Qwen3-4B-DFlash-b16 \
      --dataset "$ds" \
      --max-samples 8 \
      --max-new-tokens 256 \
      --max-model-len 2048 \
      --temperature 0.0 \
      --warmup 1 \
      --output-json /output/v5p-pr/v5p_pr_${ds}.json"

  echo "[$num/$TOTAL] $ds DONE"
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "========================================"
echo "  All $TOTAL benchmarks complete!"
echo "  Total time: ${MINS}m ${SECS}s"
echo "  Results in: $OUTPUT_DIR/"
echo "========================================"
ls -la "$OUTPUT_DIR/"

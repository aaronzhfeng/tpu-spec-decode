#!/bin/bash
# Run all 9 benchmarks inside Docker and save replay data.
# Usage: bash visualizations/scripts/run_all_replay.sh

set -e

DATASETS="gsm8k math500 aime24 aime25 humaneval mbpp mt-bench alpaca swe-bench"
SAMPLES=3
MAX_TOKENS=256
OUTPUT_DIR="visualizations/output/replay"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " DFlash Replay Capture — All 9 Benchmarks"
echo " Samples per dataset: $SAMPLES"
echo "=============================================="

for ds in $DATASETS; do
    echo ""
    echo ">>> $ds"
    sudo docker run --rm --privileged --net=host --shm-size=16g \
        -v /dev/shm/hf-cache:/root/.cache/huggingface \
        -v /home/aaronfeng/tpu-spec-decode:/workspace \
        -w /workspace \
        -e PYTHONPATH=pr-ready/vllm-lkg:pr-ready/pr_dflash \
        -e VLLM_XLA_CACHE_PATH=/root/.cache/vllm_xla_cache \
        -e HF_HOME=/root/.cache/huggingface \
        vllm/vllm-tpu:latest \
        bash -c "pip install -q flax==0.12.4 2>/dev/null && python benchmarks/standalone_dflash.py \
            --target-model Qwen/Qwen3-4B \
            --draft-model z-lab/Qwen3-4B-DFlash-b16 \
            --dataset $ds \
            --max-samples $SAMPLES \
            --max-new-tokens $MAX_TOKENS \
            --output-json $OUTPUT_DIR/raw_${ds}.json"
    echo "<<< $ds done"
done

echo ""
echo "=============================================="
echo " All benchmarks complete. Raw JSONs in $OUTPUT_DIR/"
echo "=============================================="

# Post-process to replay format
echo ""
echo "Post-processing to replay format..."
python visualizations/scripts/capture_replay.py --all --skip-benchmark

echo "Done."

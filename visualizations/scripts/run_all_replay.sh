#!/bin/bash
# Run replay benchmarks inside Docker on TPU and save replay data.
# Focuses on datasets that produce long generations for better visualization.
#
# Usage: bash visualizations/scripts/run_all_replay.sh
#        bash visualizations/scripts/run_all_replay.sh aime24   # single dataset

set -e

# Datasets that produce long chain-of-thought reasoning (best for visualization)
DATASETS="${1:-aime24 aime25 math500 mt-bench swe-bench}"
SAMPLES=5
MAX_TOKENS=2048
OUTPUT_DIR="visualizations/output/replay"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " DFlash Replay Capture"
echo " Datasets: $DATASETS"
echo " Samples per dataset: $SAMPLES"
echo " Max tokens: $MAX_TOKENS"
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
        bash -c "pip install -q flax==0.12.2 2>/dev/null && python benchmarks/standalone_dflash.py \
            --target-model Qwen/Qwen3-4B \
            --draft-model z-lab/Qwen3-4B-DFlash-b16 \
            --dataset $ds \
            --max-samples $SAMPLES \
            --max-new-tokens $MAX_TOKENS \
            --max-model-len 4096 \
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
python visualizations/scripts/capture_replay.py --all --skip-benchmark --max-new-tokens $MAX_TOKENS

echo "Done."

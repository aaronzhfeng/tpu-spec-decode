#!/usr/bin/env bash
# Run all V14 GPU verification experiments sequentially.
# Usage: bash benchmarks/gpu/run_all.sh
# Run from repo root (tpu-spec-decode/). Activate venv first.
#
# Total estimated time: ~40 min on H100, ~90 min on RTX 4090.

set -euo pipefail

RESULTS_DIR="results/v14_h100"
mkdir -p "$RESULTS_DIR"

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
echo "=== V14 GPU Verification Suite ==="
echo "GPU: $GPU_NAME"
echo "Results: $RESULTS_DIR/"
echo ""

# V14a: Full forward pass K=16,128 at L=64,256,512,1024 (replicates V7)
echo "--- V14a: Full forward pass (V7 replication) ---"
python3 benchmarks/gpu_verify_full.py \
    --k-values 16,128 \
    --context-lengths 64,256,512,1024 \
    --trials 50 \
    --output-json "$RESULTS_DIR/v14a_full_forward.json"
echo ""

# V14b: Context scaling L=256-2048 (replicates V12)
echo "--- V14b: Context scaling (V12 replication) ---"
python3 benchmarks/gpu_verify_context_scaling.py \
    --context-lengths 256,512,1024,2048 \
    --trials 30 \
    --output-json "$RESULTS_DIR/v14b_context_scaling.json"
echo ""

# V14c: Fine-grained K sweep (replicates V13)
echo "--- V14c: K-sweep fine-grained (V13 replication) ---"
python3 benchmarks/gpu_verify_full.py \
    --k-values 16,32,48,64,80,96,112,128 \
    --context-lengths 256,1024 \
    --trials 50 \
    --output-json "$RESULTS_DIR/v14c_k_sweep.json"
echo ""

# V14d: Isolated matmul scaling (replicates Doc 42)
echo "--- V14d: Isolated matmul scaling ---"
python3 benchmarks/gpu_matmul_scaling.py \
    --output-json "$RESULTS_DIR/v14d_matmul.json"
echo ""

# V14e: Draft speed (DFlash K=16 vs K=128)
echo "--- V14e: Draft speed ---"
python3 benchmarks/gpu_draft_speed.py \
    --output-json "$RESULTS_DIR/v14e_draft_speed.json"
echo ""

# V14f: Component decomposition (replicates V10)
echo "--- V14f: Component decomposition (V10 replication) ---"
python3 benchmarks/gpu_forward_decomposition.py \
    --output-json "$RESULTS_DIR/v14f_decomposition.json"
echo ""

echo "=== All V14 experiments complete ==="
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Quick summary:"
echo "  V14a (full forward):      $RESULTS_DIR/v14a_full_forward.json"
echo "  V14b (context scaling):   $RESULTS_DIR/v14b_context_scaling.json"
echo "  V14c (K-sweep):           $RESULTS_DIR/v14c_k_sweep.json"
echo "  V14d (matmul):            $RESULTS_DIR/v14d_matmul.json"
echo "  V14e (draft speed):       $RESULTS_DIR/v14e_draft_speed.json"
echo "  V14f (decomposition):     $RESULTS_DIR/v14f_decomposition.json"

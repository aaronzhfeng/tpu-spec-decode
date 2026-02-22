# Doc 38: GPU Matmul Scaling Comparison — Execution Plan

## Goal

Run the same matmul scaling benchmark on a CUDA GPU (A100, H100, or similar) to produce a direct apples-to-apples comparison with our TPU v4-8 results (Doc 37). This proves that the flat scaling we observe is TPU-specific (MXU tile amortization) and does not occur on GPU hardware.

## Why This Matters

Doc 34 cites SmartSpec (ICLR 2025) which shows GPU verification scales linearly. But that's someone else's measurement on a different workload. Running our own benchmark with **identical dimensions, identical K values, identical methodology** produces a much stronger claim:

> "On the same matmul dimensions (Qwen3-4B), GPU latency grows ~linearly from K=16 to K=128 while TPU latency is flat. The difference is architectural: TPU MXU processes 128×128 bf16 tiles as atomic units; GPU CUDA/Tensor cores process each row independently."

This is the key evidence for the paper's contribution.

---

## Context for the New Agent

### What we're measuring

Raw matrix multiplication latency at K=16, 32, 64, 128 query tokens for three operation types:

1. **DFlash FFN matmuls** — (K, 2560) × (2560, 9728) gate/up/down projections, 5 layers
2. **Target model FFN matmuls** — same dimensions, 36 layers
3. **Attention Q×K^T** — (32 heads, K, 80) × (32 heads, 80, KV_len), at KV lengths 64/256/512/1024

These dimensions come from Qwen3-4B (target model) and its DFlash drafter (Qwen3-4B-DFlash-b16). Both share hidden_size=2560, intermediate_size=9728, 32 attention heads, head_dim=80.

### What we already measured on TPU v4-8 (Doc 37)

All operations are **flat** from K=16 to K=128:

```
Component               K=16        K=128       Ratio
─────────────────────────────────────────────────────
DFlash FFN (5 layers)   1.637ms     1.556ms     0.95×
Target FFN (36 layers)  11.871ms    11.384ms    0.96×
Attention Q×K^T (KV=256) 0.202ms   0.196ms     0.97×
```

### What we expect on GPU

Linear scaling — K=128 should cost ~4-8× more than K=16 for FFN matmuls, and proportionally more for attention. This matches SmartSpec's published GPU measurements.

---

## The Benchmark Script

**File:** `benchmarks/gpu_matmul_scaling.py`

This is a **completely standalone** script. It has:
- **ZERO dependencies** beyond PyTorch (no vllm, no tpu-inference, no JAX, no HuggingFace)
- All model dimensions hardcoded (no model loading needed)
- Same K values, same operation structure, same output format as the TPU benchmark
- Optional `--output-json` flag to save structured results
- Built-in comparison against TPU reference numbers in the summary

### How to run

```bash
# Only dependency
pip install torch

# Run with defaults (20 trials, 5 warmup)
python benchmarks/gpu_matmul_scaling.py

# Run with more trials for cleaner numbers
python benchmarks/gpu_matmul_scaling.py --trials 50 --warmup 10

# Save results as JSON
python benchmarks/gpu_matmul_scaling.py --output-json gpu_results.json
```

### What the script does

1. Creates weight matrices matching Qwen3-4B dimensions in bf16 on GPU
2. For each K in [16, 32, 64, 128]:
   - Runs warmup iterations
   - Times `gate = x @ W_gate; up = x @ W_up; h = gate * up; out = h @ W_down` with `torch.cuda.synchronize()`
   - Reports per-layer latency, full-model latency (×num_layers), and ratio vs K=16
3. Repeats for target model dimensions
4. Tests attention Q×K^T at multiple KV cache lengths
5. Prints summary comparing GPU ratios against TPU reference values

### Expected output format

```
PART 1: RAW MATMUL SCALING (DFlash dimensions)
    K | Per-layer (ms) |  Full model (ms) |  vs K=16
  -------------------------------------------------------
   16 |      X.XXX ± X.XXX |            X.XXX |    1.00x
   32 |      X.XXX ± X.XXX |            X.XXX |    ~2.00x  ← expect linear
   64 |      X.XXX ± X.XXX |            X.XXX |    ~4.00x
  128 |      X.XXX ± X.XXX |            X.XXX |    ~8.00x
```

The summary section automatically compares GPU results against TPU Doc 37 numbers.

---

## Setup Instructions

### Option A: Cloud GPU (Recommended)

Any NVIDIA GPU with bf16 support works. Minimum: T4 (Turing). Ideal: A100 or H100 for relevance to serving.

```bash
# 1. Get a GPU instance (Colab, Lambda Labs, RunPod, etc.)

# 2. Copy just the benchmark script (no repo clone needed)
# The script is fully self-contained at:
#   benchmarks/gpu_matmul_scaling.py

# 3. Install PyTorch
pip install torch

# 4. Run
python gpu_matmul_scaling.py --output-json gpu_results.json
```

### Option B: Colab (Quickest)

1. Open Google Colab, select a GPU runtime (T4 free, or A100 with Pro)
2. Upload `benchmarks/gpu_matmul_scaling.py`
3. Run: `!python gpu_matmul_scaling.py --output-json gpu_results.json`
4. Download `gpu_results.json`

### Option C: From this repo on a GPU machine

```bash
git clone <this-repo>
cd tpu-spec-decode
pip install torch
python benchmarks/gpu_matmul_scaling.py --output-json gpu_results.json
```

---

## After Running: What to Do With Results

### 1. Bring results back

Copy the terminal output and/or the JSON file back to this repo. Save as:
```
results/gpu_matmul_scaling_<gpu_name>.json
```

### 2. Expected comparison table for the paper

| Component | TPU v4-8 (K=128 vs K=16) | GPU (K=128 vs K=16) |
|---|---|---|
| DFlash FFN (5 layers) | 0.95× | ~4-8× (expected) |
| Target FFN (36 layers) | 0.96× | ~4-8× (expected) |
| Attention Q×K^T (KV=256) | 0.97× | ~2-4× (expected) |

### 3. What constitutes success

- GPU shows **any ratio > 2× at K=128** for FFN matmuls → confirms linear scaling
- Combined with TPU's ~1.0× → proves the finding is hardware-specific
- The larger the GPU ratio, the stronger the contrast

### 4. Edge case: GPU is also flat

If GPU somehow shows flat scaling too (unlikely), it means:
- The matrices are too small to saturate GPU compute at K=16
- Re-run with larger dimensions or batch the matmul across layers
- This would weaken but not invalidate the TPU-specific argument (SmartSpec still shows linear scaling at serving-scale dimensions)

---

## Relationship to Existing Code

| File | Purpose | Where |
|---|---|---|
| `benchmarks/drafter_scaling.py` | TPU version (JAX, needs tpu-inference + vllm) | Already run, results in Doc 37 |
| `benchmarks/gpu_matmul_scaling.py` | GPU version (PyTorch only, standalone) | **Run this on GPU** |
| `tests/drafter_scaling.sh` | Docker wrapper for TPU benchmark | Not needed for GPU |
| `docs/37_drafter_scaling_results.md` | TPU results documentation | Reference for comparison |

The GPU script intentionally mirrors the TPU script's structure (same 3 parts, same K values, same output format) but is completely standalone — no shared code, no shared dependencies.

---

*Created: February 21, 2026*
*Status: Ready to execute — script written, instructions complete*
*Next: Run `benchmarks/gpu_matmul_scaling.py` on any CUDA GPU, bring results back*

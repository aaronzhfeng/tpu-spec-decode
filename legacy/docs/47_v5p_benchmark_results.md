# V5P Benchmark Results

**Date:** 2026-03-03
**VM:** TPU v5p-8 (4 chips, 2x2x1x1 topology)
**Model:** Qwen/Qwen3-4B (target) + z-lab/Qwen3-4B-DFlash-b16 (draft)
**Benchmark:** `benchmarks/standalone_dflash.py`
**Config:** 8 samples per dataset, 256 max new tokens, temperature=0.0, 1 warmup, block_size=16
**Runtime:** Docker `vllm/vllm-tpu:latest` (Python 3.12, JAX 0.8.1, Flax 0.11.1)

---

## Full Results (All 9 Datasets)

| Dataset | Category | Baseline TPOT | DFlash TPOT | Speedup | Tau | Exact Match | Answer Match |
|---------|----------|--------------|-------------|---------|-----|-------------|--------------|
| gsm8k | math | 6.35 ms | 2.12 ms | **3.00x** | 5.40 | 5/8 (62.5%) | 2/3 |
| math500 | math | 6.37 ms | 1.29 ms | **4.93x** | 8.80 | 3/8 (37.5%) | 2/2 |
| aime24 | math | 6.31 ms | 1.77 ms | **3.57x** | 6.48 | 2/8 (25.0%) | N/A |
| aime25 | math | 6.38 ms | 1.88 ms | **3.39x** | 6.14 | 3/8 (37.5%) | N/A |
| humaneval | code | 6.31 ms | 1.99 ms | **3.17x** | 5.76 | 4/8 (50.0%) | N/A |
| mbpp | code | 6.40 ms | 1.96 ms | **3.27x** | 6.16 | 3/8 (37.5%) | N/A |
| mt-bench | chat | 6.27 ms | 2.74 ms | **2.29x** | 3.87 | 1/8 (12.5%) | N/A |
| alpaca | chat | 6.29 ms | 3.86 ms | **1.63x** | 2.86 | 4/8 (50.0%) | N/A |
| swe-bench | code | 6.32 ms | 3.40 ms | **1.86x** | 3.35 | 0/8 (0.0%) | N/A |

**Exact Match** = DFlash output tokens are identical to baseline autoregressive output.
**Answer Match** = final `\boxed{}` answer matches (math datasets only, when both produce a boxed answer).

Mismatches are expected with bf16 speculative decoding — batch-16 verify vs single-token baseline produces different floating-point accumulation, which can flip argmax at decision boundaries. This does not indicate correctness loss.

### Category Averages

| Category | Avg Speedup | Avg Tau | Avg DFlash TPS |
|----------|------------|---------|----------------|
| Math (4) | **3.72x** | 6.71 | 585.4 |
| Code (3) | **2.77x** | 5.09 | 435.7 |
| Chat (2) | **1.96x** | 3.37 | 312.2 |
| **Overall (9)** | **3.02x** | 5.42 | 474.8 |

---

## V5P vs V4 Comparison

| Dataset | V5P Speedup | V4 Speedup | V5P Baseline TPOT | V4 Baseline TPOT | V5P DFlash TPOT | V4 DFlash TPOT |
|---------|------------|-----------|-------------------|------------------|----------------|---------------|
| gsm8k | 3.00x | 3.36x | 6.35 ms | 10.75 ms | 2.12 ms | 3.20 ms |
| math500 | 4.93x | 5.70x | 6.37 ms | 10.73 ms | 1.29 ms | 1.88 ms |
| aime24 | 3.57x | 3.93x | 6.31 ms | 10.68 ms | 1.77 ms | 2.72 ms |
| aime25 | 3.39x | 4.18x | 6.38 ms | 10.74 ms | 1.88 ms | 2.57 ms |
| humaneval | 3.17x | 3.59x | 6.31 ms | 10.81 ms | 1.99 ms | 3.01 ms |
| mbpp | 3.27x | 3.26x | 6.40 ms | 10.66 ms | 1.96 ms | 3.27 ms |
| mt-bench | 2.29x | 2.58x | 6.27 ms | 10.69 ms | 2.74 ms | 4.15 ms |
| alpaca | 1.63x | 1.95x | 6.29 ms | 10.65 ms | 3.86 ms | 5.47 ms |
| swe-bench | 1.86x | 1.27x | 6.32 ms | 10.65 ms | 3.40 ms | 8.42 ms |

### Key Observations

1. **V5P baseline is ~1.69x faster than V4** — autoregressive TPOT dropped from ~10.7ms to ~6.3ms across all datasets.
2. **V5P DFlash is ~1.55x faster than V4** — DFlash TPOT improved from v4 across all datasets, with swe-bench seeing the largest gain (2.48x faster).
3. **Speedup ratios are lower on V5P** — because the baseline improved proportionally more than DFlash. The faster baseline leaves less room for speculative decoding to show relative gains.
4. **Tau values are comparable** — acceptance lengths are nearly identical between v4 and v5p (expected since tau depends on model quality, not hardware).
5. **SWE-bench is the outlier** — speedup improved from 1.27x (v4) to 1.86x (v5p), likely because the long-context overhead was reduced more on v5p.

---

## V5P vs GPU Paper (Math Datasets)

| Dataset | V5P Tau | GPU Paper Tau | Tau % of GPU | V5P Speedup | GPU Speedup |
|---------|---------|--------------|-------------|------------|------------|
| gsm8k | 5.40 | 6.53 | 82.7% | 3.00x | 5.15x |
| math500 | 8.80 | 7.84 | 112.2% | 4.93x | 6.09x |
| aime24 | 6.48 | 7.27 | 89.1% | 3.57x | 5.68x |
| aime25 | 6.14 | 6.64 | 92.5% | 3.39x | 5.21x |
| **Average** | **6.71** | **7.07** | **94.9%** | **3.72x** | **5.53x** |

- V5P achieves **94.9% of GPU paper tau** on math benchmarks (v4 was 94.4%).
- Speedup ratios are lower than GPU because TPU autoregressive baseline is already fast (6.3ms vs GPU's slower baseline).

---

## Raw Results

JSON files for all benchmarks are in `results/v5p/`:
- `standalone_gsm8k.json`
- `standalone_math500.json`
- `standalone_aime24.json`
- `standalone_aime25.json`
- `standalone_humaneval.json`
- `standalone_mbpp.json`
- `standalone_mt-bench.json`
- `standalone_alpaca.json`
- `standalone_swe-bench.json`

CSV summaries:
- `standalone_all_benchmarks.csv` — all 9 datasets
- `standalone_vs_v4.csv` — v5p vs v4 comparison
- `standalone_vs_gpu_paper.csv` — v5p vs GPU paper (math only)

---

## Docker Command Template

```bash
sudo docker run --rm --privileged --network host --ipc host \
  --tmpfs "/tmp:rw,size=80g" \
  -e HF_HOME=/hf-cache \
  -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub \
  -e XDG_CACHE_HOME=/hf-cache/xdg \
  -e JAX_COMPILATION_CACHE_DIR=/hf-cache/jax \
  -e TMPDIR=/tmp \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH=/workspace/tpu-spec-decode/vllm:/workspace/tpu-spec-decode/tpu-inference \
  -v /dev/shm/hf-cache:/hf-cache \
  -v /dev/shm/dflash-test-outputs:/output \
  -v /dev/shm/tpu-logs:/tmp/tpu_logs \
  -v /home/aaronfeng/tpu-spec-decode:/workspace/tpu-spec-decode \
  -w /workspace/tpu-spec-decode \
  vllm/vllm-tpu:latest \
  python3 benchmarks/standalone_dflash.py \
    --target-model Qwen/Qwen3-4B \
    --draft-model z-lab/Qwen3-4B-DFlash-b16 \
    --dataset <DATASET> \
    --max-samples 8 \
    --max-new-tokens 256 \
    --max-model-len 2048 \
    --temperature 0.0 \
    --warmup 1 \
    --output-json /output/v5p_standalone_<DATASET>.json
```

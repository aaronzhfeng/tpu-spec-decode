# Doc 42: GPU Matmul Scaling Results — Both Hardware Are Memory-Bound

## Status: NEGATIVE RESULT — GPU FFN is flat, invalidating TPU-uniqueness claim at Qwen3-4B scale

---

## 1. Hardware

- **GPU:** NVIDIA RTX 2000 Ada Generation (16 GB GDDR6, Ada Lovelace)
- **CUDA:** 12.4 / PyTorch 2.4.1
- **Benchmark:** `benchmarks/gpu_matmul_scaling.py` — 50 trials, 10 warmup, bf16
- **Note:** Bug from Doc 41 Section 5 (line 178 `w_down` → `tw_down`) was already fixed in code

---

## 2. Experiment A Results: GPU Isolated FFN Matmul Scaling

### DFlash FFN (5 layers, hidden=2560, intermediate=9728)

| K | Per-layer (ms) | Full model (ms) | vs K=16 |
|---|---|---|---|
| 16 | 0.788 ± 0.003 | 3.939 | 1.00x |
| 32 | 0.763 ± 0.003 | 3.817 | 0.97x |
| 64 | 0.794 ± 0.002 | 3.968 | 1.01x |
| 128 | 0.860 ± 0.006 | 4.302 | **1.09x** |

### Target FFN (36 layers, same dimensions)

| K | Per-layer (ms) | Full model (ms) | vs K=16 |
|---|---|---|---|
| 16 | 0.782 ± 0.003 | 28.169 | 1.00x |
| 32 | 0.761 ± 0.001 | 27.394 | 0.97x |
| 64 | 0.789 ± 0.001 | 28.406 | 1.01x |
| 128 | 0.852 ± 0.002 | 30.679 | **1.09x** |

### Attention Q×K^T (32 heads, head_dim=80)

| K \ KV_len | 64 | 256 | 512 | 1024 |
|---|---|---|---|---|
| 16 | 1.00x | 1.00x | 1.00x | 1.00x |
| 32 | 1.00x | 1.03x | 1.04x | 1.02x |
| 64 | 1.00x | 1.22x | 1.35x | 1.51x |
| 128 | 1.04x | 1.58x | 1.98x | **2.51x** |

---

## 3. Experiment C Results: GPU K=129 Tile Boundary

Extended k_values to [16, 32, 64, 128, 129, 256]:

### DFlash FFN

| K | Per-layer (ms) | vs K=16 | K→K+1 jump |
|---|---|---|---|
| 128 | 0.859 | 1.09x | — |
| 129 | 0.884 | 1.12x | **1.03x** (no discontinuity) |
| 256 | 1.095 | 1.39x | — |

### Target FFN

| K | Per-layer (ms) | vs K=16 | K→K+1 jump |
|---|---|---|---|
| 128 | 0.853 | 1.09x | — |
| 129 | 0.883 | 1.13x | **1.04x** (no discontinuity) |
| 256 | 1.094 | 1.40x | — |

### Attention (KV=1024)

| K | Attn (ms) | vs K=16 | K→K+1 jump |
|---|---|---|---|
| 128 | 0.090 | 2.46x | — |
| 129 | 0.106 | 2.90x | **1.18x** (some boundary effect) |
| 256 | 0.099 | 2.70x | — |

---

## 4. Side-by-Side: GPU vs TPU at K=128/K=16

| Component | GPU (RTX 2000 Ada) | TPU (v4-8) | Expected GPU |
|---|---|---|---|
| DFlash FFN | **1.09x** | 0.95x | ~4-8x |
| Target FFN | **1.09x** | 0.96x | ~4-8x |
| Attention (KV=256) | **1.58x** | 0.97x | linear |
| K=129/K=128 FFN | **1.03x** | 0.97x | — |

---

## 5. Interpretation

### FFN matmuls are memory-bound on BOTH GPU and TPU at Qwen3-4B scale

The K=128/K=16 FFN ratio is 1.09x on GPU vs the expected 4-8x linear scaling. This confirms what Doc 41 Section 6 warned about:

> "If GPU FFN is also flat: This would mean the contrast doesn't hold at Qwen3-4B scale (both GPU and TPU are memory-bound at this model size)."

**Why:** Each FFN layer loads ~150 MB of weights from HBM. At these dimensions, weight loading dominates total time. Whether the activation matrix has 16 or 128 rows, the weight transfer time is identical. The tensor cores (GPU) and MXU (TPU) both finish the compute within the memory transfer window.

### The roofline crossover hasn't been reached

At K=256, GPU FFN shows 1.39x — the beginning of actual scaling. The crossover from memory-bound to compute-bound likely occurs somewhere between K=256 and K=512 on this GPU. On TPU (with its wider MXU), the crossover would be even later.

### Attention DOES scale differently

GPU attention shows real scaling with K, especially at longer KV lengths (2.51x at KV=1024). This is because attention compute is O(K × KV_len) — it grows in two dimensions, pushing through the roofline sooner than FFN. The TPU's 0.97x for attention suggests the MXU can absorb this extra work within its wider tiles.

### No tile boundary discontinuity on GPU either

K=128→K=129 shows only 1.03x increase for FFN (no jump), confirming both architectures are smoothly memory-bound rather than exhibiting tile-quantized behavior.

---

## 6. Impact on Research Claims

### What's invalidated:
- "GPU verification cost scales linearly with K" — **FALSE at Qwen3-4B**
- "TPU flat-K property is unique to MXU tile amortization" — **FALSE, it's generic memory-boundedness**
- The 3.47x end-to-end GPU number from dflash-wide/docs/09 was indeed entirely due to acceptance rate collapse, as Doc 41 suspected

### What still holds:
- TPU verification IS flat (confirmed by Docs 33, 37, 39)
- GPU verification IS flat for FFN too — but this isn't a contrast, it's the same behavior
- **Attention scaling IS different**: GPU attention scales with K at longer contexts while TPU attention remains flat

### Possible pivots:
1. **Larger model sizes** — At 70B parameters, weight matrices are bigger and the memory transfer window is longer, but compute grows quadratically with hidden dim. The roofline crossover may occur at smaller K, making the contrast real at scale.
2. **Attention-focused argument** — At long contexts (KV≥512), GPU attention costs 2x more at K=128, while TPU stays flat. This is a real contrast but weaker than the original "4-8x full verification" claim.
3. **Different GPU architecture** — An H100 with different memory bandwidth / tensor core ratio might cross the roofline sooner. The RTX 2000 Ada has high bandwidth-to-compute ratio.
4. **Practical overhead argument** — GPU kernel launch overhead, CUDA scheduling, and memory allocation for larger K may add practical costs beyond raw matmul. This would need a different benchmark to demonstrate.

---

## 7. Data Files

- `results/gpu_matmul_scaling.json` — Experiment A (K=16,32,64,128)
- `results/gpu_matmul_scaling_extended.json` — Experiment C (K=16,32,64,128,129,256)

---

*Created: February 27, 2026*
*Hardware: NVIDIA RTX 2000 Ada Generation, CUDA 12.4, PyTorch 2.4.1, bf16*
*Executed: Experiments A (P0) and C (P2) from Doc 41*
*Skipped: Experiment B (P1) — requires TPU, not available on this machine*
*Builds on: Doc 41 (corrected experiment plan), Doc 37 (TPU matmul data), Doc 39 (TPU tile boundary)*

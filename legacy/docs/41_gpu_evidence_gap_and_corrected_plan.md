# Doc 41: GPU Evidence Gap — Confounded Measurement and Corrected Experiment Plan

## Status: GAP IDENTIFIED — blocks all claims about GPU vs TPU contrast

---

## 1. The Problem

The core research claim requires proving two things:

1. **TPU verification cost is flat from K=16 to K=128** — PROVEN (Docs 33, 37, 39)
2. **GPU verification cost scales linearly with K** — NOT PROPERLY PROVEN

The only GPU evidence that exists is an end-to-end speculative decoding benchmark (dflash-wide/docs/09), not an isolated verification cost measurement. The reported 3.47× scaling conflates two independent effects.

---

## 2. What the GPU Data Actually Shows

Source: `dflash-wide/docs/09_gpu_validation_plan.md`, run via `benchmarks/benchmark_block_sizes.py`

| K | GPU Time/Token (ms) | vs K=16 | Acceptance Len |
|---|---|---|---|
| 16 | 6.38 | 1.00× | 8.30 |
| 32 | 9.92 | 1.55× | 5.24 |
| 64 | 12.39 | 1.94× | 4.06 |
| 128 | 22.12 | 3.47× | 2.25 |

### Why 3.47× is confounded

The 3.47× measures **time per output token**, which is:

```
time_per_output_token = total_decode_time / num_output_tokens
```

This metric is affected by two independent factors:

**Factor 1: Hardware verification cost scaling (what we want to isolate)**
- More query tokens → more compute per verification step on GPU
- This is the hardware effect that should scale linearly

**Factor 2: Acceptance rate collapse from train/test mismatch**
- The K=16-trained model was evaluated at K=128 inference
- The model was never trained to predict positions 17–128
- Acceptance length dropped from 8.30 → 2.25 (3.7× worse)
- Fewer accepted tokens per step → more wasted verification steps → higher time/token

Both effects inflate time/token. They are completely entangled in this measurement. The 3.47× number tells us nothing about how much is hardware scaling vs how much is model mismatch.

### Rough decomposition attempt

If acceptance drops from 8.30 to 2.25, each step produces ~3.7× fewer useful tokens. Even if verification cost were *constant*, time/token would increase ~3.7× from acceptance collapse alone. The measured 3.47× is *less* than the acceptance ratio, suggesting GPU verification cost might be nearly flat at these dimensions too — which would undermine the entire contrast.

This decomposition is imprecise (time/token also depends on draft time, which is flat for K≤128). But it demonstrates the confound: **we cannot separate the two effects from end-to-end data**.

---

## 3. What the TPU Data Shows (for comparison)

The TPU evidence does NOT have this problem because it uses **isolated microbenchmarks**:

### Doc 37 — drafter_scaling.py (isolated FFN matmuls, no model, no acceptance logic)

| K | DFlash FFN (5 layers) | Target FFN (36 layers) | Attention Q×K^T |
|---|---|---|---|
| 16 | 1.637ms (1.00×) | 12.201ms (1.00×) | 0.202ms (1.00×) |
| 128 | 1.556ms (0.95×) | 11.842ms (0.97×) | 0.196ms (0.97×) |

### Doc 33 — amortized_verification.py (full target forward pass, no acceptance logic)

| K | Verify latency | vs K=16 |
|---|---|---|
| 16 | 1.70 ± 0.06ms | 1.00× |
| 128 | 1.78 ± 0.06ms | 1.05× |

Both TPU benchmarks isolate the hardware cost from acceptance rates. The GPU benchmark does not.

---

## 4. Secondary Issue: Narrative Inconsistency (Tile vs Memory-Bound)

Doc 39 (K=129 experiment) found **no discontinuity** at the MXU tile boundary:

| K | DFlash FFN | K=129/K=128 |
|---|---|---|
| 128 | 1.649ms (0.98×) | — |
| 129 | 1.594ms (0.95×) | **0.97×** |
| 256 | 1.716ms (1.02×) | — |

The predicted 2× jump at K=129 does not appear. Doc 39 correctly identifies the real mechanism as **memory-bandwidth boundedness**: weight loading (~150 MB/layer from HBM at 1.2 TB/s) dominates, and MXU compute for even 2 tile rows completes within the memory transfer window.

However, proposal v3 (Section 2.3) and the dflash-wide docs still frame the finding as "128×128 tile alignment." This language needs updating before publication, though it doesn't affect the experimental plan.

---

## 5. Code Bug in gpu_matmul_scaling.py

Line 178 in Part 2 warmup loop uses `w_down` (DFlash down_proj from Part 1) instead of `tw_down` (target down_proj):

```python
# Line 178 (warmup — WRONG variable):
o = torch.mm(h, w_down)    # ← should be tw_down

# Line 188 (timed trials — correct):
o = torch.mm(h, tw_down)   # ← correct
```

Both matrices are (9728, 2560) filled with ones, so results are unaffected. Fix before running.

---

## 6. Corrected Experiment Plan

### Experiment A: GPU Isolated Matmul Scaling (MUST-DO)

**Goal:** Prove GPU FFN matmul cost scales with K, isolated from acceptance effects.

**Script:** `benchmarks/gpu_matmul_scaling.py` (exists, needs bug fix + run)

**Fix required before running:**
```python
# Line 178: change w_down → tw_down
o = torch.mm(h, tw_down)
```

**How to run:**
```bash
# Any GPU with bf16 — Colab T4 (free), A100, H100 all work
pip install torch
python benchmarks/gpu_matmul_scaling.py --trials 50 --warmup 10 --output-json results/gpu_matmul_scaling.json
```

**What it measures (3 parts):**
1. DFlash FFN: (K, 2560) × (2560, 9728) gate/up/down, ×5 layers
2. Target FFN: same dimensions, ×36 layers
3. Attention Q×K^T: (32 heads, K, 80) × (32 heads, 80, KV_len)

**K values:** 16, 32, 64, 128
**Expected results (if GPU scales linearly):**

| K | GPU FFN ratio | TPU FFN ratio (measured) |
|---|---|---|
| 16 | 1.00× | 1.00× |
| 32 | ~1.5–2.0× | 1.00× |
| 64 | ~2.0–4.0× | 0.98× |
| 128 | ~4.0–8.0× | 0.95× |

**Success criteria:**
- GPU K=128/K=16 ratio > 2× for FFN matmuls → confirms linear scaling
- Combined with TPU's 0.95× → proves the contrast is hardware-specific

**Time to execute:** <1 minute on any GPU

**If GPU FFN is also flat:** This would mean the contrast doesn't hold at Qwen3-4B scale (both GPU and TPU are memory-bound at this model size). This would be a negative result that requires changing the paper's claim. The contrast might only appear at larger models or different GPU architectures. We need to know this before publishing.

---

### Experiment B: TPU Verification at Longer Contexts (SHOULD-DO)

**Goal:** Confirm flat-K property holds when attention is a larger fraction of forward pass time.

**Script:** `benchmarks/amortized_verification.py` (exists, no changes needed)

**How to run:**
```bash
# Modify the prompt to generate longer prefill contexts, or use longer dataset samples
python benchmarks/amortized_verification.py \
    --target-model Qwen/Qwen3-4B \
    --draft-model z-lab/Qwen3-4B-DFlash-b16 \
    --dataset gsm8k --max-samples 1 \
    --context-lengths 64,256,512,1024
```

Note: The script currently uses the first sample's prompt length (~66 tokens) as context. To test longer contexts, either:
- Use a dataset with longer prompts (mt-bench, swe-bench)
- Add a `--min-context-length` flag that skips short prompts
- Extend the script to support prefilling to a target context length before measuring verification

**What to measure:**
At each context length L ∈ {64, 256, 512, 1024}:
- Verify latency at K=16, 64, 128
- Record the K=128/K=16 ratio

**Expected results:**

| Context L | K=128/K=16 ratio (predicted) |
|---|---|
| 64 | ~1.05× (measured, Doc 33) |
| 256 | ~1.05–1.10× (FFN still dominates) |
| 512 | ~1.10–1.20× (attention growing) |
| 1024 | ~1.20–1.50× (attention significant) |

At L=1024, attention cost = O(K × 1024) per head per layer. For K=128 vs K=16, attention alone costs 8× more. If attention is 30% of total forward pass at L=1024, the total ratio would be ~1 + 0.30 × 7 = ~3.1×. This would mean the flat property partially breaks at long contexts.

**Why this matters:** The proposal claims "step time is approximately constant for any K≤128." If that's only true at short contexts (L<256), the claim needs a context-length caveat. Most real generation happens at L=200–2000.

---

### Experiment C: GPU Matmul with K=129 (NICE-TO-HAVE)

**Goal:** Show whether GPU also lacks a tile boundary discontinuity (confirming both architectures are memory-bound, just with different scaling characteristics).

**Script:** `benchmarks/gpu_matmul_scaling.py` (add K=129, 256 to k_values)

**Modification:**
```python
# Line 75: extend k_values
k_values = [16, 32, 64, 128, 129, 256]
```

**Why:** If GPU also has no K=129 jump, both hardware are memory-bound. The contrast then is about *degree* of memory-boundedness: TPU's wider systolic array hides more compute behind memory transfers than GPU's smaller tensor cores.

---

## 7. Execution Priority

| Priority | Experiment | Effort | Blocks |
|---|---|---|---|
| **P0** | A: GPU isolated matmul | 5 min (fix bug + run) | All GPU vs TPU claims |
| **P1** | B: TPU long-context verify | 30 min (script mod + run) | "Step time constant" claim |
| **P2** | C: GPU K=129 | 5 min (add K values + run) | Tile boundary narrative |

**Experiment A is the critical gate.** If GPU matmuls are also flat at Qwen3-4B scale, the entire TPU-uniqueness claim needs rethinking. If they scale linearly as expected, the paper's core contribution is validated.

---

## 8. What Happens After Results

### If GPU matmul scales linearly (expected):

- The confounded 3.47× end-to-end number can be replaced with the clean matmul ratio
- The side-by-side comparison table (GPU: ~4×, TPU: 0.95×) goes directly into the paper
- Proceed with K=128 training

### If GPU matmul is also flat:

- The "TPU-unique" claim is false at Qwen3-4B scale
- Must test at larger model sizes (70B) where the roofline crossover shifts
- Or pivot the claim to: "both are flat, but GPU's other overheads (kernel launch, scheduling) make larger K suboptimal on GPU for different reasons"
- This is a significant pivot — better to discover now than after submission

### If TPU flat property breaks at long context:

- Add context-length caveat to all claims
- Proposal Section 4.3 throughput model needs T_step(K, L) instead of T_step ≈ constant
- K=128 advantage may only hold for latency-optimized short-context serving

---

*Created: February 27, 2026*
*Status: Gap identified — Experiment A (GPU matmul) must run before any GPU vs TPU claim is publishable*
*Builds on: Doc 39 (tile boundary negative result), Doc 37 (TPU matmul data), dflash-wide/docs/09 (confounded GPU data)*
*Blocks: proposal v3 Section 3.4, Section 4.1, Section 7 (all reference GPU contrast)*

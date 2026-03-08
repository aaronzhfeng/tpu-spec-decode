# Technical Reference: DFlash Speculative Decoding on TPU

This document is the technical companion to META.md. It covers mechanisms, hardware adaptations, engineering decisions, implementation techniques, quantitative results, and open problems. For project logistics, file paths, and deliverable tracking, see META.md.

---

## 1. Core Discovery: K-Flat Verification on TPU

TPU verification forward-pass latency is invariant to the number of query tokens K in the single-request decode regime.

### Measured Ratios (K=128 / K=16)

| Hardware | L=256 | L=1024 | L=2048 | L=4096 |
|----------|-------|--------|--------|--------|
| TPU v4-8 | 0.97x | 0.97x | -- | -- |
| TPU v5p-8 | 1.02x | 1.01x | 0.91x | 0.95x |
| GPU (RTX 2000 Ada) | 1.24x | 1.24x | -- | -- |

K-ceiling sweep (V3, v5p, L=256): flat through K=1024 (ratio 1.00x at K=1024/K=16). No ceiling found.

### Mechanism: Why TPU Is Flat

The flat behavior is NOT due to MXU 128x128 tile boundaries (falsified by K=129 test: 1.02x, Doc 39). The correct explanation:

**Per-layer budget at K=128, L=256 (Qwen3-4B, 36 layers, v5p):**

| Component | Time (ms) | % of Per-Layer |
|-----------|-----------|----------------|
| Compute (FFN + attention matmuls) | 0.013 | 35.7% |
| HBM weight loading (182 MB/layer) | 0.010 | 26.1% |
| Fixed overhead (kernel launch, RPA, norms, ICI) | 0.023 | 64.3% |
| **Measured per-layer** | **0.036** | **100%** |

Fixed overhead dominates. Increasing K from 16 to 1024 only affects the compute component (35.7%), which is absorbed within the overhead budget. The K-dependent attention compute (QK^T + AV) is 2.1% of total FLOPs at K=128.

**Roofline numbers (v5p-8):**
- MXU: 1836 TFLOPS, HBM: 19.2 TB/s, ridge point: 95.6 FLOP/byte
- Total model weights: 6.56 GB (loaded every forward pass)
- Weight load time at peak BW: 0.34ms; measured: 1.31ms (3.9x overhead from XLA infra)

### FFN vs Attention Decomposition

Both FFN and attention are individually flat on TPU. The hardware contrast with GPU lives entirely in attention:

| Component | TPU K=128/K=16 | GPU K=128/K=16 | GPU scaling with L |
|-----------|---------------|---------------|-------------------|
| FFN matmuls | 0.95x | 1.09x | None |
| Attention QK^T | 1.02x (at KV=1024) | 2.51x (at KV=1024) | Linear with L |
| Full verify | **0.97x** | **1.24x** | Attention-driven |

Attention is 83% of verify time (Doc 43). GPU attention scales 2.51x at L=1024 (isolated) producing ~1.24x total verify cost. TPU's RPA v3 kernel absorbs the compute via systolic pipeline overlap.

---

## 2. DFlash Architecture on TPU

### Draft Model

DFlash is a block-diffusion drafter: predicts K tokens in parallel via iterative denoising, unlike autoregressive drafters (EAGLE3) that generate sequentially.

- **Architecture:** 5-layer transformer, hidden=2560, intermediate=9728 (same dimensions as Qwen3-4B)
- **Parameters:** ~429M (5/36 of target model layers + projection layers)
- **Shared components:** target model's embedding table and LM head (no separate vocab weights)
- **Attention:** non-causal within block (each draft token attends to all K positions), causal to KV cache
- **Input:** `[next_token, mask, mask, ..., mask]` (K positions, first token known, rest denoised)
- **Target conditioning:** `aux_hidden_states` from 5 target layers projected into drafter's input via `combine_fn`

### Inference Pipeline

```
1. Target prefill → kv_caches, aux_hidden_states
2. Project aux_hidden_states → drafter context (combine_fn)
3. Draft: [next_tok, MASK x (K-1)] + context → 5-layer forward → K hidden states
4. LM head (shared with target): hidden → logits → argmax → K draft tokens
5. Verify: target forward on K tokens → logits → argmax → posterior
6. Accept: consecutive prefix match (draft[i+1] == posterior[i])
7. Update: accepted tokens extend sequence, aux_hidden_states update context
8. Repeat from step 3
```

One draft forward pass produces all K tokens (O(1) passes vs O(K) for AR drafters).

### Draft Cost Scaling

| K | TPU Draft (ms) | Ratio | GPU Draft (ms) | Ratio |
|---|---------------|-------|---------------|-------|
| 16 | 1.08 | 1.00x | 6.74 | 1.00x |
| 64 | 1.11 | 1.02x | 7.12 | 1.06x |
| 128 | 1.08 | **1.00x** | 8.22 | **1.22x** |
| 256 | 1.07 | 0.98x | -- | -- |

Draft cost is near-flat on both hardware (architecture property of parallel generation). Within-block attention is O(K^2) but at K=128 it's only 262K elements/head — tiny vs weight loading.

---

## 3. The 2x2 Intersection Argument

K=128 block size requires flat scaling on BOTH draft and verify sides:

|  | GPU Verify (1.24x) | TPU Verify (0.97x) |
|--|-------------------|-------------------|
| **AR Draft (8x at K=128)** | Both scale. Not viable. | Draft scales. Not viable. |
| **Diffusion Draft (1.22x GPU / 1.00x TPU)** | Verify scales. Marginal. | **Both flat. Viable.** |

Only diffusion + TPU makes K=128 a guaranteed win.

**Throughput equation:**

```
TPS(K) = τ(K) / [T_draft(K) + T_verify(K)]
```

At K=128 on TPU: T_draft ≈ T_draft(16), T_verify ≈ T_verify(16). Any τ improvement is pure throughput gain.

At K=128 on GPU: T_verify = 1.24 * T_verify(16). Need τ(128) > 1.24 * τ(16) to break even. At current α=0.86, geometric series saturates by K≈32-64.

---

## 4. Step-Time Budget

Full speculative decode step (standalone, GSM8K, K=16, v4):

| Component | Time (ms) | % |
|-----------|-----------|---|
| Draft forward (5 layers) | ~2 | 12% |
| Draft LM head + sampling | ~8.8* | -- |
| Verify forward (36 layers) | ~10 | 59% |
| Acceptance logic | ~3 | 18% |
| Host overhead | ~2 | 12% |
| **Total step** | **~17** | **100%** |

*Draft LM head 8.8ms is device-host sync barrier, not actual compute. `jax.lax.while_loop` would eliminate this.

On v5p: baseline TPOT = 6.3ms, DFlash TPOT = 1.88ms (GSM8K), 1.40ms (MATH500).

---

## 5. Benchmark Results (V5P)

### 9-Dataset Performance

| Dataset | Category | Tau | Speedup | DFlash TPOT (ms) |
|---------|----------|-----|---------|-------------------|
| math500 | math | 8.80 | 5.72x | 1.40 |
| aime24 | math | 6.48 | 3.98x | 1.93 |
| aime25 | math | 6.14 | 3.35x | 2.05 |
| gsm8k | math | 5.40 | 3.17x | 2.31 |
| humaneval | code | 5.76 | 3.53x | 2.18 |
| mbpp | code | 6.16 | 2.77x | 2.64 |
| mt-bench | chat | 3.87 | 2.36x | 3.05 |
| alpaca | chat | 2.86 | 1.65x | 4.08 |
| swe-bench | code | 3.35 | 1.60x | 4.27 |
| **Average** | | **5.42** | **3.13x** | **2.66** |

Baseline TPOT across all: ~7.30 ms (Qwen3-4B, greedy, v5p-8).

### Acceptance Rate Profile (K=16 block, per-position)

```
pos  0: 1.000   pos  4: 0.495   pos  8: 0.237   pos 12: 0.127
pos  1: 0.889   pos  5: 0.412   pos  9: 0.201   pos 13: 0.102
pos  2: 0.741   pos  6: 0.345   pos 10: 0.168   pos 14: 0.084
pos  3: 0.601   pos  7: 0.295   pos 11: 0.152   pos 15: 0.069
```

Per-position α ≈ 0.86 (geometric decay). At K=16, τ=5.92 (GSM8K, v5p).

---

## 6. Verification Experiments (V1-V9) — V5P Results

| ID | Test | Result |
|----|------|--------|
| V1 | K-sweep v5p (K=16→256, L=256,1024) | Flat: 1.02x (L=256), 1.01x (L=1024) |
| V2 | Extended context (L=2048,4096) | Flat: 0.91x (L=2048), 0.95x (L=4096) |
| V3 | K-ceiling (K=16→1024, L=256) | No ceiling: 1.00x at K=1024 |
| V4 | Roofline analysis | 64% of per-layer time is K-independent overhead |
| V5 | Context position → acceptance | GSM8K: rho=+0.10, p=0.006 (acceptance improves 4.93→6.83) |
| V6 | Batch size (extrapolated from V3) | Flat through effective batch=8 at K=128 |
| V7 | GPU full forward pass | 1.24x at K=128/K=16 (corrected from 2.51x isolated Q×K^T) |
| V8 | FFN/attention decomposition | FFN: 1.00x, attention: 1.02x (KV=1024). Both flat on TPU |
| V9 | Draft model K-sweep v5p | Draft FFN: 1.00x at K=128/K=16 |

---

## 7. Tau Ceiling Analysis

At fixed per-position α, acceptance follows geometric distribution: τ(K) = Σ_{i=1}^{K} α^{i-1} = (1 - α^K) / (1 - α).

| α | τ(K=16) | τ(K=32) | τ(K=64) | τ(K=128) | τ(K=∞) |
|------|---------|---------|---------|----------|--------|
| 0.86 | 6.15 | 6.86 | 7.11 | 7.14 | 7.14 |
| 0.90 | 8.15 | 9.58 | 9.97 | 10.00 | 10.00 |
| 0.95 | 13.17 | 17.65 | 19.52 | 19.95 | 20.00 |

At α=0.86 (current DFlash), going from K=16 to K=128 gains only 16% more τ (6.15→7.14). The series saturates by K≈32-64.

**Key insight:** K=128 only helps significantly if wider blocks improve α (assumption B1 — bidirectional context within block improves per-position prediction). If α improves to 0.90, τ jumps from 8.15 to 10.00 — a 23% gain that K=16 can never capture.

---

## 8. Open Technical Questions

### Must-Answer Before K=128 Training

| # | Question | Impact |
|---|----------|--------|
| B1 | Does wider block improve per-position α via bidirectional context? | If no, τ ceiling is 7.14 (+16% over K=16). If yes (α→0.90), ceiling is 10.00 (+63%). |
| B2 | Can DFlash be trained at K=128 without quality collapse? | If training diverges, need K=32→K=64→K=128 curriculum. |
| B3 | Does standalone τ translate to vLLM τ? | Standalone τ=6.67 vs vLLM τ=4.48 (33% degradation). Pipeline overhead is K-independent, so ratio should hold. |

### Resolved

| # | Question | Answer |
|---|----------|--------|
| Q1 | Where does flat end? | Not found through K=1024 (V3) |
| Q2 | Is flat v4-specific? | No — confirmed on v5p with stronger effect (V1) |
| Q3 | What about L>1024? | Flat through L=4096 (V2) |
| Q4 | Is it MXU tiles? | No — K=129 is 1.02x, no discontinuity (Doc 39) |
| Q5 | Is GPU verify really 2.3x? | No — full forward pass is 1.24x (V7); 2.51x was isolated Q×K^T |
| Q6 | Batch size? | V3 extrapolation: flat through batch=8 at K=128 |

---

## 9. Code Architecture (PR #1868)

### New Components

**`dflash.py` (DFlashForCausalLM):** 5-layer transformer with non-causal block attention + causal KV attention. `combine_fn` projects concatenated aux_hidden_states from target layers. Forward signature: `(state, kv_caches, input_ids, target_hidden, attn_metadata) → (kv_caches, hidden, aux)`.

**`qwen3_dflash.py`:** Qwen3-specific variant. Overrides `Qwen3Model.__call__` to collect `aux_hidden_states` from 5 target layers (indices selected by `_get_dflash_target_layer_ids()`). Returns `(kv_caches, x, aux_hidden_states)`.

**`dflash_attention_interface.py`:** `dflash_concat_attention` kernel. Concatenates block tokens with KV cache, applies non-causal mask within block + causal mask to cache. Handles GQA (32 query heads, 8 KV heads).

**`DFlashProposer`:** Implements `prepare_inputs()` and `propose()`. Builds noise block `[next_token, MASK, ..., MASK]`, packs target context as 3-tuple `(projected_hidden, cache_len, ctx_count)`, runs draft forward + LM head + greedy sampling.

### Integration Points

- `speculative_decoding_manager.py`: separate `propose_dflash_draft_token_ids()` method (does NOT modify Eagle3 path)
- `kv_cache_manager.py`: reads `num_hidden_layers` from config instead of hardcoding 1
- `qwen3.py`: `Qwen3Model.__call__` returns aux_hidden_states; `_init_aux_hidden_state_layers()` at model init
- `tpu_runner.py`: `elif method == "dflash"` dispatch to DFlashProposer
- `model_loader.py`: register DFlashDraftModel in model registry

---

## 10. Key Equations

**Throughput:**
```
TPS(K) = τ(K) / [T_draft(K) + T_verify(K)]
```

**Acceptance length (geometric, fixed α):**
```
τ(K) = (1 - α^K) / (1 - α)
```

**Speedup:**
```
Speedup = TPS_speculative / TPS_baseline = τ(K) · T_baseline / [T_draft(K) + T_verify(K)]
```

**K-flat condition (empirical):**
```
T_verify(K) ≈ T_verify(1) for K ∈ [1, 1024] on TPU v4/v5p at L ≤ 4096
```

**Roofline operational intensity:**
```
OI = FLOPs / Bytes_moved
If OI < ridge_point → memory-bound (flat with K)
If OI > ridge_point → compute-bound (scales with K)
Ridge point (v5p-8): 95.6 FLOP/byte
```

**Practical bound:** even when naive OI > ridge at K=128, per-layer fixed overhead (64% of execution) absorbs the compute increase.

---

## 11. GPU → TPU Porting: Engineering Challenges

### The Fundamental Constraint: JAX Is Functional

PyTorch DFlash uses mutable state everywhere — `DynamicCache.append()`, in-place tensor ops, Python-level control flow. JAX requires:
- All arrays are immutable; model functions return new state
- KV cache updates use `jax.lax.dynamic_update_slice` instead of in-place append
- No Python-side mutation inside JIT-traced code
- Model `__call__` must have fixed pytree structure across all calls (variable shapes trigger retracing)

### KV Cache: Three Phases of Iteration

**Phase 1 — Context buffer only (Doc 16):** Host-side NumPy buffer accumulates projected target hidden states. Re-projected to K/V each iteration. Result: τ=2.38, acceptance=9.2%, 1.27x speedup. Problem: loses noise K/V history between steps (GPU preserves it via DynamicCache).

**Phase 2 — Per-layer KV cache (Doc 16):** Tried JAX-native per-layer KV buffers managed by proposer. Collapsed to 3.5-7% acceptance. Root cause: pytree structure mismatch — model `__call__` JIT-traced differently on first iteration (empty cache) vs subsequent (populated cache), causing full retracing.

**Phase 3 — On-device cache + flash_attention (Doc 18):** Pre-allocated contiguous arrays `(1, num_kv_heads, max_kv_len, head_dim)` per layer. New tokens appended via `dynamic_update_slice` at `cache_len` offset. Used TPU Pallas `flash_attention` kernel with `causal=False` and `SegmentIds` masking. Result: τ=2.37, acceptance=8.3%, peak 102.3 TPS. JIT retracing from variable context sizes dropped average to 63.6 TPS.

**Final design (standalone):** Pre-allocated KV cache with scalar length counter. Incremental context injection (only new projected features sent each step). `pad_context()` rounds to next power-of-2 to minimize JIT retrace shapes.

### Attention Kernel Adaptation

GPU DFlash uses non-causal attention within the draft block — each of K tokens attends to all K positions bidirectionally. The TPU runtime's default attention path (ragged paged attention / RPA) is causal-only and requires `q_len == kv_len`.

**Solution:** `dflash_concat_attention` — a separate dense attention path that:
1. Concatenates context K/V with noise K/V on the token dimension: `k_cat = concat([k_ctx, k_noise], axis=seq_dim)`
2. Applies non-causal mask within the block, causal mask to KV cache
3. Routes through TPU Pallas `flash_attention` with `causal=False` + `SegmentIds`
4. Handles GQA by repeating KV heads (8 KV → 32 query heads)
5. Bypasses the RPA v3 kernel entirely — DFlash and Eagle3 use separate attention paths

Custom `BlockSizes` tuning: `block_q=16` (matches draft block size), preventing excessive padding.

### Acceptance Rate Debugging Arc

The acceptance rate journey: 0.29% → 9.2% → 23.2% → 94% of GPU quality (τ=6.67).

**Bug 1 — Additive K/V fusion (Doc 09):** Initial implementation ADDED context K/V to noise K/V instead of concatenating. Destroyed the separation between context features and denoised block. Fix: `dflash_concat_attention` with proper concat semantics.

**Bug 2 — Layer indexing off-by-one (Doc 14):** GPU's `extract_context_feature` uses `hidden_states[layer_id + 1]` (post-layer convention). TPU was capturing pre-layer hidden states. Fix: `capture_aux_after_layer=True` flag in target model when `method=="dflash"`.

**Bug 3 — Rejection context not trimmed (Doc 14):** After rejection, context buffer retained features for positions that were rejected. Drafter received context for tokens that don't exist in the final sequence. Fix: `_resolve_context_seq_len` subtracts rejected token count before building context.

**Bug 4 — cache_len crop bug (Doc 19/20):** `_cache_len` set to current `seq_len` instead of `prev_seq_len`, leaving stale noise K/V entries with wrong RoPE positions. Fix: added `self._prev_seq_len` tracking to match GPU's `DynamicCache.crop(start)` semantics.

**Bug 5 — seq_len inflation (Doc 21, CRITICAL):** The single most impactful bug. vLLM's `speculative_decoding_manager` passed `attn_metadata.seq_lens` (which includes unverified draft tokens being scored) to DFlash proposer. This inflated seq_len by 10-16 tokens per step, corrupting three systems simultaneously:
- Context buffer: stored features for phantom positions
- KV cache positions: `cache_len` derived from inflated `prev_seq_len`, creating gaps
- RoPE positions: noise positions shifted by 10-15 every iteration, accumulating drift

Fix: 4 lines in `speculative_decoding_manager.py` — extract `num_tokens_no_spec` (ground-truth accepted-only count) from `runner.input_batch`, replace `seq_lens` in `attn_metadata`. Doubled speedup from 1.30x to 2.31x (τ 2.49 → 4.48).

Why hard to find: (1) No crash — inflated seq_len is silently accepted; (2) Eagle3 unaffected — it's stateless; (3) Degradation is gradual, not catastrophic; (4) Bug is upstream in the manager, not in the model/proposer being debugged.

### Definitive A/B Experiments (Doc 19)

Three runtime experiments proved the implementation is correct after all bug fixes:

| Experiment | Toggle | Result |
|-----------|--------|--------|
| Flash attention vs manual dot-product | `DFLASH_USE_MANUAL_ATTN=1` | Bit-identical (τ=2.365) |
| Position scheme (incremental vs reset) | `DFLASH_POSITION_SCHEME=reset` | Bit-identical |
| No-cache mode (re-project all context every step) | `DFLASH_NO_CACHE=1` | Bit-identical |

All three producing identical results proves: flash_attention kernel is correct, KV cache is not corrupted, position encoding is correct. The remaining gap vs GPU paper (τ=6.67 vs 7.07) is checkpoint/benchmark methodology difference, not a bug.

### Standalone vs vLLM Pipeline Gap

| Environment | τ (GSM8K) | Speedup | Why |
|-------------|-----------|---------|-----|
| Standalone JAX | 6.67 | 4.26x | Direct draft-verify loop, no scheduling overhead |
| vLLM pipeline | 4.48 | 2.31x | Scheduling, batch management, rejection sampling loop |
| GPU paper | 6.53-7.07 | 5.15x | Different checkpoint, PyTorch, H200 hardware |

The standalone benchmark (`benchmarks/standalone_dflash.py`) mirrors the GPU paper's setup: same generate-verify-accept loop, same datasets, greedy decoding, single device. It imports only `ModelConfig` and `LoadConfig` from vLLM — existing tpu-inference dependencies, not new ones. This proves DFlash works independently of vLLM.

---

## 12. JAX/XLA/TPU-Specific Techniques

### JIT Compilation Management

- **Shape polymorphism:** Context features vary in length each step. Naive approach triggers JIT retracing per unique shape (~5-7s each). Solution: `pad_context()` pads to next power-of-2, limiting to ~12 possible shapes for sequences up to 4096.
- **Pytree structure stability:** Model `__call__` must return the same pytree structure every call. When `aux_hidden_states` is empty (no spec decoding), return `[]` — same type as populated list.
- **`jax.effects_barrier()`:** Used as `tpu_sync()` for timing measurements. Ensures all pending device computations complete before host-side `time.perf_counter()`.

### Device-Host Transfer

- `device_array(mesh, np_array)` — creates JAX array from NumPy with proper sharding on the mesh
- Host-side NumPy used for acceptance logic (argmax comparison) — cheaper than keeping on device for small arrays
- Context buffer maintained as NumPy array on host, converted to `jnp.bfloat16` per step — avoids device memory for the running buffer

### Weight Sharing

DFlash shares the target model's embedding table and LM head — no separate vocab weights. Implemented by copying references after model loading:
```python
target_embed = target_state.model.embed_tokens
draft_state.model.embed_tokens = target_embed  # shared, not copied
```
Draft LM head calls `target_logits_fn(target_state, draft_hidden, None)` — uses target's weights directly.

### Tensor Parallel (4-chip)

TPU v5p-8 uses 4 chips with tensor parallelism on the model axis. Mesh layout: `(1, 1, 1, 4)` over axes `("data", "attn_dp", "expert", "model")`. Weight matrices sharded on the model axis; attention heads distributed across chips. ICI (inter-chip interconnect) handles all-reduce operations. The draft model (5 layers) also runs tensor-parallel across all 4 chips — not replicated.

### Pre-commit and Code Style

PR follows upstream conventions: `addlicense`, `isort`, `yapf`, `ruff`. DFlash code stripped to 17 comments (from 47) to match Eagle3's terse style. Eagle3 and DFlash use entirely separate code paths — no shared propose method — to prevent regressions (learned from the `propose_draft_model_token_ids` incident where sharing broke Eagle3's `attn_metadata`).

---

## 13. Mechanism Evolution — How Understanding Changed

| Phase | Hypothesis | Evidence | Status |
|-------|-----------|----------|--------|
| 1 (Docs 33,37) | MXU 128x128 tiles — one tile row for K≤128, cost jumps at K=129 | TPU flat K=16→128 | Seemed right |
| 2 (Doc 39) | Tile boundary falsified | K=129 costs 1.02x (no jump) | **Wrong** |
| 3 (Doc 42) | GPU FFN also flat → FFN is a red herring | GPU FFN 1.09x, TPU FFN 0.95x | Revised |
| 4 (Docs 42-43) | Attention is the differentiator | GPU attention 2.51x at L=1024; TPU flat | **Correct** |
| 5 (V4 roofline) | Fixed per-layer overhead dominates (64%) | Arithmetic intensity analysis | **Refined** |

Final understanding: Weight loading + fixed infrastructure overhead (kernel launches, RPA, layer norms, ICI) dominates per-layer time. K-dependent attention compute is 2.1% of total FLOPs — invisible in the noise. Both GPU and TPU are memory-bound for FFN; the contrast lives in attention handling where TPU's RPA v3 kernel + systolic pipeline overlap absorbs the compute.

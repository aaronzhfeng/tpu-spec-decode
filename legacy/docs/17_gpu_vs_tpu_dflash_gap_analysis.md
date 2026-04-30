# Doc 17 — GPU vs TPU DFlash Gap Analysis

## Purpose

This document provides a systematic comparison of the reference GPU DFlash
implementation against our TPU/JAX port, identifying the specific ecosystem
features and architectural patterns that CUDA/PyTorch provides but TPU/JAX
lacks.  Understanding these gaps is essential for closing the acceptance-rate
and throughput delta between the two platforms.

**Current TPU baseline (Phase 1 context buffer):** tau ≈ 2.38, acceptance ≈ 10.2 %, speedup ≈ 1.27×  
**GPU reference (DFlash paper):** tau ≈ 5–6, acceptance ≈ 60–70 %, speedup ≈ 5–6×

---

## Gap 1 — DynamicCache (KV Cache): THE Critical Gap

### What GPU has

The reference GPU DFlash (`zhongyan_dev/dflash/model/dflash.py`) uses
PyTorch's `transformers.DynamicCache` — a mutable, dynamically-growing cache
that stores per-layer K/V projections across speculative-decoding iterations:

```python
# GPU: Qwen3DFlashAttention.forward (lines 83-85)
if past_key_values is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
```

On the GPU the flow is:

| Iteration | What gets projected | What attention sees |
|---|---|---|
| 1 | K/V for `[ctx_1, noise_block_1]` | Full K/V → stored in `DynamicCache` |
| 2 | K/V for **only** `noise_block_2` (new tokens) | Cached K/V from iter 1 **concatenated** with new K/V |
| N | K/V for **only** new tokens | Full accumulated K/V history |

After rejection the proposer calls `past_key_values_draft.crop(start)` to
trim the cache to the accepted prefix.

### What TPU lacks

`DynamicCache` uses **in-place mutation** (`list.append`, `torch.cat`).  This
is fundamentally incompatible with JAX's functional programming model where
JIT compilation requires:

- Pure functions (no side effects)
- Static pytree structures (shape/type of all inputs must be known at trace
  time)

Our TPU implementation only caches the **projected context hidden states**
(after `fc` + `hidden_norm`) in a host-side NumPy buffer.  It **re-projects
K/V for the entire context on every single iteration** and never caches the
per-layer K/V projections themselves.

### Impact

This is the **#1 cause** of the acceptance-rate gap.  The GPU sees attention
over the **full accumulated K/V history** (context + all past noise blocks),
while our TPU version only sees `[current_context_buffer, current_noise_block]`
— it completely loses all information from past noise blocks' K/V projections.

---

## Gap 2 — Attention Backend: Non-Causal with KV Cache

### What GPU has

The GPU DFlash dispatches via `ALL_ATTENTION_FUNCTIONS` from HuggingFace
transformers to one of:

- **FlashAttention2** — CUDA kernel supporting non-causal attention with
  variable Q/KV lengths via `cu_seqlens`.
- **SDPA** (`torch.nn.functional.scaled_dot_product_attention`) — supports
  `is_causal=False`.

Both operate on 4D batched tensors `[batch, heads, seq, dim]` and integrate
seamlessly with `DynamicCache`.

### What TPU has

| Kernel | Causal support | Q ≠ KV length | Status for DFlash |
|---|---|---|---|
| `ragged_paged_attention` (v3 Pallas) | **Causal-only** | No (`q_len == kv_len` assumed) | ❌ Not usable |
| `flash_attention` (Pallas) | `causal=True/False` | Yes (`q_seq_len ≠ kv_seq_len`) | ✅ **Available but unused** |
| `dflash_concat_attention` (pure JAX) | Non-causal ✅ | Same-length only | Slow (`fori_loop` + einsum) |
| Our `dflash.py` attention | Non-causal ✅ | Yes | Raw `jnp.einsum` — correct but unoptimised |

### Impact

The attention computation itself is correct (non-causal works), but:

- **Throughput** — our raw einsum is significantly slower than FlashAttention2.
- **Acceptance rate** — not directly affected by the kernel choice, but by
  Gap 1 (what data gets fed to attention).

The **TPU `flash_attention` kernel already supports `causal=False` and allows
`q_seq_len ≠ kv_seq_len`** — it exists in the codebase at
`tpu_inference/kernels/flash_attention/kernel.py` but is not wired into our
DFlash model.

---

## Gap 3 — Context Accumulation: Noise History Lost

### What GPU does

In `spec_generate`:

```python
# GPU: DFlashDraftModel.spec_generate (lines 240-243)
position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
past_key_values=past_key_values_draft,
use_cache=True,
```

`position_ids` starts from `past_key_values_draft.get_seq_length()` — the
cache already holds K/V for positions 0…(start-1).  The model only processes
new tokens but the cache carries forward the full K/V history (context + all
accepted noise blocks).

### What our TPU does

- Context buffer stores `projected_hidden[0:seq_len]` — just the `fc` +
  `hidden_norm` output of target hidden states.
- Each iteration re-projects the **entire** buffer through K/V projections.
- **Past noise blocks' K/V contributions are completely lost** — the model
  only sees noise from the current block.

### Impact

This is the same core issue as Gap 1 from the proposer's perspective: the
GPU's `DynamicCache` integrates both context **and** noise K/V seamlessly;
our buffer only preserves context features, not the per-layer K/V projections
that include noise history.

---

## Gap 4 (Minor) — Position Embedding Handling

**GPU:** Applies RoPE to the full concatenated K (context + noise) with
positions spanning `[0, total_seq_len)`.  `DynamicCache` stores post-RoPE
K/V, so positions are correctly accumulated.

**TPU:** Our code derives `ctx_positions` as
`arange(ctx_len) + max(first_noise_pos - ctx_len, 0)` each iteration.  Since
we re-project K/V and re-apply RoPE every time, positions are redundantly
recomputed but not incorrect.  However, positions for context tokens shift as
the sequence grows, which could introduce subtle numerical drift.

---

## Gap 5 (Minor) — `qwen3_dflash.py`: Google's Parallel Attempt

There is a **second** DFlash implementation in the codebase:
`tpu_inference/models/jax/qwen3_dflash.py` (`Qwen3DFlashForCausalLM`).  This:

- Uses `ragged_paged_attention` for KV cache updates (causal-only — wrong
  for DFlash)
- Has a `concat_dense` mode using `dflash_concat_attention` (correct
  non-causal, but slow `fori_loop`)
- Requires `target_hidden_states.shape[0] == hidden_states.shape[0]` — same
  number of context and noise tokens — which violates DFlash semantics
  (context can be much larger than noise)

This model **is not currently loaded** (model registry maps
`"DFlashDraftModel"` → our `DFlashForCausalLM` in `dflash.py`), but it shows
that Google's team is also wrestling with the same gaps.

---

## Summary Table

| Feature | GPU (CUDA / PyTorch) | TPU (JAX) | Impact |
|---|---|---|---|
| **Mutable KV Cache** | `DynamicCache` — in-place append/crop per layer | No equivalent; JAX is functional | **Critical** — #1 cause of acceptance gap |
| **FlashAttention2** | CUDA kernel, non-causal, variable Q/KV lengths | Pallas `flash_attention` exists but **unused** | **Moderate** — affects throughput, not correctness |
| **SDPA** | `torch.nn.functional.scaled_dot_product_attention` | No direct equivalent; must use custom kernels | **Moderate** — affects throughput |
| **In-place tensor ops** | `torch.cat`, list append for cache growth | Must use `dynamic_update_slice` with pre-allocated buffers | **Critical** — blocks KV cache implementation |
| **Dynamic shapes** | Tensors can change size between calls | JIT requires static shapes or retraces | **High** — forces padding / recompilation |

---

## Recommended Path Forward

The **single most impactful change** is to implement a **static-shape,
pre-allocated KV cache for the draft model** on-device that mimics
`DynamicCache` behaviour:

1. **Pre-allocate** contiguous arrays of shape
   `(max_model_len, num_kv_heads, head_dim)` per layer for K and V on TPU
   HBM.
2. **Maintain a length counter** (scalar integer on host).
3. **Each iteration:** project K/V for **only** the new tokens (noise block),
   write them into the buffer via `jax.lax.dynamic_update_slice`, and use the
   TPU's `flash_attention` kernel (`causal=False`) to attend over the full
   buffer up to the length counter.
4. **On rejection:** decrement the length counter — stale data beyond the
   counter gets masked out by the attention kernel.
5. **Position handling:** store post-RoPE K/V so positions are baked in at
   insertion time, matching GPU semantics exactly.

This avoids the pytree-structure-change problem (always the same shape
arrays), avoids host↔device transfers for K/V data, and matches the GPU's
`DynamicCache` semantics.

The key insight is to use the **TPU's existing `flash_attention` kernel**
(`tpu_inference/kernels/flash_attention/kernel.py`) which already supports
`causal=False` and allows `q_seq_len ≠ kv_seq_len`, rather than
`ragged_paged_attention` (causal-only) or raw `jnp.einsum` (slow).

---

## File References

| File | Role |
|---|---|
| `zhongyan_dev/dflash/model/dflash.py` | Reference GPU DFlash model (PyTorch) |
| `zhongyan_dev/dflash/model/utils.py` | GPU helper functions (`extract_context_feature`, `sample`) |
| `tpu-inference/tpu_inference/models/jax/dflash.py` | Our TPU DFlash model (JAX/Flax NNX) |
| `tpu-inference/tpu_inference/spec_decode/jax/dflash.py` | Our TPU DFlash proposer |
| `tpu-inference/tpu_inference/models/jax/qwen3_dflash.py` | Google's alternative DFlash model |
| `tpu-inference/tpu_inference/layers/common/dflash_attention_interface.py` | `dflash_concat_attention` (pure JAX) |
| `tpu-inference/tpu_inference/kernels/flash_attention/kernel.py` | TPU flash attention Pallas kernel (**target**) |
| `tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py` | TPU ragged paged attention (causal-only) |
| `tpu-inference/tpu_inference/layers/jax/attention/attention.py` | Standard TPU attention layer (uses ragged paged) |

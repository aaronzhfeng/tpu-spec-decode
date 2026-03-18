# Doc 56: DFlash Concat Attention — Technical Understanding

Purpose: Document my understanding of `dflash_attention_interface.py` for cross-validation
before making any code changes to address kyuyeunk's PR review.

---

## What the reviewer asked

On `tpu_inference/layers/common/dflash_attention_interface.py`:

> can you elaborate what kind of feature is missing from existing attention implementation that it requires its own separate code? if it's due to bi-directional attention, we already have an implementation for that.

On `dflash_concat_attention` function:

> in general, i think this function is lacking a lot of comments explaning what each line does.

---

## My understanding of why separate code is needed

### The standard attention path

`attention()` in `attention_interface.py:370` takes a single `(q, k, v)` tuple and passes it
into `sharded_ragged_paged_attention`, which:
- Writes k, v into a **single paged KV cache** addressed by `block_tables`
- Reads from that same cache to compute attention output
- Returns `(output, updated_kv_cache)`

This is a fused write-then-read operation over one unified KV source.

### What DFlash needs

DFlash speculative decoding has two KV sources per layer:
- **context** (`k_ctx`, `v_ctx`): projected from target model hidden states (what the target model already computed)
- **noise** (`k_noise`, `v_noise`): projected from the draft/noise token embeddings (the noisy candidates being denoised)

The DFlash paper's semantics require each query token to attend over **both streams concatenated along the token axis**: `K = [k_ctx ; k_noise]`, `V = [v_ctx ; v_noise]`. The KV sequence length is 2x the query length.

After attention is computed, **only the noise stream** is persisted to the draft paged KV cache. The context stream is not stored in the draft cache — the proposer uploads fresh context tensors each `prepare_inputs()` call.

### Why the existing paged-cache `attention()` path can't express this

1. `sharded_ragged_paged_attention` addresses KV through block tables into a single cache. There is no interface to say "concatenate two separate KV tensors at attention time, then attend over the combined sequence."
2. The fused write+read means both streams would be cached. DFlash needs the attention output from the concatenated KV but must only cache the noise stream.

### Is this bi-directional attention?

Not exactly. The codebase does have a non-causal attention primitive — `sharded_flash_attention(..., causal=False)` in `attention_interface.py` — used by llama4, qwen2.5_vl, and even `models/jax/dflash.py` (the standalone proposer). So non-causal / bi-directional attention is not the missing feature.

The distinguishing feature is **dual-stream KV concatenation**: two separate KV sources must be concatenated at attention time, while only one stream (noise) is persisted to the draft KV cache. The existing non-causal attention primitives still operate on a single KV source and wouldn't express this dual-stream + selective-caching pattern.

### The additive_legacy alternative

There is an alternative mode (`additive_legacy` in `qwen3_dflash.py:181-203`) that **does** reuse the standard `attention()`:
```
k = k_ctx + k_noise   # element-wise addition
v = v_ctx + v_noise
attention(kv_cache, q, k, v, ...)
```

This works with the existing path because it collapses two streams into one before attention. But it's mathematically different from concatenation:
- Additive: `softmax(Q @ (K_ctx + K_noise)^T) @ (V_ctx + V_noise)`
- Concat:   `softmax(Q @ [K_ctx ; K_noise]^T) @ [V_ctx ; V_noise]`

The concat variant matches the DFlash paper and is used for correctness validation.

---

## Line-by-line understanding of `dflash_concat_attention`

Source: `tpu_inference/layers/common/dflash_attention_interface.py:27-129`

### Function signature (lines 27-38)

```python
@functools.partial(jax.jit, static_argnames=("max_query_len", ))
def dflash_concat_attention(
    q: jax.Array,        # [T, N, H]  — query from noise stream
    k_ctx: jax.Array,    # [T, K, H]  — keys from target hidden states
    k_noise: jax.Array,  # [T, K, H]  — keys from noise embeddings
    v_ctx: jax.Array,    # [T, K, H]  — values from target hidden states
    v_noise: jax.Array,  # [T, K, H]  — values from noise embeddings
    attention_metadata: AttentionMetadata,
    *,
    max_query_len: int,  — static for XLA compilation (used for fixed-size slicing)
    sm_scale: float,     — 1/sqrt(head_dim)
) -> jax.Array:
```

- T = total tokens across all requests in the batch (ragged)
- N = num query heads, K = num KV heads, H = head dim
- `max_query_len` is marked `static_argnames` so JIT recompiles per distinct value — needed because `lax.dynamic_slice_in_dim` requires a compile-time-known slice size

### Validation (lines 44-56)

Checks: max_query_len > 0, all tensors share same T, num_heads divisible by num_kv_heads.

### GQA/MQA head expansion (lines 58-63)

```python
kv_repeat = num_heads // num_kv_heads
if kv_repeat > 1:
    k_ctx = jnp.repeat(k_ctx, kv_repeat, axis=1)
    ...
```

For grouped-query attention (e.g. Qwen3 uses GQA), KV has fewer heads than Q. This repeats each KV head to match the query head count so the einsum dimensions align. After this, all tensors are `[T, N, H]`.

### Padding (lines 65-70)

```python
pad_len = max_query_len
q = jnp.pad(q, ((0, pad_len), (0, 0), (0, 0)))
```

Appends `max_query_len` zero rows along the token axis to every tensor. This ensures `lax.dynamic_slice_in_dim(x, start, max_query_len)` inside `fori_loop` never reads out of bounds — even for the last request whose tokens sit near the end of T.

JAX's `lax.fori_loop` requires all operations inside `_body` to have static shapes. The padding guarantees that every slice is exactly `[max_query_len, N, H]` regardless of which request we're processing.

### Per-request boundaries (lines 72-78)

```python
query_start_loc = attention_metadata.query_start_loc
req_lens = query_start_loc[1:] - query_start_loc[:-1]
```

`query_start_loc` is a prefix-sum array: request i's tokens start at `query_start_loc[i]` and its length is `query_start_loc[i+1] - query_start_loc[i]`. This is the standard ragged-batch addressing used throughout the codebase.

`request_distribution[2]` gives the active request count when request distribution metadata is present.

### Index ranges and mask value (lines 80-84)

```python
arange_q = jnp.arange(max_query_len)
arange_kv = jnp.arange(2 * max_query_len)
```

Pre-computed index arrays for building validity masks. KV range is 2x because after concatenation, each request has up to `2 * req_len` KV tokens (context + noise).

`mask_value = -0.7 * float(jnp.finfo(jnp.float32).max)` — large negative for masking out padding positions before softmax. The 0.7 factor avoids overflow when exponentiating.

### The per-request loop body (lines 86-126)

```python
def _body(i: int, current: jax.Array) -> jax.Array:
```

Processes one request at a time via `lax.fori_loop`. Each iteration:

**Step 1 — Slice this request's tokens (lines 87-107):**
```python
start = query_start_loc[i]
req_len = jnp.clip(req_len, 0, max_query_len)
q_blk = lax.dynamic_slice_in_dim(q, start, max_query_len, axis=0)
```
Extracts a fixed-size `[max_query_len, N, H]` window for each of q, k_ctx, k_noise, v_ctx, v_noise. Positions beyond `req_len` are padding (zeros from the pad step).

**Step 2 — Concatenate context and noise KV (lines 109-110):**
```python
k_blk = jnp.concatenate([k_ctx_blk, k_noise_blk], axis=0)  # [2*max_query_len, N, H]
v_blk = jnp.concatenate([v_ctx_blk, v_noise_blk], axis=0)
```
This is the core DFlash operation. Each query now has `2 * max_query_len` KV positions to attend over.

**Step 3 — Build validity masks (lines 112-114):**
```python
q_valid = arange_q < req_len           # which query positions are real
kv_valid_len = jnp.maximum(2 * req_len, 1)  # ctx + noise token count
kv_valid = arange_kv < kv_valid_len    # which KV positions are real
```
Masks ensure padding zeros don't contribute to attention.

**Step 4 — Scaled dot-product attention (lines 116-122):**
```python
logits = jnp.einsum("qnh,knh->nqk", q_blk, k_blk)  # [N, max_query_len, 2*max_query_len]
logits = logits * sm_scale
logits = jnp.where(kv_valid[None, None, :], logits, mask_value)  # mask padding KV
probs = jax.nn.softmax(logits, axis=-1)
out_blk = jnp.einsum("nqk,knh->qnh", probs, v_blk)  # [max_query_len, N, H]
out_blk = jnp.where(q_valid[:, None, None], out_blk, 0)  # zero out padding query outputs
```

Standard scaled dot-product, but over the concatenated KV. No causal mask — DFlash attention is non-causal (all positions attend to all positions).

**Step 5 — Write back (line 126):**
```python
return lax.dynamic_update_slice_in_dim(current, out_blk, start, axis=0)
```
Places this request's output into the correct position in the output buffer.

### Final output (lines 128-129)

```python
outputs = lax.fori_loop(0, num_reqs, _body, outputs)
return outputs[:num_tokens]   # strip padding added earlier
```

---

## How the caller uses it (`qwen3_dflash.py:204-239`)

The `concat_dense` path in `Qwen3DFlashAttention`:
1. Calls `dflash_concat_attention(q, k_ctx, k_noise, v_ctx, v_noise, ...)` → gets attention **output**
2. Calls standard `attention(kv_cache, q, k_noise, v_noise, ...)` → gets updated **KV cache** (discards output)

This two-call pattern exists because:
- The attention output needs concat semantics (both streams)
- The KV cache should only store the noise stream; context is not stored in the draft paged KV cache and is supplied by the proposer each `prepare_inputs()` call
- The standard `attention()` function handles the paged cache write mechanics

---

## Corrections applied after cross-validation (doc 57)

1. **Fixed:** Removed false claim that no non-causal/bidirectional attention exists. The codebase has `sharded_flash_attention(..., causal=False)` used by multiple models.
2. **Narrowed:** "The existing path can't do this" → "The existing paged-cache `attention()` path can't express this." The issue is specifically about dual-stream concat + selective caching, not about non-causal masking.
3. **Tightened:** "Context is transient/recomputed each step" → "Context is not stored in the draft paged KV cache; the proposer uploads fresh context each `prepare_inputs()` call."

# Doc 59: Proposed Inline Comments for `dflash_attention_interface.py`

Repo comment style (from `attention_interface.py`, `attention.py`, `qwen3_dflash.py`):
- Short, terse — shape annotations like `# q: (T, N, H)`
- What-not-why, brief context where non-obvious
- No long prose or paragraph-style comments

---

## Proposed comments (line references from current file)

```python
# Line 58-63: GQA head expansion
    # Expand KV heads to match query head count for GQA/MQA.
    kv_repeat = num_heads // num_kv_heads
    if kv_repeat > 1:
        ...

# Line 65-70: Padding
    # Pad so dynamic_slice_in_dim always has static size inside fori_loop.
    pad_len = max_query_len
    ...

# Line 72-78: Per-request boundaries
    # Per-request token offsets and lengths.
    query_start_loc = attention_metadata.query_start_loc
    req_lens = query_start_loc[1:] - query_start_loc[:-1]
    ...

# Line 80-81: Index ranges
    # KV range is 2x because context and noise are concatenated.
    arange_q = jnp.arange(max_query_len)
    arange_kv = jnp.arange(2 * max_query_len)

# Line 83: Mask value
    # Large negative for masking out padding positions before softmax.
    mask_value = -0.7 * float(jnp.finfo(jnp.float32).max)

# Line 86: Loop body
    def _body(i: int, current: jax.Array) -> jax.Array:
        # Process one request: slice, concat ctx+noise KV, attend, write back.

# Line 91-107: Slicing (no comment needed — straightforward dynamic_slice)

# Line 109-110: Concatenation
        # Concat context and noise KV along token axis. [2*max_query_len, N, H]
        k_blk = jnp.concatenate([k_ctx_blk, k_noise_blk], axis=0)
        v_blk = jnp.concatenate([v_ctx_blk, v_noise_blk], axis=0)

# Line 112-114: Masks
        # Mask out padding positions for both Q and KV.
        q_valid = arange_q < req_len
        ...

# Line 116-124: Attention (no comment needed — standard scaled dot-product)

# Line 126: Write back (no comment needed — obvious from function name)
```

---

## What NOT to comment

- Validation checks (lines 44-56) — self-explanatory
- Softmax + einsum (lines 116-122) — standard attention, anyone reading this code knows
- `lax.dynamic_slice_in_dim` calls (lines 91-107) — obvious from context
- Final return (lines 128-129) — obvious

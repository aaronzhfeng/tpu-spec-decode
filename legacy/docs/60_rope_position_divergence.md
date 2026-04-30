# Doc 60: RoPE Position Divergence Between GPU and TPU DFlash

Date: 2026-03-14

---

## Finding

The GPU and TPU DFlash implementations assign **different position embeddings**
to context and noise keys. This is a potential correctness issue.

---

## GPU code (`z-lab/dflash/model/dflash.py`)

```python
# Line 77-78: Concat BEFORE RoPE
k = torch.cat([k_ctx, k_noise], dim=1)  # [bsz, ctx_len + q_len, kv_heads, head_dim]

# Line 82: RoPE applied to concatenated k
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

The `apply_rotary_pos_emb` function (line 22-28):
```python
q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
k_embed = (k * cos) + (rotate_half(k) * sin)
```

- `q` gets the **last q_len** positions from cos/sin (noise positions)
- `k` gets the **full** cos/sin range (context positions followed by noise positions)

So in the GPU code:
- **context keys** get positions `[0, 1, ..., ctx_len-1]`
- **noise keys** get positions `[ctx_len, ctx_len+1, ..., ctx_len+q_len-1]`
- **query** gets positions `[ctx_len, ctx_len+1, ..., ctx_len+q_len-1]` (same as noise)

Context and noise keys have **different** position embeddings.

### Where position_ids come from

`spec_generate` (line 241):
```python
position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size]
```

`position_ids` is `torch.arange(max_length + block_size)` (line 212). So the draft
model receives absolute positions spanning from cache_length to start + block_size.

`DFlashDraftModel.forward` (line 178):
```python
position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

This produces cos/sin of length = len(position_ids) = (start + block_size) - cache_length.
On first decode step: len = num_input_tokens + block_size = ctx_len + q_len. This matches
the concatenated k length exactly.

---

## TPU code (`qwen3_dflash.py:168-178`)

```python
k_ctx = self.k_proj(target_hidden_states)
k_ctx = self.k_norm(k_ctx)
k_ctx = apply_rope(k_ctx, md.input_positions, ...)   # same positions

k_noise = self.k_proj(hidden_states)
k_noise = self.k_norm(k_noise)
k_noise = apply_rope(k_noise, md.input_positions, ...)  # same positions
```

Both k_ctx and k_noise use **`md.input_positions`** — the same position IDs.

So in the TPU code:
- **context keys** get positions `[p0, p1, ..., p_{q_len-1}]`
- **noise keys** get the **same** positions `[p0, p1, ..., p_{q_len-1}]`
- **query** gets the **same** positions

Context and noise keys have **identical** position embeddings.

---

## The difference

| | GPU | TPU |
|---|---|---|
| Context key positions | `[0, 1, ..., ctx_len-1]` | Same as noise |
| Noise key positions | `[ctx_len, ..., ctx_len+q_len-1]` | `md.input_positions` |
| Query positions | Same as noise | `md.input_positions` |

In the GPU code, ctx_len can be different from q_len (context comes from target model's
accepted tokens, noise is the current block). The position_ids span the full range, and
context keys get earlier positions while noise keys get later positions.

In the TPU code, both streams share the same positions, so context keys are positioned
as if they occupy the same slot as noise keys.

---

## Why this could matter

RoPE encodes relative distance between tokens. If context keys have positions 0-15 and
noise keys have positions 16-31, a query at position 20 will compute different attention
scores to a context key at position 5 (distance=15) vs a noise key at position 20
(distance=0). With identical positions, both would have distance=0, which changes the
relative attention pattern.

---

## Confirmed: standalone TPU proposer does it correctly

The standalone proposer (`models/jax/dflash.py:529-531`) constructs separate positions:

```python
ctx_positions = jnp.arange(T_padded) + pos_offset
noise_positions = jnp.arange(T_noise) + pos_offset + actual_ctx_count
```

Then concatenates them before applying RoPE on the combined key (`dflash.py:183-191`):

```python
new_positions = jnp.concatenate([ctx_positions, noise_positions], axis=0)
k_new = apply_rope(k_new, new_positions, ...)
```

This matches the GPU reference: context keys get earlier positions, noise keys get
later positions offset by `actual_ctx_count`.

## Confirmed: `qwen3_dflash.py` does NOT do this

`qwen3_dflash.py:170-178` applies the same `md.input_positions` to both:

```python
k_ctx = apply_rope(k_ctx, md.input_positions, ...)
k_noise = apply_rope(k_noise, md.input_positions, ...)
```

`md.input_positions` comes from `tpu_runner.py:1591-1594`:

```python
np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)
```

These are absolute sequence positions for the scheduled tokens, but they are the
**same array** applied to both context and noise keys. Context keys should have
earlier positions (matching the target model's hidden state positions), not the
noise block's positions.

## Impact

All three reference implementations (GPU repo, standalone TPU proposer) give
context and noise keys **different** RoPE positions. `qwen3_dflash.py` gives them
**identical** positions. This changes the relative distance encoding in attention
and is likely a correctness bug in the `concat_dense` path.

The `additive_legacy` path may be unaffected since it adds k_ctx + k_noise
element-wise (both with same positions), which is a different mathematical
operation anyway.

Our benchmark results (tau=6.67, 4.26x speedup) were measured with the standalone
proposer, not the `qwen3_dflash.py` path, so they do not validate the
`concat_dense` path's position handling.

## Recommended fix

`qwen3_dflash.py` needs separate position arrays for context and noise, similar
to the standalone proposer. This requires either:

1. Passing context positions through `AttentionMetadata` or `target_hidden_states`
2. Computing them in `Qwen3DFlashAttention` from the metadata (e.g. from
   `query_start_loc` and a context offset)

This is a code change beyond commenting and should be addressed separately from
the PR review response.

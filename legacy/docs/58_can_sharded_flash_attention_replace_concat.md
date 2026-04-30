# Doc 58: Can `sharded_flash_attention` Replace `dflash_concat_attention`?

Purpose: Investigate whether we can remove `dflash_concat_attention` and use the
existing `sharded_flash_attention(..., causal=False)` instead, to address the
reviewer's concern about introducing a separate attention implementation.

Date: 2026-03-13

---

## Question

The reviewer asks why DFlash needs its own attention code when the repo already
has non-causal attention (`sharded_flash_attention(..., causal=False)`).

Could we concat `k_ctx`/`k_noise` upstream in `qwen3_dflash.py` and pass the
result into the existing primitive?

---

## What `sharded_flash_attention` supports

Source: `attention_interface.py:51-75`, kernel at `kernels/flash_attention/kernel.py`

| Feature                        | Supported? |
|--------------------------------|------------|
| KV seq_len ≠ Q seq_len (2x)   | Yes        |
| Non-causal (`causal=False`)    | Yes        |
| segment_ids masking            | Yes        |
| GQA                            | Indirectly — kernel requires equal head counts (`kernel.py:95`), so caller must pre-expand KV heads (as `dflash.py:193` does) |
| Ragged per-request batching    | **No**     |
| 3D tensor input [T, N, H]     | **No** — requires 4D [B, N, T, H] |
| Pure JAX (no mesh required)    | **No** — requires mesh + shard_map |

---

## What `dflash_concat_attention` needs that the existing path lacks

### 1. Ragged per-request batching

`dflash_concat_attention` processes multiple requests packed into a single token
dimension, using `query_start_loc` to find each request's boundaries:

```
Request 0: tokens 0-2   (3 queries, 6 KV after concat)
Request 1: tokens 3-7   (5 queries, 10 KV after concat)
Request 2: tokens 8-9   (2 queries, 4 KV after concat)
Total T = 10
```

It loops over requests via `lax.fori_loop`, slicing and processing each one
independently. The KV concat happens **per-request** — each request's context
and noise are concatenated separately.

`sharded_flash_attention` has no `query_start_loc` or per-request loop. It
treats the batch dimension as uniform — every entry in `[B, N, T, H]` has
the same sequence length.

### 2. Per-request KV concatenation

The concat must happen per-request inside the attention loop, not globally.
A naive global `concat([k_ctx, k_noise], axis=0)` would produce `[2T, K, H]`
which conflates all requests' context and noise tokens together. Each query
would attend over all requests' KV, which is wrong.

### 3. Shape format

`dflash_concat_attention` works on 3D `[T, N, H]` ragged tensors.
`sharded_flash_attention` requires 4D `[B, N, T, H]` with explicit batch.

---

## Could we restructure inputs to make it work?

### Option A: Expand batch dim to num_requests

Reshape the packed `[T, N, H]` into `[num_reqs, max_query_len, N, H]`, pad
each request, concat KV per-request, and call `sharded_flash_attention` with
`batch=num_reqs`.

**Problems:**
- Requires unpacking ragged → padded batched format, then repacking output
- Adds a reshape + pad layer around the call, roughly equivalent complexity
  to what `dflash_concat_attention` already does
- Introduces mesh dependency where none currently exists
- Gains the Pallas kernel performance, but for small seq lengths (K=4-16
  speculative tokens), einsum may already be faster than kernel launch overhead

### Option B: Use segment_ids to isolate requests

Keep everything in a single batch entry `[1, N, T, H]`, use segment_ids to
prevent cross-request attention.

This is more feasible than initially assumed — `segment_ids` assigns per-token
IDs and the kernel masks attention between tokens with different IDs
(`kernel.py:21`, `kernel.py:685`). So assigning each request a distinct
segment ID would prevent cross-request attention.

**Remaining problems:**
- Still doesn't handle per-request KV concatenation. Each request's context
  and noise tokens must be interleaved correctly in the packed sequence, with
  matching segment IDs on both Q and KV sides.
- Packing the concatenated KV (2x length per request) into a single flat
  sequence with correct segment IDs requires the same kind of per-request
  reshape/pad logic we're trying to avoid.
- Padding within requests (positions beyond `req_len`) would need a reserved
  segment ID to mask out, adding further bookkeeping.

### Option C: Call per-request in a loop

Loop over requests, call `sharded_flash_attention` once per request.

**Problems:**
- Loses cross-request batching (same as current fori_loop, but heavier per call)
- Each call goes through shard_map + Pallas dispatch — overhead likely exceeds
  the simple einsum for small K

---

## What the standalone proposer does (`models/jax/dflash.py`)

The standalone DFlash proposer (`models/jax/dflash.py:247`) already uses
`flash_attention` with `causal=False`. But its setup is different:

- It maintains its **own on-device KV cache** (not paged)
- It processes **one request at a time** (batch=1, no ragged batching)
- Q and KV have the same sequence context (no dual-stream concat needed —
  it writes both context and noise into the same cache before attending)

So the standalone proposer doesn't face the dual-stream or ragged batching
problems. It can use `flash_attention` directly because it has a simpler
single-stream, single-request setup.

---

## Assessment

Replacing `dflash_concat_attention` with `sharded_flash_attention` is possible
but likely not worth the added packing/sharding complexity for this path. Each
option (A, B, C) requires reshape/pad/segment-ID scaffolding of similar or
greater complexity to what `dflash_concat_attention` already does, plus takes
on mesh/sharding dependencies that the current pure-JAX path avoids.

For speculative decoding query lengths (typically K=4-16 tokens per request),
the pure JAX einsum in `dflash_concat_attention` is likely comparable or faster
than Pallas kernel dispatch overhead.

---

## Recommendation for the reviewer response

Don't claim it's impossible to use the existing path. Instead:

1. Acknowledge that the repo has `sharded_flash_attention(..., causal=False)`
   and that non-causal masking is not the reason for separate code.
2. Explain the actual gap: the existing path doesn't support ragged per-request
   batching with dual-stream KV concatenation. The paged-cache `attention()`
   path fuses cache write+read over a single KV source.
3. Note that adapting `sharded_flash_attention` would require reshape/pad
   scaffolding of similar complexity, and for small speculative decoding
   sequence lengths the einsum path is efficient.
4. Offer to explore integration if the reviewer prefers — we're open to it
   if there's a cleaner way we're not seeing.

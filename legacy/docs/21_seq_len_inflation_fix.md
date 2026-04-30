# Doc 21: The seq_len Inflation Bug — Game-Changing Fix for DFlash TPU

## Result Summary

A single bug fix in `speculative_decoding_manager.py` nearly doubled DFlash performance:

| Metric | Before Fix | After Fix | Change |
|---|---|---|---|
| **Overall Speedup** | 1.30x | **2.31x** | +77% |
| **τ (tau)** | 2.49 | **4.48** | +80% |
| **Draft Acceptance Rate** | 9.9% | **23.2%** | +134% |
| **TPS (DFlash)** | 120.8 | **212.4** | +76% |
| TPS (Baseline) | 92.7 | 92.1 | stable |

Per-dataset speedup:

| Dataset | Before Fix | After Fix |
|---|---|---|
| AIME24 | 1.45x | **2.52x** |
| AIME25 | 1.19x | **2.56x** |
| Math500 | 1.35x | **2.18x** |
| GSM8K | 1.25x | **2.05x** |

Per-position acceptance rate:

| Position | Before Fix | After Fix |
|---|---|---|
| 0 | 48.2% | **77.9%** |
| 1 | 30.2% | **58.0%** |
| 2 | 19.5% | **43.1%** |
| 3 | 13.9% | **33.8%** |
| 4 | 10.3% | **26.8%** |

---

## The Bug

### Root Cause: Inflated Sequence Lengths Passed to the Proposer

In the vLLM speculative decoding pipeline, `attn_metadata.seq_lens` is computed as:

```
seq_lens = num_computed_tokens + num_scheduled_tokens
```

During a verification step, `num_scheduled_tokens` includes **all tokens being verified**:
the 1 bonus token + all 15 draft tokens. So if a request has 100 accepted tokens and
15 draft tokens under verification:

```
seq_lens[0] = 100 + 16 = 116    (inflated)
actual accepted count = 100       (correct)
```

The proposer receives `attn_metadata.seq_lens[0] = 116` when the true accepted
sequence length is only 100. This happens **every single iteration**.

### Where It Went Wrong

In `speculative_decoding_manager.py`, the `propose_eagle3_draft_token_ids()` method
(shared by both Eagle3 and DFlash) passed the raw `attn_metadata` directly to
`prepare_inputs()`:

```python
# BEFORE (broken):
target_hidden_states, input_ids, last_token_indices, attn_metadata = \
    self.runner.drafter.prepare_inputs(
        attn_metadata,       # ← contains inflated seq_lens
        input_ids,
        aux_hidden_states,
        next_token_ids,
        num_rejected_tokens,
    )
```

### Cascading Damage Inside the Proposer

The DFlash proposer (`spec_decode/jax/dflash.py`) reads `attn_metadata.seq_lens[0]`
as the authoritative sequence length. The inflated value corrupts three things:

1. **Context buffer pollution**: `_update_context_buffer(projected, seq_len)` uses
   `seq_len` to compute how many new tokens have appeared since the last call.
   With `seq_len=116` instead of `100`, it stores features for ~16 phantom positions
   that don't correspond to real tokens. These garbage entries enter the context
   buffer and get fed to the DFlash model as "target features".

2. **KV cache position corruption**: `self._prev_seq_len = seq_len` stores the
   inflated value. On the next iteration, the cache crop logic
   (`self._cache_len = self._prev_seq_len`) sets the cache write position to 116
   instead of 100, leaving a gap of stale K/V entries from the previous noise
   generation that never get overwritten.

3. **RoPE position drift**: Noise positions are computed as
   `arange(block_size) + cache_len`, where `cache_len` is derived from the inflated
   prev_seq_len. This shifts all rotary embeddings by ~10-15 positions every
   iteration, accumulating errors throughout generation.

### Diagnostic Confirmation

We added a temporary diagnostic to log the mismatch on every iteration:

```
seq_lens[0]=108, num_tokens_no_spec[0]=98,  delta=10, accepted=5
seq_lens[0]=121, num_tokens_no_spec[0]=109, delta=12, accepted=4
seq_lens[0]=134, num_tokens_no_spec[0]=120, delta=14, accepted=4
seq_lens[0]=147, num_tokens_no_spec[0]=131, delta=16, accepted=4
```

Every iteration showed a delta of 10-16 phantom tokens. The delta equals
`1 (bonus) + num_draft_tokens - num_accepted_tokens` — exactly the number of
rejected draft tokens that should NOT be counted.

---

## The Fix

### One Change, Four Lines

File: `tpu-inference/tpu_inference/runner/speculative_decoding_manager.py`

```python
# AFTER (fixed):

# Use the actual accepted seq_len (num_tokens_no_spec) instead of
# attn_metadata.seq_lens which includes unverified draft tokens.
accepted_seq_lens = self.runner.input_batch.num_tokens_no_spec[
    :attn_metadata.seq_lens.shape[0]].copy()
accepted_attn_metadata = replace(
    attn_metadata,
    seq_lens=device_array(
        self.runner.mesh,
        accepted_seq_lens.astype(np.int32)),
)

target_hidden_states, input_ids, last_token_indices, attn_metadata = \
    self.runner.drafter.prepare_inputs(
        accepted_attn_metadata,   # ← corrected seq_lens
        input_ids,
        aux_hidden_states,
        next_token_ids,
        num_rejected_tokens,
    )
```

### What `num_tokens_no_spec` Is

In `tpu_runner.py`, after rejection sampling completes (lines 984-994), the runner
updates `num_tokens_no_spec[req_idx] = end_idx` with **only** the accepted token
count. This is the ground truth for how many tokens the request actually has,
excluding any unverified draft tokens.

Meanwhile, `num_computed_tokens_cpu` in `persistent_batch_manager.py` (line 246)
is set from the scheduler's `num_computed_tokens`, which is also the accepted-only
count. But `attn_metadata.seq_lens` adds `num_scheduled_tokens` on top (line 1315-1317
of `tpu_runner.py`), which includes all draft tokens being verified.

### Why This Was Hard to Find

1. **No visible errors**: The inflated seq_len doesn't cause crashes or NaN outputs.
   The proposer still produces draft tokens — just bad ones.

2. **Eagle3 doesn't care**: The Eagle3 proposer (the other user of this code path)
   doesn't maintain a persistent context buffer or KV cache across iterations.
   It only uses `attn_metadata` for the current step, so the inflation is harmless.
   DFlash is the only proposer that tracks state across iterations.

3. **Gradual degradation**: The corruption accumulates gradually. Early in generation
   (short sequences), the delta is small relative to total length. As sequences grow,
   the corruption becomes a larger fraction of the context, but by then the draft
   quality has already been dragged down.

4. **Doc 20's investigation was correct**: All 13 verification checks in the prior
   investigation (A/B tests on flash attention, position scheme, no-cache mode)
   were valid. The bug was **upstream** of all those components — in the manager
   that calls the proposer, not in the proposer or model themselves.

---

## Comparison with GPU Paper Results

The DFlash paper (Table 1) reports for Qwen3-4B at temperature=0:

| Dataset | GPU τ | GPU Speedup | TPU τ (after fix) | TPU Speedup |
|---|---|---|---|---|
| GSM8K | 6.53 | 5.15x | 4.48 | 2.05x |
| Math500 | 7.84 | 6.09x | 4.48 | 2.18x |
| AIME24 | 7.27 | 5.68x | 4.48 | 2.52x |
| AIME25 | 6.64 | 5.21x | 4.48 | 2.56x |

Our τ=4.48 is now **69% of GPU quality** (up from 36% before the fix).

Note: The GPU paper results come from a standalone `spec_generate()` loop, not the
vLLM pipeline. The vLLM GPU path (`gpu_model_runner.py:467-470`) only enables
aux_hidden_states for `method == "eagle3"`, excluding DFlash. Our TPU implementation
runs inside the full vLLM pipeline, which adds overhead from scheduling, rejection
sampling, and batch management.

The remaining τ gap (4.48 vs 6.53) likely comes from:
- Pipeline overhead (vLLM scheduler loop vs standalone generation)
- Numerical precision differences (TPU bf16 vs GPU fp16/fp32)
- flash_attention kernel differences (TPU Pallas vs GPU FlashAttention-2)

---

## Files Modified

| File | Change |
|---|---|
| `tpu_inference/runner/speculative_decoding_manager.py` | Added `replace` import; create `accepted_attn_metadata` with corrected seq_lens from `num_tokens_no_spec`; pass to `prepare_inputs()` |

No changes to the proposer or model. The fix is entirely in the speculative decoding
manager — the orchestration layer between the target model runner and the draft proposer.

---

## Verification

- **Smoke test**: τ=3.90, acceptance=19.4%, TPS=112.3 (3 prompts, GSM8K)
- **Full benchmark**: τ=4.48, acceptance=23.2%, TPS=212.4, speedup=2.31x (32 prompts, 4 datasets)
- **Baseline stable**: 92.1 TPS (no regression from the fix)
- **All 32 prompts completed** with 0 errors

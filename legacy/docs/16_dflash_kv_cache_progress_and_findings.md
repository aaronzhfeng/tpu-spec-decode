# DFlash KV Cache Progress and Findings

Status: **In progress**
Date: 2026-02-12

## Current State

### Working Implementation (Phase 1 — Context Buffer)

- **Files**: zhongyan_dev-style `dflash.py` (model) + `dflash.py` (proposer)
- **Approach**: Proposer accumulates projected target hidden states in a host-side NumPy buffer. Each draft iteration, the full buffer (padded to power-of-2) is uploaded to the device. The model re-projects all context through k_proj/v_proj each time.
- **Results**: tau=2.38, acceptance=9.2%, pos0=44.7%, 1.27x speedup over baseline

### Benchmark Comparison (TPU v4-8)

| Method | Model | Speedup | Tau | Pos-0 Accept | TPS |
|--------|-------|---------|-----|-------------|-----|
| Baseline (AR) | Qwen3-4B | 1.00x | -- | -- | 92.9 |
| DFlash | Qwen3-4B | **1.27x** | 2.53 | 48.8% | 118.3 |
| Eagle3 | Llama3.1-8B | 1.53x | 2.18 | 59.7% | 78.4 |

### Config Fix

Fixed `num_speculative_tokens` from 16 to 15 (matching block_size=16, since position 0 is the known token and positions 1-15 are draft). Previous runs wasted position 15 with guaranteed 0% acceptance.

## DFlash Paper Findings (arXiv:2602.06036)

Key insight from the paper's "Target Knows Best" design:

> "DFlash directly injects the context feature into the Key and Value projections of every layer in the draft model. These projected features are stored in the draft model's KV cache and reused across drafting iterations, providing strong, consistent conditioning."

The reference DFlash uses `DynamicCache` that stores **both context AND noise K/V** from all iterations. After rejection, the cache is cropped to remove rejected tokens. This means:

1. Context K/V (from target hidden states projected through draft k_proj/v_proj) accumulate across iterations
2. **Noise K/V** from accepted draft tokens also remain in the cache
3. The draft model's attention sees the full history of its own K/V projections

## Gap Analysis: Why Tau=2.5 vs Reference Tau=6.5

### What's the same
- Model architecture (5 layers, same weights)
- Context feature extraction (post-layer aux hidden states from target)
- fc + hidden_norm projection
- Non-causal attention with RoPE
- Dense einsum attention

### What's different (identified root cause)

**Our context buffer only stores projected target hidden states.** The noise K/V from previous iterations are NOT preserved. In the reference:

- Iteration 1: cache = [k_ctx_prompt, k_noise_block1, ...] (then crop to accepted)
- Iteration 2: cache = [k_ctx_prompt_accepted, k_noise_accepted, k_ctx_new, k_noise_block2, ...]
- The accepted noise K/V from block1 remain in the cache and inform block2's attention

In our implementation:
- Each iteration: model receives FULL accumulated projected context + NEW noise block
- The noise K/V from previous iterations are LOST — only the target's context features survive
- The draft model can't attend to its own previous predictions' K/V representations

This means our draft model has less information about recent token history, leading to lower acceptance at later positions in the block.

## Failed KV Cache Attempt (Phase 2)

### What we tried
- Rewrote model attention to accept per-layer cached K/V buffers
- Changed `target_hidden_states` parameter to a tuple: `(new_ctx_hidden, cached_k_list, cached_v_list, cache_len, new_ctx_positions)`
- Proposer manages per-layer KV buffers on host (NumPy), uploads padded to device each iteration
- Model projects only new tokens' K/V, concatenates with cached, runs attention with masking

### What happened
- Acceptance collapsed from 44.7% to 3.5-7% at position 0 (worse than torchax baseline)
- Throughput dropped from ~80 TPS to ~30 TPS
- Fixed dtype (float32 buffers vs bfloat16 model) — no improvement
- Reverted to working Phase 1

### Root cause (not yet fixed)
The KV cache approach changed the model's `__call__` interface significantly by packing a tuple into `target_hidden_states`. The `model_fn` (JIT-compiled) may not trace correctly through this pytree structure, or there's a semantic bug in how the first-iteration context extraction works when `_kv_len=0` and `num_new_ctx = seq_len`.

### What's needed to debug
1. **Isolated test harness**: Run the DFlash attention on known inputs and compare output between context-buffer approach and KV-cache approach
2. **Shape tracing**: Print intermediate tensor shapes at each step to verify the new_ctx_hidden extraction is correct
3. **First-iteration focus**: The first draft iteration (prefill → first block) is where acceptance is worst — this is the iteration where the KV cache is empty and ALL context is "new"

## Recommended Next Steps

### Option A: Minimal KV Cache (noise-only)
Instead of caching ALL K/V (context + noise), only cache the noise K/V from accepted blocks. The context buffer continues to provide projected hidden states. This is simpler because:
- Context K/V re-projection is cheap (5 small layers)
- Only noise K/V need to be cached (block_size entries per iteration)
- No change to the model's `target_hidden_states` interface
- The noise KV buffer can be managed entirely in the proposer

### Option B: Full Debug of Phase 2
Fix the per-layer KV cache by:
1. Adding logging to trace tensor shapes and values at each step
2. Running with a single prompt and comparing intermediate values
3. Testing the model's `__call__` with the tuple interface in isolation

### Option C: On-Device KV State
Store KV buffers as `nnx.Variable` on the model's layers (stays on device, no host roundtrip). Each layer accumulates its own K/V across calls. Requires understanding nnx state mutation through model_fn.

## Files Modified

- `tpu-inference/tpu_inference/models/jax/dflash.py` — zhongyan_dev version (context buffer + padding-aware masking)
- `tpu-inference/tpu_inference/spec_decode/jax/dflash.py` — zhongyan_dev version (JAX-native proposer with context accumulation)
- `tpu-inference/tpu_inference/runner/kv_cache_manager.py` — comment update for DFlash dummy layer
- `tests/configs/*.json` — fixed num_speculative_tokens from 16 to 15
- `tests/configs/eagle3_llama_math.json` — Eagle3 benchmark config (new)
- `tests/configs/benchmark_math_dflash_only.json` — DFlash-only benchmark config (new)

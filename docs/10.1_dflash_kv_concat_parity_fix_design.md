# DFlash K/V Concatenation Parity Fix Design

## Goal

Fix the confirmed DFlash parity bug in TPU implementation where DFlash attention currently **adds** target/noise K/V projections instead of **concatenating** them along token axis.

This document defines a concrete implementation plan for the fix.

## Date and Scope

- Date: **February 6, 2026**
- Codebase scope: nested `tpu-inference/`
- Bug focus: `tpu-inference/tpu_inference/models/jax/qwen3_dflash.py`

## Problem Statement

Current TPU DFlash attention behavior:

- `k = k_proj(target_hidden_states) + k_proj(hidden_states)`
- `v = v_proj(target_hidden_states) + v_proj(hidden_states)`

Reference DFlash behavior (`external/dflash/model/dflash.py`):

- `k = cat([k_ctx, k_noise], dim=1)`
- `v = cat([v_ctx, v_noise], dim=1)`

Addition destroys separable context/noise token structure. Concatenation preserves both streams and changes attention distribution fundamentally.

## Constraints in Current TPU Runtime

1. Current default attention path is ragged paged attention through `tpu_inference/layers/common/attention_interface.py`.
2. Ragged paged kernel static validation (`tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`) requires equal token count for q/k/v first dimension.
3. DFlash concatenation requires `kv_seq_len = ctx_len + q_len`, so generally `kv_seq_len != q_len`.
4. Current DFlash integration reused EAGLE3 proposer loop and metadata contracts, which were built for causal incremental decode assumptions.

## Design Requirements

1. Enforce token-axis K/V concatenation semantics for DFlash attention.
2. Support `q_len != kv_len` in DFlash path.
3. Keep non-DFlash attention paths unchanged.
4. Keep DFlash changes scoped and testable.
5. Preserve a safe fallback while migrating.

## Chosen Approach

Implement a **DFlash-specific dense attention path** that supports `q_len != kv_len`, and route only DFlash model attention to this path.

### Why this approach

1. It directly models concatenation semantics.
2. It avoids invasive changes to ragged paged kernel validation and cache semantics.
3. TPU already has a flash attention kernel that supports different q/kv sequence lengths (`tpu_inference/kernels/flash_attention/kernel.py`).

## Architecture Changes

### 1) Add DFlash-specific attention interface

Create:

- `tpu-inference/tpu_inference/layers/common/dflash_attention_interface.py`

Add function (name can be finalized during implementation):

- `dflash_attention(...)`

Expected behavior:

1. Accept q, k_cat, v_cat with `q_len != kv_len`.
2. Use dense flash attention backend with `causal=False`.
3. Return attention outputs for q tokens only.
4. Keep DFlash isolated from default ragged paged path.

### 2) Update DFlash model attention to concat K/V

Modify:

- `tpu-inference/tpu_inference/models/jax/qwen3_dflash.py`

Changes:

1. Replace additive fusion with token-axis concat:
   - `k_cat = concat([k_ctx, k_noise], axis=token_dim)`
   - `v_cat = concat([v_ctx, v_noise], axis=token_dim)`
2. Update rotary handling for concatenated K tokens.
3. Call new DFlash attention interface instead of common ragged attention path.
4. Gate behavior with runtime option for rollback:
   - `dflash_attention_impl in {"concat_dense", "additive_legacy"}`
   - default: `concat_dense`

### 3) Keep proposer/manager contracts stable in this fix

No API contract changes in:

- `tpu-inference/tpu_inference/spec_decode/jax/dflash.py`
- `tpu-inference/tpu_inference/runner/speculative_decoding_manager.py`

This fix targets attention correctness first, minimizing blast radius.

## KV Cache Handling in This Fix

Use correctness-first behavior for the DFlash concat path:

1. Treat concatenated context K/V as ephemeral per forward call.
2. Do not force these context tokens into ragged draft KV cache format.
3. Keep existing draft KV cache object wiring unchanged for compatibility with runner interfaces.

Note: this may be less efficient than a fully cache-aware DFlash block implementation; efficiency tuning is a follow-up phase.

## Implementation Plan

### Phase A: Functional parity fix

1. Add DFlash attention interface file.
2. Switch DFlash attention from add to concat.
3. Add config flag and default to concat implementation.
4. Add focused unit tests.

### Phase B: Performance/semantics hardening

1. Optimize batching and sharding behavior in DFlash attention interface.
2. Evaluate whether DFlash should get a dedicated block-level proposer flow rather than inherited EAGLE3 loop.
3. Benchmark compile latency and throughput impacts.

## File-Level Change Plan

### New files

1. `tpu-inference/tpu_inference/layers/common/dflash_attention_interface.py`
2. `tpu-inference/tests/models/jax/test_qwen3_dflash_attention.py`

### Modified files

1. `tpu-inference/tpu_inference/models/jax/qwen3_dflash.py`
2. `tpu-inference/tpu_inference/runner/compilation_manager.py` (if new helper compilation hooks are needed)
3. `tpu-inference/tests/spec_decode/test_dflash.py`
4. `tpu-inference/tests/models/jax/test_qwen3_dflash.py`
5. Optional docs/support matrix updates after verification.

## Test Plan for the Fix

### Unit tests

1. `test_qwen3_dflash_attention_concat_not_add`:
   - deterministic tensors where additive and concatenative outputs differ.
   - assert implementation matches concat reference path.
2. `test_qwen3_dflash_attention_supports_q_kv_len_mismatch`:
   - q_len != kv_len should run successfully.
3. `test_dflash_attention_impl_flag`:
   - both `concat_dense` and `additive_legacy` switch correctly.

### Integration tests

1. existing runner/spec manager tests must remain green.
2. DFlash smoke path (`verification/sh/run_tpu_inference_dflash_smoke.sh`) must run.
3. DFlash eval script (`verification/sh/run_tpu_inference_dflash_eval.sh`) must run at least on small sample.

### Optional parity micro-test

If dependency environment allows, compare tiny-shape DFlash attention output against PyTorch reference implementation from `external/dflash/model/dflash.py`.

## Acceptance Criteria

1. No additive K/V fusion in default DFlash path.
2. DFlash attention path executes with `q_len != kv_len`.
3. New concat-focused unit tests pass.
4. Existing DFlash integration tests continue to pass.
5. Smoke/eval scripts run (at least dry-run in constrained environments; full runs in TPU env).

## Risks

1. Dense DFlash attention may increase memory/compile time versus ragged paged path.
2. Rotary-position handling for concatenated K sequence must be validated carefully.
3. Current proposer loop is still EAGLE-shaped; this fix is attention-correctness first, not full algorithmic parity completion.

## Open Questions

1. Should DFlash remain on inherited EAGLE proposal loop, or move to a dedicated block-diffusion proposer in next phase?
2. Do we require strict numerical parity with external DFlash now, or staged parity with measurable convergence targets?
3. Should `additive_legacy` fallback remain after initial validation, or be removed immediately to avoid silent regressions?

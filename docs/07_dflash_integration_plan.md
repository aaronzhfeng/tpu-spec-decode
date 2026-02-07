# DFlash Integration Plan for `tpu-inference`

## Objective

Integrate DFlash speculative drafting into the local `tpu-inference/` runtime so DFlash can be selected as a speculative method (similar to `ngram` and `eagle3`) in vLLM TPU execution.

This document is based on the repository snapshot reviewed on **February 6, 2026**.

---

## Implementation Status Snapshot (February 6, 2026)

Phase 1 wiring is now implemented in code under `tpu-inference/`:

- Added `dflash` method routing in runner and speculative manager.
- Added `DFlashProposer` in `tpu_inference/spec_decode/jax/dflash.py`.
- Added draft model registration (`DFlashDraftModel`, `Qwen3DFlashForCausalLM`).
- Added new JAX draft model in `tpu_inference/models/jax/qwen3_dflash.py`.
- Added Qwen3 target aux-hidden-state export path for `method == "dflash"`.
- Extended KV cache draft-layer allocation for multi-layer DFlash drafts.
- Extended speculative precompile helper path for `dflash`.
- Added DFlash-focused unit-test coverage (runner/spec_decode/model helpers).

Remaining to close full parity/perf objectives:

- DFlash algorithmic parity track (notably non-causal draft-attention behavior).
- Confirmed K/V fusion mismatch to external DFlash in current Phase 1 code:
  - current TPU path uses additive K/V fusion,
  - external DFlash uses token-axis concatenation.
- End-to-end TPU benchmark runs and baseline-retention validation.
- Support-matrix and user-facing docs updates after benchmark validation.

---

## Repositories Reviewed

- DFlash source:
  - `external/dflash/README.md`
  - `external/dflash/model/dflash.py`
  - `external/dflash/model/utils.py`
  - `external/dflash/benchmark.py`
- TPU runtime source:
  - `tpu-inference/tpu_inference/runner/tpu_runner.py`
  - `tpu-inference/tpu_inference/runner/speculative_decoding_manager.py`
  - `tpu-inference/tpu_inference/spec_decode/jax/eagle3.py`
  - `tpu-inference/tpu_inference/models/common/model_loader.py`
  - `tpu-inference/tpu_inference/models/jax/llama_eagle3.py`
  - `tpu-inference/tpu_inference/models/jax/qwen3.py`
  - `tpu-inference/tpu_inference/runner/kv_cache_manager.py`
  - `tpu-inference/tpu_inference/runner/compilation_manager.py`
  - `tpu-inference/tests/spec_decode/test_eagle3.py`
  - `tpu-inference/tests/runner/test_speculative_decoding_manager.py`
  - `tpu-inference/tests/runner/test_kv_cache_manager.py`
  - `tpu-inference/tests/e2e/test_speculative_decoding.py`

---

## DFlash: Functional Summary

From `external/dflash/model/dflash.py`, DFlash is a draft model that:

- Uses target-model hidden states from selected target layers (`target_layer_ids`).
- Projects concatenated target hidden states through an FC layer (`fc`) and normalization.
- Runs a draft stack over noise/token embeddings.
- Produces draft-token logits using the **target model LM head**.
- Uses block-wise speculative decode:
  - block input includes one known token + masked/noise tokens,
  - draft predicts multiple tokens in parallel,
  - target verifies, computes acceptance length, and continues.

Important details:

- DFlash currently targets Qwen3-family checkpoints.
- DFlash uses separate draft KV cache (`past_key_values_draft`) and crops it after accept/reject updates.
- DFlash attention is configured with `is_causal=False` in its PyTorch implementation.
- DFlash requires a mask token for block denoising.

---

## Current `tpu-inference` Spec Decode Architecture

Current methods:

- `ngram`
- `eagle3`

Current control points:

- Method selection in `tpu-inference/tpu_inference/runner/tpu_runner.py` (`_init_speculative_decoding`).
- Method dispatch in `tpu-inference/tpu_inference/runner/speculative_decoding_manager.py`.
- EAGLE3 proposer in `tpu-inference/tpu_inference/spec_decode/jax/eagle3.py`.
- Draft model load path via `get_model(..., is_draft_model=True)` in `tpu-inference/tpu_inference/models/common/model_loader.py`.
- Extra draft KV cache specs added only for `eagle3` in `tpu-inference/tpu_inference/runner/kv_cache_manager.py`.
- EAGLE3-specific precompile helpers in `tpu-inference/tpu_inference/runner/compilation_manager.py`.

Design constraints already encoded:

- Rejection/acceptance is handled centrally in runner/rejection sampler.
- Proposer must only return draft token IDs for active requests.
- Target model must expose auxiliary hidden states if proposer needs them.
- Draft model is integrated as a JAX model class + weight loader + registry entry.

---

## Fit Analysis: DFlash -> TPU Runtime

### Concept Mapping

- DFlash draft method -> new speculative method value (`"dflash"`).
- DFlash proposer loop -> new proposer class under `spec_decode/jax`.
- DFlash draft architecture -> new JAX draft model class (Qwen3-based).
- DFlash target hidden-state extraction -> target model aux-hidden-state export path.
- DFlash block size -> `num_speculative_tokens + 1` mapping.
- DFlash LM head usage -> use target compute-logits path for draft hidden states.

### Gaps to Close

1. No DFlash method plumbing exists (runner + manager + compile hooks).
2. No DFlash draft model exists in `tpu-inference/models/jax`.
3. No Qwen3 target aux-hidden-state extraction path for speculative methods.
4. KV cache manager currently assumes exactly one draft layer for `eagle3`.
5. Precompile manager has only EAGLE3-specific helper coverage.
6. Test suite does not include DFlash unit or e2e coverage.

### Key Technical Risk

DFlash PyTorch draft attention is explicitly non-causal. Current text decoder attention path in TPU runtime is centered on causal ragged paged attention. This is the main parity risk and should be treated as a dedicated implementation track.

---

## Recommended Integration Strategy

Implement DFlash using the same integration pattern as EAGLE3, but with Qwen3-specific draft/target logic and a phased parity plan.

### Phase 1 (Functional Integration, Method + Model Wiring)

Goal: end-to-end runnable `"dflash"` method in TPU runtime with deterministic draft token proposal flow and full scheduler compatibility.

Deliver:

- Method registration and dispatch.
- DFlash proposer skeleton integrated with runner metadata path.
- Qwen3 DFlash draft model class + weight loading.
- Target aux-hidden-state export in Qwen3 model path.
- KV cache allocation for draft layers based on draft model layer count.
- Unit tests for runner/manager/model plumbing.

### Phase 2 (Algorithmic Parity + Performance)

Goal: converge toward DFlash algorithm semantics and speedup targets.

Deliver:

- Non-causal draft attention parity path (or documented approximation if not feasible initially).
- DFlash-specific precompile coverage.
- e2e correctness and performance tests.
- support matrix/docs updates.

---

## File-by-File Change Plan

### 1) Add DFlash proposer

Create:

- `tpu-inference/tpu_inference/spec_decode/jax/dflash.py`

Responsibilities:

- `load_model`: load draft model via `get_model(..., is_draft_model=True)`.
- Share/route embeddings and logits decode path with target model as needed.
- `prepare_inputs`: reuse rejection-aware selection logic pattern from EAGLE3.
- `propose`: generate `num_speculative_tokens` draft IDs per request using DFlash draft forward.
- Maintain/update draft KV cache group(s) consistently with runner cache.

### 2) Runner method registration

Modify:

- `tpu-inference/tpu_inference/runner/tpu_runner.py`

Changes:

- In `_init_speculative_decoding`, add `"dflash"` branch creating `DFlashProposer`.
- Keep rejection sampler flow unchanged.

### 3) Spec decode manager dispatch

Modify:

- `tpu-inference/tpu_inference/runner/speculative_decoding_manager.py`

Changes:

- Add dispatch branch for `"dflash"` in `propose_draft_token_ids`.
- Add a `propose_dflash_draft_token_ids(...)` helper similar to EAGLE3 path.
- Reuse existing metadata path; keep output contract as `list[list[int]]`.

### 4) Draft model implementation

Create:

- `tpu-inference/tpu_inference/models/jax/qwen3_dflash.py`

Design:

- Qwen3-based draft model with:
  - DFlash attention/layers,
  - `combine_hidden_states` projection head (`fc`),
  - draft forward API compatible with `model_loader` / proposer,
  - `load_weights` mapping for DFlash checkpoint parameter names.

Also modify:

- `tpu-inference/tpu_inference/models/common/model_loader.py`

Changes:

- Register DFlash architecture key(s), likely including `DFlashDraftModel` (confirm from checkpoint config).
- Ensure `is_draft_model=True` resolves DFlash class for corresponding architecture.

### 5) Target hidden-state extraction for Qwen3

Modify:

- `tpu-inference/tpu_inference/models/jax/qwen3.py`

Changes:

- Add aux-hidden-state capture path for method `"dflash"` (similar to Llama EAGLE3 pattern).
- Use layer selection from draft config (e.g., explicit `target_layer_ids` if provided; otherwise deterministic builder).
- Return selected aux hidden states in model forward output tuple.

### 6) KV cache spec for DFlash draft layers

Modify:

- `tpu-inference/tpu_inference/runner/kv_cache_manager.py`

Changes:

- Extend method handling to include `"dflash"`.
- Allocate `draft_layer.{i}` specs for `i in range(draft_num_layers)` instead of hardcoding one layer.
- Derive `draft_num_layers` from `draft_model_config.hf_config`.

### 7) Compilation prewarm support

Modify:

- `tpu-inference/tpu_inference/runner/compilation_manager.py`

Changes:

- Extend speculative precompile dispatch to `"dflash"`.
- Add `_precompile_dflash_helpers()` for DFlash proposer/model helper functions.

### 8) Tests

Add/modify:

- `tpu-inference/tests/spec_decode/test_dflash.py` (new)
- `tpu-inference/tests/runner/test_speculative_decoding_manager.py`
- `tpu-inference/tests/runner/test_kv_cache_manager.py`
- `tpu-inference/tests/runner/test_tpu_runner.py`
- `tpu-inference/tests/models/jax/test_qwen3_dflash.py` (new)
- `tpu-inference/tests/e2e/test_speculative_decoding.py`

Test scope:

- Method dispatch and proposer contract.
- Input preparation under rejection/non-rejection.
- Draft token shape and deterministic token generation mocks.
- Draft KV cache spec sizing for multi-layer draft.
- End-to-end correctness and speedup sanity.

### 9) Documentation/support matrix updates

Modify after implementation stabilizes:

- `tpu-inference/support_matrices/feature_support_matrix.csv`
- `tpu-inference/support_matrices/nightly/feature_support_matrix.csv`
- `tpu-inference/docs/*` speculative decoding docs pages if present.

---

## Attention Compatibility Plan (Critical)

### Problem

DFlash draft attention uses non-causal behavior in its PyTorch implementation; current TPU text decode path is optimized around causal ragged paged attention.

### Plan

1. MVP path:
   - Implement DFlash draft method with existing causal-friendly primitives to validate integration plumbing and scheduler compatibility.
2. Parity path:
   - Add/route a non-causal attention path for DFlash draft layers (likely flash-attention based), while preserving KV-cache correctness.
3. Acceptance gate:
   - Compare acceptance length and output parity against reference DFlash runs before claiming parity.

If parity path is too costly initially, explicitly document MVP as "DFlash-inspired draft method on TPU" until non-causal parity is complete.

---

## Validation and Exit Criteria

Functional completion criteria:

- `speculative_config={"method":"dflash", ...}` initializes and runs.
- Proposer returns draft IDs for all active requests without shape errors.
- Rejection sampler path works with DFlash metadata and scheduler loops.
- No KV cache index/group errors with draft model enabled.

Quality criteria:

- Unit tests added for new branches and model/proposer behavior.
- e2e correctness test compares baseline vs DFlash speculative output behavior for deterministic settings.
- Performance benchmark shows measurable speedup vs non-spec baseline on at least one supported Qwen3 setup.

Companion artifacts:

- `docs/08_dflash_test_plan_and_reported_baselines.md`
- `verification/README.md`
- `verification/py/dflash_reported_baselines.py`
- `verification/py/compare_results_to_baseline.py`
- `verification/py/run_tpu_dflash_eval.py`
- `verification/py/check_tpu_inference_scope.py`
- `verification/sh/*.sh`
- `verification/config/tpu_inference_dflash_allowlist.txt`

---

## Open Questions to Resolve During Implementation

1. Exact architecture name(s) in DFlash checkpoint `config.json` for registry matching.
2. Source of mask token ID in TPU runtime (checkpoint config vs tokenizer metadata).
3. Whether draft block size must equal `num_speculative_tokens + 1` strictly for released checkpoints.
4. Non-causal attention implementation choice and acceptable parity threshold.
5. Initial support scope:
   - Qwen3-only first (recommended),
   - or generic hook for future model families.

---

## Recommended Execution Order

1. Add `"dflash"` method wiring (`tpu_runner` + manager + tests).
2. Add draft model registration + load path (`model_loader` + `qwen3_dflash.py`).
3. Add Qwen3 aux hidden-state extraction path.
4. Add KV cache and compile helper support.
5. Add unit tests for draft model/proposer.
6. Run deterministic correctness pass.
7. Tune parity/performance and update support matrices/docs.

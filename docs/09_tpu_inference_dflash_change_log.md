# TPU-Inference DFlash Change Log

## Purpose

Document all DFlash integration changes currently made inside the nested repository:

- `tpu-inference/`

This file is intended to be the single implementation record for the current DFlash integration and parity-fix work.

## Snapshot

- Date: **February 6, 2026**
- Nested repo: `tpu-inference/`
- Baseline commit referenced for the integration frame: `2b2780fa43a2ea6c6c3f6751b83771744e76b1a8`

## Scope Guard

Change scope is constrained by:

- `verification/config/tpu_inference_dflash_allowlist.txt`

Validation command:

```bash
bash verification/sh/check_tpu_inference_change_scope.sh
```

## File-By-File Inventory

### Speculative decoding method plumbing

1. `tpu-inference/tpu_inference/runner/tpu_runner.py`
- Added `DFlashProposer` import.
- Extended `_init_speculative_decoding` with `method == "dflash"` branch.
- `self.drafter` now supports DFlash construction in the same lifecycle as `ngram` and `eagle3`.

2. `tpu-inference/tpu_inference/runner/speculative_decoding_manager.py`
- Added `DFlashProposer` import.
- Added method dispatch for `dflash` in `propose_draft_token_ids`.
- Added `propose_dflash_draft_token_ids(...)`.
- Refactored model-based proposer path into shared helper `_propose_draft_token_ids_with_model(...)` used by both EAGLE3 and DFlash.

3. `tpu-inference/tpu_inference/spec_decode/jax/dflash.py` (new)
- Added `DFlashProposer(Eagle3Proposer)`.
- Reuses Eagle3 proposer loop/prepare/propose contract.
- Overrides `_get_draft_token_ids` to use **target model logits**:
  - calls `self.runner.compute_logits_fn(self.runner.state, hidden_states, lora_metadata)`
  - argmax over target logits
- This is the core DFlash-specific token-selection behavior in proposer logic.

### Draft model registration and implementation

4. `tpu-inference/tpu_inference/models/common/model_loader.py`
- Registered new architecture entries:
  - `DFlashDraftModel`
  - `Qwen3DFlashForCausalLM`
- Both resolve to `Qwen3DFlashForCausalLM` for draft-model loading path.

5. `tpu-inference/tpu_inference/models/jax/qwen3_dflash.py` (new)
- Added DFlash-specific Qwen3 draft model stack:
  - `_build_target_layer_ids(...)`
  - `_get_dflash_target_layer_ids(...)`
  - `Qwen3DFlashAttention`
  - `Qwen3DFlashDecoderLayer`
  - `Qwen3DFlashModel`
  - `Qwen3DFlashWeightLoader`
  - `Qwen3DFlashForCausalLM`
- Key implemented behavior:
  - Default DFlash attention path now uses concatenation semantics:
    - projects `k_ctx`, `k_noise`, `v_ctx`, `v_noise` separately
    - computes outputs with concat K/V semantics using `dflash_concat_attention`
  - Added runtime switch:
    - `dflash_attention_impl="concat_dense"` (default)
    - `dflash_attention_impl="additive_legacy"` (fallback)
  - Keeps existing draft KV-cache update wiring by updating cache from the noise stream.
  - Supports draft hidden-state conditioning via `combine_hidden_states` (`fc` + `hidden_norm`).
  - Returns residual list for compatibility with existing proposer loop shape expectations.
- Weight loading specifics:
  - Added filtered loading for DFlash-required weights.
  - Filter/mapping supports both key styles:
    - raw (`layers.*`, `fc.*`, ...)
    - prefixed (`model.layers.*`, `model.fc.*`, ...)

### Target-model aux hidden-state export

6. `tpu-inference/tpu_inference/models/jax/qwen3.py`
- Added DFlash layer-selection helpers:
  - `_build_target_layer_ids(...)`
  - `_get_dflash_target_layer_ids(...)`
- `Qwen3Model.__init__` now sets `self.aux_hidden_state_layers` when speculative method is `dflash`.
- Overrode `Qwen3Model.__call__` to collect and return auxiliary hidden states.
- Updated `Qwen3ForCausalLM.__call__` to return collected aux hidden states instead of always returning `[]`.
- Layer capture semantics align with DFlash intent:
  - global layer indexing (`self.start_layer + i`)
  - captures hidden state **after** selected layer execution.

### KV cache and compilation support

7. `tpu-inference/tpu_inference/runner/kv_cache_manager.py`
- Extended draft KV cache spec path from `eagle3` only to `("eagle3", "dflash")`.
- For DFlash, draft KV cache layers are dynamic:
  - `num_draft_layers = draft_hf_config.num_hidden_layers`
- Generates `draft_layer.{i}` specs for all draft layers.

8. `tpu-inference/tpu_inference/runner/compilation_manager.py`
- Extended speculative helper precompile dispatch to include `dflash`.
- In `_precompile_eagle3_helpers`:
  - added dynamic `num_aux_hidden_states` for DFlash based on draft model `target_layer_ids`
  - replaced fixed-size aux-hidden dummy tensors with list comprehensions using dynamic count
- This allows precompile shape coverage to match DFlash aux hidden-state cardinality.

### DFlash-specific attention interface

9. `tpu-inference/tpu_inference/layers/common/dflash_attention_interface.py` (new)
- Added `dflash_concat_attention(...)`.
- Implements DFlash concat attention semantics with per-request token slicing:
  - concatenates context/noise K/V on token axis per request
  - masks by valid query/KV lengths
  - supports GQA by repeating KV heads to match query heads.
- Returns outputs only for query-token rows.

## Test Changes in `tpu-inference/tests`

10. `tpu-inference/tests/runner/test_speculative_decoding_manager.py`
- Added dispatch test for `dflash` branch.
- Generalized model-based proposal test to parameterize both:
  - `eagle3`
  - `dflash`
- Verifies both methods share the same manager contract and invoke drafter correctly.

11. `tpu-inference/tests/runner/test_kv_cache_manager.py`
- Added `test_get_kv_cache_spec_with_dflash_multi_layer`.
- Verifies `draft_layer.0..N-1` spec creation for multi-layer DFlash draft config.

12. `tpu-inference/tests/runner/test_tpu_runner.py`
- Added `test_init_speculative_decoding_dflash`.
- Verifies `_init_speculative_decoding` instantiates `DFlashProposer` and rejection sampler.

13. `tpu-inference/tests/spec_decode/test_dflash.py` (new)
- Added proposer-unit tests for DFlash `_get_draft_token_ids`:
  - ensures target model logits path is used
  - validates output shape/type for draft token IDs.

14. `tpu-inference/tests/models/jax/test_qwen3_dflash.py` (new)
- Added tests for DFlash target-layer-id helper logic:
  - default layer-id layout
  - explicit `dflash_config.target_layer_ids` override
  - fallback parity between `qwen3.py` and `qwen3_dflash.py` helper implementations.

15. `tpu-inference/tests/models/jax/test_qwen3_dflash_attention.py` (new)
- Added concat-parity tests for DFlash attention:
  - concat output matches dense concat reference
  - concat path is different from additive behavior on deterministic tensors
  - GQA KV-head repeat behavior is validated
  - `Qwen3DFlashAttention` impl switch coverage:
    - `concat_dense` path calls concat interface and cache-update attention
    - `additive_legacy` path bypasses concat interface
    - invalid impl raises `ValueError`.

## End-to-End Runtime Behavior After These Changes

With speculative config `method="dflash"`:

1. Runner initializes `DFlashProposer`.
2. Speculative manager dispatches to DFlash proposal path.
3. Qwen3 target forward returns selected aux hidden states.
4. DFlash draft model combines target context features, runs draft layers, and uses concat K/V semantics by default in draft attention.
5. Draft token IDs are selected from target LM-head logits through proposer override.
6. KV cache manager allocates per-draft-layer draft cache specs.
7. Compilation manager precompiles helper paths with DFlash-appropriate aux hidden-state tensor counts.

## Validation Status in Current Workspace

Completed:

- `python -m py_compile` on modified/new integration and test files (including concat-parity test file): **pass**.
- `bash verification/sh/check_tpu_inference_change_scope.sh`: **pass**.
- Dry-run script checks:
  - `DRY_RUN=1 bash verification/sh/run_tpu_inference_dflash_smoke.sh`: **pass**.
  - `DRY_RUN=1 CATEGORY=math MAX_SAMPLES=4 bash verification/sh/run_tpu_inference_dflash_eval.sh`: **pass**.

Not completed in this environment:

- `pytest` execution (module missing).
- Full runtime import checks requiring `vllm` (module missing).

## Known Gaps / Follow-Up Work

1. Full parity to original DFlash algorithm still needs explicit validation in TPU runtime, including broader decode-loop behavior beyond attention K/V concat.
2. `concat_dense` currently assumes request-local query lengths fit the configured bound (`num_speculative_tokens + 1`), so stress testing on partial-prefill-heavy workloads is still required.
3. TPU benchmark runs are still needed for real observed speedup/tau comparison against extracted DFlash references.
4. After benchmarks pass, support matrices and user-facing feature docs should be updated to mark DFlash support level.

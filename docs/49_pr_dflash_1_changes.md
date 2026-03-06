# PR DFlash 1: Changes Applied

**Date:** 2026-03-05
**Branch:** `pr_dflash_1` on `aaronzhfeng/tpu-inference`
**Base:** `origin/main` (commit 39f24d73)

---

## Commit

`[Jax] Add DFlash speculative decoding support`

14 files changed, +2310 / -16 lines.

---

## New Files (8)

### Model & Proposer
- `tpu_inference/models/jax/dflash.py` (632 lines) -- DFlash JAX draft model (DFlashForCausalLM)
- `tpu_inference/models/jax/qwen3_dflash.py` (529 lines) -- Qwen3-specific DFlash variant with attention
- `tpu_inference/layers/common/dflash_attention_interface.py` (129 lines) -- dflash_concat_attention kernel
- `tpu_inference/spec_decode/jax/dflash.py` (374 lines) -- DFlashProposer (prepare_inputs, propose, sampling)

### Tests
- `tests/models/jax/test_qwen3_dflash.py` (51 lines) -- target layer ID selection tests
- `tests/models/jax/test_qwen3_dflash_attention.py` (289 lines) -- attention concat/additive/GQA tests
- `tests/spec_decode/test_dflash.py` (68 lines) -- proposer sampling tests

### CI
- `.buildkite/features/Speculative_Decoding-_DFlash.yml` (68 lines) -- correctness + performance CI pipeline

---

## Modified Files (6)

### `tpu_inference/models/common/model_loader.py` (+2)
- Registered `DFlashDraftModel` -> `DFlashForCausalLM` in `_MODEL_REGISTRY`

### `tpu_inference/models/jax/qwen3.py` (+85)
- Added `_build_target_layer_ids()` and `_get_dflash_target_layer_ids()` helpers
- Added `_init_aux_hidden_state_layers()` to `Qwen3Model` -- configures which target layers to capture for Eagle3 or DFlash
- Overrode `Qwen3Model.__call__()` to collect `aux_hidden_states` during forward pass
- Updated `Qwen3ForCausalLM.__call__()` to pass `aux_hidden_states` through (was returning `[]`)

### `tpu_inference/runner/tpu_runner.py` (+3)
- Added `dflash` case to drafter initialization -- imports and instantiates `DFlashProposer`

### `tpu_inference/runner/speculative_decoding_manager.py` (+39/-16)
- Added `dflash` method dispatch in `propose_draft_token_ids()`
- Renamed `propose_eagle3_draft_token_ids` -> `propose_draft_model_token_ids` (shared by Eagle3 and DFlash)
- Used `accepted_seq_lens` from `num_tokens_no_spec` to pass correct seq_lens to drafter
- Removed unused `Eagle3Proposer` import and `logger`
- Removed dead `draft_token_probs` code (neither proposer returns probs)

### `tpu_inference/runner/kv_cache_manager.py` (+7/-7)
- Extended draft KV cache allocation to cover `dflash` method (was eagle3-only)
- Changed `draft_num_layers = 1` -> `getattr(hf_config, 'num_hidden_layers', 1)` to read from config

### `tests/e2e/test_speculative_decoding.py` (+50)
- Added `get_dflash_test_prompts()` and registered in `get_test_prompts()`
- Added `test_dflash_correctness()` -- Qwen3-4B + z-lab/Qwen3-4B-DFlash-b16
- Added `test_dflash_performance()` -- min 1.5x speedup threshold

---

## Pre-commit Fixes Applied

All files were auto-formatted by the pre-commit pipeline:
- **isort**: fixed import ordering in `dflash.py`, `qwen3_dflash.py`, `test_qwen3_dflash.py`
- **yapf**: reformatted all 13 Python files
- **ruff**: auto-fixed lint issues

Final run: all hooks pass.

---

## Issues from Review (docs/48_pr_dflash_1_review.md) -- Resolution

| Issue | Status |
|-------|--------|
| C1. Nothing committed | Staged, pending user commit |
| C2. No DCO sign-off | `git commit -s` ready |
| C3. License headers on test files | Fixed (full Google LLC block) |
| C4. __pycache__ not staged | Verified clean |
| M1. No e2e test | Added correctness + performance |
| M2. No Buildkite CI YAML | Added |
| M3. Commit structure | Single commit |
| O1. Misleading function name | Renamed to propose_draft_model_token_ids |
| O2. Fragile tuple unpacking | Cleaned up, dead code removed |
| O3. qwen3.py __call__ override | Skipped (user: no comments) |
| O4. Hardcoded draft_num_layers | Reads from hf_config |
| O5. Pre-commit not run | Run, all passing |

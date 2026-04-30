# PR Readiness Review: DFlash Integration into tpu-inference

**Reviewed:** 2026-03-05
**Branch:** `pr_dflash_1` (at `origin/main`, zero commits ahead)
**Target:** `vllm-project/tpu-inference` main

---

## Status: NOT READY

All 12 changed/new files are uncommitted. Several structural and convention issues must be fixed.

---

## CRITICAL Issues

### C1. Nothing is committed
The branch is at the exact same commit as `origin/main`. 5 modified files are unstaged, 7 new files are untracked. Must commit before creating a PR.

### C2. No DCO sign-off
Repo requires `Signed-off-by:` on all commits (pre-commit hook + DCO bot). Use `git commit -s`.

### C3. License header mismatch on test files
The 3 new test files use SPDX-style headers:
```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```
All upstream files use the full Google LLC Apache 2.0 block. The `addlicense` pre-commit hook will reject these.

**Files to fix:**
- `tests/models/jax/test_qwen3_dflash.py`
- `tests/models/jax/test_qwen3_dflash_attention.py`
- `tests/spec_decode/test_dflash.py`

### C4. `__pycache__` / `.pyc` files present
~30+ `.pyc` files in working tree. `.gitignore` should block them, but verify they don't get staged.

---

## MAJOR Issues

### M1. No e2e test for DFlash
`tests/e2e/test_speculative_decoding.py` has no DFlash test. Eagle3 and Ngram both have `test_*_correctness` and `test_*_performance`. Must add at minimum `test_dflash_correctness`.

CONTRIBUTING.md: *"When checking in a new feature, we expect that you add relevant unit tests as well as CI tests."*

### M2. No Buildkite CI feature YAML
Eagle3 has `.buildkite/features/Speculative_Decoding-_Eagle3.yml`. DFlash needs a matching `Speculative_Decoding-_DFlash.yml`.

### M3. Commit structure undefined
PR_PROGRESS.md planned 5 commits. Currently all changes are lumped together. Recommended split:
- **Commit 1:** New DFlash files (pure additions) -- `dflash.py`, `qwen3_dflash.py`, `dflash_attention_interface.py`, `spec_decode/jax/dflash.py`, all 3 test files
- **Commit 2:** Integration into pipeline (modifications) -- `model_loader.py`, `tpu_runner.py`, `speculative_decoding_manager.py`, `kv_cache_manager.py`, `qwen3.py`

---

## MODERATE Issues

### O1. `speculative_decoding_manager.py` -- misleading function reuse
Line 90-96: DFlash method branch calls `self.propose_eagle3_draft_token_ids(...)`. Works because they share the proposer interface, but function name is misleading. Either rename to something generic (e.g., `propose_draft_model_token_ids`) or add a comment.

### O2. `speculative_decoding_manager.py` -- fragile tuple unpacking
Lines 163-168: `if len(propose_output) == 3` is brittle. Better to have proposers always return a consistent 3-tuple (with `None` for probs when N/A), or use a named return type.

### O3. `qwen3.py` -- full `__call__` override duplicates parent logic
The `Qwen3Model.__call__` override duplicates the entire forward loop from `Qwen2Model` just to add aux hidden state collection. Will break if parent changes. Add a comment noting this coupling.

### O4. `kv_cache_manager.py` -- hardcoded `draft_num_layers = 1`
Line 161: `draft_num_layers = 1` is hardcoded. DFlash models can have multiple layers. Should read from `hf_config.num_hidden_layers`.

### O5. Pre-commit hooks not run
Repo uses `yapf`, `isort`, `ruff`, `addlicense`, and others. New files haven't been formatted. Known issue: `qwen3_dflash.py` has backslash-continuation imports that isort will flag.

---

## What's in GOOD shape

- All 4 new source files have proper Apache 2.0 Google LLC license headers
- No debug `print()` statements
- No TODO/FIXME comments in new code
- 3 test files with 7 test functions covering attention, layer IDs, proposer sampling
- Code follows existing patterns (mirrors Eagle3 in tpu_runner.py, model_loader.py)
- `torchax/` bridge correctly excluded (not present upstream)
- `dflash_attention_interface.py` correctly placed in `layers/common/`

---

## Fix Checklist

```
[ ] C1. Stage and commit all files
[ ] C2. Use `git commit -s` for DCO sign-off
[ ] C3. Fix license headers on 3 test files (use full Google LLC Apache 2.0 block)
[ ] C4. Verify __pycache__ not staged
[ ] M1. Add e2e test(s) to tests/e2e/test_speculative_decoding.py
[ ] M2. Add .buildkite/features/Speculative_Decoding-_DFlash.yml
[ ] M3. Decide and implement commit structure
[ ] O1. Rename or comment propose_eagle3_draft_token_ids reuse
[ ] O2. Standardize propose() return type
[ ] O3. Add comment on qwen3.py __call__ override coupling
[ ] O4. Fix hardcoded draft_num_layers = 1 in kv_cache_manager.py
[ ] O5. Run `pre-commit run --all-files` and fix formatting
[ ] Write PR description following template (Description, Tests, Checklist)
```

---

## Files Changed (12 total)

### New files (7):
- `tpu_inference/layers/common/dflash_attention_interface.py` -- 119 lines
- `tpu_inference/models/jax/dflash.py` -- 602 lines
- `tpu_inference/models/jax/qwen3_dflash.py` -- 525 lines
- `tpu_inference/spec_decode/jax/dflash.py` -- 390 lines
- `tests/models/jax/test_qwen3_dflash.py` -- 42 lines
- `tests/models/jax/test_qwen3_dflash_attention.py` -- 279 lines
- `tests/spec_decode/test_dflash.py` -- 58 lines

### Modified files (5, +161 lines):
- `tpu_inference/models/common/model_loader.py` -- +2 lines (DFlashDraftModel registry)
- `tpu_inference/models/jax/qwen3.py` -- +85 lines (aux_hidden_states restoration)
- `tpu_inference/runner/kv_cache_manager.py` -- +6/-6 lines (DFlash KV cache compat)
- `tpu_inference/runner/speculative_decoding_manager.py` -- +79/-13 lines (DFlash dispatch + draft_token_probs)
- `tpu_inference/runner/tpu_runner.py` -- +3 lines (DFlashProposer init)

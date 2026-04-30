# DFlash Parity Findings and Remediation

Status: **In progress (core parity fixes applied, re-validation running)**
Date: 2026-02-11

## Purpose

Capture the recent parity audit between our TPU integration and `external/dflash`, the key behavioral findings, and the exact code changes made to remediate them.

This document complements:

- `docs/13_dflash_integration_backport_report.md` (runnability backport)
- `docs/09_tpu_inference_dflash_change_log.md` (broader file inventory)

## Scope

Focused on the most recent investigation cycle in these areas:

- `tpu-inference/tpu_inference/models/jax/qwen3.py`
- `tpu-inference/tpu_inference/spec_decode/jax/dflash.py`
- `verification/contribution/py/run_matrix.py`
- `verification/contribution/manifests/*.json`
- `verification/contribution/sh/run_contribution_matrix.sh`

## What We Discovered

### 1) Rejection-aware context trimming was missing in DFlash proposer

Observation:

- `SpeculativeDecodingManager` computes and passes `num_rejected_tokens`.
- DFlash proposer accepted that argument but did not use it to trim context.

Why it matters:

- After target verification, rejected draft tail tokens should not remain in the next draft context.
- Keeping stale tail context can degrade draft/target agreement and acceptance.

### 2) Target aux hidden-state indexing likely mismatched reference semantics

Reference behavior (`external/dflash/model/utils.py`):

- Uses `hidden_states[layer_id + 1]` (post-layer state with embedding offset).

Our target export path originally:

- Captured hidden states before each selected layer pass.

Why it matters:

- Feeding shifted layer features into DFlash conditioning (`fc + hidden_norm`) can materially reduce acceptance quality.

### 3) Harness prompt formatting differed from external benchmark flow

Reference behavior (`external/dflash/benchmark.py`):

- Prompts are wrapped with tokenizer chat template (`add_generation_prompt=True`, `enable_thinking=False`).

Harness behavior before fix:

- Prompt text was passed directly to `llm.generate(...)`.

Why it matters:

- Prompt formatting differences can shift output-token distribution and acceptance behavior.

### 4) One parity gap remains open

Current DFlash torchax wrapper call path still uses:

- `past_key_values=None`
- `use_cache=False`

This differs from external benchmark behavior where draft cache is used and cropped per step.

## Behavior Observed During Diagnostics

From diagnostic run:

- `verification/outputs/contribution/contrib_diag_20260211_211807/summaries/overall.json`

Key metrics (method-level):

- baseline TPOT: `0.0116009`
- dflash TPOT: `0.0163496`
- speedup: `0.7096x` (slowdown)
- `tau`: `1.0434`
- `draft_acceptance_rate`: `0.002894` (0.289%)
- `num_draft_tokens`: `446044`
- `num_accepted_tokens`: `1291`

Interpretation:

- This is acceptance collapse (very low accepted draft-token fraction), consistent with the parity drifts identified above.

## Changes We Made

### A) Align target aux hidden-state capture for DFlash

File:

- `tpu-inference/tpu_inference/models/jax/qwen3.py`

Change:

- Added method-aware capture mode:
  - `capture_aux_after_layer=False` by default
  - set `capture_aux_after_layer=True` for `method == "dflash"`
- DFlash aux states are now captured after selected layer execution.

Effect:

- Closer to `external/dflash` `hidden_states[layer_id + 1]` semantics.

### B) Make DFlash proposer rejection-aware and trim padded context

File:

- `tpu-inference/tpu_inference/spec_decode/jax/dflash.py`

Changes:

- Added `_resolve_context_seq_len(...)` to:
  - derive effective sequence length from metadata,
  - subtract rejected-token count when present,
  - clamp to valid bounds.
- Updated `_build_noise_block_and_context(...)` to:
  - trim aux hidden states to effective `seq_len`,
  - build context+noise position ids from trimmed length.
- Added guard for current supported batch shape:
  - `max_num_seqs=1` path.

Effect:

- DFlash draft context now excludes rejected tail tokens and padding.

### C) Add chat-template parity mode to contribution harness

File:

- `verification/contribution/py/run_matrix.py`

Changes:

- Added manifest field support:
  - `use_chat_template` (default `false`)
- Added tokenizer-driven prompt formatter:
  - applies chat template when enabled
  - fallback path when tokenizer/options differ.
- Applied same formatting to warmup prompts and measured prompts.
- Added prompt trace fields:
  - `generation_prompt_sha256`
  - `used_chat_template`

Effect:

- Harness behavior can match external benchmark prompt construction.

### D) Enable chat-template mode in key manifests

Files:

- `verification/contribution/manifests/dflash_diagnostic_near_demo.json`
- `verification/contribution/manifests/dflash_demo_tasks_medium.json`

Change:

- Set `"use_chat_template": true`.

### E) Prevent host Python-version mismatch regressions

File:

- `verification/contribution/sh/run_contribution_matrix.sh`

Changes:

- Auto-resolve Python interpreter with Python 3.11+ requirement.
- Respect `PYTHON_BIN` override with validation.
- Emit explicit error if only older Python is available.

Also documented in:

- `verification/contribution/README.md`

## Validation Status

Completed:

- Linter checks on modified files: pass.
- Host-run failure mode for Python 3.10 is now prevented by runner guard.

Pending:

- Full benchmark rerun after environmental constraints (disk/cache) are stabilized.
- Compare new `tau` and acceptance against the previous diagnostic baseline.

## Expected Outcome of This Remediation

These fixes target acceptance collapse at the integration level:

- Better conditioning features (aux layer alignment),
- Correct context state after rejection events,
- Prompt-construction parity with external benchmark.

They do not yet address draft KV-cache parity, which remains a candidate for additional acceptance/speedup improvement.

## Next Recommended Step

After the active run completes successfully:

1. Compare method summary and per-dataset summary against `contrib_diag_20260211_211807`.
2. Check whether:
   - `draft_acceptance_rate` increases materially,
   - `tau` moves away from ~1.0,
   - TPOT speedup approaches or exceeds 1.0.
3. If still low, prioritize draft-cache parity in torchax DFlash wrapper.


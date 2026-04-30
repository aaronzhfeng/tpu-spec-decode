# PR DFlash 1: Progress & Problems

**Date:** 2026-03-05
**Branch:** `pr_dflash_1` on `aaronzhfeng/tpu-inference`
**Target:** `vllm-project/tpu-inference` main
**PR:** https://github.com/vllm-project/tpu-inference/pull/1868

---

## Timeline

### Phase 1: PR Readiness Review

Reviewed the full repo structure, upstream PR conventions (Eagle3 PRs), CONTRIBUTING.md, pre-commit config, and Buildkite CI setup. Produced `docs/48_pr_dflash_1_review.md` with 5 critical, 3 major, and 5 moderate issues.

### Phase 2: Benchmark Verification (parallel agent)

A separate agent ran 9 benchmark datasets to verify DFlash speedup on TPU. All passed with acceptable speedup ratios. Results confirmed before proceeding to fixes.

### Phase 3: Issue Fixes

All issues from the review were addressed:

- **C1-C4 (Critical):** Staged files, prepared DCO sign-off, fixed license headers on 3 test files (SPDX -> full Google LLC Apache 2.0 block), verified no __pycache__ staged.
- **M1-M3 (Major):** Added e2e tests (`test_dflash_correctness`, `test_dflash_performance`), added Buildkite CI YAML, decided on single-commit structure.
- **O1-O5 (Moderate):** Renamed `propose_eagle3_draft_token_ids` -> `propose_draft_model_token_ids`, removed dead `draft_token_probs` code and unused imports, fixed hardcoded `draft_num_layers`, ran pre-commit hooks.

### Phase 4: Commit & Push (problems began)

### Phase 5: Eagle3 regression fix

Discovered that sharing `propose_draft_model_token_ids` between Eagle3 and DFlash changed Eagle3's behavior -- we passed `accepted_attn_metadata` (with overridden seq_lens) instead of the original `attn_metadata` to Eagle3's `prepare_inputs`.

**Fix:** Restored Eagle3's original method `propose_eagle3_draft_token_ids` untouched. Added a separate `propose_dflash_draft_token_ids` with the `accepted_attn_metadata` logic. Restored `Eagle3Proposer` import. The diff against upstream now only adds code, no Eagle3 behavior change.

### Phase 6: PR Split

Reviewer (Lumosis) requested splitting the large 14-file PR into smaller PRs. Studied how Eagle3 was split (#591 skeleton, #671 model, #863 e2e tests, #1178 CI).

Split into 3 PRs:

| PR | Branch | Files | Content |
|----|--------|-------|---------|
| #1868 (updated) | `pr_dflash_1` / `pr_dflash_1a` | 7 new files | Model + proposer + unit tests |
| PR 2 | `pr_dflash_1b` | 5 modified files | Pipeline integration |
| PR 3 | `pr_dflash_1c` | 2 files (1 new + 1 modified) | E2E tests + Buildkite CI |

Force-pushed `pr_dflash_1a` to `pr_dflash_1` to update #1868. Updated PR title to `[Spec Decoding] Add DFlash model and proposer` and description. Replied to reviewer explaining the split and why PR 1 can't be split further (model, proposer, and attention kernel are tightly coupled).

---

## Problems Encountered

### Problem 1: AI-like commit message

**What happened:** Initial commit message was multi-line with bullet points describing every change -- looked obviously AI-generated.

**Fix:** User rejected it. Switched to single-line format matching upstream Eagle3 convention: `[Spec Decoding] Add DFlash block-diffusion draft model`.

### Problem 2: Removed upstream comments

**What happened:** While cleaning up `speculative_decoding_manager.py`, pre-existing upstream comments were accidentally removed:
- "Cached draft tokens."
- "Common case."
- "Partial prefill (rare case)."
- "Get the next token id from the request state."
- "Pad the batch size to match with existing padding for target model"

Same issue in `kv_cache_manager.py`: removed "Eagle3 has only 1 layer" comment.

**Fix:** Restored all original comments. The rule: never remove comments that existed before our changes.

### Problem 3: Excessive comments in dflash.py

**What happened:** `tpu_inference/models/jax/dflash.py` had 47 non-license comments -- numbered step narrations, section headers, obvious commentary. Eagle3's equivalent file has ~24 comments. The verbose style looked AI-generated.

**Fix:** Stripped down to 17 comments. Removed all numbered step comments, section headers, and obvious narration. Kept only comments that add information not obvious from the code itself. Reworded the one kept comment to match Eagle3's terse style.

### Problem 4: Force push went one commit too far

**What happened:** After pushing the initial (bad) commit and wanting to revert, `git reset --soft HEAD~1` was run twice -- once by the user, once by the agent. This landed at commit `6230d35d`, one behind `origin/main` (`39f24d73`).

**Fix:** `git reset --soft origin/main` to realign with the correct base, then `git push origin pr_dflash_1 --force` to fix the remote.

### Problem 5: Rebase picked up upstream changes

**What happened:** After `origin/main` advanced (5 new upstream commits since `39f24d73`), running `git reset --soft origin/main` staged 24 files instead of 14 -- our 14 DFlash files plus 10 upstream files that had changed.

**Fix:** Identified the 10 upstream-only files, ran `git reset HEAD` on each, then `git checkout` to restore them. Verified only our 14 files remained staged.

### Problem 6: Pre-commit required multiple passes

**What happened:** First `pre-commit run --all-files` auto-fixed formatting (isort, yapf, ruff). But auto-fixes meant the staged content no longer matched, so a second run was needed after re-staging.

**Fix:** Re-staged fixed files, ran pre-commit again. All hooks passed on second run.

### Problem 7: Eagle3 regression in shared method

**What happened:** Renamed `propose_eagle3_draft_token_ids` to `propose_draft_model_token_ids` and added `accepted_attn_metadata` logic for DFlash. This changed the `attn_metadata` passed to Eagle3's `prepare_inputs`, breaking Eagle3's behavior.

**Fix:** Kept Eagle3's original method untouched. Added separate `propose_dflash_draft_token_ids` for DFlash. No shared method -- each has its own code path.

---

## Current State -- All 3 PRs Submitted

| PR | Branch | Title | Files | Status |
|----|--------|-------|-------|--------|
| [#1868](https://github.com/vllm-project/tpu-inference/pull/1868) | `pr_dflash_1` | [Spec Decoding] Add DFlash model and proposer | 7 new files (~2000 lines) | Submitted, awaiting review |
| [#1869](https://github.com/vllm-project/tpu-inference/pull/1869) | `pr_dflash_1b` | [Spec Decoding] Integrate DFlash into speculative decoding pipeline | 5 modified files (~230 lines) | Submitted, depends on #1868 |
| [#1870](https://github.com/vllm-project/tpu-inference/pull/1870) | `pr_dflash_1c` | [Spec Decoding] Add DFlash e2e tests and Buildkite CI | 2 files (~120 lines) | Submitted, depends on #1869 |

**CI notes:** Buildkite pipeline upload fails in 9 seconds on all PRs -- this is a first-time contributor workflow approval issue, not a code problem. DCO passes on all PRs. Maintainer needs to approve workflow runs.

**Merge order:** #1868 → #1869 → #1870 (sequential dependency)

---

## Lessons

1. Match the repo's commit message style exactly -- check recent upstream PRs before writing.
2. Never remove or modify pre-existing comments when editing upstream files.
3. Keep new code comments minimal and terse -- match the density of comparable files in the repo.
4. When multiple people are operating on git state, coordinate resets carefully to avoid double-stepping.
5. After rebasing onto a moved upstream main, verify the staged file list before committing.
6. Pre-commit auto-fixes require a re-stage + re-run cycle.
7. Don't merge code paths for different spec decode methods -- keep Eagle3 and DFlash separate to avoid regressions.
8. When asked to split a PR, study how similar features were split in the same repo.

---

## Related Docs

- `docs/48_pr_dflash_1_review.md` -- PR readiness review
- `docs/49_pr_dflash_1_changes.md` -- Detailed change log
- `docs/51_pr_dflash_1_description.md` -- PR descriptions for all 3 PRs
- `docs/52_pr_dflash_1_reply.md` -- Reply to reviewer on PR split

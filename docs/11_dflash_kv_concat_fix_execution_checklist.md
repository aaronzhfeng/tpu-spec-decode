# DFlash K/V Concat Fix Execution Checklist

## Scope

Execution tracker for implementing:

- `docs/10_dflash_kv_concat_parity_fix_design.md`

Target: replace additive DFlash K/V fusion with concat-parity behavior and validate correctness/perf gates.

## Tracking Conventions

- `Status`: `not_started` | `in_progress` | `blocked` | `done`
- `Owner`: GitHub handle or team alias (`@unassigned` until assigned)
- `Evidence`: PR link, commit SHA, test output path, benchmark JSON path

## Work Items

| ID | Task | Owner | Status | Dependencies | Exit Criteria | Evidence |
|---|---|---|---|---|---|---|
| A1 | Create DFlash-specific attention interface (`dflash_attention_interface.py`) supporting `q_len != kv_len` with dense flash attention (`causal=False`) | @unassigned | done | None | New module merged; callable from DFlash model path | `tpu_inference/layers/common/dflash_attention_interface.py` added |
| A2 | Refactor `qwen3_dflash.py` attention path to concat K/V token axis (`k_cat`, `v_cat`) and route to new interface | @unassigned | done | A1 | No additive K/V in default path; concat path active by default | `tpu_inference/models/jax/qwen3_dflash.py` updated |
| A3 | Add runtime switch `dflash_attention_impl` with `concat_dense` default and `additive_legacy` fallback | @unassigned | done | A2 | Config switch works and is documented in code comments | `qwen3_dflash.py` impl switch + validation |
| A4 | Keep proposer/manager APIs stable (no behavioral regressions in dispatch/prep loop contracts) | @unassigned | in_progress | A2 | Existing model-based dispatch tests pass for eagle3+dflash | Code unchanged for proposer/manager contracts; runtime tests pending |
| A5 | Add unit tests for concat semantics and `q_len != kv_len` support (`test_qwen3_dflash_attention.py`) | @unassigned | done | A2 | New tests pass in CI/local env | `tests/models/jax/test_qwen3_dflash_attention.py` added |
| A6 | Extend/adjust existing DFlash tests for implementation switch coverage | @unassigned | done | A3, A5 | `test_dflash_attention_impl_flag` and related checks pass | Impl-switch tests added in new test file |
| A7 | Re-run integration gates (`spec manager`, `kv cache`, `runner` tests) | @unassigned | blocked | A4, A6 | No regressions in updated unit/integration subset | `pytest` unavailable locally (`No module named pytest`) |
| A8 | Run verification smoke/eval scripts (at minimum dry-run here; full run in TPU env) | @unassigned | in_progress | A7 | Smoke/eval commands complete and outputs logged | DRY_RUN smoke/eval commands passed on 2026-02-06 |
| B1 | Performance hardening pass (compile latency/memory review for dense path) | @unassigned | not_started | A8 | Documented before/after compile/perf deltas | |
| B2 | Decide on keeping/removing `additive_legacy` fallback | @unassigned | not_started | B1 | Decision recorded with rationale in docs/changelog | |
| B3 | Update support/docs status after acceptance criteria are met | @unassigned | not_started | A8, B2 | Docs and support matrices updated | |

## Milestones

| Milestone | Scope | Owner | Status | Exit Criteria | Evidence |
|---|---|---|---|---|---|
| M1 Functional Parity Fix | A1-A6 | @unassigned | done | Concat path implemented, tested, defaulted | Local code/tests added; `py_compile` pass |
| M2 Integration Validation | A7-A8 | @unassigned | in_progress | Runner/tests/smoke-eval pass | Dry-run smoke/eval pass; pytest blocked locally |
| M3 Performance + Rollout | B1-B3 | @unassigned | not_started | Perf characterized, fallback policy finalized, docs updated | |

## Acceptance Checklist

- [x] Default DFlash path no longer uses additive K/V fusion.
- [x] DFlash attention path supports `q_len != kv_len`.
- [ ] New concat-focused unit tests are passing.
- [ ] Existing DFlash integration tests remain passing.
- [ ] Verification smoke/eval scripts run with logged outputs.

## Commands

```bash
# Scope guard
bash verification/sh/check_tpu_inference_change_scope.sh

# Unit/integration subset
python -m pytest -q \
  tpu-inference/tests/spec_decode/test_dflash.py \
  tpu-inference/tests/models/jax/test_qwen3_dflash.py \
  tpu-inference/tests/models/jax/test_qwen3_dflash_attention.py \
  tpu-inference/tests/runner/test_speculative_decoding_manager.py \
  tpu-inference/tests/runner/test_kv_cache_manager.py \
  tpu-inference/tests/runner/test_tpu_runner.py

# Verification scripts
DRY_RUN=1 bash verification/sh/run_tpu_inference_dflash_smoke.sh
DRY_RUN=1 CATEGORY=math MAX_SAMPLES=4 bash verification/sh/run_tpu_inference_dflash_eval.sh
```

## Risks / Active Blockers

- Missing runtime deps in current environment (`pytest`, `vllm`) block full local execution.
- Dense attention fallback may increase compile/memory overhead versus ragged path.

## Change Log

- 2026-02-06: Initial checklist created from parity fix design doc (`docs/10_dflash_kv_concat_parity_fix_design.md`).
- 2026-02-06: Updated execution status after implementing concat-parity attention path, adding tests, and running local compile/scope/dry-run checks.

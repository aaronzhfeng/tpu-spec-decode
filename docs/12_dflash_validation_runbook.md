# DFlash Validation Runbook

## Goal

Run DFlash validation in a fixed sequence so TPU-inference integration can be checked consistently against:

- local code correctness gates
- TPU runtime behavior
- extracted DFlash reported baselines

## Validation Order

1. Preflight environment and repo layout.
2. Scope guard for nested `tpu-inference/` edits.
3. Python syntax checks.
4. DFlash-focused pytest subset.
5. TPU smoke run (sanity wiring).
6. TPU eval + baseline comparison.
7. Optional external DFlash reference benchmark.

## Single-Command Runner

Use:

```bash
bash verification/sh/run_dflash_validation_matrix.sh
```

Main toggles:

- `STRICT_PRECHECK=1|0` (default `1`)
- `RUN_PYTEST=1|0` (default `1`)
- `RUN_SMOKE=1|0` (default `1`)
- `RUN_EVAL=1|0` (default `1`)
- `RUN_EXTERNAL=1|0` (default `0`)

Example (dry run for smoke/eval path only):

```bash
DRY_RUN=1 RUN_PYTEST=0 RUN_EXTERNAL=0 \
  bash verification/sh/run_dflash_validation_matrix.sh
```

## Direct Stage Commands

Preflight:

```bash
python verification/py/preflight_dflash_validation.py \
  --require-pytest \
  --require-vllm \
  --require-external-dflash
```

Scope guard:

```bash
bash verification/sh/check_tpu_inference_change_scope.sh
```

Pytest subset:

```bash
bash verification/sh/run_pytest_dflash_subset.sh
```

TPU smoke:

```bash
bash verification/sh/run_tpu_inference_dflash_smoke.sh
```

TPU eval + comparison:

```bash
CATEGORY=math MAX_SAMPLES=16 PROMPT_SOURCE=external \
  bash verification/sh/run_tpu_inference_dflash_eval.sh
```

Optional external reference:

```bash
bash verification/sh/run_external_dflash_reference.sh
```

## Result Artifacts

- TPU smoke outputs: `verification/outputs/tpu_smoke/`
- TPU eval outputs: `verification/outputs/tpu_eval/`
- External reference logs: `verification/outputs/external/`

## Baseline Source

Reported DFlash baselines are extracted into:

- `verification/py/dflash_reported_baselines.py`

Comparison logic:

- `verification/py/compare_results_to_baseline.py`

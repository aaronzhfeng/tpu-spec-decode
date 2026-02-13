# Verification Harness

This folder contains repo-root validation utilities for DFlash integration.

## Structure

```text
verification/
  config/
    tpu_inference_dflash_allowlist.txt
  contribution/
    manifests/
      default.json
      quick_math.json
    py/
      run_matrix.py
    sh/
      run_contribution_matrix.sh
  py/
    preflight_dflash_validation.py
    dflash_reported_baselines.py
    compare_results_to_baseline.py
    run_tpu_dflash_eval.py
    check_tpu_inference_scope.py
  sh/
    check_tpu_inference_change_scope.sh
    run_pytest_dflash_subset.sh
    run_dflash_validation_matrix.sh
    run_external_dflash_reference.sh
    run_tpu_inference_dflash_smoke.sh
    run_tpu_inference_dflash_eval.sh
```

## Usage

0. Preflight (strict):

```bash
python verification/py/preflight_dflash_validation.py \
  --require-pytest \
  --require-vllm \
  --require-external-dflash
```

1. Scope guard:

```bash
bash verification/sh/check_tpu_inference_change_scope.sh
```

2. Run DFlash pytest subset:

```bash
bash verification/sh/run_pytest_dflash_subset.sh
```

3. Run smoke test:

```bash
bash verification/sh/run_tpu_inference_dflash_smoke.sh
```

4. Run eval and compare to extracted baselines:

```bash
CATEGORY=math MAX_SAMPLES=16 PROMPT_SOURCE=external \
  bash verification/sh/run_tpu_inference_dflash_eval.sh
```

5. Direct comparator usage (if you already have a results JSON):

```bash
python verification/py/compare_results_to_baseline.py \
  --results-json /path/to/results.json \
  --category math \
  --model-key Qwen3-8B-DFlash-b16 \
  --temperature 0 \
  --metric speedup \
  --min-retention 0.70
```

`min-retention=0.70` means observed values must be at least 70% of the reported reference value.

Run one-shot validation matrix (all stages):

```bash
STRICT_PRECHECK=1 \
RUN_PYTEST=1 \
RUN_SMOKE=1 \
RUN_EVAL=1 \
RUN_EXTERNAL=0 \
bash verification/sh/run_dflash_validation_matrix.sh
```

Dry run (print settings only):

```bash
DRY_RUN=1 \
RUN_PYTEST=0 \
RUN_EXTERNAL=0 \
bash verification/sh/run_dflash_validation_matrix.sh
```

Contribution-focused matrix (per-prompt logs + summaries):

```bash
MANIFEST=verification/contribution/manifests/quick_math.json \
  bash verification/contribution/sh/run_contribution_matrix.sh
```

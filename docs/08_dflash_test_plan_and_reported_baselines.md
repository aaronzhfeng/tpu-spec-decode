# DFlash Test Plan and Reported Baselines

## Purpose

Define how to validate DFlash integration in `tpu-inference` with:

- basic correctness gates (things run),
- integration gates (spec decode path behaves correctly),
- parity/perf gates (TPU results compared against DFlash-reported references).

This plan uses a repo-root harness:

- Python utilities in `verification/py/`
- Shell runners in `verification/sh/`

---

## Root Test Harness Layout

```text
verification/
  config/
    tpu_inference_dflash_allowlist.txt
  README.md
  py/
    dflash_reported_baselines.py
    compare_results_to_baseline.py
    run_tpu_dflash_eval.py
    check_tpu_inference_scope.py
  sh/
    check_tpu_inference_change_scope.sh
    run_external_dflash_reference.sh
    run_tpu_inference_dflash_smoke.sh
    run_tpu_inference_dflash_eval.sh
```

Roles:

- `verification/py/`: machine-readable baselines and result comparators.
- `verification/sh/`: reproducible command entrypoints for external DFlash and local TPU runs.
- `verification/config/`: implementation guardrails (allowed nested repo change scope).

---

## Environment Prerequisites

- `python` with `pytest` installed for unit tests in `tpu-inference/tests`.
- TPU runtime dependencies for integration/eval scripts.
- Optional: external DFlash benchmarking dependencies if reproducing GPU-side references.

Quick check:

```bash
python -m pytest --version
```

---

## Pre-Change Frame Lock

Nested `tpu-inference` base commit before integration:

- `2b2780fa43a2ea6c6c3f6751b83771744e76b1a8`

Run before and during implementation:

```bash
bash verification/sh/check_tpu_inference_change_scope.sh
```

This enforces that changed files inside nested `tpu-inference/` remain inside:

- `verification/config/tpu_inference_dflash_allowlist.txt`

---

## Test Stages

## Stage 0: External Reference Reproduction (Optional but Preferred)

Goal:

- Re-run external DFlash benchmark flow to verify local environment can reproduce their pipeline shape.

Command:

- `verification/sh/run_external_dflash_reference.sh`

Checks:

- benchmark script runs end-to-end for selected datasets,
- output logs are generated in a stable location for downstream parsing.

## Stage 1: TPU Smoke (Basic)

Goal:

- Validate the new `method="dflash"` path is wired and runnable.

Command:

- `verification/sh/run_tpu_inference_dflash_smoke.sh`

Checks:

- model + draft model load succeeds,
- one short prompt generates output,
- no runtime failure in drafter prep/propose/rejection path,
- output token count > 0.

## Stage 2: TPU Functional Regression (Integration)

Goal:

- Validate scheduler and speculative metadata logic under realistic batching.

Command:

- unit tests in `tpu-inference/tests` (new DFlash tests),
- plus `verification/sh/run_tpu_inference_dflash_eval.sh` on a small dataset slice.

Checks:

- draft token shape consistency per request,
- rejection/acceptance loop stability,
- KV cache updates remain valid across iterations.

Recommended command set:

```bash
python -m pytest -q \
  tpu-inference/tests/spec_decode/test_dflash.py \
  tpu-inference/tests/models/jax/test_qwen3_dflash.py \
  tpu-inference/tests/runner/test_speculative_decoding_manager.py \
  tpu-inference/tests/runner/test_kv_cache_manager.py \
  tpu-inference/tests/runner/test_tpu_runner.py
```

## Stage 3: TPU Advanced Parity and Performance

Goal:

- Compare TPU observed results against reported DFlash baselines.

Command:

- `verification/sh/run_tpu_inference_dflash_eval.sh` to produce result JSON,
- `python verification/py/compare_results_to_baseline.py ...` to compare.

Reference command sequence:

```bash
# 1) scope guard (should pass)
bash verification/sh/check_tpu_inference_change_scope.sh

# 2) smoke (synthetic prompt, no baseline run)
bash verification/sh/run_tpu_inference_dflash_smoke.sh

# 3) eval + baseline comparison (category-level)
CATEGORY=math MAX_SAMPLES=16 PROMPT_SOURCE=external \
  bash verification/sh/run_tpu_inference_dflash_eval.sh
```

Checks:

- speedup and acceptance-length (`tau`) comparisons by dataset,
- pass/fail threshold using retention rule (for example 70-90% of reported speedup, hardware-dependent),
- category-level average deltas tracked over time.

---

## Result File Contract

The comparator expects a JSON like:

```json
{
  "datasets": {
    "GSM8K": {"speedup": 5.10, "tau": 6.50},
    "Math500": {"speedup": 6.09, "tau": 7.80}
  },
  "average": {"speedup": 5.79, "tau": 7.27}
}
```

`tau` can be omitted if a run only reports speedup.

---

## Reported DFlash Baselines (Extracted)

Source artifacts:

- `external/dflash/assets/dflash_results.png`
- `external/dflash/assets/speedup.png`
- `external/dflash/README.md` (states results tested on NVIDIA B200 GPUs)

Important context:

- These are reported GPU-side numbers, not TPU numbers.
- Use them as relative targets, not strict equality targets.

## A) `dflash_results.png` (speedup / tau)

### Math benchmarks

| Method | Temp | GSM8K | Math500 | AIME24 | AIME25 | Average |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B-speculator.eagle3 | 0 | 2.13 / 2.89 | 2.18 / 2.94 | 2.25 / 3.04 | 2.18 / 2.93 | 2.19 / 2.95 |
| Qwen3-4B-DFlash-b16 | 0 | 5.17 / 6.50 | 6.19 / 7.84 | 6.00 / 7.47 | 5.79 / 7.28 | 5.79 / 7.27 |
| Qwen3-8B-DFlash-b16 | 0 | 5.20 / 6.55 | 6.17 / 7.87 | 5.91 / 7.48 | 5.85 / 7.31 | 5.78 / 7.30 |
| Qwen3-8B-speculator.eagle3 | 1 | 2.07 / 2.79 | 2.03 / 2.75 | 1.88 / 2.54 | 1.81 / 2.44 | 1.95 / 2.63 |
| Qwen3-4B-DFlash-b16 | 1 | 4.73 / 5.98 | 5.14 / 6.67 | 3.84 / 4.97 | 3.89 / 5.01 | 4.40 / 5.66 |
| Qwen3-8B-DFlash-b16 | 1 | 4.78 / 6.04 | 5.02 / 6.57 | 3.87 / 5.06 | 3.84 / 5.03 | 4.38 / 5.68 |

### Code benchmarks

| Method | Temp | Humaneval | MBPP | LiveCodeBench | SWE-Bench | Average |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B-speculator.eagle3 | 0 | 2.48 / 3.36 | 2.27 / 3.08 | 2.24 / 3.16 | 1.90 / 2.55 | 2.22 / 3.04 |
| Qwen3-4B-DFlash-b16 | 0 | 5.26 / 6.63 | 4.87 / 6.19 | 5.41 / 6.97 | 2.97 / 3.70 | 4.63 / 5.87 |
| Qwen3-8B-DFlash-b16 | 0 | 5.20 / 6.55 | 4.75 / 6.00 | 5.43 / 7.12 | 2.92 / 3.69 | 4.58 / 5.84 |
| Qwen3-8B-speculator.eagle3 | 1 | 2.30 / 3.11 | 2.15 / 2.92 | 2.17 / 3.00 | 1.66 / 2.21 | 2.07 / 2.81 |
| Qwen3-4B-DFlash-b16 | 1 | 4.80 / 6.05 | 4.35 / 5.55 | 5.00 / 6.60 | 2.51 / 3.09 | 4.17 / 5.32 |
| Qwen3-8B-DFlash-b16 | 1 | 4.35 / 5.40 | 4.07 / 5.17 | 5.15 / 6.79 | 2.30 / 2.82 | 3.97 / 5.05 |

### Chat benchmarks

| Method | Temp | MT-Bench | Alpaca | Average |
|---|---:|---:|---:|---:|
| Qwen3-8B-speculator.eagle3 | 0 | 1.94 / 2.72 | 1.88 / 2.68 | 1.91 / 2.70 |
| Qwen3-4B-DFlash-b16 | 0 | 2.87 / 4.35 | 2.23 / 3.10 | 2.55 / 3.73 |
| Qwen3-8B-DFlash-b16 | 0 | 2.79 / 4.25 | 2.27 / 3.16 | 2.53 / 3.71 |
| Qwen3-8B-speculator.eagle3 | 1 | 1.81 / 2.55 | 1.79 / 2.56 | 1.80 / 2.56 |
| Qwen3-4B-DFlash-b16 | 1 | 2.63 / 4.03 | 2.16 / 2.99 | 2.40 / 3.51 |
| Qwen3-8B-DFlash-b16 | 1 | 2.50 / 3.74 | 2.11 / 2.88 | 2.31 / 3.31 |

## B) `speedup.png` (Qwen3-8B charted speedups)

Values in the bar chart:

| Dataset | Eagle-3 | DFlash |
|---|---:|---:|
| GSM8K | 2.13 | 5.10 |
| Math500 | 2.18 | 6.09 |
| AIME24 | 2.25 | 5.73 |
| AIME25 | 2.18 | 5.75 |
| Humaneval | 2.48 | 5.18 |
| MBPP | 2.27 | 4.67 |
| LiveCodeBench | 2.24 | 5.51 |
| SWE-Bench | 1.90 | 2.88 |
| MT-Bench | 1.94 | 2.76 |
| Alpaca | 1.88 | 2.21 |

---

## Suggested TPU Comparison Policy

Use a two-level threshold:

1. Bring-up threshold:
   - `speedup >= 1.0` on all datasets in smoke/eval slices.
2. Target threshold:
   - category average speedup at least `X%` of reported baseline (start with `70%`, tighten over time).

For acceptance length (`tau`):

- require monotonic improvement over non-spec baseline,
- compare against reported `tau` with broader tolerance than speedup because kernel/runtime differences can shift acceptance dynamics.

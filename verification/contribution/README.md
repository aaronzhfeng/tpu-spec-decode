# Contribution Validation Harness

This folder contains contribution-focused validation for DFlash behavior.

Unlike basic smoke/eval checks, this harness logs per-prompt artifacts so runs can be audited later.

## Goals

- Validate that DFlash runs end-to-end.
- Measure contribution metrics versus baseline:
  - time per output token (TPOT)
  - tokens per second (TPS)
  - speedup vs baseline
- Persist outputs, timings, and errors for post-run analysis.

## Layout

```text
verification/contribution/
  manifests/
    default.json
    quick_math.json
    dflash_diagnostic_near_demo.json
  py/
    run_matrix.py
  sh/
    run_contribution_matrix.sh
```

Outputs are written to:

```text
verification/outputs/contribution/<run_id>/
  env_snapshot.json
  run_manifest.json
  prompt_records/
    baseline.<dataset>.jsonl
    dflash.<dataset>.jsonl
    ...
  summaries/
    by_dataset_method.csv
    overall.json
    comparator_dflash.json
    report.md
```

## Quick Start

Requires Python 3.11+ on host. Override interpreter explicitly if needed:

```bash
PYTHON_BIN=python3.11 bash verification/contribution/sh/run_contribution_matrix.sh
```

Run the small manifest:

```bash
MANIFEST=verification/contribution/manifests/quick_math.json \
  bash verification/contribution/sh/run_contribution_matrix.sh
```

Run default manifest:

```bash
bash verification/contribution/sh/run_contribution_matrix.sh
```

Run diagnostic manifest (near demo settings, includes tau/acceptance metrics):

```bash
MANIFEST=verification/contribution/manifests/dflash_diagnostic_near_demo.json \
  bash verification/contribution/sh/run_contribution_matrix.sh
```

Dry-run:

```bash
DRY_RUN=1 \
  bash verification/contribution/sh/run_contribution_matrix.sh
```

## Manifest Notes

Manifest fields:

- `model`: target model (for example `Qwen/Qwen3-8B`)
- `draft_model`: draft model (for example `z-lab/Qwen3-8B-DFlash-b16`)
- `methods`: list such as `["baseline", "dflash"]`
- `datasets`: dataset ids
- `max_samples`: prompts per dataset
- `prompt_source`: `external` or `synthetic`
- `use_chat_template`: when true, wrap prompts with tokenizer chat template (near `external/dflash` behavior)
- `sampling`: generation parameters
- `runtime`: execution parameters (TP size, max length, speculative tokens, warmup)
- For speculative diagnostics (`tau`, acceptance), set `runtime.disable_log_stats=false`.

## Prompt Record Format (JSONL)

Each line in `prompt_records/*.jsonl` contains:

- run metadata (`run_id`, `method`, `dataset`, `sample_idx`)
- prompt text/hash
- generation status and error text
- elapsed time and token count
- output text

## Speculative Diagnostics

When `disable_log_stats=false`, the summary includes method-level speculative metrics:

- `tau`: mean acceptance length, computed as `1 + accepted_tokens / num_drafts`
- `draft_acceptance_rate`: accepted draft tokens divided by drafted tokens
- raw counters: `num_drafts`, `num_draft_tokens`, `num_accepted_tokens`


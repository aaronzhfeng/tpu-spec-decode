# DFlash Runtime Behavior and Environment Playbook

Status: **Active**
Date: 2026-02-11

## Purpose

Document the execution behavior we repeatedly observed while validating DFlash, with concrete mitigations that make long benchmark runs reliable.

This is an operations-focused companion to:

- `docs/14_dflash_parity_findings_and_remediation.md`
- `docs/12_dflash_validation_runbook.md`
- `verification/contribution/README.md`

## Repeated Runtime Behaviors Observed

### A) Host Python parser mismatch

Symptom:

- `SyntaxError` in fused MoE kernel code (`w1_vmem[*w_slices]` style syntax).

Cause:

- Host interpreter was Python 3.10, while parser support requires Python 3.11+.

Mitigation now in repo:

- `verification/contribution/sh/run_contribution_matrix.sh` auto-selects Python 3.11+ and fails fast otherwise.

### B) Host dependency/platform mismatch

Symptoms:

- External prompt loading warning: `No module named 'datasets'` and fallback to synthetic prompts.
- TPU runtime/platform failure: `RuntimeError: Device string must not be empty`.

Cause:

- Host environment lacked required packages and TPU runtime assumptions.

Operational rule:

- Use the validated Docker runtime for contribution runs.

### C) Disk exhaustion during model/materialization

Symptoms:

- HuggingFace snapshot/data processing failures with `No space left on device`.
- `tee`/log writes failing.

Cause:

- Docker overlay/root filesystem saturation (often despite model count seeming small).

Contributors to high footprint:

- Multiple model revisions and shard sets.
- Duplicate caches across host and container paths.
- Root-owned artifacts from `sudo docker` runs.
- Intermediate download/staging footprint exceeding steady-state model size.

### D) Permission-denied cleanup loops

Symptom:

- `rm -rf` fails on prior output/cache directories.

Cause:

- Artifacts created by `sudo docker` are root-owned.

Mitigation:

- Cleanup with `sudo rm -rf ...` or reclaim ownership via `sudo chown -R ...`.

## Stable Execution Pattern

Use Docker with explicit mounts and controlled cache/output locations.

Recommended practices:

- Keep HuggingFace cache and run outputs on large host storage or `tmpfs` when root disk is constrained.
- Ensure output and cache paths are writable for the chosen container user mode.
- Avoid mixing many historical runs in the same output root without cleanup.

## Minimal Pre-Run Checklist

1. Interpreter/runtime
   - Run contribution harness in Docker, not host, for benchmark passes.
2. Storage
   - Verify free space in Docker root and mounted cache/output targets.
3. Cache strategy
   - Decide whether to use persistent host cache or ephemeral `tmpfs`.
4. Permissions
   - Confirm no root-owned stale output path blocks current run.
5. Manifest correctness
   - Confirm model/draft pair and `use_chat_template` settings.

## Practical Cleanup Strategy (when runs fail with ENOSPC)

1. Remove stale run outputs (especially failed partial runs).
2. Remove obsolete local HF/pip caches no longer needed.
3. Prune Docker data if safe for your workflow.
4. Re-run with:
   - explicit mount for HF cache,
   - explicit mount for contribution outputs,
   - optional `tmpfs` for temporary high-churn paths.

## Validation Artifacts to Check After Any Recovery Run

Primary:

- `verification/outputs/contribution/<run_id>/runner.log`
- `verification/outputs/contribution/<run_id>/summaries/overall.json`
- `verification/outputs/contribution/<run_id>/summaries/comparator_dflash.json`

Secondary:

- per-dataset JSONL prompt records for partial failures/retries.

## Known Open Risk

Even with stable execution, performance parity is not guaranteed:

- environment stability only ensures runs complete,
- model-mechanics parity (acceptance behavior) determines speedup.

Track both dimensions independently:

- **Ops health**: no crashes, no ENOSPC, reproducible artifact generation.
- **Algorithm health**: acceptance metrics (`tau`, `draft_acceptance_rate`) and TPOT speedup.


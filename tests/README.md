# Tests

One-line commands for validating DFlash on TPU. Every script runs from the repo root.

---

## Quick Reference

| Command | What it does | Time |
|---------|-------------|------|
| `bash tests/preflight.sh` | Check repos, Docker, vLLM dflash support | ~30s |
| `bash tests/tpu_sanity.sh` | Verify TPU hardware (matmul + SDPA) | ~1min |
| `bash tests/smoke.sh` | Quick DFlash end-to-end (3 prompts, greedy) | ~5min |
| `bash tests/smoke.sh stochastic` | Same but with temperature=1 sampling | ~5min |
| `bash tests/benchmark.sh math` | Math benchmarks (GSM8K, Math500, AIME24, AIME25) | ~30min |
| `bash tests/benchmark.sh code` | Code benchmarks (Humaneval, MBPP, etc.) | ~30min |
| `bash tests/benchmark.sh chat` | Chat benchmarks (MT-Bench, Alpaca) | ~20min |
| `bash tests/benchmark.sh full` | All 10 datasets | ~1.5hr |
| `bash tests/compare.sh latest` | Compare most recent benchmark to GPU baselines | ~1s |
| `bash tests/cleanup.sh` | Interactive cleanup of outputs/caches | ~1s |

---

## Recommended Order

For a fresh environment, run these in sequence:

```bash
# 1. Check everything is set up correctly
bash tests/preflight.sh

# 2. Verify TPU hardware
bash tests/tpu_sanity.sh

# 3. Quick smoke test — does DFlash run at all?
bash tests/smoke.sh

# 4. Run benchmarks
bash tests/benchmark.sh math

# 5. Compare against reported GPU baselines
bash tests/compare.sh latest
```

---

## Scripts

### `preflight.sh` — Environment Check

Verifies repos are cloned, Docker image is available, and vLLM accepts the `"dflash"` speculative method.

```bash
bash tests/preflight.sh          # Check inside Docker (default)
bash tests/preflight.sh host     # Check on host directly
```

### `tpu_sanity.sh` — TPU Hardware Check

Runs a small PyTorch/XLA matmul and scaled dot-product attention to confirm the TPU device is accessible and functioning.

```bash
bash tests/tpu_sanity.sh         # Inside Docker (default)
bash tests/tpu_sanity.sh host    # On host directly
```

### `smoke.sh` — Quick Smoke Test

Runs DFlash speculative decoding on 3 prompts from GSM8K. Validates the full pipeline: target model load, draft model load, proposal, verification, token generation.

```bash
bash tests/smoke.sh              # Greedy (temperature=0)
bash tests/smoke.sh greedy       # Same as above
bash tests/smoke.sh stochastic   # Temperature=1 sampling
```

Dry run (validates config without loading models):
```bash
DRY_RUN=1 bash tests/smoke.sh
```

### `benchmark.sh` — Performance Benchmarks

Measures DFlash vs autoregressive baseline across datasets. Produces per-prompt JSONL records, dataset-level CSV summaries, speedup comparisons, and speculative metrics (tau, acceptance rate).

```bash
bash tests/benchmark.sh          # Math (default)
bash tests/benchmark.sh math     # Math datasets
bash tests/benchmark.sh code     # Code datasets
bash tests/benchmark.sh chat     # Chat datasets
bash tests/benchmark.sh full     # All 10 datasets
```

Custom config:
```bash
bash tests/benchmark.sh tests/configs/my_custom.json
```

Override sample count:
```bash
MAX_SAMPLES=4 bash tests/benchmark.sh math
```

### `compare.sh` — Baseline Comparison

Compares benchmark results against the DFlash paper's reported GPU speedups (NVIDIA B200). Reports per-dataset retention percentage and pass/fail against a configurable threshold (default 70%).

```bash
bash tests/compare.sh latest                  # Most recent benchmark run
bash tests/compare.sh bench_math_20260212_*   # Specific run ID
bash tests/compare.sh /path/to/overall.json   # Direct file path
```

Adjust the passing threshold:
```bash
THRESHOLD=0.50 bash tests/compare.sh latest   # More lenient (50% of GPU speedup)
```

### `cleanup.sh` — Clean Outputs and Caches

```bash
bash tests/cleanup.sh            # Interactive (shows sizes, asks what to delete)
bash tests/cleanup.sh outputs    # Delete test outputs only
bash tests/cleanup.sh cache      # Delete HF/JAX model cache only
bash tests/cleanup.sh docker     # Prune Docker images/containers
bash tests/cleanup.sh all        # Delete everything
```

---

## Configs

Test configurations live in `tests/configs/`. Each is a JSON manifest that controls what models, datasets, and sampling parameters to use.

| Config | Target Model | Datasets | Samples | Methods |
|--------|-------------|----------|---------|---------|
| `smoke_greedy.json` | Qwen3-4B | GSM8K | 3 | dflash |
| `smoke_stochastic.json` | Qwen3-4B | GSM8K | 3 | dflash |
| `benchmark_math.json` | Qwen3-4B | GSM8K, Math500, AIME24, AIME25 | 8 | baseline, dflash |
| `benchmark_code.json` | Qwen3-4B | Humaneval, MBPP, LiveCodeBench, SWE-Bench | 8 | baseline, dflash |
| `benchmark_chat.json` | Qwen3-4B | MT-Bench, Alpaca | 8 | baseline, dflash |
| `benchmark_full.json` | Qwen3-4B | All 10 datasets | 16 | baseline, dflash |

To use Qwen3-8B instead of 4B, edit the `model` and `draft_model` fields:
```json
{
  "model": "Qwen/Qwen3-8B",
  "draft_model": "z-lab/Qwen3-8B-DFlash-b16"
}
```

---

## Output Artifacts

All test outputs go to `HOST_OUTPUT_DIR` (default: `/dev/shm/dflash-test-outputs`). Each run creates a timestamped directory:

```
/dev/shm/dflash-test-outputs/
  smoke_greedy_20260212_143000/
    env_snapshot.json          # Environment, git state, package versions
    run_manifest.json          # Config used for this run
    runner.log                 # Full stdout
    prompt_records/
      dflash.gsm8k.jsonl      # Per-prompt timing and output
    summaries/
      overall.json             # Method-level aggregates
      by_dataset_method.csv    # Dataset x method table
      comparator_dflash.json   # Speedup comparison data
      report.md                # Human-readable report
```

---

## Environment Variables

All scripts have sensible defaults. Override via environment if needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCKER_IMAGE` | `vllm/vllm-tpu:latest` | Docker image for TPU runtime |
| `HOST_HF_CACHE` | `/dev/shm/hf-cache` | HuggingFace model cache on host |
| `HOST_OUTPUT_DIR` | `/dev/shm/dflash-test-outputs` | Where outputs are written |
| `HOST_TPU_LOG_DIR` | `/dev/shm/tpu-logs` | TPU log directory |
| `TMPFS_SIZE` | `80g` | tmpfs size inside Docker container |
| `TPU_INFERENCE_DIR` | `<repo>/tpu-inference` | Path to tpu-inference checkout |
| `VLLM_DIR` | `<repo>/vllm` | Path to vllm checkout |
| `DRY_RUN` | `0` | Set to `1` to validate config without running |
| `PYTHON_BIN` | auto-detected | Python 3.11+ interpreter |
| `THRESHOLD` | `0.70` | Retention threshold for baseline comparison |

---

## File Layout

```
tests/
  README.md               # This file
  preflight.sh            # Environment checks
  tpu_sanity.sh           # TPU hardware verification
  smoke.sh                # Quick end-to-end smoke test
  benchmark.sh            # Performance benchmarks
  compare.sh              # GPU baseline comparison
  cleanup.sh              # Output/cache cleanup
  configs/                # Test configurations
    smoke_greedy.json
    smoke_stochastic.json
    benchmark_math.json
    benchmark_code.json
    benchmark_chat.json
    benchmark_full.json
  lib/                    # Shared helpers (not called directly)
    common.sh             # Path resolution, output helpers, Docker detection
    docker_run.sh         # Docker execution wrapper
```

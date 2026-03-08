# Cleanup Notes

Documentation of required repo parts for v4/v5p benchmarks and cleanup recommendations.

---

## Executive Summary

- **V4 reproduced:** 2026-03-08 on son-v4-node (v4-8). Standalone and vLLM pipeline benchmarks succeeded.
- **V5P:** TBD (run on v5p hardware).
- **GPU:** Keep as-is. Do not remove `benchmarks/gpu/`, `results/v14_h100/`, or v4 GPU comparison files.

---

## V4 Hardware (Executed 2026-03-08)

### Step 0: Preparation

| Item | Details |
|------|---------|
| Script | `bash preparation/clone_repos.sh --skip-pr --skip-ext` |
| Required repos | `tpu-inference/`, `vllm/` (root-level, Group 1 only) |
| Branches | `dflash-integration` (tpu-inference), `dflash-speculative-config` (vllm) |
| Environment | Docker `vllm/vllm-tpu:latest`; no host venv for v4 |
| Note | If `tpu-inference` or `vllm` exist as empty dirs, remove them first so the script can clone |

### Step 1: Standalone TPU Benchmark

| Item | Details |
|------|---------|
| Entry | `benchmarks/standalone_dflash.py` |
| Run | `bash tests/standalone_benchmark.sh [--max-samples N] [--output-json /output/path.json]` |
| Output | Writes to `/output` (mounted from `/dev/shm/dflash-test-outputs`) |
| Copy to results | `cp /dev/shm/dflash-test-outputs/<file>.json results/v4/standalone_<dataset>.json` |
| Required parts | `tpu-inference/`, `benchmarks/standalone_dflash.py`, `tests/standalone_benchmark.sh`, `tests/lib/common.sh`, `tests/lib/docker_run.sh` |

**Command used:**
```bash
bash tests/standalone_benchmark.sh --max-samples 2 --max-new-tokens 64 --output-json /output/v4_standalone_gsm8k.json
cp /dev/shm/dflash-test-outputs/v4_standalone_gsm8k.json results/v4/standalone_gsm8k.json
```

### Step 2: vLLM Pipeline Benchmark

| Item | Details |
|------|---------|
| Entry | `verification/contribution/sh/run_contribution_matrix.sh` → `verification/contribution/py/run_matrix.py` |
| Run | `MAX_SAMPLES=2 bash tests/benchmark.sh math` (or `code`, `chat`, `full`, `eagle3_qwen3`) |
| Config | `tests/configs/benchmark_math.json` (gsm8k, math500, aime24, aime25) |
| Output | `/dev/shm/dflash-test-outputs/bench_math_<timestamp>/summaries/` |
| CSV generation | `python results/v4/generate_csvs.py` reads bench_math_* and writes `results/v4/vllm_pipeline_results.csv` |
| Required parts | `tpu-inference/`, `vllm/`, `verification/contribution/`, `tests/benchmark.sh`, `tests/configs/`, `results/v4/generate_csvs.py` |

**Command used:**
```bash
MAX_SAMPLES=2 bash tests/benchmark.sh math
python results/v4/generate_csvs.py
```

**Note:** Without `external/dflash`, prompts fall back to synthetic. Benchmark still runs.

### Step 3: GPU

Keep all GPU-related code and results unchanged.

### Step 4: V4 Required Parts Summary

| Category | Paths |
|----------|-------|
| Repos | `tpu-inference/`, `vllm/` |
| Scripts | `benchmarks/standalone_dflash.py`, `tests/standalone_benchmark.sh`, `tests/benchmark.sh`, `verification/contribution/` |
| Config | `tests/configs/*.json`, `verification/contribution/manifests/*.json` |
| Lib | `tests/lib/common.sh`, `tests/lib/docker_run.sh` |
| Prep | `preparation/clone_repos.sh` |
| Results | `results/v4/` (including `generate_csvs.py`) |

---

## V5P Hardware

TBD. Run same steps on v5p node with `preparation/setup_v5p_safe.sh` first.

---

## Cleanup Recommendations

After v5p is documented, determine final removals. Candidates (from plan):

| Category | Candidates |
|----------|------------|
| Symlinks/dirs | `report`, `.Claude`, `brainstorm-*`, `dflash-wide`, `pr-ready`, `zhongyan_dev`, `slides`, `visualizations` (if unused) |
| clone_repos groups | Groups 2–7 optional for benchmarks; only Group 1 (tpu-inference, vllm) required |
| External | `external/dflash` optional (synthetic prompts if missing) |
| Verification outputs | `verification/outputs/` – historical artifacts, can be pruned |
| Capstone/report | Keep if part of deliverables |

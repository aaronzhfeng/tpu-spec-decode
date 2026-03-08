# Benchmarks, Tests, and Verification — Reference

Descriptions of each file in `benchmarks/`, `tests/`, and `verification/`, including purpose, what it measures, and hardware target (v4, v5p, GPU).

---

## benchmarks/

### TPU benchmarks (v4, v5p — run via Docker or host venv)

| File | Purpose | What it measures / experiment | Hardware |
|------|---------|------------------------------|----------|
| **standalone_dflash.py** | Primary DFlash benchmark. Replicates GPU paper numbers in JAX without vLLM. Runs prefill → draft → verify → accept loop. | τ (acceptance length), speedup vs AR baseline, per-position acceptance rate, output quality. Apples-to-apples with DFlash paper. | v4, v5p |
| **drafter_scaling.py** | Tests MXU tile amortization for the drafter. | (1) DFlash forward at K=16, (2) raw matmul latency at K=16,32,64,128 for DFlash/target layers, (3) target matmul at same K. Validates both drafter and target benefit from amortization. | v4, v5p |
| **amortized_verification.py** | Tests whether verifying more tokens at once costs less per token on TPU. | Part 1: target verify latency vs query count (16–128). Part 2: multi-block speculation (draft 2+ blocks, verify together). Flat latency ⇒ MXU amortization. | v4, v5p |
| **verify_context_scaling.py** | Experiment B: Does flat-K hold at longer contexts? | Verify latency vs K (16,64,128) and L (64,256,512,1024). Tests if attention scaling at long context changes the flat-K property. | v4, v5p |
| **layer_truncation.py** | How many target layers does verification need? | Captures hidden states at every layer; simulates acceptance (τ) at truncation points 6,12,18,24,30,36. Explores early-exit verification. | v4, v5p |
| **tree_speculation.py** | Multi-candidate drafting using MXU amortization. | Draft K candidate blocks (different first tokens), verify each, pick best. Measures best τ, draft/verify time, theoretical throughput with batched verify. | v4, v5p |
| **pipeline_profiling.py** | Fine-grained timing of draft-verify-accept loop. | Draft latency, verify latency, acceptance logic, context update, cache management, host-device transfer. Identifies overhead. | v4, v5p |
| **ablation_study.py** | Isolates optimization targets. | H1: LM head matmul cost. H2: Host loop overhead. H3: vLLM vs standalone τ gap. H4: Component overlap. | v4, v5p |
| **fused_dflash.py** | Direction 1 optimization: fused JIT to remove host-device roundtrips. | Fuses draft→logits→argmax and verify→logits→acceptance into fewer device calls. Compares τ, latency, throughput vs unfused. | v4, v5p |
| **iterative_refinement.py** | Direction 2: iterative draft refinement. | Runs DFlash with k=0,1,2,3 refinement steps (feed predictions back as input). Measures τ, latency, throughput. | v4, v5p |
| **v11_roofline_analysis.py** | Analytical script; no hardware. | Arithmetic intensity per forward component at varying K; compares to ridge points (TPU v5p, H100, RTX). Predicts memory- vs compute-bound. | N/A (analysis) |

### GPU benchmarks (keep as-is)

| File | Purpose | What it measures / experiment | Hardware |
|------|---------|------------------------------|----------|
| **gpu_matmul_scaling.py** | Companion to drafter_scaling (TPU). | Raw matmul latency at K=16,32,64,128. Proves GPU scales linearly while TPU is flat. | GPU |
| **gpu_verify_full.py** | V7: Full target forward at K=16 vs K=128. | End-to-end verification penalty on GPU. Validates Doc 42 component measurements compose. | GPU |
| **gpu_verify_context_scaling.py** | V12: GPU verify vs context length. | Verify latency at K=16,64,128 across L=256–2048. Tests 1.24× penalty constancy. | GPU |
| **gpu_draft_speed.py** | DFlash draft forward at K=16 vs K=128. | Draft model latency; single-pass vs 8×K=16 multi-pass. | GPU |
| **gpu_forward_decomposition.py** | V10: Component split (FFN, attention, etc.). | Profiles real forward with CUDA events. Resolves 1.24× vs 1.33× discrepancy. | GPU |
| **benchmark_block_sizes.py** | Block size sweep on GPU (PyTorch, external/dflash). | Runs DFlash at block_sizes 16,32,64,128. Uses `model` from external/dflash. | GPU |
| **gpu/setup.sh** | One-shot setup for GPU benchmarks. | Installs PyTorch, downloads Qwen3-4B. | GPU |
| **gpu/run_all.sh** | Runs all V14 GPU experiments. | V14a–f: full forward, context scaling, K-sweep, matmul, draft speed, decomposition. | GPU |
| **gpu/README.md** | Documentation for GPU verification suite. | | |

---

## tests/

### Shell wrappers (invoke benchmarks via Docker; v4, v5p)

| File | Purpose | Invokes | Hardware |
|------|---------|---------|----------|
| **standalone_benchmark.sh** | Single-dataset standalone DFlash (default: gsm8k, 3 samples). Quick run; not a duplicate of run_standalone_all_v5p. | `standalone_dflash.py` | TPU v4, v5p |
| **run_standalone_all_v5p.sh** | Batch variant: runs `standalone_dflash.py` on 8 datasets (math500, aime24, aime25, humaneval, mbpp, swe-bench, mt-bench, alpaca). Uses 8 samples, 256 tokens; outputs per-dataset JSONs. | Same script, loop over datasets; outputs to `HOST_OUTPUT_DIR/v5p-pr/`. | TPU v5p |
| **benchmark.sh** | Run vLLM pipeline benchmarks. | `verification/contribution/sh/run_contribution_matrix.sh` with config (math/code/chat/full/eagle3_qwen3) | TPU v4, v5p |
| **verify_context_scaling.sh** | Experiment B: verify vs K and L. | `verify_context_scaling.py` | TPU v4, v5p |
| **drafter_scaling.sh** | Drafter forward scaling. | `drafter_scaling.py` | TPU v4, v5p |
| **amortized_verification.sh** | Amortized verification experiment. | `amortized_verification.py` | TPU v4, v5p |
| **layer_truncation.sh** | Layer-truncated verification. | `layer_truncation.py` | TPU v4, v5p |
| **tree_speculation.sh** | Tree speculation. | `tree_speculation.py` | TPU v4, v5p |
| **pipeline_profiling.sh** | Pipeline profiling. | `pipeline_profiling.py` | TPU v4, v5p |
| **ablation_study.sh** | Ablation study. | `ablation_study.py` | TPU v4, v5p |
| **iterative_refinement.sh** | Iterative refinement. | `iterative_refinement.py` | TPU v4, v5p |
| **fused_benchmark.sh** | Fused vs unfused DFlash. | `fused_dflash.py` | TPU v4, v5p |
| **smoke.sh** | Quick end-to-end smoke test. | Contribution matrix with smoke config (3 prompts, greedy or stochastic) | TPU v4, v5p |
| **preflight.sh** | Environment check. | Verifies repos, Docker, vLLM dflash support | TPU v4, v5p (Docker) |
| **tpu_sanity.sh** | TPU hardware check. | Matmul + SDPA sanity | TPU v4, v5p |
| **compare.sh** | Compare results to GPU baselines. | Reads comparator JSON, checks speedup/tau vs DFlash paper | Host only |
| **cleanup.sh** | Clean outputs/caches. | Interactive cleanup | Host only |
| **verify_all_wrappers.sh** | Run every wrapper to verify repo on v4 or v5p. | Assumes fresh clone; runs clone_repos, then all wrappers in sequence. Use `v4` or `v5p` arg. | TPU v4 or v5p |

### Lib and configs

| File | Purpose |
|------|---------|
| **lib/common.sh** | Paths (REPO_ROOT, TPU_INFERENCE_DIR, VLLM_DIR), Docker detection, output dirs, helpers |
| **lib/docker_run.sh** | Docker execution wrapper; mounts repo, HF cache, output dirs |
| **configs/benchmark_math.json** | Math: GSM8K, Math500, AIME24, AIME25; baseline + dflash |
| **configs/benchmark_code.json** | Code: Humaneval, MBPP, LiveCodeBench, SWE-Bench |
| **configs/benchmark_chat.json** | Chat: MT-Bench, Alpaca |
| **configs/benchmark_full.json** | All 10 datasets |
| **configs/benchmark_math_dflash_only.json** | Math, dflash only |
| **configs/smoke_greedy.json** | Smoke: 3 prompts, greedy |
| **configs/smoke_stochastic.json** | Smoke: 3 prompts, temperature=1 |
| **configs/eagle3_qwen3_math.json** | Eagle3 (Qwen3-4B) on math |
| **configs/eagle3_llama_math.json** | Eagle3 (LLaMA) on math |

---

## verification/

### Contribution harness (run by tests/benchmark.sh)

| File | Purpose | Hardware |
|------|---------|----------|
| **contribution/py/run_matrix.py** | Main contribution runner. Loads manifest, runs baseline + dflash per dataset, writes per-prompt JSONL and summaries. | v4, v5p |
| **contribution/sh/run_contribution_matrix.sh** | Entry point for contribution validation. Resolves Python, sets env, calls run_matrix.py. | v4, v5p |
| **contribution/manifests/*.json** | Manifests: default, quick_math, dflash_only_tmp, dflash_demo_tasks_medium, dflash_diagnostic_near_demo, dflash_acceptance_smoke, dflash_acceptance_smoke_greedy | |

### Validation and comparison

| File | Purpose | Hardware |
|------|---------|----------|
| **py/compare_results_to_baseline.py** | Compares benchmark output to DFlash-reported baselines. | N/A |
| **py/dflash_reported_baselines.py** | Extracted DFlash paper reference values. | N/A |
| **py/preflight_dflash_validation.py** | Preflight: require pytest, vLLM, external/dflash. | N/A |
| **py/run_tpu_dflash_eval.py** | Runs DFlash eval on TPU; used by run_tpu_inference_dflash_eval.sh | v4, v5p |
| **py/check_tpu_inference_scope.py** | Checks tpu-inference change scope. | N/A |
| **sh/run_dflash_validation_matrix.sh** | One-shot validation: pytest, smoke, eval. | v4, v5p |
| **sh/run_tpu_inference_dflash_eval.sh** | Runs TPU DFlash eval. | v4, v5p |
| **sh/run_tpu_inference_dflash_smoke.sh** | Runs TPU DFlash smoke. | v4, v5p |
| **sh/run_pytest_dflash_subset.sh** | Runs pytest DFlash subset. | v4, v5p |
| **sh/check_tpu_inference_change_scope.sh** | Scope guard for tpu-inference changes. | N/A |
| **sh/run_external_dflash_reference.sh** | Runs external DFlash reference (GPU). | GPU |
| **config/tpu_inference_dflash_allowlist.txt** | Allowlist for scope checks. | N/A |

---

## Quick reference: run commands by hardware

| Goal | v4 | v5p |
|------|----|-----|
| Standalone DFlash | `bash tests/standalone_benchmark.sh` | `bash tests/standalone_benchmark.sh` or `bash tests/run_standalone_all_v5p.sh` |
| vLLM pipeline | `bash tests/benchmark.sh math` | `bash tests/benchmark.sh math` |
| Smoke test | `bash tests/smoke.sh` | `bash tests/smoke.sh` |
| Verify context scaling | `bash tests/verify_context_scaling.sh` | `bash tests/verify_context_scaling.sh` |
| Pipeline profiling | `bash tests/pipeline_profiling.sh` | `bash tests/pipeline_profiling.sh` |
| Compare to GPU | `bash tests/compare.sh latest` | `bash tests/compare.sh latest` |

GPU benchmarks: run from `benchmarks/gpu/` (see `gpu/README.md`).

### Verify all wrappers (v4 or v5p)

To run every shell wrapper in sequence on a fresh clone:

```bash
bash tests/verify_all_wrappers.sh v5p          # Full run on v5p
bash tests/verify_all_wrappers.sh v4 --quick   # Minimal samples on v4
bash tests/verify_all_wrappers.sh v5p --skip-prep --no-cleanup  # Repos already cloned; keep outputs
```

Logs: `.verify_wrappers_log/verify_<hw>_<timestamp>.log`

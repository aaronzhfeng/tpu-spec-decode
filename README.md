# Toward Accelerated LLM Inference

### Porting and Evaluating Diffusion-Based Speculative Decoding on TPU

---

## Abstract

Large language model (LLM) inference is bottlenecked by sequential autoregressive decoding. **DFlash** replaces autoregressive drafting with a block diffusion model, enabling parallel generation of 16 draft tokens per step and substantial speedups on GPU. We port DFlash to TPU by reimplementing it in JAX within the `tpu-inference` framework and integrating it into the vLLM serving pipeline—our primary deliverable. The vLLM pipeline achieves 2.85× speedup and 265 TPS on TPU v4 across four math datasets. A head-to-head comparison with Eagle3 (both using Qwen3-4B) shows DFlash's block diffusion advantage (2.85× vs. 1.30×). The standalone benchmark reaches 94.9% of GPU draft quality (τ) on TPU v5p, 3.02× overall speedup, and 773 TPS peak on Math500. We further report experimentations on verification cost scaling (K-flat on datacenter hardware), block size theory, and iterative refinement. Output quality is preserved. A pull request to `tpu-inference` has been submitted.

---

## Team & Links

| | |
|---|---|
| **Team** | Aaron Feng, Zhongyan Luo, Son Nguyen, Andy Huang |
| **Advisors** | Hao Zhang, Yiming Zhao (UC San Diego) |
| **Website** | [DFlash on TPU](https://zhongyan0721.github.io/tpu-dflash/) |
| **Code** | [tpu-spec-decode](https://github.com/aaronzhfeng/tpu-spec-decode) · [tpu-inference (PR #1868)](https://github.com/vllm-project/tpu-inference/pull/1868) |

---

## Key Results

| Metric | Value |
|--------|-------|
| TPU standalone τ (avg, v5p) | 5.42 (94.9% of GPU) |
| TPU v5p 9-dataset speedup | 3.13× |
| vLLM pipeline τ (v4) | 4.75 |
| vLLM pipeline speedup (v4) | 2.85× (265 TPS) |
| DFlash vs Eagle3 on TPU | 2.19× higher speedup |
| Verify latency K=16→K=256 | Flat (0.97×) — *memory-bound* |
| Peak throughput (Math500, v5p) | 773 TPS |

### K-Flat Verification Property

TPU verification forward-pass latency is invariant to the number of query tokens K (single-request regime). GPU scales 1.24× at K=128 vs K=16; TPU remains flat (0.97×). This enables risk-free wide-block drafting on TPU—a key hardware-regime insight.

| Hardware | K=128/K=16 | K-flat through | L-flat through |
|----------|------------|----------------|----------------|
| TPU v4-8 | 0.97× | K=1024 | L=1024 |
| TPU v5p-8 | 0.97× | K=1024 | L=4096 |
| GPU (RTX 2000 Ada) | 1.24× | K=128 | — |

---

## Quick Start

### TPU v5p

```bash
git clone --recurse-submodules https://github.com/aaronzhfeng/tpu-spec-decode.git
cd tpu-spec-decode
```

**Option A — Host setup (bare-metal venv):**

```bash
bash preparation/setup_v5p_safe.sh
source ~/venv/bin/activate
bash preparation/run_dflash_acceptance_smoke.sh host
```

**Option B — Docker-based verification:**

```bash
bash preparation/clone_repos.sh
bash tests/verify_all_wrappers.sh v5p --quick
```

For v5p, do not use `--skip-pr` when cloning; pr-ready (pr/dflash, vllm-lkg) is required. See `preparation/V5P_SETUP_MANUAL.md` for troubleshooting.

### TPU v4

v4 uses Flax 0.11.1. Root tpu-inference/vllm (dflash-integration) is sufficient; `--skip-pr` is allowed. Use Docker (host setup is v5p-only).

```bash
bash preparation/clone_repos.sh --skip-pr
FLAX_VERSION=0.11.1 bash tests/verify_all_wrappers.sh v4
```

Docker default is `FLAX_VERSION=0.11.1`; v4 runs all wrappers including the vLLM pipeline (unlike v5p).

---

## Repo Structure

```
tpu-spec-decode/
├── preparation/        # Setup scripts, clone_repos, V5P manual
├── benchmarks/         # standalone_dflash.py, amortized_verification, ...
├── tests/              # Shell wrappers (standalone_benchmark, smoke, verify_all_wrappers)
├── docs/               # Experiment docs (00–52)
├── results/            # JSON/CSV benchmark outputs (v4, v5p)
├── tpu-inference/      # Cloned (dflash-integration or pr-ready/pr)
└── vllm/               # Cloned (dflash-speculative-config or pr-ready/vllm-lkg)
```

---

## Technical Highlights

- **Dual KV cache architecture:** Target uses paged attention; draft uses static JAX arrays with `dynamic_update_slice` and non-causal `flash_attention`
- **Context buffer management:** Projected target hidden states fed to draft; power-of-2 padding to limit JIT retracing
- **Seq_len inflation fix:** vLLM manager passed inflated `seq_lens`; switching to `num_tokens_no_spec` doubled speedup (1.30× → 2.31×)
- **Standalone benchmark:** Isolates algorithm quality; mirrors GPU paper setup for direct comparison
- **K-flat mechanism:** Per-layer fixed overhead (64%) dominates; K-dependent attention compute is 2.1% of FLOPs—absorbed within overhead

---

## Deliverables

| Deliverable | Location |
|-------------|----------|
| tpu-inference PR #1868 | `pr-ready/pr` (pr/dflash branch) |
| Capstone report | `capstone_report/final/` |
| Benchmark scripts | `benchmarks/`, `tests/` |
| Experiment docs | `docs/` (52 docs) |

---

## Research Direction

Post-port findings show verification cost is flat from K=16 to K=1024 on TPU. DFlash's block_size=16 is a GPU design constraint; TPU's K-flat property enables risk-free wide-block drafting. Proposed next step: train DFlash with block_size=128 (TPU-native optimal).

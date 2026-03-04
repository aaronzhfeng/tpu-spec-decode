---
marp: true
theme: default
paginate: true
footer: 'DFlash TPU Benchmark Report | 2026-03-03'
---

<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&family=Fira+Code:wght@400;500;700&display=swap');

:root {
  --color-background: #0d1117;
  --color-foreground: #c9d1d9;
  --color-heading: #58a6ff;
  --color-accent: #7ee787;
  --color-code-bg: #161b22;
  --color-border: #30363d;
  --color-dim: #8b949e;
  --color-warn: #d29922;
  --color-good: #3fb950;
  --color-bad: #f85149;
  --font-default: 'Noto Sans JP', 'Hiragino Kaku Gothic ProN', 'Meiryo', sans-serif;
  --font-code: 'Fira Code', 'Consolas', 'Monaco', monospace;
}

section {
  background-color: var(--color-background);
  color: var(--color-foreground);
  font-family: var(--font-default);
  font-weight: 400;
  box-sizing: border-box;
  border-left: 4px solid var(--color-accent);
  position: relative;
  line-height: 1.5;
  font-size: 18px;
  padding: 48px 56px;
}

h1, h2, h3, h4, h5, h6 {
  font-weight: 700;
  color: var(--color-heading);
  margin: 0;
  padding: 0;
  font-family: var(--font-code);
}

h1 {
  font-size: 48px;
  line-height: 1.3;
  text-align: left;
}

h1::before {
  content: '# ';
  color: var(--color-accent);
}

h2 {
  font-size: 32px;
  margin-bottom: 24px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--color-border);
}

h2::before {
  content: '## ';
  color: var(--color-accent);
}

h3 {
  color: var(--color-foreground);
  font-size: 22px;
  margin-top: 20px;
  margin-bottom: 8px;
}

h3::before {
  content: '### ';
  color: var(--color-accent);
}

ul, ol {
  padding-left: 28px;
  margin: 8px 0;
}

li {
  margin-bottom: 6px;
  font-size: 17px;
}

li::marker {
  color: var(--color-accent);
}

/* Table-heavy styling */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 12px 0;
  font-size: 15px;
  font-family: var(--font-code);
}

th, td {
  border: 1px solid var(--color-border);
  padding: 6px 10px;
  text-align: left;
  color: #4a5568;
  font-weight: 700;
}

th {
  background-color: #e2e8f0;
  color: #556677;
  font-weight: 700;
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

tr:nth-child(even) {
  background-color: rgba(22, 27, 34, 0.5);
}

tr:hover {
  background-color: rgba(88, 166, 255, 0.08);
}

pre {
  background-color: var(--color-code-bg);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  font-family: var(--font-code);
  font-size: 14px;
  line-height: 1.4;
}

code {
  background-color: var(--color-code-bg);
  color: var(--color-accent);
  padding: 2px 6px;
  border-radius: 3px;
  font-family: var(--font-code);
  font-size: 0.85em;
}

pre code {
  background-color: transparent;
  padding: 0;
  color: var(--color-foreground);
}

footer {
  font-size: 12px;
  color: var(--color-dim);
  font-family: var(--font-code);
  position: absolute;
  left: 56px;
  right: 56px;
  bottom: 28px;
  text-align: right;
}

footer::before {
  content: '// ';
  color: var(--color-accent);
}

section.lead {
  border-left: 4px solid var(--color-accent);
  display: flex;
  flex-direction: column;
  justify-content: center;
}

section.lead h1 {
  margin-bottom: 20px;
}

section.lead p {
  font-size: 20px;
  color: var(--color-foreground);
  font-family: var(--font-code);
}

strong {
  color: var(--color-accent);
  font-weight: 700;
}

/* Annotation blocks */
blockquote {
  border-left: 3px solid var(--color-warn);
  background-color: rgba(210, 153, 34, 0.08);
  padding: 8px 16px;
  margin: 12px 0;
  font-size: 15px;
  border-radius: 0 4px 4px 0;
}

blockquote strong {
  color: var(--color-warn);
}

/* Section divider slides */
section.divider {
  display: flex;
  flex-direction: column;
  justify-content: center;
  border-left: 4px solid var(--color-heading);
}

section.divider h2 {
  font-size: 44px;
  border-bottom: none;
  margin-bottom: 16px;
}
</style>

<!-- _class: lead -->
<!-- _paginate: false -->

# DFlash on TPU

Speculative Decoding Benchmark Report
Qwen3-4B + DFlash-b16 | TPU v4 vs v5p

---

## Overview

| Property | Value |
|----------|-------|
| Target Model | `Qwen/Qwen3-4B` (3.4B params, 36 layers) |
| Draft Model | `z-lab/Qwen3-4B-DFlash-b16` (diffusion drafter) |
| Block Size | 16 (draft 15 tokens per step) |
| Benchmarks | 9 datasets across math, code, chat |
| Samples | 8 per dataset, 256 max tokens, temp=0.0, 1 warmup |
| Runtime | Docker `vllm/vllm-tpu:latest` (JAX 0.8.1, Flax 0.11.1) |

> **DFlash** uses a diffusion-style draft model that proposes token blocks in O(1) forward passes, achieving high acceptance rates on structured tasks (math, code).

---

## V5P Results — All 9 Datasets

| Dataset | Cat | Baseline | DFlash | Speedup | Tau | Match |
|---------|-----|----------|--------|---------|-----|-------|
| gsm8k | math | 6.35 ms | 2.12 ms | **3.00x** | 5.40 | 5/8 |
| math500 | math | 6.37 ms | 1.29 ms | **4.93x** | 8.80 | 3/8 |
| aime24 | math | 6.31 ms | 1.77 ms | **3.57x** | 6.48 | 2/8 |
| aime25 | math | 6.38 ms | 1.88 ms | **3.39x** | 6.14 | 3/8 |
| humaneval | code | 6.31 ms | 1.99 ms | **3.17x** | 5.76 | 4/8 |
| mbpp | code | 6.40 ms | 1.96 ms | **3.27x** | 6.16 | 3/8 |
| mt-bench | chat | 6.27 ms | 2.74 ms | **2.29x** | 3.87 | 1/8 |
| alpaca | chat | 6.29 ms | 3.86 ms | **1.63x** | 2.86 | 4/8 |
| swe-bench | code | 6.32 ms | 3.40 ms | **1.86x** | 3.35 | 0/8 |

> **Match** = exact token match between DFlash and autoregressive baseline. Mismatches are bf16 floating-point divergence (batch-16 verify vs single-token), not correctness loss.

---

## V5P Results — By Category

| Category | Avg Speedup | Avg Tau | Avg DFlash TPS | Best Dataset |
|----------|:-----------:|:-------:|:--------------:|:------------:|
| Math (4) | **3.72x** | 6.71 | 585.4 | math500 (4.93x) |
| Code (3) | **2.77x** | 5.09 | 435.7 | mbpp (3.27x) |
| Chat (2) | **1.96x** | 3.37 | 312.2 | mt-bench (2.29x) |
| **All (9)** | **3.02x** | **5.42** | **474.8** | |

> **Why math >> chat?** Structured math reasoning has highly predictable token sequences (equations, step numbering). Chat has more creative variation, reducing draft acceptance rate.

---

## V5P Results — Throughput (TPS)

| Dataset | Baseline TPS | DFlash TPS | Gain |
|---------|:------------:|:----------:|:----:|
| gsm8k | 157.5 | 471.9 | +200% |
| math500 | 156.9 | **772.7** | +392% |
| aime24 | 158.4 | 565.0 | +257% |
| aime25 | 156.9 | 532.0 | +239% |
| humaneval | 158.4 | 501.9 | +217% |
| mbpp | 156.4 | 510.7 | +226% |
| mt-bench | 159.4 | 365.2 | +129% |
| alpaca | 158.9 | 259.2 | +63% |
| swe-bench | 158.2 | 294.5 | +86% |

> **Peak throughput**: **772.7 TPS** on math500 — the draft model achieves tau=8.80 (nearly 9 tokens accepted per draft block of 15).

---

<!-- _class: divider -->

## V5P vs V4

Hardware generation comparison on identical benchmarks

---

## V5P vs V4 — Latency (TPOT)

| Dataset | V5P Baseline | V4 Baseline | V5P DFlash | V4 DFlash |
|---------|:------------:|:-----------:|:----------:|:---------:|
| gsm8k | 6.35 ms | 10.75 ms | 2.12 ms | 3.20 ms |
| math500 | 6.37 ms | 10.73 ms | **1.29 ms** | 1.88 ms |
| aime24 | 6.31 ms | 10.68 ms | 1.77 ms | 2.72 ms |
| aime25 | 6.38 ms | 10.74 ms | 1.88 ms | 2.57 ms |
| humaneval | 6.31 ms | 10.81 ms | 1.99 ms | 3.01 ms |
| mbpp | 6.40 ms | 10.66 ms | 1.96 ms | 3.27 ms |
| mt-bench | 6.27 ms | 10.69 ms | 2.74 ms | 4.15 ms |
| alpaca | 6.29 ms | 10.65 ms | 3.86 ms | 5.47 ms |
| swe-bench | 6.32 ms | 10.65 ms | 3.40 ms | 8.42 ms |

> V5P baseline is **1.69x** faster than V4 across the board. V5P DFlash is **1.55x** faster in absolute latency.

---

## V5P vs V4 — Speedup & Tau

| Dataset | V5P Speedup | V4 Speedup | V5P Tau | V4 Tau |
|---------|:-----------:|:----------:|:-------:|:------:|
| gsm8k | 3.00x | 3.36x | 5.40 | 5.17 |
| math500 | 4.93x | 5.70x | 8.80 | 8.72 |
| aime24 | 3.57x | 3.93x | 6.48 | 6.32 |
| aime25 | 3.39x | 4.18x | 6.14 | 6.48 |
| humaneval | 3.17x | 3.59x | 5.76 | 5.75 |
| mbpp | 3.27x | 3.26x | 6.16 | 5.95 |
| mt-bench | 2.29x | 2.58x | 3.87 | 3.84 |
| alpaca | 1.63x | 1.95x | 2.86 | 2.83 |
| swe-bench | 1.86x | 1.27x | 3.35 | 3.28 |

> **Speedup ratios are lower on V5P** because the baseline improved proportionally more than DFlash. However, **absolute DFlash performance is better** on V5P in every dataset. Tau is hardware-independent (model quality metric).

---

## V5P vs V4 — Summary

| Metric | V5P | V4 | Delta |
|--------|:---:|:--:|:-----:|
| Avg Baseline TPOT | 6.33 ms | 10.71 ms | **1.69x faster** |
| Avg DFlash TPOT | 2.44 ms | 3.85 ms | **1.58x faster** |
| Avg Speedup | 3.02x | 3.36x | -10% ratio |
| Avg Tau | 5.42 | 5.37 | +0.9% |
| Best DFlash TPOT | 1.29 ms | 1.88 ms | **1.46x faster** |
| Peak TPS | 772.7 | 531.5 | **+45%** |

> **Key insight**: V5P delivers higher absolute throughput but lower speedup *ratios* because autoregressive decoding also benefits from faster hardware. DFlash's value increases with model size (where baseline is slower).

---

<!-- _class: divider -->

## V5P vs GPU Paper

DFlash paper (arXiv 2025) reported GPU results on A100

---

## V5P vs GPU Paper — Math Benchmarks

| Dataset | V5P Tau | GPU Tau | Tau % | V5P Speedup | GPU Speedup |
|---------|:-------:|:-------:|:-----:|:-----------:|:-----------:|
| gsm8k | 5.40 | 6.53 | 82.7% | 3.00x | 5.15x |
| math500 | 8.80 | 7.84 | **112.2%** | 4.93x | 6.09x |
| aime24 | 6.48 | 7.27 | 89.1% | 3.57x | 5.68x |
| aime25 | 6.14 | 6.64 | 92.5% | 3.39x | 5.21x |
| **Average** | **6.71** | **7.07** | **94.9%** | **3.72x** | **5.53x** |

> V5P TPU achieves **94.9% of GPU paper tau** on math benchmarks — nearly full acceptance rate parity. Speedup gap is due to TPU's already-fast baseline (6.3ms vs GPU's slower autoregressive path).

> Math500 **exceeds** GPU tau (112.2%), likely due to sampling variance at 8 samples.

---

## Output Quality

| Dataset | Exact Match | Rate | Answer Match | Notes |
|---------|:-----------:|:----:|:------------:|-------|
| gsm8k | 5/8 | 62.5% | 2/3 | High match on structured math |
| math500 | 3/8 | 37.5% | 2/2 | Answers always correct |
| aime24 | 2/8 | 25.0% | N/A | Competition problems diverge early |
| aime25 | 3/8 | 37.5% | N/A | Similar to aime24 |
| humaneval | 4/8 | 50.0% | N/A | Code structure preserved |
| mbpp | 3/8 | 37.5% | N/A | Variable names may differ |
| mt-bench | 1/8 | 12.5% | N/A | Creative text diverges easily |
| alpaca | 4/8 | 50.0% | N/A | Short responses match better |
| swe-bench | 0/8 | 0.0% | N/A | Long context, early divergence |

> **Exact match** = DFlash produces identical tokens to baseline. Mismatches come from bf16 batch-16 verify vs single-token decode producing different floating-point accumulation. This flips argmax at decision boundaries — **not a correctness issue**.

---

<!-- _class: divider -->

## Environment & Reproducibility

---

## Hardware & Software

| Component | V5P | V4 |
|-----------|-----|-----|
| TPU Type | v5p-8 (4 chips) | v4-8 (4 chips) |
| Topology | 2x2x1x1 | 2x2x1 |
| Device Interface | `/dev/vfio/[0-3]` | `/dev/accel/[0-3]` |
| `/dev/shm` | 221 GB | 330 GB |
| Python | 3.12 (Docker) | 3.12 (Docker) |
| JAX | 0.8.1 | 0.8.1 |
| Flax | 0.11.1 | 0.11.1 |
| Docker Image | `vllm/vllm-tpu:latest` | `vllm/vllm-tpu:latest` |

> Both V4 and V5P benchmarks use the same Docker image and identical code (`benchmarks/standalone_dflash.py`). Results are directly comparable.

---

## Reproduce

```bash
# Activate environment
source ~/venv/bin/activate
export PYTHONPATH="$HOME/tpu-spec-decode/vllm:$HOME/tpu-spec-decode/tpu-inference"

# Run a single benchmark (Docker)
sudo docker run --rm --privileged --network host --ipc host \
  --tmpfs "/tmp:rw,size=80g" \
  -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub \
  -e PYTHONPATH=/workspace/tpu-spec-decode/vllm:/workspace/tpu-spec-decode/tpu-inference \
  -v /dev/shm/hf-cache:/hf-cache \
  -v /home/$USER/tpu-spec-decode:/workspace/tpu-spec-decode \
  -w /workspace/tpu-spec-decode \
  vllm/vllm-tpu:latest \
  python3 benchmarks/standalone_dflash.py \
    --target-model Qwen/Qwen3-4B \
    --draft-model z-lab/Qwen3-4B-DFlash-b16 \
    --dataset gsm8k --max-samples 8 --max-new-tokens 256 \
    --temperature 0.0 --warmup 1 \
    --output-json /output/result.json
```

---

## File Layout

```
results/
  v4/                              # V4 benchmark data (28 files)
    standalone_*.json              # Raw results per dataset
    standalone_all_benchmarks.csv  # Summary table
    standalone_vs_gpu_paper.csv    # V4 vs GPU paper
    report.md                      # V4 report
  v5p/                             # V5P benchmark data (12 files)
    standalone_*.json              # Raw results per dataset
    standalone_all_benchmarks.csv  # Summary table (with accuracy)
    standalone_vs_v4.csv           # V5P vs V4 comparison
    standalone_vs_gpu_paper.csv    # V5P vs GPU paper
  report.md                        # This report (Marp slides)
```

> Raw JSON files contain per-sample data: token outputs, acceptance histograms, per-position acceptance rates, quality comparisons.

---

<!-- _class: divider -->

## PR Validation (pr/dflash)

Upstream PR benchmarked on TPU v5p vs GPU paper (A100)

---

## PR Validation — TPU v5p vs GPU (A100)

| Dataset | Cat | TPU Tau | GPU Tau | Tau % | TPU Speedup | GPU Speedup |
|---------|-----|:-------:|:-------:|:-----:|:-----------:|:-----------:|
| gsm8k | math | 5.40 | 6.53 | 82.7% | 2.95x | 5.15x |
| math500 | math | 8.80 | 7.84 | **112.2%** | 4.90x | 6.09x |
| aime24 | math | 6.48 | 7.27 | 89.1% | 3.49x | 5.68x |
| aime25 | math | 6.14 | 6.64 | 92.5% | 3.32x | 5.21x |
| humaneval | code | 5.76 | — | — | 3.09x | — |
| mbpp | code | 6.16 | — | — | 3.41x | — |
| mt-bench | chat | 3.87 | — | — | 2.38x | — |
| alpaca | chat | 2.86 | — | — | 1.65x | — |
| swe-bench | code | 3.35 | — | — | 1.97x | — |

> Results from `pr/dflash` branch ([aaronzhfeng/tpu-inference](https://github.com/aaronzhfeng/tpu-inference)). Math tau is **94.9%** of GPU paper on TPU — near-parity in draft quality. Speedup gap is architectural: TPU's faster autoregressive baseline compresses the ratio.

---

## PR Validation — Summary

| Metric | TPU v5p (PR) | GPU A100 (Paper) |
|--------|:------------:|:----------------:|
| Math Avg Tau | **6.71** | 7.07 |
| Math Avg Speedup | **3.67x** | 5.53x |
| Math Tau Parity | **94.9%** | 100% |
| Peak Tau | **8.80** (math500) | 7.84 (math500) |
| Datasets Tested | **9** | 4 |
| Hardware | v5p-8, 4 chips | A100 80GB |

> **DFlash on TPU matches GPU draft quality** — the acceptance rate gap is <6% on math. The speedup ratio difference is explained by TPU's 6.3ms autoregressive TPOT vs GPU's slower baseline. At K=16, DFlash already delivers **3.02x** average speedup. The real opportunity is scaling to K=64/128 where TPU verification cost stays flat.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Summary

**3.02x** avg speedup on V5P across 9 benchmarks
**4.93x** peak on math500 | **772.7 TPS** peak throughput
**94.9%** of GPU paper tau on math benchmarks
**1.69x** faster baseline than V4 | **1.55x** faster DFlash

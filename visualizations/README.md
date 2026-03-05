# DFlash Speculative Decoding — Visualizations

This folder contains a Python script that generates 15 publication-ready visualizations from the TPU v4 and v5p DFlash speculative decoding benchmark results. Figures 1–10 cover exploratory analysis and ablations; figures R1–R5 present the **final findings** matching the slide deck in `results/report.md`.

## Quick Start

```bash
pip install matplotlib numpy

python visualizations/generate_visualizations.py
```

All 15 PNG figures will be written into the `visualizations/` folder alongside the script.

## How the Code Works

`generate_visualizations.py` is a single self-contained script structured as follows:

1. **Path setup** — It locates data files relative to its own location: `../results/v4/` and `../results/v5p/`.
2. **Helper functions** — `_read_csv()` and `_read_json()` load data; `_bar_label()` adds value annotations to bar charts.
3. **Analysis figures** (`fig1_*` through `fig10_*`) — Deep-dive visualizations covering speedup, throughput, profiling, ablations, and method comparisons.
4. **Report figures** (`figR1_*` through `figR5_*`) — Final-findings visualizations aligned with the report slide deck: category performance, latency comparison, GPU parity, output quality, and a summary dashboard.
5. **Main block** — Calls all fifteen functions in sequence and prints progress.

No external configuration is needed. The script uses only `matplotlib` and `numpy` (plus the Python standard library).

---

## Figure Catalog

### Figure 1 — DFlash Speedup: TPU v4 vs v5p

| | |
|---|---|
| **File** | `fig1_speedup_v4_vs_v5p.png` |
| **Data source** | `results/v4/standalone_all_benchmarks.csv`, `results/v5p/standalone_all_benchmarks.csv` |
| **What it shows** | Side-by-side bar chart of DFlash speedup over autoregressive baseline for all 9 benchmarks (gsm8k, math500, aime24, aime25, humaneval, mbpp, mt-bench, alpaca, swe-bench) on both TPU v4 and TPU v5p. Bars are color-coded by benchmark category (math, code, chat). |
| **Why it matters** | This is the headline result. It answers: "How much faster is speculative decoding than normal autoregressive generation?" Speedups range from 1.3x (swe-bench) to 5.7x (math500), demonstrating that DFlash delivers substantial acceleration on TPU hardware, with math-heavy benchmarks benefiting most due to higher draft acceptance rates. |

---

### Figure 2 — Absolute Throughput (Tokens per Second)

| | |
|---|---|
| **File** | `fig2_throughput.png` |
| **Data source** | `results/v4/standalone_all_benchmarks.csv`, `results/v5p/standalone_all_benchmarks.csv` |
| **What it shows** | Four-bar grouped chart showing baseline TPS and DFlash TPS for both v4 and v5p across all 9 benchmarks. |
| **Why it matters** | While speedup is a ratio, absolute throughput tells you the real-world generation speed. TPU v5p + DFlash hits 773 tokens/sec on math500 — over 8x the v4 baseline of 93 TPS. This figure shows the compounding benefit of better hardware (v5p) and better algorithms (DFlash). |

---

### Figure 3 — Tau Comparison: TPU v4 vs v5p vs GPU Paper

| | |
|---|---|
| **File** | `fig3_tau_tpu_vs_gpu.png` |
| **Data source** | `results/v4/standalone_vs_gpu_paper.csv`, `results/v5p/standalone_vs_gpu_paper.csv` |
| **What it shows** | Grouped bars comparing tau (average number of accepted draft tokens per speculative step) between TPU v4, TPU v5p, and the GPU reference numbers from the DFlash paper, on the 4 math benchmarks. |
| **Why it matters** | Tau is a hardware-independent measure of draft model quality. If the TPU tau matches the GPU paper tau, it confirms the implementation is correct and the draft model works as expected on TPU. The TPU tau averages ~94% of the GPU paper tau, validating the port while highlighting where further optimization is possible. |

---

### Figure 4 — Acceptance Rate Decay by Draft Position

| | |
|---|---|
| **File** | `fig4_acceptance_decay.png` |
| **Data source** | `results/v4/standalone_acceptance_per_pos.csv` |
| **What it shows** | Line chart with one curve per dataset showing how the probability of accepting the i-th draft token decays as the position increases (0 through 15). |
| **Why it matters** | This is the fundamental characteristic of speculative decoding. Position 0 always has 100% acceptance (the first draft token is always checked). By position 15, acceptance drops to 2–21% depending on the dataset. Math500 has the gentlest decay (high tau), while alpaca and mt-bench decay faster (lower tau). Understanding this curve is critical for choosing the optimal block size — the sweet spot where the draft model still adds value before the acceptance rate drops too low. |

---

### Figure 5 — Profiling: Step Time Breakdown

| | |
|---|---|
| **File** | `fig5_profiling_breakdown.png` |
| **Data source** | `results/v4/profiling_gsm8k.json` |
| **What it shows** | Dual panel: a pie chart showing the proportion of time each component consumes, and a horizontal bar chart showing absolute latency in ms. Components include draft forward, draft sample, verify forward, acceptance checking, host-device transfer, auxiliary projection, context update, and cache management. |
| **Why it matters** | Reveals where time is actually spent inside a DFlash step. The acceptance kernel (8.3 ms) dominates at ~48% of total step time, far exceeding the verify forward pass (2.6 ms) and draft forward pass (0.5 ms). This is a critical finding: the bottleneck is not the neural network inference but the token acceptance logic. Optimizing the acceptance kernel (e.g., fusing it with the verify pass) is the highest-leverage path to further speedups. |

---

### Figure 6 — Effect of Refinement Steps on Tau and Speedup

| | |
|---|---|
| **File** | `fig6_refinement_impact.png` |
| **Data source** | `results/v4/refinement_gsm8k.json` |
| **What it shows** | Dual-axis bar chart comparing tau (left axis) and end-to-end speedup (right axis) for k=0, 1, 2, and 3 refinement steps on GSM8K. |
| **Why it matters** | Refinement steps are an optional post-processing stage where rejected draft tokens are re-sampled. Counterintuitively, k=0 (no refinement) is the clear winner with tau=6.18 and 3.93x speedup, while k=1 drops to tau=2.48 and only 1.36x speedup. Each refinement step adds latency (~2.7 ms) that far outweighs the marginal benefit of recovering a few extra tokens. This tells practitioners to skip refinement on TPU. |

---

### Figure 7 — Verification Latency vs Context Length

| | |
|---|---|
| **File** | `fig7_context_scaling.png` |
| **Data source** | `results/v4/verify_context_scaling.json` |
| **What it shows** | Line chart with error bars showing mean verification latency (ms) vs context length (64, 256, 512, 1024 tokens) for K=16, 64, and 128 draft tokens. |
| **Why it matters** | A key question for production deployment: does verification get slower as the context grows? The answer is no — latency stays flat around 1.8 ms regardless of context length, and is also nearly identical across K values. This happens because TPU verification processes all K draft tokens in a single batched forward pass. It means DFlash's speedup advantage holds even for long-context applications. |

---

### Figure 8 — Cross-Comparison: Standalone vs vLLM Pipeline vs GPU Paper

| | |
|---|---|
| **File** | `fig8_cross_comparison.png` |
| **Data source** | `results/v4/cross_comparison.csv` |
| **What it shows** | Three-group bar chart comparing tau between the standalone TPU implementation, the vLLM-integrated pipeline, and the GPU paper reference, on 4 math benchmarks. |
| **Why it matters** | The standalone implementation achieves tau close to the GPU paper (93.7% on average), confirming algorithmic correctness. The vLLM pipeline has lower tau (~4.48 across all datasets) due to the overhead of integrating with the full serving stack (batching, scheduling, KV cache management). This gap quantifies the "integration tax" and motivates further optimization of the vLLM integration path. |

---

### Figure 9 — TPU v5p over v4: Improvement Factors

| | |
|---|---|
| **File** | `fig9_v5p_improvement.png` |
| **Data source** | `results/v5p/standalone_vs_v4.csv` |
| **What it shows** | Side-by-side bars showing how much faster v5p is compared to v4 for both the baseline (autoregressive) and DFlash modes, across all 9 benchmarks. |
| **Why it matters** | Baseline improves by a consistent ~1.69x across all benchmarks (matching the expected hardware improvement). DFlash improvement varies from 1.37x to 2.48x. SWE-bench sees the largest DFlash improvement (2.48x) because its low acceptance rate on v4 meant the verification step dominated — v5p's faster verification disproportionately helps these worst-case workloads. |

---

### Figure 10 — Speculative Decoding Methods Compared

| | |
|---|---|
| **File** | `fig10_method_comparison.png` |
| **Data source** | `results/v4/standalone_all_benchmarks.csv`, `results/v4/vllm_pipeline_results.csv`, `results/v4/eagle3_llama_results.csv` |
| **What it shows** | Three-group bar chart comparing speedup across DFlash standalone, DFlash vLLM pipeline, and Eagle3 (Llama-based) on the 4 math benchmarks. |
| **Why it matters** | Puts DFlash into context against alternative speculative decoding methods. DFlash standalone achieves 3.4–5.7x speedup, the vLLM pipeline achieves 2.0–2.6x, and Eagle3 achieves 1.5–1.6x. DFlash's advantage comes from its architecture-specific draft model (DFlash-b16 shares weights with the target), whereas Eagle3 uses a separate Llama draft model. This comparison helps practitioners choose the right method for their deployment scenario. |

---

## Report Figures (Final Findings)

These five figures map directly to the key sections in `results/report.md` and present the final conclusions.

---

### Figure R1 — V5P Performance by Task Category

| | |
|---|---|
| **File** | `figR1_category_summary.png` |
| **Data source** | `results/v5p/standalone_all_benchmarks.csv` |
| **Report section** | "V5P Results — By Category" |
| **What it shows** | Three-panel chart showing average speedup, average tau, and average DFlash TPS broken down by task category (math, code, chat) on TPU v5p. |
| **Why it matters** | Distills the 9-benchmark suite into a clear story: **math (3.72x, tau 6.71, 585 TPS) >> code (2.77x) >> chat (1.96x)**. The ordering is explained by task predictability — structured math reasoning produces highly predictable token sequences that the draft model can anticipate, while creative chat text has more variation that reduces acceptance rates. This is the most concise way to communicate the category-dependent nature of speculative decoding benefits. |

---

### Figure R2 — Latency (TPOT): V5P vs V4

| | |
|---|---|
| **File** | `figR2_latency_v5p_vs_v4.png` |
| **Data source** | `results/v5p/standalone_vs_v4.csv` |
| **Report section** | "V5P vs V4 — Latency (TPOT)" |
| **What it shows** | Four-bar grouped chart of time-per-output-token (TPOT in ms) for v4 baseline, v4 DFlash, v5p baseline, and v5p DFlash across all 9 datasets. DFlash latency values are annotated. |
| **Why it matters** | Visualizes the report's key insight that v5p baseline is 1.69x faster than v4 across the board, while v5p DFlash reaches as low as **1.29 ms** TPOT on math500. This is the absolute latency view — complementing the speedup ratios shown elsewhere — and demonstrates that v5p + DFlash delivers sub-2ms token generation for most tasks. |

---

### Figure R3 — TPU V5P vs GPU A100: Tau Parity and Speedup Gap

| | |
|---|---|
| **File** | `figR3_tpu_vs_gpu_parity.png` |
| **Data source** | `results/v5p/standalone_vs_gpu_paper.csv` |
| **Report section** | "V5P vs GPU Paper — Math Benchmarks" |
| **What it shows** | Dual-panel figure. Left panel: tau comparison between TPU v5p and GPU A100 paper, with percentage annotations colored green (>=100%), yellow (>=90%), or red (<90%). Right panel: end-to-end speedup comparison. |
| **Why it matters** | This is the validation figure. It shows that TPU achieves **94.9% of GPU tau** on math (near-parity in draft quality), with math500 actually exceeding the GPU (112.2%). The speedup gap (3.72x vs 5.53x) is *not* because the TPU draft model is worse — it's because the TPU autoregressive baseline is already fast (6.3 ms vs GPU's slower path), compressing the speedup ratio. This nuanced story is critical for fair comparison against the GPU paper. |

---

### Figure R4 — Output Quality: Exact Token Match Rates

| | |
|---|---|
| **File** | `figR4_output_quality.png` |
| **Data source** | `results/v5p/standalone_all_benchmarks.csv` |
| **Report section** | "Output Quality" |
| **What it shows** | Bar chart of exact token match rates (%) between DFlash and autoregressive baseline across all 9 datasets, color-coded by category. Each bar is annotated with the raw match count (e.g., "5/8"). |
| **Why it matters** | Addresses the most important question for production deployment: does speculative decoding change the output? Match rates range from 0% (swe-bench) to 62.5% (gsm8k). The note at the bottom explains that mismatches are **bf16 floating-point divergence** from batch-16 verification vs single-token decode — not correctness errors. On tasks where answers are extractable (gsm8k, math500), the final answers match even when token sequences diverge. |

---

### Figure R5 — Final Summary Dashboard

| | |
|---|---|
| **File** | `figR5_summary_dashboard.png` |
| **Data source** | `results/v5p/standalone_all_benchmarks.csv`, `results/v5p/standalone_vs_gpu_paper.csv` |
| **Report section** | Final summary slide |
| **What it shows** | Dark-themed dashboard with four headline metrics (avg speedup, peak speedup, peak TPS, GPU tau parity) displayed as large numbers at the top, and a horizontal bar chart showing per-dataset speedup with the average line marked. |
| **Why it matters** | This is the "one figure to rule them all" — designed for presentations and executive summaries. The four headline numbers (**3.02x** avg speedup, **4.93x** peak, **773 TPS** peak throughput, **94.9%** GPU tau parity) capture the entire story of the DFlash TPU port in a single glance. The bar chart below provides the per-dataset detail for audiences who want to dig deeper. |

---

## Data Sources Summary

All data files live under `results/` in the repository root:

| Directory | Contents |
|---|---|
| `results/v4/` | TPU v4 benchmark results: standalone per-benchmark JSONs, aggregated CSVs, vLLM pipeline results, Eagle3 comparison, profiling, refinement ablation, context scaling, quality checks |
| `results/v5p/` | TPU v5p benchmark results: standalone per-benchmark JSONs, aggregated CSV, v4 comparison CSV, GPU paper comparison CSV |

The benchmarks evaluate DFlash speculative decoding using **Qwen3-4B** as the target model and **Qwen3-4B-DFlash-b16** as the draft model across 9 datasets spanning math (gsm8k, math500, aime24, aime25), code (humaneval, mbpp, swe-bench), and chat (mt-bench, alpaca) tasks.

## Dependencies

- Python 3.8+
- `matplotlib`
- `numpy`

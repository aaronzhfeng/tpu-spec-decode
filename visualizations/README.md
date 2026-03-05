# DFlash Speculative Decoding — Visualizations

This folder contains a Python script that generates 12 publication-ready visualizations from the TPU DFlash speculative decoding benchmark results. **Figures 1–8 focus on V5P results** (including comparisons against GPU A100), while **Figures 9–12 provide secondary V4 context** such as V4-vs-V5P comparisons, profiling, and alternative method benchmarks.

## Quick Start

```bash
pip install matplotlib numpy

python visualizations/generate_visualizations.py
```

All 12 PNG figures will be written into the `visualizations/` folder alongside the script.

## How the Code Works

`generate_visualizations.py` is a single self-contained script structured as follows:

1. **Path setup** — It locates data files relative to its own location: `../results/v4/` and `../results/v5p/`.
2. **Helper functions** — `_read_csv()` and `_read_json()` load data; `_bar_label()` adds value annotations to bar charts.
3. **Primary figures** (`fig1` through `fig8`) — V5P-focused visualizations covering speedup, throughput, category breakdown, GPU parity, acceptance decay, latency, output quality, and a summary dashboard.
4. **Secondary figures** (`fig9` through `fig12`) — V4 and comparison visualizations: V5P-vs-V4 speedup, improvement factors, profiling breakdown, and method comparison.
5. **Main block** — Calls all twelve functions in sequence and prints progress.

No external configuration is needed. The script uses only `matplotlib` and `numpy` (plus the Python standard library).

---

## Figure Catalog

### Primary Figures — V5P Results

---

#### Figure 1 — V5P DFlash Speedup

| | |
|---|---|
| **File** | `fig1_v5p_speedup.png` |
| **Data source** | `results/v5p/standalone_all_benchmarks.csv` |
| **What it shows** | Bar chart of DFlash speedup over autoregressive baseline for all 9 benchmarks on TPU V5P, color-coded by category (math, code, chat), with the average speedup marked. |
| **Why it matters** | This is the headline result. Speedups range from 1.63x (alpaca) to 4.93x (math500), demonstrating substantial acceleration. Math benchmarks benefit most due to higher draft acceptance rates from predictable token sequences. |

---

#### Figure 2 — V5P Throughput

| | |
|---|---|
| **File** | `fig2_v5p_throughput.png` |
| **Data source** | `results/v5p/standalone_all_benchmarks.csv` |
| **What it shows** | Grouped bar chart comparing baseline TPS and DFlash TPS for all 9 benchmarks on TPU V5P. DFlash TPS values are annotated. |
| **Why it matters** | Absolute throughput tells you the real-world generation speed. V5P + DFlash hits 773 tokens/sec on math500, nearly 5x the baseline of 157 TPS. |

---

#### Figure 3 — V5P Performance by Task Category

| | |
|---|---|
| **File** | `fig3_v5p_category_summary.png` |
| **Data source** | `results/v5p/standalone_all_benchmarks.csv` |
| **What it shows** | Three-panel chart showing average speedup, average tau, and average DFlash TPS broken down by task category (math, code, chat). |
| **Why it matters** | Distills the 9-benchmark suite into a clear story: **math (3.72x, τ 6.71, 585 TPS) >> code (2.77x) >> chat (1.96x)**. Structured math reasoning produces highly predictable sequences that the draft model anticipates well, while creative chat text reduces acceptance rates. |

---

#### Figure 4 — V5P vs GPU A100: Tau Parity & Speedup Gap

| | |
|---|---|
| **File** | `fig4_v5p_vs_gpu.png` |
| **Data source** | `results/v5p/standalone_vs_gpu_paper.csv` |
| **What it shows** | Dual-panel figure. Left panel: tau comparison between TPU V5P and GPU A100, with percentage annotations colored green (≥100%), yellow (≥90%), or red (<90%). Right panel: end-to-end speedup comparison. |
| **Why it matters** | The validation figure. TPU achieves **94.9% of GPU tau** on math (near-parity), with math500 exceeding the GPU at 112.2%. The speedup gap (3.72x vs 5.53x) is because the TPU autoregressive baseline is already fast (6.3 ms vs GPU's slower path), compressing the ratio. |

---

#### Figure 5 — V5P Acceptance Rate Decay

| | |
|---|---|
| **File** | `fig5_v5p_acceptance_decay.png` |
| **Data source** | `results/v5p/standalone_*.json` (all 9 per-benchmark JSONs) |
| **What it shows** | Line chart with one curve per dataset showing how the probability of accepting the i-th draft token decays from position 0 through 15 on TPU V5P. |
| **Why it matters** | The fundamental characteristic of speculative decoding on V5P. Position 0 always has 100% acceptance; by position 15, acceptance drops to 2–6%. Math500 has the gentlest decay (τ=8.8), while alpaca decays fastest (τ=2.9). Understanding this curve is critical for choosing the optimal block size. |

---

#### Figure 6 — V5P Latency (TPOT)

| | |
|---|---|
| **File** | `fig6_v5p_latency.png` |
| **Data source** | `results/v5p/standalone_all_benchmarks.csv` |
| **What it shows** | Grouped bar chart of time-per-output-token (TPOT in ms) for baseline vs DFlash on TPU V5P across all 9 benchmarks. An annotation highlights the best DFlash latency. |
| **Why it matters** | The absolute latency view. V5P + DFlash reaches **1.29 ms** TPOT on math500, delivering sub-2ms token generation for most math and code tasks. |

---

#### Figure 7 — V5P Output Quality

| | |
|---|---|
| **File** | `fig7_v5p_output_quality.png` |
| **Data source** | `results/v5p/standalone_all_benchmarks.csv` |
| **What it shows** | Bar chart of exact token match rates (%) between DFlash and autoregressive baseline across all 9 datasets, with raw match counts annotated (e.g., "5/8"). |
| **Why it matters** | Addresses whether speculative decoding changes the output. Match rates range from 0% (swe-bench) to 62.5% (gsm8k). Mismatches are bf16 floating-point divergence from batch-16 verification, not correctness errors. |

---

#### Figure 8 — Summary Dashboard

| | |
|---|---|
| **File** | `fig8_summary_dashboard.png` |
| **Data source** | `results/v5p/standalone_all_benchmarks.csv`, `results/v5p/standalone_vs_gpu_paper.csv` |
| **What it shows** | Dark-themed dashboard with four headline metrics (avg speedup, peak speedup, peak TPS, GPU tau parity) and a horizontal bar chart of per-dataset speedup with the average line. |
| **Why it matters** | The "one figure to rule them all" — designed for presentations. The four headline numbers (**3.02x** avg, **4.93x** peak, **773 TPS** peak throughput, **94.9%** GPU tau parity) capture the entire V5P story at a glance. |

---

### Secondary Figures — V4 & Comparison

---

#### Figure 9 — V5P vs V4 Speedup

| | |
|---|---|
| **File** | `fig9_v5p_vs_v4_speedup.png` |
| **Data source** | `results/v4/standalone_all_benchmarks.csv`, `results/v5p/standalone_all_benchmarks.csv` |
| **What it shows** | Side-by-side bars of DFlash speedup on V4 vs V5P across all 9 benchmarks, color-coded by category. |
| **Why it matters** | Generational comparison showing that V5P matches or exceeds V4 speedup on most benchmarks, with the notable exception that V4 has slightly higher speedup on some math tasks due to a slower baseline amplifying the ratio. |

---

#### Figure 10 — V5P over V4: Improvement Factors

| | |
|---|---|
| **File** | `fig10_v5p_over_v4_improvement.png` |
| **Data source** | `results/v5p/standalone_vs_v4.csv` |
| **What it shows** | Side-by-side bars showing how much faster V5P is compared to V4 for both baseline and DFlash modes across all 9 benchmarks. |
| **Why it matters** | Baseline improves by a consistent ~1.69x (matching hardware improvement), while DFlash improvement varies from 1.37x to 2.48x. SWE-bench sees the largest DFlash improvement (2.48x) because V5P's faster verification disproportionately helps workloads with low acceptance rates. |

---

#### Figure 11 — Profiling: Step Time Breakdown (V4)

| | |
|---|---|
| **File** | `fig11_profiling_breakdown.png` |
| **Data source** | `results/v4/profiling_gsm8k.json` |
| **What it shows** | Dual panel: pie chart of time proportions and horizontal bar chart of absolute latencies for each DFlash step component (draft forward, verify forward, acceptance, etc.). |
| **Why it matters** | Reveals the bottleneck: the acceptance kernel (~48% of step time) dominates far beyond neural network inference. This insight is hardware-general and motivates future optimization of the acceptance logic, regardless of TPU generation. |

---

#### Figure 12 — Speculative Decoding Methods Compared (V4)

| | |
|---|---|
| **File** | `fig12_method_comparison.png` |
| **Data source** | `results/v4/standalone_all_benchmarks.csv`, `results/v4/vllm_pipeline_results.csv`, `results/v4/eagle3_llama_results.csv` |
| **What it shows** | Three-group bar chart comparing speedup across DFlash standalone, DFlash vLLM pipeline, and Eagle3 (Llama-based) on the 4 math benchmarks. |
| **Why it matters** | Puts DFlash into context: standalone achieves 3.4–5.7x, vLLM pipeline 2.0–2.6x, Eagle3 1.5–1.6x. DFlash's advantage comes from its architecture-specific draft model that shares weights with the target. |

---

## Data Sources Summary

All data files live under `results/` in the repository root:

| Directory | Contents |
|---|---|
| `results/v5p/` | **Primary.** TPU V5P standalone per-benchmark JSONs, aggregated CSV, V4 comparison CSV, GPU paper comparison CSV |
| `results/v4/` | **Secondary.** TPU V4 benchmark results: standalone CSVs, vLLM pipeline results, Eagle3 comparison, profiling, and more |

The benchmarks evaluate DFlash speculative decoding using **Qwen3-4B** as the target model and **Qwen3-4B-DFlash-b16** as the draft model across 9 datasets spanning math (gsm8k, math500, aime24, aime25), code (humaneval, mbpp, swe-bench), and chat (mt-bench, alpaca) tasks.

## Dependencies

- Python 3.8+
- `matplotlib`
- `numpy`

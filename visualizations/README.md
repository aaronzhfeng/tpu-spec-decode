# TPU DFlash Benchmark Visualizations

This folder contains code to generate visualizations from TPU DFlash benchmark results (TPU v4 and v5p). The visualizations help compare speculative decoding performance across hardware, benchmarks, and configurations.

---

## Quick Start

```bash
# From the project root (zvs/)
pip install pandas matplotlib seaborn
python visualizations/generate_visualizations.py
```

Generated PNG files will appear in this `visualizations/` folder.

---

## Data Sources

All data is read from the `results/` directory:

| Source | Path | Contents |
|--------|------|----------|
| **TPU v4** | `results/v4/` | Benchmark JSON and CSV files from TPU v4 runs |
| **TPU v5p** | `results/v5p/` | Benchmark JSON and CSV files from TPU v5p runs |

### Key Files Used

| File | Used For |
|------|----------|
| `standalone_all_benchmarks.csv` | Per-dataset speedup, tau, TPS (v4 and v5p) |
| `standalone_vs_v4.csv` | v5p vs v4 improvement (baseline and DFlash) |
| `standalone_vs_gpu_paper.csv` | TPU vs GPU paper reference (tau, speedup) |
| `standalone_acceptance_per_pos.csv` | Acceptance rate by position in draft block |
| `cross_comparison.csv` | Standalone vs vLLM pipeline vs GPU paper |
| `gpu_matmul_scaling.json` | GPU matmul scaling with context length K |
| `standalone_gsm8k.json` | Acceptance histogram (per-draft length distribution) |

---

## How the Code Works

The script `generate_visualizations.py`:

1. **Loads data** from `results/v4/` and `results/v5p/` using pandas (CSV) and json (JSON).
2. **Builds 10 plots** in sequence, each saved as a PNG.
3. **Writes outputs** to the same `visualizations/` folder.

Each plot function is self-contained: it loads its required files, filters/transforms data, and saves a figure. Missing files are handled gracefully (that plot is skipped).

---

## Visualizations and What They Show

### 1. `01_benchmark_speedup_v4_v5p.png`
**What:** Bar chart of DFlash speedup (×) by benchmark dataset for TPU v4 and v5p.  
**Why:** Compare how much faster DFlash is vs baseline autoregressive decoding on each benchmark across hardware generations.

### 2. `02_tau_comparison_v4_v5p.png`
**What:** Bar chart of tau (average accepted draft length) by dataset for v4 and v5p.  
**Why:** Tau reflects how many draft tokens are accepted per verification step. Higher tau means better speculative decoding efficiency.

### 3. `03_tps_comparison.png`
**What:** Grouped bars of baseline TPS vs DFlash TPS for v4 and v5p.  
**Why:** Shows raw throughput (tokens/second) before and after DFlash. Highlights the throughput gain from speculative decoding.

### 4. `04_v5p_vs_v4_improvement.png`
**What:** Bar chart of baseline and DFlash improvement (×) when moving from v4 to v5p.  
**Why:** Quantifies how much v5p improves both baseline and DFlash throughput compared to v4.

### 5. `05_acceptance_rate_by_position.png`
**What:** Line chart of draft acceptance rate vs position (0–15) in the draft block.  
**Why:** First positions are typically accepted more often; later positions drop off. This shows how acceptance degrades over the block and helps tune block size.

### 6. `06_tpu_vs_gpu_paper.png`
**What:** Grouped bars of tau and speedup for TPU v4, v5p, and GPU (paper reference).  
**Why:** Compares TPU performance to published GPU numbers from the DFlash paper.

### 7. `07_cross_comparison.png`
**What:** Bar chart of speedup for standalone TPU, vLLM pipeline TPU, and GPU paper.  
**Why:** Shows how standalone runs differ from vLLM pipeline runs and how both compare to GPU.

### 8. `08_gpu_matmul_scaling.png`
**What:** Line chart of matmul time ratio vs context length K for draft and target model.  
**Why:** Shows how compute scales with context length; useful for understanding performance at different K.

### 9. `09_speedup_by_category.png`
**What:** Bar chart of average speedup by task category (math, code, chat).  
**Why:** Reveals whether DFlash benefits more for certain task types (e.g., math vs chat).

### 10. `10_acceptance_histogram.png`
**What:** Bar chart of the fraction of drafts accepted at each length (0–16) for GSM8K.  
**Why:** Shows the distribution of accepted lengths; complements per-position acceptance rate.

---

## Why These Visualizations Matter

- **Hardware comparison:** v4 vs v5p and TPU vs GPU help choose hardware and set expectations.
- **Benchmark behavior:** Per-dataset speedup and tau show which benchmarks benefit most from DFlash.
- **Block tuning:** Acceptance rate by position and acceptance histogram guide block size and draft strategy.
- **Pipeline impact:** Cross comparison shows how different deployment modes (standalone vs vLLM) affect speedup.
- **Scaling:** GPU matmul scaling shows how context length affects performance.

---

## Requirements

- Python 3.8+
- pandas
- matplotlib
- seaborn

Install with: `pip install pandas matplotlib seaborn`

---

## Troubleshooting

- **Missing figures:** Ensure `results/v4/` and `results/v5p/` contain the expected CSV/JSON files. The script skips plots when data is missing.
- **Style warnings:** If seaborn styles are unavailable, the script falls back to default matplotlib styling.
- **Path issues:** Run from the project root (`zvs/`) so relative paths resolve correctly.

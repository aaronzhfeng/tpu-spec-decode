#!/usr/bin/env python3
"""Generate visualizations from TPU DFlash benchmark results (v4 and v5p).

Reads data from:
  - results/v4/  (TPU v4 benchmark JSON/CSV files)
  - results/v5p/ (TPU v5p benchmark JSON/CSV files)

Outputs PNG figures to: visualizations/

Usage:
    python visualizations/generate_visualizations.py

Requirements:
    pip install pandas matplotlib seaborn
"""

import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_V4 = PROJECT_ROOT / "results" / "v4"
RESULTS_V5P = PROJECT_ROOT / "results" / "v5p"
OUTPUT_DIR = SCRIPT_DIR

# Style (seaborn-v0_8-whitegrid for matplotlib 3.6+)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass
COLORS = {
    "v4": "#2ecc71",
    "v5p": "#3498db",
    "gpu": "#e74c3c",
    "math": "#9b59b6",
    "code": "#1abc9c",
    "chat": "#f39c12",
}


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV, return empty DataFrame if missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    """Load JSON, return empty dict if missing."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Visualization 1: Benchmark Speedup Comparison (v4 vs v5p)
# ---------------------------------------------------------------------------
def plot_benchmark_speedup():
    """Bar chart: Speedup by dataset for v4 and v5p."""
    v4_df = load_csv(RESULTS_V4 / "standalone_all_benchmarks.csv")
    v5p_df = load_csv(RESULTS_V5P / "standalone_all_benchmarks.csv")

    if v4_df.empty and v5p_df.empty:
        print("  [SKIP] No benchmark data for speedup plot")
        return

    # Filter out AVERAGE rows if present
    v4_df = v4_df[v4_df["dataset"] != "AVERAGE"] if not v4_df.empty else v4_df
    v5p_df = v5p_df[v5p_df["dataset"] != "AVERAGE"] if not v5p_df.empty else v5p_df

    datasets = v5p_df["dataset"].tolist() if not v5p_df.empty else v4_df["dataset"].tolist()
    if not datasets:
        return

    x = range(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    if not v4_df.empty:
        v4_speedup = [v4_df[v4_df["dataset"] == d]["tpu_speedup"].values[0] if d in v4_df["dataset"].values else 0 for d in datasets]
        ax.bar([i - width/2 for i in x], v4_speedup, width, label="TPU v4", color=COLORS["v4"])
    if not v5p_df.empty:
        v5p_speedup = [v5p_df[v5p_df["dataset"] == d]["tpu_speedup"].values[0] if d in v5p_df["dataset"].values else 0 for d in datasets]
        ax.bar([i + width/2 for i in x], v5p_speedup, width, label="TPU v5p", color=COLORS["v5p"])

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("DFlash Speedup: TPU v4 vs v5p by Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_benchmark_speedup_v4_v5p.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 01_benchmark_speedup_v4_v5p.png")


# ---------------------------------------------------------------------------
# Visualization 2: Tau (Acceptance Length) by Dataset
# ---------------------------------------------------------------------------
def plot_tau_comparison():
    """Bar chart: Tau (average accepted draft length) by dataset."""
    v4_df = load_csv(RESULTS_V4 / "standalone_all_benchmarks.csv")
    v5p_df = load_csv(RESULTS_V5P / "standalone_all_benchmarks.csv")

    if v4_df.empty and v5p_df.empty:
        return

    v4_df = v4_df[v4_df["dataset"] != "AVERAGE"] if not v4_df.empty else v4_df
    v5p_df = v5p_df[v5p_df["dataset"] != "AVERAGE"] if not v5p_df.empty else v5p_df

    datasets = v5p_df["dataset"].tolist() if not v5p_df.empty else v4_df["dataset"].tolist()
    x = range(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    if not v4_df.empty:
        v4_tau = [v4_df[v4_df["dataset"] == d]["tpu_tau"].values[0] if d in v4_df["dataset"].values else 0 for d in datasets]
        ax.bar([i - width/2 for i in x], v4_tau, width, label="TPU v4", color=COLORS["v4"])
    if not v5p_df.empty:
        v5p_tau = [v5p_df[v5p_df["dataset"] == d]["tpu_tau"].values[0] if d in v5p_df["dataset"].values else 0 for d in datasets]
        ax.bar([i + width/2 for i in x], v5p_tau, width, label="TPU v5p", color=COLORS["v5p"])

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Tau (avg accepted draft length)")
    ax.set_title("DFlash Tau: TPU v4 vs v5p by Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_tau_comparison_v4_v5p.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 02_tau_comparison_v4_v5p.png")


# ---------------------------------------------------------------------------
# Visualization 3: Tokens Per Second (TPS) by Dataset
# ---------------------------------------------------------------------------
def plot_tps_comparison():
    """Grouped bar: Baseline TPS vs DFlash TPS for v4 and v5p."""
    v4_df = load_csv(RESULTS_V4 / "standalone_all_benchmarks.csv")
    v5p_df = load_csv(RESULTS_V5P / "standalone_all_benchmarks.csv")

    if v4_df.empty and v5p_df.empty:
        return

    v4_df = v4_df[v4_df["dataset"] != "AVERAGE"] if not v4_df.empty else v4_df
    v5p_df = v5p_df[v5p_df["dataset"] != "AVERAGE"] if not v5p_df.empty else v5p_df

    datasets = v5p_df["dataset"].tolist() if not v5p_df.empty else v4_df["dataset"].tolist()
    x = range(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    if not v4_df.empty:
        v4_baseline = [v4_df[v4_df["dataset"] == d]["tpu_baseline_tps"].values[0] if d in v4_df["dataset"].values else 0 for d in datasets]
        v4_dflash = [v4_df[v4_df["dataset"] == d]["tpu_dflash_tps"].values[0] if d in v4_df["dataset"].values else 0 for d in datasets]
        ax.bar([i - 1.5*width for i in x], v4_baseline, width, label="v4 Baseline", color=COLORS["v4"], alpha=0.6)
        ax.bar([i - 0.5*width for i in x], v4_dflash, width, label="v4 DFlash", color=COLORS["v4"])
    if not v5p_df.empty:
        v5p_baseline = [v5p_df[v5p_df["dataset"] == d]["tpu_baseline_tps"].values[0] if d in v5p_df["dataset"].values else 0 for d in datasets]
        v5p_dflash = [v5p_df[v5p_df["dataset"] == d]["tpu_dflash_tps"].values[0] if d in v5p_df["dataset"].values else 0 for d in datasets]
        ax.bar([i + 0.5*width for i in x], v5p_baseline, width, label="v5p Baseline", color=COLORS["v5p"], alpha=0.6)
        ax.bar([i + 1.5*width for i in x], v5p_dflash, width, label="v5p DFlash", color=COLORS["v5p"])

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Tokens per Second")
    ax.set_title("Throughput: Baseline vs DFlash on TPU v4 and v5p")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend(ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_tps_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 03_tps_comparison.png")


# ---------------------------------------------------------------------------
# Visualization 4: v5p vs v4 Improvement
# ---------------------------------------------------------------------------
def plot_v5p_vs_v4_improvement():
    """Bar chart: Baseline and DFlash improvement from v4 to v5p."""
    df = load_csv(RESULTS_V5P / "standalone_vs_v4.csv")
    if df.empty:
        return

    datasets = df["dataset"].tolist()
    x = range(len(datasets))
    width = 0.35

    # Parse "1.69x" style strings
    def parse_improvement(s):
        if isinstance(s, str) and "x" in s:
            return float(s.replace("x", ""))
        return float(s) if s else 0

    baseline_imp = [parse_improvement(df[df["dataset"] == d]["baseline_improvement"].values[0]) if d in df["dataset"].values else 0 for d in datasets]
    dflash_imp = [parse_improvement(df[df["dataset"] == d]["dflash_improvement"].values[0]) if d in df["dataset"].values else 0 for d in datasets]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], baseline_imp, width, label="Baseline improvement", color=COLORS["v4"], alpha=0.8)
    ax.bar([i + width/2 for i in x], dflash_imp, width, label="DFlash improvement", color=COLORS["v5p"], alpha=0.8)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Improvement (×)")
    ax.set_title("v5p vs v4: Baseline and DFlash Throughput Improvement")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_v5p_vs_v4_improvement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 04_v5p_vs_v4_improvement.png")


# ---------------------------------------------------------------------------
# Visualization 5: Acceptance Rate by Position (from v4)
# ---------------------------------------------------------------------------
def plot_acceptance_rate_by_position():
    """Line chart: Draft acceptance rate vs position in block (0–15)."""
    df = load_csv(RESULTS_V4 / "standalone_acceptance_per_pos.csv")
    if df.empty:
        return

    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    positions = list(range(len(pos_cols)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        rates = [row[c] for c in pos_cols]
        ax.plot(positions, rates, marker="o", markersize=4, label=row["dataset"])

    ax.set_xlabel("Position in draft block (0–15)")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("DFlash Acceptance Rate by Position in Block (TPU v4)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_acceptance_rate_by_position.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 05_acceptance_rate_by_position.png")


# ---------------------------------------------------------------------------
# Visualization 6: TPU vs GPU Paper Comparison
# ---------------------------------------------------------------------------
def plot_tpu_vs_gpu():
    """Grouped bar: TPU tau/speedup vs GPU paper reference."""
    v4_df = load_csv(RESULTS_V4 / "standalone_vs_gpu_paper.csv")
    v5p_df = load_csv(RESULTS_V5P / "standalone_vs_gpu_paper.csv")

    if v4_df.empty and v5p_df.empty:
        return

    v4_df = v4_df[v4_df["dataset"] != "AVERAGE"] if not v4_df.empty else v4_df
    v5p_df = v5p_df[v5p_df["dataset"] != "AVERAGE"] if not v5p_df.empty else v5p_df

    datasets = ["gsm8k", "math500", "aime24", "aime25"]
    datasets = [d for d in datasets if d in (v4_df["dataset"].tolist() if not v4_df.empty else []) or d in (v5p_df["dataset"].tolist() if not v5p_df.empty else [])]
    if not datasets:
        return

    x = range(len(datasets))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Tau
    if not v4_df.empty:
        v4_tau = [v4_df[v4_df["dataset"] == d]["tpu_tau"].values[0] if d in v4_df["dataset"].values else 0 for d in datasets]
        ax1.bar([i - width for i in x], v4_tau, width, label="TPU v4", color=COLORS["v4"])
    if not v5p_df.empty:
        v5p_tau = [v5p_df[v5p_df["dataset"] == d]["v5p_tau"].values[0] if d in v5p_df["dataset"].values else 0 for d in datasets]
        ax1.bar([i for i in x], v5p_tau, width, label="TPU v5p", color=COLORS["v5p"])
    gpu_tau = [v4_df[v4_df["dataset"] == d]["gpu_paper_tau"].values[0] if not v4_df.empty and d in v4_df["dataset"].values else (v5p_df[v5p_df["dataset"] == d]["gpu_paper_tau"].values[0] if not v5p_df.empty and d in v5p_df["dataset"].values else 0) for d in datasets]
    ax1.bar([i + width for i in x], gpu_tau, width, label="GPU (paper)", color=COLORS["gpu"])
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Tau")
    ax1.set_title("Tau: TPU vs GPU Paper")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(bottom=0)

    # Speedup
    if not v4_df.empty:
        v4_spd = [v4_df[v4_df["dataset"] == d]["tpu_speedup"].values[0] if d in v4_df["dataset"].values else 0 for d in datasets]
        ax2.bar([i - width for i in x], v4_spd, width, label="TPU v4", color=COLORS["v4"])
    if not v5p_df.empty:
        v5p_spd = [v5p_df[v5p_df["dataset"] == d]["v5p_speedup"].values[0] if d in v5p_df["dataset"].values else 0 for d in datasets]
        ax2.bar([i for i in x], v5p_spd, width, label="TPU v5p", color=COLORS["v5p"])
    gpu_spd = [v4_df[v4_df["dataset"] == d]["gpu_paper_speedup"].values[0] if not v4_df.empty and d in v4_df["dataset"].values else (v5p_df[v5p_df["dataset"] == d]["gpu_paper_speedup"].values[0] if not v5p_df.empty and d in v5p_df["dataset"].values else 0) for d in datasets]
    ax2.bar([i + width for i in x], gpu_spd, width, label="GPU (paper)", color=COLORS["gpu"])
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("Speedup: TPU vs GPU Paper")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha="right")
    ax2.legend()
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_tpu_vs_gpu_paper.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 06_tpu_vs_gpu_paper.png")


# ---------------------------------------------------------------------------
# Visualization 7: Cross Comparison (Standalone vs vLLM vs GPU)
# ---------------------------------------------------------------------------
def plot_cross_comparison():
    """Bar chart: Standalone vs vLLM vs GPU paper (v4 only)."""
    df = load_csv(RESULTS_V4 / "cross_comparison.csv")
    if df.empty:
        return

    df = df[df["dataset"] != "AVERAGE"]
    datasets = df["dataset"].tolist()
    x = range(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width for i in x], df["standalone_speedup"], width, label="Standalone TPU", color=COLORS["v4"])
    ax.bar([i for i in x], df["vllm_speedup"], width, label="vLLM Pipeline TPU", color=COLORS["v5p"], alpha=0.8)
    ax.bar([i + width for i in x], df["gpu_paper_speedup"], width, label="GPU (paper)", color=COLORS["gpu"])

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Speedup Comparison: Standalone TPU vs vLLM Pipeline vs GPU Paper")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_cross_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 07_cross_comparison.png")


# ---------------------------------------------------------------------------
# Visualization 8: GPU Matmul Scaling (from v4 JSON)
# ---------------------------------------------------------------------------
def plot_gpu_matmul_scaling():
    """Line chart: GPU matmul time ratio vs K (context length)."""
    data = load_json(RESULTS_V4 / "gpu_matmul_scaling.json")
    if not data:
        return

    part1 = data.get("part1_dflash", [])
    part2 = data.get("part2_target", [])

    if not part1 and not part2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if part1:
        K = [p["K"] for p in part1]
        ratio = [p["ratio"] for p in part1]
        ax1.plot(K, ratio, marker="o", color=COLORS["v4"], linewidth=2)
        ax1.set_xlabel("K (context length)")
        ax1.set_ylabel("Time ratio (vs K=16)")
        ax1.set_title("Draft model (part1) scaling")
        ax1.set_xticks(K)
        ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylim(bottom=0.9)

    if part2:
        K = [p["K"] for p in part2]
        ratio = [p["ratio"] for p in part2]
        ax2.plot(K, ratio, marker="s", color=COLORS["v5p"], linewidth=2)
        ax2.set_xlabel("K (context length)")
        ax2.set_ylabel("Time ratio (vs K=16)")
        ax2.set_title("Target model (part2) scaling")
        ax2.set_xticks(K)
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylim(bottom=0.9)

    fig.suptitle("GPU Matmul Scaling: Draft vs Target Model (from v4 profiling)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_gpu_matmul_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 08_gpu_matmul_scaling.png")


# ---------------------------------------------------------------------------
# Visualization 9: Speedup by Category (math, code, chat)
# ---------------------------------------------------------------------------
def plot_speedup_by_category():
    """Bar chart: Average speedup by task category."""
    v5p_df = load_csv(RESULTS_V5P / "standalone_all_benchmarks.csv")
    if v5p_df.empty:
        v5p_df = load_csv(RESULTS_V4 / "standalone_all_benchmarks.csv")
    if v5p_df.empty:
        return

    v5p_df = v5p_df[v5p_df["dataset"] != "AVERAGE"]
    if "category" not in v5p_df.columns:
        v5p_df["category"] = "other"

    cat_avg = v5p_df.groupby("category")["tpu_speedup"].mean().sort_values(ascending=False)
    colors = [COLORS.get(c, "#95a5a6") for c in cat_avg.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cat_avg.index, cat_avg.values, color=colors)
    ax.set_xlabel("Task category")
    ax.set_ylabel("Average speedup (×)")
    ax.set_title("DFlash Speedup by Task Category")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_speedup_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 09_speedup_by_category.png")


# ---------------------------------------------------------------------------
# Visualization 10: Acceptance Histogram (from JSON)
# ---------------------------------------------------------------------------
def plot_acceptance_histogram():
    """Bar chart: Distribution of acceptance lengths from GSM8K JSON."""
    v4_data = load_json(RESULTS_V4 / "standalone_gsm8k.json")
    v5p_data = load_json(RESULTS_V5P / "standalone_gsm8k.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (data, title) in enumerate([(v4_data, "TPU v4"), (v5p_data, "TPU v5p")]):
        if not data or "summary" not in data:
            axes[idx].text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        hist = data["summary"].get("acceptance_histogram", [])
        if not hist:
            continue
        positions = list(range(len(hist)))
        axes[idx].bar(positions, hist, color=COLORS["v4"] if idx == 0 else COLORS["v5p"], alpha=0.8)
        axes[idx].set_xlabel("Accepted length (tokens)")
        axes[idx].set_ylabel("Fraction of drafts")
        axes[idx].set_title(f"Acceptance length distribution — {title} (GSM8K)")
        axes[idx].set_xticks(positions)

    fig.suptitle("DFlash Acceptance Length Histogram")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_acceptance_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  wrote 10_acceptance_histogram.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Generating visualizations from results/v4 and results/v5p")
    print("=" * 60)
    ensure_output_dir()
    print()

    plot_benchmark_speedup()
    plot_tau_comparison()
    plot_tps_comparison()
    plot_v5p_vs_v4_improvement()
    plot_acceptance_rate_by_position()
    plot_tpu_vs_gpu()
    plot_cross_comparison()
    plot_gpu_matmul_scaling()
    plot_speedup_by_category()
    plot_acceptance_histogram()

    print()
    print("Done. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""
Generate all visualizations from TPU v5p (primary) and v4 (secondary)
DFlash speculative decoding benchmark results.

Usage:
    python generate_visualizations.py

Outputs PNG files into the same directory as this script.
"""

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
V4_DIR = RESULTS_DIR / "v4"
V5P_DIR = RESULTS_DIR / "v5p"
OUT_DIR = SCRIPT_DIR

CATEGORY_COLORS = {"math": "#4C72B0", "code": "#55A868", "chat": "#C44E52"}
CATEGORY_ORDER = ["math", "code", "chat"]

BENCHMARK_JSONS = [
    "standalone_gsm8k.json",
    "standalone_math500.json",
    "standalone_aime24.json",
    "standalone_aime25.json",
    "standalone_humaneval.json",
    "standalone_mbpp.json",
    "standalone_mt-bench.json",
    "standalone_alpaca.json",
    "standalone_swe-bench.json",
]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _bar_label(ax, bars, fmt="{:.2f}", fontsize=8):
    for bar in bars:
        h = bar.get_height()
        if h > 0 and not np.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.04 * ax.get_ylim()[1],
                fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize,
            )


# ══════════════════════════════════════════════
# PRIMARY FIGURES (1–8): V5p-focused
# ══════════════════════════════════════════════


def fig1_v5p_speedup():
    """V5p DFlash speedup across all 9 benchmarks."""
    rows = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")

    datasets = [r["dataset"] for r in rows]
    categories = [r["category"] for r in rows]
    speedups = [float(r["tpu_speedup"]) for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = [CATEGORY_COLORS[c] for c in categories]
    bars = ax.bar(datasets, speedups, color=colors, edgecolor="white", width=0.6)
    _bar_label(ax, bars, fontsize=10)

    for i, cat in enumerate(categories):
        ax.text(i, -0.35, cat, ha="center", fontsize=8,
                color=CATEGORY_COLORS[cat], fontweight="bold")

    ax.set_ylabel("Speedup over autoregressive baseline")
    ax.set_title("DFlash Speculative Decoding Speedup (TPU V5P)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.grid(axis="y", alpha=0.2)

    avg = np.mean(speedups)
    ax.axhline(avg, color="#DD8452", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.text(len(datasets) - 0.5, avg + 0.1, f"avg {avg:.2f}x",
            fontsize=9, color="#DD8452", fontweight="bold", ha="right")

    fig.savefig(OUT_DIR / "fig1_v5p_speedup.png")
    plt.close(fig)
    print("  [1/12] fig1_v5p_speedup.png")


def fig2_v5p_throughput():
    """V5p baseline vs DFlash throughput (TPS) across all 9 benchmarks."""
    rows = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")
    datasets = [r["dataset"] for r in rows]
    categories = [r["category"] for r in rows]

    base_tps = [float(r["tpu_baseline_tps"]) for r in rows]
    df_tps = [float(r["tpu_dflash_tps"]) for r in rows]

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5))

    b1 = ax.bar(x - w / 2, base_tps, w, label="Baseline", color="#A6CEE3", edgecolor="white")
    b2 = ax.bar(x + w / 2, df_tps, w, label="DFlash", color="#DD8452", edgecolor="white")

    for bar, val in zip(b2, df_tps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{val:.0f}", ha="center", fontsize=8, fontweight="bold", color="#DD8452")

    ax.set_ylabel("Tokens per Second (TPS)")
    ax.set_title("Throughput: Baseline vs DFlash (TPU V5P)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(base_tps), max(df_tps)) * 1.2)
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(OUT_DIR / "fig2_v5p_throughput.png")
    plt.close(fig)
    print("  [2/12] fig2_v5p_throughput.png")


def fig3_v5p_category_summary():
    """V5p performance by task category (math / code / chat)."""
    rows = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")

    cats = {"math": [], "code": [], "chat": []}
    for r in rows:
        cats[r["category"]].append(r)

    cat_names = ["Math", "Code", "Chat"]
    cat_keys = ["math", "code", "chat"]
    avg_speedup, avg_tau, avg_tps = [], [], []
    for ck in cat_keys:
        rr = cats[ck]
        avg_speedup.append(np.mean([float(r["tpu_speedup"]) for r in rr]))
        avg_tau.append(np.mean([float(r["tpu_tau"]) for r in rr]))
        avg_tps.append(np.mean([float(r["tpu_dflash_tps"]) for r in rr]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = [CATEGORY_COLORS[c] for c in cat_keys]

    titles = ["Avg Speedup", "Avg Tau", "Avg DFlash TPS"]
    data = [avg_speedup, avg_tau, avg_tps]
    ylabels = ["Speedup (x)", "Tau (tokens)", "Tokens / sec"]

    for ax, title, vals, ylabel in zip(axes, titles, data, ylabels):
        bars = ax.bar(cat_names, vals, color=colors, edgecolor="white", width=0.55)
        _bar_label(ax, bars, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("V5P DFlash Performance by Task Category",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_v5p_category_summary.png")
    plt.close(fig)
    print("  [3/12] fig3_v5p_category_summary.png")


def fig4_v5p_vs_gpu():
    """V5p vs GPU A100 — tau parity and speedup gap (math benchmarks)."""
    rows = _read_csv(V5P_DIR / "standalone_vs_gpu_paper.csv")
    data_rows = [r for r in rows if r["dataset"] != "AVERAGE"]
    avg_candidates = [r for r in rows if r["dataset"] == "AVERAGE"]
    avg_row = avg_candidates[0] if avg_candidates else None

    datasets = [r["dataset"] for r in data_rows]
    v5_tau = [float(r["v5p_tau"]) for r in data_rows]
    gpu_tau = [float(r["gpu_paper_tau"]) for r in data_rows]
    tau_pct = [float(r["tau_pct_of_gpu"]) for r in data_rows]
    v5_speedup = [float(r["v5p_speedup"]) for r in data_rows]
    gpu_speedup = [float(r["gpu_paper_speedup"]) for r in data_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(datasets))
    w = 0.35

    b1 = ax1.bar(x - w / 2, v5_tau, w, label="TPU V5P", color="#DD8452")
    b2 = ax1.bar(x + w / 2, gpu_tau, w, label="GPU A100 (paper)", color="#55A868")
    for i, pct in enumerate(tau_pct):
        color = "#3fb950" if pct >= 100 else ("#d29922" if pct >= 90 else "#f85149")
        ax1.text(i, max(v5_tau[i], gpu_tau[i]) + 0.35, f"{pct:.0f}%",
                 ha="center", fontsize=10, fontweight="bold", color=color)
    ax1.set_ylabel("Tau (avg accepted tokens)")
    ax1.set_title("Draft Quality Parity (Tau)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.set_ylim(0, max(max(v5_tau), max(gpu_tau)) * 1.35)
    avg_v5_tau = float(avg_row["v5p_tau"]) if avg_row else np.mean(v5_tau)
    ax1.axhline(avg_v5_tau, color="#DD8452", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.grid(axis="y", alpha=0.2)

    b3 = ax2.bar(x - w / 2, v5_speedup, w, label="TPU V5P", color="#DD8452")
    b4 = ax2.bar(x + w / 2, gpu_speedup, w, label="GPU A100 (paper)", color="#55A868")
    _bar_label(ax2, b3)
    _bar_label(ax2, b4)
    ax2.set_ylabel("Speedup over baseline")
    ax2.set_title("End-to-End Speedup", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.set_ylim(0, max(max(v5_speedup), max(gpu_speedup)) * 1.3)
    ax2.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle("TPU V5P vs GPU A100 — Math Benchmarks",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_v5p_vs_gpu.png")
    plt.close(fig)
    print("  [4/12] fig4_v5p_vs_gpu.png")


def fig5_v5p_acceptance_decay():
    """V5p acceptance rate decay by draft token position."""
    positions = list(range(16))
    fig, ax = plt.subplots(figsize=(10, 5.5))

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.85, len(BENCHMARK_JSONS)))

    for i, jf in enumerate(BENCHMARK_JSONS):
        data = _read_json(V5P_DIR / jf)
        ds = data["config"]["dataset"]
        all_rates = data["summary"]["acceptance_rate_per_pos"]
        rates = all_rates[:16]
        pos = positions[:len(rates)]
        tau = data["summary"]["tau"]
        ax.plot(pos, rates, marker="o", markersize=4,
                label=f"{ds} (\u03c4={tau:.1f})", color=colors[i])

    ax.set_xlabel("Draft Token Position")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Acceptance Rate Decay by Draft Position (TPU V5P)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(positions)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT_DIR / "fig5_v5p_acceptance_decay.png")
    plt.close(fig)
    print("  [5/12] fig5_v5p_acceptance_decay.png")


def fig6_v5p_latency():
    """V5p latency (TPOT) — baseline vs DFlash across all 9 benchmarks."""
    rows = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")
    datasets = [r["dataset"] for r in rows]
    categories = [r["category"] for r in rows]

    base_tpot = [float(r["tpu_baseline_tpot_ms"]) for r in rows]
    df_tpot = [float(r["tpu_dflash_tpot_ms"]) for r in rows]

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.bar(x - w / 2, base_tpot, w, label="Baseline", color="#A6CEE3", edgecolor="white")
    b_df = ax.bar(x + w / 2, df_tpot, w, label="DFlash", color="#DD8452", edgecolor="white")

    for bar, val in zip(b_df, df_tpot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f"{val:.2f}", ha="center", fontsize=8, fontweight="bold", color="#DD8452")

    ax.set_ylabel("Time per Output Token (ms) \u2014 lower is better")
    ax.set_title("Latency (TPOT): Baseline vs DFlash (TPU V5P)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(loc="upper right")
    y_max = max(max(base_tpot), max(df_tpot))
    ax.set_ylim(0, y_max * 1.2)
    ax.grid(axis="y", alpha=0.2)

    best_idx = df_tpot.index(min(df_tpot))
    n = len(datasets)
    text_x = best_idx + 2 if best_idx < n - 3 else best_idx - 2
    ax.annotate(f"best: {min(df_tpot):.2f} ms ({datasets[best_idx]})",
                xy=(best_idx + w / 2, min(df_tpot)),
                xytext=(text_x, y_max * 0.5),
                fontsize=9, color="#DD8452", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.2))

    fig.savefig(OUT_DIR / "fig6_v5p_latency.png")
    plt.close(fig)
    print("  [6/12] fig6_v5p_latency.png")


def fig7_v5p_output_quality():
    """V5p exact token match rates between DFlash and baseline."""
    rows = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")
    datasets = [r["dataset"] for r in rows]
    categories = [r["category"] for r in rows]

    match_rates = [float(r["match_rate"].replace("%", "")) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [CATEGORY_COLORS[c] for c in categories]
    bars = ax.bar(datasets, match_rates, color=colors, edgecolor="white", width=0.6)

    for bar, rate, r in zip(bars, match_rates, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{r['exact_match']}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Exact Token Match Rate (%)")
    ax.set_title("DFlash Output Quality \u2014 Exact Token Match vs Baseline (V5P)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CATEGORY_COLORS[c], label=c.capitalize())
                       for c in CATEGORY_ORDER]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.text(0.5, -0.18,
            "Mismatches are bf16 floating-point divergence "
            "(batch-16 verify vs single-token), not correctness errors",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#8b949e", style="italic")

    fig.savefig(OUT_DIR / "fig7_v5p_output_quality.png")
    plt.close(fig)
    print("  [7/12] fig7_v5p_output_quality.png")


def fig8_summary_dashboard():
    """Dark-themed summary dashboard with headline V5p metrics."""
    v5p = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")
    gpu_rows = _read_csv(V5P_DIR / "standalone_vs_gpu_paper.csv")
    gpu_data = [r for r in gpu_rows if r["dataset"] != "AVERAGE"]
    avg_candidates = [r for r in gpu_rows if r["dataset"] == "AVERAGE"]

    speedups = [float(r["tpu_speedup"]) for r in v5p]
    tps_vals = [float(r["tpu_dflash_tps"]) for r in v5p]
    datasets = [r["dataset"] for r in v5p]

    avg_speedup = np.mean(speedups)
    peak_speedup = max(speedups)
    peak_speedup_ds = datasets[speedups.index(peak_speedup)]
    peak_tps = max(tps_vals)
    peak_tps_ds = datasets[tps_vals.index(peak_tps)]
    if avg_candidates:
        tau_pct = float(avg_candidates[0]["tau_pct_of_gpu"])
    else:
        tau_pct = np.mean([float(r["tau_pct_of_gpu"]) for r in gpu_data])

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")

    metrics = [
        ("Avg Speedup", f"{avg_speedup:.2f}x", "across 9 benchmarks", "#58a6ff"),
        ("Peak Speedup", f"{peak_speedup:.2f}x", f"on {peak_speedup_ds}", "#7ee787"),
        ("Peak Throughput", f"{peak_tps:.0f}", f"TPS on {peak_tps_ds}", "#d29922"),
        ("GPU Tau Parity", f"{tau_pct:.1f}%", "math avg (vs A100)", "#f778ba"),
    ]

    for i, (label, value, subtitle, color) in enumerate(metrics):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.set_facecolor("#0d1117")
        ax.text(0.5, 0.6, value, transform=ax.transAxes, fontsize=36,
                fontweight="bold", ha="center", va="center", color=color,
                fontfamily="monospace")
        ax.text(0.5, 0.25, label, transform=ax.transAxes, fontsize=13,
                fontweight="bold", ha="center", va="center", color="#c9d1d9")
        ax.text(0.5, 0.08, subtitle, transform=ax.transAxes, fontsize=10,
                ha="center", va="center", color="#8b949e", style="italic")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax_bar = fig.add_subplot(2, 1, 2)
    ax_bar.set_facecolor("#161b22")
    colors_bar = [CATEGORY_COLORS[r["category"]] for r in v5p]
    bars = ax_bar.barh(datasets[::-1], speedups[::-1],
                       color=colors_bar[::-1], edgecolor="#30363d", height=0.6)
    for bar, s in zip(bars, speedups[::-1]):
        ax_bar.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                    f"{s:.2f}x", va="center", fontsize=10, color="#c9d1d9",
                    fontweight="bold")
    ax_bar.set_xlabel("Speedup", color="#c9d1d9", fontsize=12)
    ax_bar.set_title("DFlash Speedup by Dataset (TPU V5P)", color="#58a6ff",
                     fontsize=13, fontweight="bold", pad=10)
    ax_bar.axvline(1, color="#30363d", linewidth=1, linestyle="--")
    ax_bar.axvline(avg_speedup, color="#58a6ff", linewidth=1.2, linestyle="--", alpha=0.7)
    from matplotlib.transforms import blended_transform_factory
    ax_bar.text(avg_speedup + 0.05, -0.02, f"avg {avg_speedup:.2f}x",
                fontsize=9, color="#58a6ff", style="italic",
                transform=blended_transform_factory(ax_bar.transData, ax_bar.transAxes),
                va="top")
    ax_bar.tick_params(colors="#c9d1d9")
    ax_bar.set_xlim(0, max(speedups) * 1.25)
    for spine in ax_bar.spines.values():
        spine.set_color("#30363d")
    ax_bar.grid(axis="x", alpha=0.15, color="#c9d1d9")

    fig.suptitle("DFlash on TPU V5P \u2014 Final Results Summary",
                 fontsize=18, fontweight="bold", color="#c9d1d9", y=0.98,
                 fontfamily="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "fig8_summary_dashboard.png", facecolor="#0d1117")
    plt.close(fig)
    print("  [8/12] fig8_summary_dashboard.png")


# ══════════════════════════════════════════════
# SECONDARY FIGURES (9–12): V4 & comparison
# ══════════════════════════════════════════════


def fig9_v5p_vs_v4_speedup():
    """Side-by-side speedup comparison between V4 and V5p."""
    v4 = _read_csv(V4_DIR / "standalone_all_benchmarks.csv")
    v5p = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")

    datasets = [r["dataset"] for r in v5p]
    categories = [r["category"] for r in v5p]
    v4_by_ds = {r["dataset"]: r for r in v4}
    v4_speedup = [float(v4_by_ds[d]["tpu_speedup"]) if d in v4_by_ds else np.nan
                  for d in datasets]
    v5p_speedup = [float(r["tpu_speedup"]) for r in v5p]

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - w / 2, v4_speedup, w, label="TPU V4", color="#4C72B0", edgecolor="white")
    b2 = ax.bar(x + w / 2, v5p_speedup, w, label="TPU V5P", color="#DD8452", edgecolor="white")
    _bar_label(ax, b1)
    _bar_label(ax, b2)

    for i, cat in enumerate(categories):
        ax.text(i, -0.45, cat, ha="center", fontsize=8,
                color=CATEGORY_COLORS[cat], fontweight="bold")

    ax.set_ylabel("Speedup over autoregressive baseline")
    ax.set_title("DFlash Speedup: TPU V4 vs V5P", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax.legend()
    all_vals = [v for v in v4_speedup + v5p_speedup if not np.isnan(v)]
    ax.set_ylim(0, max(all_vals) * 1.25)
    fig.savefig(OUT_DIR / "fig9_v5p_vs_v4_speedup.png")
    plt.close(fig)
    print("  [9/12] fig9_v5p_vs_v4_speedup.png")


def fig10_v5p_over_v4_improvement():
    """V5p-over-V4 improvement factors for baseline and DFlash."""
    rows = _read_csv(V5P_DIR / "standalone_vs_v4.csv")
    datasets = [r["dataset"] for r in rows]
    baseline_imp = [float(r["baseline_improvement"].replace("x", "")) for r in rows]
    dflash_imp = [float(r["dflash_improvement"].replace("x", "")) for r in rows]

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w / 2, baseline_imp, w, label="Baseline Improvement",
                color="#A6CEE3", edgecolor="white")
    b2 = ax.bar(x + w / 2, dflash_imp, w, label="DFlash Improvement",
                color="#DD8452", edgecolor="white")
    _bar_label(ax, b1)
    _bar_label(ax, b2)

    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Improvement Factor (V5P / V4)")
    ax.set_title("TPU V5P over V4: Baseline and DFlash Latency Improvement",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(baseline_imp), max(dflash_imp)) * 1.25)
    fig.savefig(OUT_DIR / "fig10_v5p_over_v4_improvement.png")
    plt.close(fig)
    print("  [10/12] fig10_v5p_over_v4_improvement.png")


def fig11_profiling_breakdown():
    """V4 profiling: step time breakdown (pie + bar). Explains the bottleneck."""
    prof = _read_json(V4_DIR / "profiling_gsm8k.json")
    s = prof["summary"]

    components = [
        ("Draft Forward", s["draft_forward"]["mean_ms"]),
        ("Draft Sample", s["draft_sample"]["mean_ms"]),
        ("Verify Forward", s["verify_forward"]["mean_ms"]),
        ("Acceptance", s["acceptance"]["mean_ms"]),
        ("Host\u2194Device Xfer", s["host_device_xfer"]["mean_ms"]),
        ("Aux Projection", s["aux_projection"]["mean_ms"]),
        ("Ctx Update", s["ctx_update"]["mean_ms"]),
        ("Cache Mgmt", s["cache_mgmt"]["mean_ms"]),
    ]
    labels, values = zip(*components)
    total = sum(values)

    cmap = plt.cm.Set2
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.5))

    wedges, texts, autotexts = ax_pie.pie(
        values, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        colors=colors, startangle=140, pctdistance=0.8,
    )
    ax_pie.legend(wedges, labels, loc="center left",
                  bbox_to_anchor=(-0.35, 0.5), fontsize=9)
    ax_pie.set_title("Proportion of Step Time")

    bars = ax_bar.barh(labels[::-1], values[::-1],
                       color=colors[::-1], edgecolor="white")
    ax_bar.set_xlabel("Mean Latency (ms)")
    ax_bar.set_title(f"Per-Component Latency (total step \u2248 {total:.1f} ms)")
    for bar, val in zip(bars, values[::-1]):
        ax_bar.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f} ms", va="center", fontsize=9)

    fig.suptitle("DFlash Step Profiling Breakdown \u2014 GSM8K on TPU V4",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig11_profiling_breakdown.png")
    plt.close(fig)
    print("  [11/12] fig11_profiling_breakdown.png")


def fig12_method_comparison():
    """DFlash vs Eagle3 vs vLLM pipeline (V4, math benchmarks)."""
    vllm = _read_csv(V4_DIR / "vllm_pipeline_results.csv")
    eagle = _read_csv(V4_DIR / "eagle3_llama_results.csv")
    v4 = _read_csv(V4_DIR / "standalone_all_benchmarks.csv")

    math_datasets = ["gsm8k", "math500", "aime24", "aime25"]

    standalone_speedup = {r["dataset"]: float(r["tpu_speedup"])
                          for r in v4 if r["dataset"] in math_datasets}
    vllm_speedup = {r["dataset"]: float(r["speedup"])
                    for r in vllm
                    if r["method"] == "dflash" and r["dataset"] in math_datasets}
    eagle_speedup = {r["dataset"]: float(r["speedup"])
                     for r in eagle
                     if r["method"] == "eagle3" and r["dataset"] in math_datasets}

    datasets = [d for d in math_datasets if d in standalone_speedup]
    s_vals = [standalone_speedup[d] for d in datasets]
    v_vals = [vllm_speedup.get(d, np.nan) for d in datasets]
    e_vals = [eagle_speedup.get(d, np.nan) for d in datasets]

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w, s_vals, w, label="DFlash Standalone", color="#4C72B0")
    b2 = ax.bar(x, v_vals, w, label="DFlash vLLM Pipeline", color="#C44E52")
    b3 = ax.bar(x + w, e_vals, w, label="Eagle3 (Llama)", color="#55A868")
    _bar_label(ax, b1)
    _bar_label(ax, b2)
    _bar_label(ax, b3)

    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Speedup over Baseline")
    ax.set_title("Speculative Decoding Methods Compared (Math Benchmarks, TPU V4)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    all_vals = [v for v in s_vals + v_vals + e_vals if not np.isnan(v)]
    ax.set_ylim(0, max(all_vals) * 1.3)
    fig.savefig(OUT_DIR / "fig12_method_comparison.png")
    plt.close(fig)
    print("  [12/12] fig12_method_comparison.png")


def fig13_cost_efficiency():
    """Cost per million tokens: V5p vs V4 vs GPU A100 estimate."""
    COST_V5P = 2.10
    COST_V4 = 3.22
    COST_A100 = 5.07
    GPU_BASELINE_TPS = 100

    v5p = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")
    v4 = _read_csv(V4_DIR / "standalone_all_benchmarks.csv")
    gpu_rows = _read_csv(V5P_DIR / "standalone_vs_gpu_paper.csv")
    gpu_by_ds = {r["dataset"]: r for r in gpu_rows if r["dataset"] != "AVERAGE"}

    datasets = [r["dataset"] for r in v5p]
    categories = [r["category"] for r in v5p]

    v5p_tps = [float(r["tpu_dflash_tps"]) for r in v5p]
    v4_by_ds = {r["dataset"]: r for r in v4}
    v4_tps = [float(v4_by_ds[d]["tpu_dflash_tps"]) if d in v4_by_ds else np.nan
              for d in datasets]

    gpu_tps = [GPU_BASELINE_TPS * float(gpu_by_ds[d]["gpu_paper_speedup"])
               if d in gpu_by_ds else np.nan
               for d in datasets]

    v5p_cost = [COST_V5P / (tps * 3600) * 1e6 for tps in v5p_tps]
    v4_cost = [COST_V4 / (tps * 3600) * 1e6 if not np.isnan(tps) else np.nan
               for tps in v4_tps]
    gpu_cost = [COST_A100 / (tps * 3600) * 1e6 if not np.isnan(tps) else np.nan
                for tps in gpu_tps]

    fig, (ax_rate, ax_token) = plt.subplots(1, 2, figsize=(16, 6),
                                            gridspec_kw={"width_ratios": [1, 2.6]})

    # --- Left panel: hourly chip/GPU rate ---
    hw = ["TPU V5P", "TPU V4", "GPU A100"]
    rates = [COST_V5P, COST_V4, COST_A100]
    hw_colors = ["#DD8452", "#4C72B0", "#55A868"]
    bars_hw = ax_rate.bar(hw, rates, color=hw_colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars_hw, rates):
        ax_rate.text(bar.get_x() + bar.get_width() / 2, val + 0.12,
                     f"${val:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax_rate.set_ylabel("On-Demand Cost ($/hour)")
    ax_rate.set_title("Chip / GPU Hourly Rate\n(GCP on-demand)", fontsize=12,
                      fontweight="bold")
    ax_rate.set_ylim(0, max(rates) * 1.3)
    ax_rate.grid(axis="y", alpha=0.2)

    # --- Right panel: $/M tokens per benchmark ---
    x = np.arange(len(datasets))
    w = 0.25
    b1 = ax_token.bar(x - w, v5p_cost, w, label="TPU V5P  ($2.10/hr)",
                      color="#DD8452", edgecolor="white")
    b2 = ax_token.bar(x, v4_cost, w, label="TPU V4  ($3.22/hr)",
                      color="#4C72B0", edgecolor="white")
    b3 = ax_token.bar(x + w, gpu_cost, w, label="GPU A100  ($5.07/hr, est.)",
                      color="#55A868", edgecolor="white", hatch="//", alpha=0.85)
    _bar_label(ax_token, b1, fmt="${:.2f}")
    _bar_label(ax_token, b2, fmt="${:.2f}")
    _bar_label(ax_token, b3, fmt="${:.2f}")

    for i, cat in enumerate(categories):
        ax_token.text(i, -0.25, cat, ha="center", fontsize=8,
                      color=CATEGORY_COLORS[cat], fontweight="bold")

    ax_token.set_ylabel("Cost per Million Tokens ($)")
    ax_token.set_title("DFlash Cost Efficiency by Benchmark", fontsize=12,
                       fontweight="bold")
    ax_token.set_xticks(x)
    ax_token.set_xticklabels(datasets, rotation=30, ha="right")
    ax_token.legend(loc="upper right", fontsize=9)
    all_costs = [c for c in v5p_cost + v4_cost + gpu_cost if not np.isnan(c)]
    ax_token.set_ylim(0, max(all_costs) * 1.25)
    ax_token.grid(axis="y", alpha=0.2)

    avg_v5p = np.mean(v5p_cost)
    ax_token.axhline(avg_v5p, color="#DD8452", linewidth=1.2, linestyle="--", alpha=0.6)
    ax_token.text(len(datasets) - 0.5, avg_v5p + 0.05,
                  f"V5P avg ${avg_v5p:.2f}/M",
                  fontsize=9, color="#DD8452", fontweight="bold", ha="right")

    ax_token.text(0.5, -0.22,
                  f"GPU A100 estimates assume ~{GPU_BASELINE_TPS} TPS autoregressive "
                  "baseline for Qwen3-4B (bf16, single-stream).  "
                  "Hatched bars = math benchmarks only (GPU paper data).",
                  transform=ax_token.transAxes, ha="center", fontsize=8,
                  color="#8b949e", style="italic")

    fig.suptitle("Cost Analysis — TPU V5P vs V4 vs GPU A100  (GCP On-Demand Pricing)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig13_cost_efficiency.png")
    plt.close(fig)
    print("  [13/13] fig13_cost_efficiency.png")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reading results from: {RESULTS_DIR}")
    print(f"Writing figures to:   {OUT_DIR}\n")

    print("=== Primary: V5P Results ===")
    fig1_v5p_speedup()
    fig2_v5p_throughput()
    fig3_v5p_category_summary()
    fig4_v5p_vs_gpu()
    fig5_v5p_acceptance_decay()
    fig6_v5p_latency()
    fig7_v5p_output_quality()
    fig8_summary_dashboard()

    print("\n=== Secondary: V4 & Comparison ===")
    fig9_v5p_vs_v4_speedup()
    fig10_v5p_over_v4_improvement()
    fig11_profiling_breakdown()
    fig12_method_comparison()
    fig13_cost_efficiency()

    print("\nAll 13 figures generated successfully.")

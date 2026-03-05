"""
Generate all visualizations from TPU v4 and v5p DFlash speculative decoding results.

Usage:
    python generate_visualizations.py

Outputs PNG files into the same directory as this script.
"""

import json
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
V4_DIR = RESULTS_DIR / "v4"
V5P_DIR = RESULTS_DIR / "v5p"
OUT_DIR = SCRIPT_DIR

CATEGORY_COLORS = {"math": "#4C72B0", "code": "#55A868", "chat": "#C44E52"}
CATEGORY_ORDER = ["math", "code", "chat"]

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


# ──────────────────────────────────────────────
# Figure 1 – DFlash Speedup: v4 vs v5p
# ──────────────────────────────────────────────
def fig1_speedup_v4_vs_v5p():
    v4 = _read_csv(V4_DIR / "standalone_all_benchmarks.csv")
    v5p = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")

    datasets = [r["dataset"] for r in v4]
    categories = [r["category"] for r in v4]
    v4_speedup = [float(r["tpu_speedup"]) for r in v4]
    v5p_speedup = [float(r["tpu_speedup"]) for r in v5p]

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - w / 2, v4_speedup, w, label="TPU v4", color="#4C72B0", edgecolor="white")
    b2 = ax.bar(x + w / 2, v5p_speedup, w, label="TPU v5p", color="#DD8452", edgecolor="white")
    _bar_label(ax, b1)
    _bar_label(ax, b2)

    for i, cat in enumerate(categories):
        ax.text(i, -0.45, cat, ha="center", fontsize=8, color=CATEGORY_COLORS[cat], fontweight="bold")

    ax.set_ylabel("Speedup over autoregressive baseline")
    ax.set_title("DFlash Speculative Decoding Speedup: TPU v4 vs v5p")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.set_ylim(0, max(max(v4_speedup), max(v5p_speedup)) * 1.25)
    fig.savefig(OUT_DIR / "fig1_speedup_v4_vs_v5p.png")
    plt.close(fig)
    print("  [1/10] fig1_speedup_v4_vs_v5p.png")


# ──────────────────────────────────────────────
# Figure 2 – Absolute Throughput (TPS)
# ──────────────────────────────────────────────
def fig2_throughput():
    v4 = _read_csv(V4_DIR / "standalone_all_benchmarks.csv")
    v5p = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")
    datasets = [r["dataset"] for r in v4]

    v4_base = [float(r["tpu_baseline_tps"]) for r in v4]
    v4_df = [float(r["tpu_dflash_tps"]) for r in v4]
    v5_base = [float(r["tpu_baseline_tps"]) for r in v5p]
    v5_df = [float(r["tpu_dflash_tps"]) for r in v5p]

    x = np.arange(len(datasets))
    w = 0.2
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - 1.5 * w, v4_base, w, label="v4 Baseline", color="#A6CEE3")
    ax.bar(x - 0.5 * w, v4_df, w, label="v4 DFlash", color="#4C72B0")
    ax.bar(x + 0.5 * w, v5_base, w, label="v5p Baseline", color="#FDBF6F")
    ax.bar(x + 1.5 * w, v5_df, w, label="v5p DFlash", color="#DD8452")

    ax.set_ylabel("Tokens per Second (TPS)")
    ax.set_title("Throughput: Baseline vs DFlash on TPU v4 and v5p")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(v5_df), max(v4_df)) * 1.15)
    fig.savefig(OUT_DIR / "fig2_throughput.png")
    plt.close(fig)
    print("  [2/10] fig2_throughput.png")


# ──────────────────────────────────────────────
# Figure 3 – Tau: TPU v4 vs v5p vs GPU Paper
# ──────────────────────────────────────────────
def fig3_tau_tpu_vs_gpu():
    v4_rows = _read_csv(V4_DIR / "standalone_vs_gpu_paper.csv")
    v5p_rows = _read_csv(V5P_DIR / "standalone_vs_gpu_paper.csv")

    v4_rows = [r for r in v4_rows if r["dataset"] != "AVERAGE"]
    v5p_by_ds = {r["dataset"]: r for r in v5p_rows if r["dataset"] != "AVERAGE"}

    datasets = [r["dataset"] for r in v4_rows]
    v4_tau = [float(r["tpu_tau"]) for r in v4_rows]
    v5_tau = [float(v5p_by_ds[d]["v5p_tau"]) if d in v5p_by_ds else np.nan for d in datasets]
    gpu_tau = [float(r["gpu_paper_tau"]) for r in v4_rows]

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w, v4_tau, w, label="TPU v4", color="#4C72B0")
    b2 = ax.bar(x, v5_tau, w, label="TPU v5p", color="#DD8452")
    b3 = ax.bar(x + w, gpu_tau, w, label="GPU (paper)", color="#55A868")
    _bar_label(ax, b1)
    _bar_label(ax, b2)
    _bar_label(ax, b3)

    ax.set_ylabel("Tau (avg accepted tokens per draft step)")
    ax.set_title("Draft Acceptance (Tau): TPU v4 vs v5p vs GPU Paper")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    all_tau = [v for v in v4_tau + v5_tau + gpu_tau if not np.isnan(v)]
    ax.set_ylim(0, max(all_tau) * 1.3)
    fig.savefig(OUT_DIR / "fig3_tau_tpu_vs_gpu.png")
    plt.close(fig)
    print("  [3/10] fig3_tau_tpu_vs_gpu.png")


# ──────────────────────────────────────────────
# Figure 4 – Acceptance Rate Decay by Position
# ──────────────────────────────────────────────
def fig4_acceptance_decay():
    rows = _read_csv(V4_DIR / "standalone_acceptance_per_pos.csv")
    positions = list(range(16))
    fig, ax = plt.subplots(figsize=(10, 5.5))

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.85, len(rows)))

    for i, row in enumerate(rows):
        ds = row["dataset"]
        rates = [float(row[f"pos_{p}"]) for p in positions]
        ax.plot(positions, rates, marker="o", markersize=4, label=f"{ds} (τ={row['tau']})", color=colors[i])

    ax.set_xlabel("Draft Token Position")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Acceptance Rate Decay by Draft Position (TPU v4)")
    ax.set_xticks(positions)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT_DIR / "fig4_acceptance_decay.png")
    plt.close(fig)
    print("  [4/10] fig4_acceptance_decay.png")


# ──────────────────────────────────────────────
# Figure 5 – Profiling: Step Time Breakdown
# ──────────────────────────────────────────────
def fig5_profiling_breakdown():
    prof = _read_json(V4_DIR / "profiling_gsm8k.json")
    s = prof["summary"]

    components = [
        ("Draft Forward", s["draft_forward"]["mean_ms"]),
        ("Draft Sample", s["draft_sample"]["mean_ms"]),
        ("Verify Forward", s["verify_forward"]["mean_ms"]),
        ("Acceptance", s["acceptance"]["mean_ms"]),
        ("Host↔Device Xfer", s["host_device_xfer"]["mean_ms"]),
        ("Aux Projection", s["aux_projection"]["mean_ms"]),
        ("Ctx Update", s["ctx_update"]["mean_ms"]),
        ("Cache Mgmt", s["cache_mgmt"]["mean_ms"]),
    ]
    labels, values = zip(*components)
    total = sum(values)
    pcts = [v / total * 100 for v in values]

    cmap = plt.cm.Set2
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.5))

    wedges, texts, autotexts = ax_pie.pie(
        values, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        colors=colors, startangle=140, pctdistance=0.8,
    )
    ax_pie.legend(wedges, labels, loc="center left", bbox_to_anchor=(-0.35, 0.5), fontsize=9)
    ax_pie.set_title("Proportion of Step Time")

    bars = ax_bar.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax_bar.set_xlabel("Mean Latency (ms)")
    ax_bar.set_title(f"Per-Component Latency (total step ≈ {total:.1f} ms)")
    for bar, val in zip(bars, values[::-1]):
        ax_bar.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f} ms", va="center", fontsize=9)

    fig.suptitle("DFlash Step Profiling Breakdown — GSM8K on TPU v4", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_profiling_breakdown.png")
    plt.close(fig)
    print("  [5/10] fig5_profiling_breakdown.png")


# ──────────────────────────────────────────────
# Figure 6 – Refinement Steps vs Tau & Speedup
# ──────────────────────────────────────────────
def fig6_refinement_impact():
    data = _read_json(V4_DIR / "refinement_gsm8k.json")
    ks = sorted(data["results_by_k"].keys(), key=int)
    taus = [data["results_by_k"][k]["tau"] for k in ks]
    speedups = [data["results_by_k"][k]["speedup"] for k in ks]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ks))
    color1 = "#4C72B0"
    color2 = "#DD8452"

    b1 = ax1.bar(x - 0.18, taus, 0.35, color=color1, label="Tau")
    ax1.set_ylabel("Tau (accepted tokens)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    _bar_label(ax1, b1)

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + 0.18, speedups, 0.35, color=color2, label="Speedup")
    ax2.set_ylabel("Speedup", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    _bar_label(ax2, b2)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k={k}" for k in ks])
    ax1.set_xlabel("Number of Refinement Steps")
    ax1.set_title("Effect of Refinement Steps on Tau and Speedup (GSM8K, TPU v4)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.savefig(OUT_DIR / "fig6_refinement_impact.png")
    plt.close(fig)
    print("  [6/10] fig6_refinement_impact.png")


# ──────────────────────────────────────────────
# Figure 7 – Verification Latency vs Context Length
# ──────────────────────────────────────────────
def fig7_context_scaling():
    data = _read_json(V4_DIR / "verify_context_scaling.json")
    ctx_lens = data["context_lengths"]
    k_vals = data["k_values"]

    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "D"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, k in enumerate(k_vals):
        means = [data["results"][str(c)][str(k)]["mean_ms"] for c in ctx_lens]
        stds = [data["results"][str(c)][str(k)]["std_ms"] for c in ctx_lens]
        ax.errorbar(ctx_lens, means, yerr=stds, marker=markers[i], color=colors[i],
                    label=f"K={k}", capsize=4, linewidth=2, markersize=7)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Verification Latency (ms)")
    ax.set_title("Verification Latency vs Context Length (TPU v4)")
    ax.legend()
    ax.set_ylim(0, max(ax.get_ylim()[1], 3.0))
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT_DIR / "fig7_context_scaling.png")
    plt.close(fig)
    print("  [7/10] fig7_context_scaling.png")


# ──────────────────────────────────────────────
# Figure 8 – Cross-Comparison: Standalone vs vLLM vs GPU
# ──────────────────────────────────────────────
def fig8_cross_comparison():
    rows = _read_csv(V4_DIR / "cross_comparison.csv")
    rows = [r for r in rows if r["dataset"] != "AVERAGE"]

    datasets = [r["dataset"] for r in rows]
    standalone_tau = [float(r["standalone_tau"]) for r in rows]
    vllm_tau = [float(r["vllm_tau"]) for r in rows]
    gpu_tau = [float(r["gpu_paper_tau"]) for r in rows]

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w, standalone_tau, w, label="TPU Standalone", color="#4C72B0")
    b2 = ax.bar(x, vllm_tau, w, label="TPU vLLM Pipeline", color="#C44E52")
    b3 = ax.bar(x + w, gpu_tau, w, label="GPU (paper)", color="#55A868")
    _bar_label(ax, b1)
    _bar_label(ax, b2)
    _bar_label(ax, b3)

    ax.set_ylabel("Tau")
    ax.set_title("Tau Comparison: Standalone vs vLLM Pipeline vs GPU Paper (TPU v4)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, max(max(standalone_tau), max(vllm_tau), max(gpu_tau)) * 1.3)
    fig.savefig(OUT_DIR / "fig8_cross_comparison.png")
    plt.close(fig)
    print("  [8/10] fig8_cross_comparison.png")


# ──────────────────────────────────────────────
# Figure 9 – v5p vs v4 Improvement Factors
# ──────────────────────────────────────────────
def fig9_v5p_improvement():
    rows = _read_csv(V5P_DIR / "standalone_vs_v4.csv")
    datasets = [r["dataset"] for r in rows]
    baseline_imp = [float(r["baseline_improvement"].replace("x", "")) for r in rows]
    dflash_imp = [float(r["dflash_improvement"].replace("x", "")) for r in rows]

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w / 2, baseline_imp, w, label="Baseline Improvement", color="#A6CEE3", edgecolor="white")
    b2 = ax.bar(x + w / 2, dflash_imp, w, label="DFlash Improvement", color="#DD8452", edgecolor="white")
    _bar_label(ax, b1)
    _bar_label(ax, b2)

    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Improvement Factor (v5p / v4)")
    ax.set_title("TPU v5p over v4: Baseline and DFlash Latency Improvement")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(baseline_imp), max(dflash_imp)) * 1.25)
    fig.savefig(OUT_DIR / "fig9_v5p_improvement.png")
    plt.close(fig)
    print("  [9/10] fig9_v5p_improvement.png")


# ──────────────────────────────────────────────
# Figure 10 – DFlash vs Eagle3 vs vLLM Pipeline
# ──────────────────────────────────────────────
def fig10_method_comparison():
    vllm = _read_csv(V4_DIR / "vllm_pipeline_results.csv")
    eagle = _read_csv(V4_DIR / "eagle3_llama_results.csv")
    v4 = _read_csv(V4_DIR / "standalone_all_benchmarks.csv")

    math_datasets = ["gsm8k", "math500", "aime24", "aime25"]

    standalone_speedup = {r["dataset"]: float(r["tpu_speedup"]) for r in v4 if r["dataset"] in math_datasets}
    vllm_speedup = {r["dataset"]: float(r["speedup"]) for r in vllm if r["method"] == "dflash" and r["dataset"] in math_datasets}
    eagle_speedup = {r["dataset"]: float(r["speedup"]) for r in eagle if r["method"] == "eagle3" and r["dataset"] in math_datasets}

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
    ax.set_title("Speculative Decoding Methods Compared (Math Benchmarks, TPU v4)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    all_vals = [v for v in s_vals + v_vals + e_vals if not np.isnan(v)]
    ax.set_ylim(0, max(all_vals) * 1.3)
    fig.savefig(OUT_DIR / "fig10_method_comparison.png")
    plt.close(fig)
    print("  [10/10] fig10_method_comparison.png")


# ══════════════════════════════════════════════
# REPORT FIGURES (R1–R5): Final-findings visuals
# matching the sections in report.md
# ══════════════════════════════════════════════


# ──────────────────────────────────────────────
# R1 – V5P Performance by Category
# ──────────────────────────────────────────────
def figR1_category_summary():
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

    fig.suptitle("V5P DFlash Performance by Task Category", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figR1_category_summary.png")
    plt.close(fig)
    print("  [R1] figR1_category_summary.png")


# ──────────────────────────────────────────────
# R2 – V5P vs V4 Latency (TPOT) Comparison
# ──────────────────────────────────────────────
def figR2_latency_v5p_vs_v4():
    rows = _read_csv(V5P_DIR / "standalone_vs_v4.csv")
    datasets = [r["dataset"] for r in rows]

    v5_base = [float(r["v5p_baseline_tpot_ms"]) for r in rows]
    v5_df = [float(r["v5p_dflash_tpot_ms"]) for r in rows]
    v4_base = [float(r["v4_baseline_tpot_ms"]) for r in rows]
    v4_df = [float(r["v4_dflash_tpot_ms"]) for r in rows]

    x = np.arange(len(datasets))
    w = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - 1.5 * w, v4_base, w, label="V4 Baseline", color="#A6CEE3", edgecolor="white")
    b_v4df = ax.bar(x - 0.5 * w, v4_df, w, label="V4 DFlash", color="#4C72B0", edgecolor="white")
    ax.bar(x + 0.5 * w, v5_base, w, label="V5P Baseline", color="#FDBF6F", edgecolor="white")
    b_v5df = ax.bar(x + 1.5 * w, v5_df, w, label="V5P DFlash", color="#DD8452", edgecolor="white")

    for bar, val in zip(b_v4df, v4_df):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.1f}", ha="center", fontsize=7.5, color="#4C72B0", fontweight="bold")
    for bar, val in zip(b_v5df, v5_df):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.1f}", ha="center", fontsize=7.5, color="#DD8452", fontweight="bold")

    ax.set_ylabel("Time per Output Token (ms) — lower is better")
    ax.set_title("Latency (TPOT): V4 vs V5P — Baseline and DFlash", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(v4_base) * 1.2)
    ax.grid(axis="y", alpha=0.2)

    ax.annotate("V5P baseline 1.69x faster",
                xy=(4, np.mean(v5_base)), fontsize=9, color="#8b949e",
                ha="center", style="italic")

    fig.savefig(OUT_DIR / "figR2_latency_v5p_vs_v4.png")
    plt.close(fig)
    print("  [R2] figR2_latency_v5p_vs_v4.png")


# ──────────────────────────────────────────────
# R3 – V5P vs GPU Paper: Tau Parity & Speedup Gap
# ──────────────────────────────────────────────
def figR3_tpu_vs_gpu_parity():
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
    ax2.set_ylim(0, max(gpu_speedup) * 1.3)
    ax2.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle("TPU V5P vs GPU A100 — Math Benchmarks", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figR3_tpu_vs_gpu_parity.png")
    plt.close(fig)
    print("  [R3] figR3_tpu_vs_gpu_parity.png")


# ──────────────────────────────────────────────
# R4 – Output Quality: Exact Match & Answer Match
# ──────────────────────────────────────────────
def figR4_output_quality():
    rows = _read_csv(V5P_DIR / "standalone_all_benchmarks.csv")
    datasets = [r["dataset"] for r in rows]
    categories = [r["category"] for r in rows]

    match_rates = []
    for r in rows:
        val = r["match_rate"].replace("%", "")
        match_rates.append(float(val))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [CATEGORY_COLORS[c] for c in categories]
    bars = ax.bar(datasets, match_rates, color=colors, edgecolor="white", width=0.6)

    for bar, rate, r in zip(bars, match_rates, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{r['exact_match']}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Exact Token Match Rate (%)")
    ax.set_title("DFlash Output Quality — Exact Token Match vs Baseline (V5P)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CATEGORY_COLORS[c], label=c.capitalize()) for c in CATEGORY_ORDER]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.text(0.5, -0.18,
            "Mismatches are bf16 floating-point divergence (batch-16 verify vs single-token), not correctness errors",
            transform=ax.transAxes, ha="center", fontsize=9, color="#8b949e", style="italic")

    fig.savefig(OUT_DIR / "figR4_output_quality.png")
    plt.close(fig)
    print("  [R4] figR4_output_quality.png")


# ──────────────────────────────────────────────
# R5 – Final Summary Dashboard
# ──────────────────────────────────────────────
def figR5_summary_dashboard():
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
        ax.text(0.5, 0.6, value, transform=ax.transAxes, fontsize=36, fontweight="bold",
                ha="center", va="center", color=color, fontfamily="monospace")
        ax.text(0.5, 0.25, label, transform=ax.transAxes, fontsize=13, fontweight="bold",
                ha="center", va="center", color="#c9d1d9")
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
                    f"{s:.2f}x", va="center", fontsize=10, color="#c9d1d9", fontweight="bold")
    ax_bar.set_xlabel("Speedup", color="#c9d1d9", fontsize=12)
    ax_bar.set_title("DFlash Speedup by Dataset (TPU V5P)", color="#58a6ff",
                     fontsize=13, fontweight="bold", pad=10)
    ax_bar.axvline(1, color="#30363d", linewidth=1, linestyle="--")
    ax_bar.axvline(avg_speedup, color="#58a6ff", linewidth=1.2, linestyle="--", alpha=0.7)
    ax_bar.text(avg_speedup + 0.05, -0.6, f"avg {avg_speedup:.2f}x",
                fontsize=9, color="#58a6ff", style="italic")
    ax_bar.tick_params(colors="#c9d1d9")
    ax_bar.set_xlim(0, max(speedups) * 1.25)
    for spine in ax_bar.spines.values():
        spine.set_color("#30363d")
    ax_bar.grid(axis="x", alpha=0.15, color="#c9d1d9")

    fig.suptitle("DFlash on TPU — Final Results Summary",
                 fontsize=18, fontweight="bold", color="#c9d1d9", y=0.98,
                 fontfamily="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "figR5_summary_dashboard.png", facecolor="#0d1117")
    plt.close(fig)
    print("  [R5] figR5_summary_dashboard.png")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reading results from: {RESULTS_DIR}")
    print(f"Writing figures to:   {OUT_DIR}\n")
    fig1_speedup_v4_vs_v5p()
    fig2_throughput()
    fig3_tau_tpu_vs_gpu()
    fig4_acceptance_decay()
    fig5_profiling_breakdown()
    fig6_refinement_impact()
    fig7_context_scaling()
    fig8_cross_comparison()
    fig9_v5p_improvement()
    fig10_method_comparison()
    print()
    figR1_category_summary()
    figR2_latency_v5p_vs_v4()
    figR3_tpu_vs_gpu_parity()
    figR4_output_quality()
    figR5_summary_dashboard()
    print("\nAll 15 figures generated successfully.")

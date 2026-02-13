#!/usr/bin/env python3
"""Generate CSV result tables from benchmark JSON logs.

Reads directly from:
  - Standalone benchmark JSON: results/standalone_*.json
  - vLLM pipeline JSON:       /dev/shm/dflash-test-outputs/bench_math_*/summaries/

GPU paper numbers are from DFlash paper Table 1 (Qwen3-4B, temperature=0).

Usage:
    python results/generate_csvs.py
"""

import csv
import glob
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# GPU paper reference numbers (DFlash paper Table 1, Qwen3-4B, temp=0)
# ---------------------------------------------------------------------------
GPU_PAPER = {
    "gsm8k":  {"tau": 6.53, "speedup": 5.15},
    "math500": {"tau": 7.84, "speedup": 6.09},
    "aime24": {"tau": 7.27, "speedup": 5.68},
    "aime25": {"tau": 6.64, "speedup": 5.21},
}

# ---------------------------------------------------------------------------
# Discover data sources
# ---------------------------------------------------------------------------

def find_standalone_jsons():
    pattern = os.path.join(SCRIPT_DIR, "standalone_*.json")
    paths = sorted(glob.glob(pattern))
    results = {}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        ds = data["config"]["dataset"]
        results[ds] = data
    return results


def find_vllm_json():
    """Find the most recent vLLM bench_math run."""
    pattern = "/dev/shm/dflash-test-outputs/bench_math_*/summaries/overall.json"
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("[WARN] No vLLM benchmark JSON found, skipping vLLM tables.")
        return None, None
    latest = paths[-1]
    with open(latest) as f:
        overall = json.load(f)
    # Also load comparator for per-dataset speedup
    comp_path = os.path.join(os.path.dirname(latest), "comparator_dflash.json")
    comparator = None
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparator = json.load(f)
    return overall, comparator


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def generate_standalone_summary(standalone):
    """Table 1: Per-dataset standalone results vs GPU paper."""
    headers = [
        "dataset",
        "tpu_tau", "tpu_speedup", "tpu_baseline_tps", "tpu_dflash_tps",
        "gpu_paper_tau", "gpu_paper_speedup",
        "tau_pct_of_gpu",
    ]
    rows = []
    for ds in ["gsm8k", "math500", "aime24", "aime25"]:
        if ds not in standalone:
            continue
        s = standalone[ds]["summary"]
        gpu = GPU_PAPER.get(ds, {})
        gpu_tau = gpu.get("tau", "")
        gpu_spd = gpu.get("speedup", "")
        tau_pct = (s["tau"] / gpu_tau * 100) if gpu_tau else ""
        rows.append([
            ds,
            round(s["tau"], 2),
            round(s["speedup"], 2),
            round(s["baseline_tps"], 1),
            round(s["dflash_tps"], 1),
            gpu_tau,
            gpu_spd,
            round(tau_pct, 1) if tau_pct else "",
        ])

    # Average row
    if rows:
        n = len(rows)
        avg_tau = sum(r[1] for r in rows) / n
        avg_spd = sum(r[2] for r in rows) / n
        avg_bl = sum(r[3] for r in rows) / n
        avg_df = sum(r[4] for r in rows) / n
        avg_gpu_tau = sum(r[5] for r in rows if r[5]) / n
        avg_gpu_spd = sum(r[6] for r in rows if r[6]) / n
        avg_pct = (avg_tau / avg_gpu_tau * 100) if avg_gpu_tau else ""
        rows.append([
            "AVERAGE",
            round(avg_tau, 2), round(avg_spd, 2),
            round(avg_bl, 1), round(avg_df, 1),
            round(avg_gpu_tau, 2), round(avg_gpu_spd, 2),
            round(avg_pct, 1) if avg_pct else "",
        ])

    write_csv(os.path.join(SCRIPT_DIR, "standalone_vs_gpu_paper.csv"), headers, rows)


def generate_standalone_acceptance(standalone):
    """Table 2: Per-position acceptance rates from standalone runs."""
    # Find max block_size across datasets
    block_size = 16
    headers = ["dataset", "tau"] + [f"pos_{i}" for i in range(block_size)]
    rows = []
    for ds in ["gsm8k", "math500", "aime24", "aime25"]:
        if ds not in standalone:
            continue
        s = standalone[ds]["summary"]
        rates = s["acceptance_rate_per_pos"]
        row = [ds, round(s["tau"], 2)]
        for i in range(block_size):
            row.append(round(rates[i], 4) if i < len(rates) else "")
        rows.append(row)
    write_csv(os.path.join(SCRIPT_DIR, "standalone_acceptance_per_pos.csv"), headers, rows)


def generate_standalone_per_sample(standalone):
    """Table 3: Per-sample detail from standalone runs."""
    headers = [
        "dataset", "sample_index", "is_warmup",
        "num_input_tokens",
        "baseline_tpot_ms", "baseline_tps",
        "dflash_tpot_ms", "dflash_tps",
        "tau", "num_drafts",
    ]
    rows = []
    for ds in ["gsm8k", "math500", "aime24", "aime25"]:
        if ds not in standalone:
            continue
        for sample in standalone[ds]["per_sample"]:
            rows.append([
                ds,
                sample["sample_index"],
                sample["is_warmup"],
                sample["num_input_tokens"],
                round(sample["baseline_tpot_ms"], 2),
                round(sample["baseline_tps"], 1),
                round(sample["dflash_tpot_ms"], 2),
                round(sample["dflash_tps"], 1),
                round(sample["tau"], 2),
                sample["num_drafts"],
            ])
    write_csv(os.path.join(SCRIPT_DIR, "standalone_per_sample.csv"), headers, rows)


def generate_vllm_summary(overall, comparator):
    """Table 4: vLLM pipeline results."""
    headers = [
        "dataset", "method", "num_prompts", "num_ok", "num_error",
        "tpot_ms", "tps", "speedup",
    ]
    rows = []
    for entry in overall.get("datasets", []):
        ds = entry["dataset"]
        method = entry["method"]
        spd = entry.get("speedup_vs_baseline", "")
        rows.append([
            ds, method, entry["num_prompts"], entry["num_ok"], entry["num_error"],
            round(entry["time_per_output_token"] * 1000, 2),
            round(entry["tokens_per_second"], 1),
            round(spd, 2) if spd else "",
        ])

    # Overall row for each method
    for method_name, mdata in overall.get("methods", {}).items():
        spd = mdata.get("speedup_vs_baseline", "")
        rows.append([
            "OVERALL", method_name,
            mdata["num_prompts"], mdata["num_ok"], mdata["num_error"],
            round(mdata["time_per_output_token"] * 1000, 2),
            round(mdata["tokens_per_second"], 1),
            round(spd, 2) if spd else "",
        ])

    write_csv(os.path.join(SCRIPT_DIR, "vllm_pipeline_results.csv"), headers, rows)


def generate_vllm_acceptance(overall):
    """Table 5: vLLM pipeline acceptance rates."""
    dflash = overall.get("methods", {}).get("dflash", {})
    rates = dflash.get("acceptance_rate_per_pos", [])
    if not rates:
        return
    headers = ["metric", "value"]
    rows = [
        ["tau", round(dflash.get("tau", 0), 2)],
        ["draft_acceptance_rate", round(dflash.get("draft_acceptance_rate", 0), 4)],
        ["num_drafts", dflash.get("num_drafts", 0)],
        ["num_draft_tokens", dflash.get("num_draft_tokens", 0)],
        ["num_accepted_tokens", dflash.get("num_accepted_tokens", 0)],
    ]
    for i, r in enumerate(rates):
        rows.append([f"acceptance_pos_{i}", round(r, 4)])
    write_csv(os.path.join(SCRIPT_DIR, "vllm_pipeline_acceptance.csv"), headers, rows)


def generate_cross_comparison(standalone, overall, comparator):
    """Table 6: Standalone vs vLLM pipeline vs GPU paper — the money table."""
    headers = [
        "dataset",
        "standalone_tau", "standalone_speedup",
        "vllm_tau", "vllm_speedup",
        "gpu_paper_tau", "gpu_paper_speedup",
        "standalone_tau_pct_of_gpu", "vllm_tau_pct_of_gpu",
    ]

    # vLLM overall tau (same across datasets in our run)
    vllm_tau = overall["methods"]["dflash"]["tau"] if overall else None
    vllm_per_ds = {}
    if comparator:
        for ds_name, vals in comparator.get("datasets", {}).items():
            vllm_per_ds[ds_name.lower()] = vals

    rows = []
    for ds in ["gsm8k", "math500", "aime24", "aime25"]:
        gpu = GPU_PAPER.get(ds, {})
        gpu_tau = gpu.get("tau")
        gpu_spd = gpu.get("speedup")

        # Standalone
        s_tau = standalone[ds]["summary"]["tau"] if ds in standalone else None
        s_spd = standalone[ds]["summary"]["speedup"] if ds in standalone else None

        # vLLM
        ds_display = {"gsm8k": "gsm8k", "math500": "math500", "aime24": "aime24", "aime25": "aime25"}[ds]
        v_spd = vllm_per_ds.get(ds_display, {}).get("speedup")
        v_tau = vllm_tau  # same overall tau for our run

        s_pct = round(s_tau / gpu_tau * 100, 1) if (s_tau and gpu_tau) else ""
        v_pct = round(v_tau / gpu_tau * 100, 1) if (v_tau and gpu_tau) else ""

        rows.append([
            ds,
            round(s_tau, 2) if s_tau else "",
            round(s_spd, 2) if s_spd else "",
            round(v_tau, 2) if v_tau else "",
            round(v_spd, 2) if v_spd else "",
            gpu_tau or "",
            gpu_spd or "",
            s_pct,
            v_pct,
        ])

    # Average
    n = len(rows)
    def avg(idx):
        vals = [r[idx] for r in rows if r[idx] != ""]
        return round(sum(vals) / len(vals), 2) if vals else ""

    rows.append([
        "AVERAGE",
        avg(1), avg(2), avg(3), avg(4), avg(5), avg(6), avg(7), avg(8),
    ])

    write_csv(os.path.join(SCRIPT_DIR, "cross_comparison.csv"), headers, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Generating CSV tables from benchmark run logs")
    print("=" * 60)
    print()

    # Load data
    standalone = find_standalone_jsons()
    print(f"Found standalone results: {list(standalone.keys())}")

    overall, comparator = find_vllm_json()
    if overall:
        print(f"Found vLLM pipeline results: {list(overall.get('methods', {}).keys())}")
    print()

    # Generate tables
    print("Generating tables:")

    if standalone:
        generate_standalone_summary(standalone)
        generate_standalone_acceptance(standalone)
        generate_standalone_per_sample(standalone)

    if overall:
        generate_vllm_summary(overall, comparator)
        generate_vllm_acceptance(overall)

    if standalone and overall:
        generate_cross_comparison(standalone, overall, comparator)

    print()
    print("Done.")


if __name__ == "__main__":
    main()

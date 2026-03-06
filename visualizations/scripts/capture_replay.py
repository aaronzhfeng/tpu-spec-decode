#!/usr/bin/env python3
"""Capture live inference replay data for the token replay visualization.

Runs the existing standalone_dflash benchmark and post-processes the results
to extract per-token text and timestamps for faithful replay.

Usage (inside Docker with TPU):
    python visualizations/scripts/capture_replay.py \
        --dataset gsm8k --max-samples 3

    # All benchmarks:
    python visualizations/scripts/capture_replay.py --all

Output: visualizations/output/replay/replay_{dataset}.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ALL_DATASETS = ["gsm8k", "math500", "aime24", "aime25",
                "humaneval", "mbpp", "mt-bench", "alpaca", "swe-bench"]


def run_benchmark(dataset, max_samples=3, max_new_tokens=256):
    """Run standalone_dflash.py and return the JSON result path."""
    output_dir = PROJECT_ROOT / "visualizations" / "output" / "replay"
    os.makedirs(output_dir, exist_ok=True)
    raw_json = output_dir / f"raw_{dataset}.json"

    cmd = [
        sys.executable, str(PROJECT_ROOT / "benchmarks" / "standalone_dflash.py"),
        "--target-model", "Qwen/Qwen3-4B",
        "--draft-model", "z-lab/Qwen3-4B-DFlash-b16",
        "--dataset", dataset,
        "--max-samples", str(max_samples),
        "--max-new-tokens", str(max_new_tokens),
        "--output-json", str(raw_json),
    ]

    print(f"\n{'='*60}")
    print(f"Running benchmark: {dataset} ({max_samples} samples)")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Benchmark failed for {dataset}")
        return None
    return raw_json


def postprocess_to_replay(raw_json_path, dataset):
    """Convert benchmark JSON to replay format with per-token text.

    The benchmark JSON already has:
    - acceptance_lengths per sample (the burst pattern)
    - baseline_tpot_ms, dflash_tpot_ms (timing)
    - output token IDs (via quality_mismatches, but not full text)

    We reconstruct timestamps from TPOT and acceptance_lengths.
    For text, we need the tokenizer — load it once.
    """
    with open(raw_json_path) as f:
        data = json.load(f)

    output_dir = Path(raw_json_path).parent
    replay_path = output_dir / f"replay_{dataset}.json"

    # We can't decode tokens here (no tokenizer outside Docker).
    # Instead, save the raw data in replay-ready format.
    # The actual token decoding happens inside Docker where the tokenizer is available.
    samples = []
    for s in data["per_sample"]:
        if s.get("is_warmup", False):
            continue

        bl_tpot = s["baseline_tpot_ms"]
        df_tpot = s["dflash_tpot_ms"]
        n_bl = s["baseline_num_output_tokens"]
        n_df = s["dflash_num_output_tokens"]
        acceptance = s.get("acceptance_lengths", [])
        tau = s.get("tau", 0)

        # Reconstruct baseline per-token timestamps (uniform spacing)
        bl_timestamps = [round(i * bl_tpot, 3) for i in range(n_bl)]

        # Reconstruct DFlash per-step timestamps from acceptance_lengths
        # Each step takes roughly the same wall time
        n_steps = len(acceptance)
        if n_steps > 0:
            total_df_time = df_tpot * n_df  # total decode time in ms
            step_time = total_df_time / n_steps
            df_steps = []
            for i, accepted in enumerate(acceptance):
                df_steps.append({
                    "ms": round((i + 1) * step_time, 3),
                    "tokens_accepted": int(accepted),
                })
        else:
            df_steps = []

        samples.append({
            "baseline": {
                "token_timestamps_ms": bl_timestamps,
                "num_output_tokens": n_bl,
                "tpot_ms": round(bl_tpot, 3),
                "tps": round(1000 / bl_tpot, 1) if bl_tpot > 0 else 0,
                "total_time_ms": round(bl_tpot * n_bl, 3),
            },
            "dflash": {
                "step_timestamps": df_steps,
                "acceptance_lengths": [int(a) for a in acceptance],
                "num_output_tokens": n_df,
                "tpot_ms": round(df_tpot, 3),
                "tps": round(1000 / df_tpot, 1) if df_tpot > 0 else 0,
                "tau": round(tau, 2),
                "total_time_ms": round(df_tpot * n_df, 3),
            },
            "speedup": round(bl_tpot / df_tpot, 2) if df_tpot > 0 else 0,
        })

    replay = {
        "config": {
            "target_model": data["config"]["target_model"],
            "draft_model": data["config"]["draft_model"],
            "dataset": dataset,
            "block_size": data["config"]["block_size"],
            "max_new_tokens": data["config"]["max_new_tokens"],
            "num_samples": len(samples),
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": data["summary"],
        "samples": samples,
    }

    with open(replay_path, "w") as f:
        json.dump(replay, f, indent=2, ensure_ascii=False)
    print(f"Replay data saved: {replay_path} ({len(samples)} samples)")
    return replay_path


def decode_tokens_into_replay(replay_path, raw_json_path):
    """Add decoded token text to replay JSON. Requires tokenizer (run inside Docker)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  (tokenizer not available — text will be added when run inside Docker)")
        return

    with open(raw_json_path) as f:
        raw = json.load(f)
    with open(replay_path) as f:
        replay = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        raw["config"]["target_model"], trust_remote_code=True)

    measured_idx = 0
    for s in raw["per_sample"]:
        if s.get("is_warmup", False):
            continue
        if measured_idx >= len(replay["samples"]):
            break

        # Get output token IDs from the raw benchmark data
        # The benchmark saves mismatches but not full output — reconstruct from
        # the fact that baseline and dflash produce similar output
        # For now, note that the text field will be populated when we modify
        # standalone_dflash.py to save output_ids, or we can re-run with a
        # modified version. The timestamps are the critical data.
        replay["samples"][measured_idx]["needs_text"] = True
        measured_idx += 1

    with open(replay_path, "w") as f:
        json.dump(replay, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Capture DFlash replay data")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true", help="Run all 9 datasets")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip benchmark, only postprocess existing raw JSON")
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.all else [args.dataset]

    for ds in datasets:
        raw_json = PROJECT_ROOT / "visualizations" / "output" / "replay" / f"raw_{ds}.json"

        if not args.skip_benchmark:
            raw_json = run_benchmark(ds, args.max_samples, args.max_new_tokens)
            if raw_json is None:
                continue

        if raw_json and raw_json.exists() if isinstance(raw_json, Path) else os.path.exists(raw_json):
            replay_path = postprocess_to_replay(raw_json, ds)
            decode_tokens_into_replay(replay_path, raw_json)
        else:
            print(f"No raw JSON for {ds}, skipping postprocess")

    print(f"\nDone. Replay files in visualizations/output/replay/")


if __name__ == "__main__":
    main()

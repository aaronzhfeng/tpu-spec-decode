#!/usr/bin/env python3
"""Compare benchmark outputs against extracted DFlash-reported baselines.

Expected result JSON format:
{
  "datasets": {
    "GSM8K": {"speedup": 5.1, "tau": 6.5},
    ...
  },
  "average": {"speedup": 5.7, "tau": 7.2}
}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from dflash_reported_baselines import (PLOT_SPEEDUP_BASELINE_QWEN3_8B,
                                       TABLE_BASELINES)


def _norm_name(name: str) -> str:
    return name.strip().replace("_", "-").lower()


def _norm_result_map(datasets: dict[str, Any]) -> dict[str, Any]:
    return {_norm_name(k): v for k, v in datasets.items()}


def _load_results(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    datasets = data.get("datasets", {})
    average = data.get("average", {})
    if not isinstance(datasets, dict):
        raise ValueError("results JSON must contain an object field 'datasets'")
    if not isinstance(average, dict):
        average = {}
    return datasets, average


def _select_baseline_table(category: str, model_key: str,
                           temperature: int) -> dict[str, dict[str, float]]:
    try:
        return TABLE_BASELINES[category][model_key][temperature]
    except KeyError as exc:
        raise KeyError(
            f"Unknown baseline selection category={category} model={model_key} temperature={temperature}"
        ) from exc


def _select_baseline_plot() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for dataset, vals in PLOT_SPEEDUP_BASELINE_QWEN3_8B.items():
        out[dataset] = {"speedup": vals["dflash"]}
    return out


def _safe_ratio(obs: float, ref: float) -> float:
    if math.isclose(ref, 0.0):
        return math.inf if not math.isclose(obs, 0.0) else 1.0
    return obs / ref


def compare(
    results_json: str,
    category: str,
    model_key: str,
    temperature: int,
    metric: str,
    min_retention: float,
) -> int:
    datasets, average = _load_results(results_json)
    observed = _norm_result_map(datasets)

    if category == "plot":
        baseline = _select_baseline_plot()
    else:
        baseline = _select_baseline_table(category, model_key, temperature)

    failures = []
    rows = []
    for dataset, expected in baseline.items():
        if dataset == "Average":
            obs_val = average.get(metric)
            ref_val = expected.get(metric)
            label = "Average"
        else:
            key = _norm_name(dataset)
            obs_obj = observed.get(key)
            obs_val = None if obs_obj is None else obs_obj.get(metric)
            ref_val = expected.get(metric)
            label = dataset

        if ref_val is None:
            continue
        if obs_val is None:
            failures.append((label, "missing"))
            rows.append((label, None, ref_val, None))
            continue

        ratio = _safe_ratio(float(obs_val), float(ref_val))
        rows.append((label, float(obs_val), float(ref_val), ratio))
        if ratio < min_retention:
            failures.append((label, f"ratio={ratio:.3f}"))

    print(
        f"Comparison metric={metric} category={category} model={model_key} temp={temperature} min_retention={min_retention}"
    )
    print("dataset, observed, baseline, ratio")
    for label, obs_val, ref_val, ratio in rows:
        obs_s = "NA" if obs_val is None else f"{obs_val:.4f}"
        ratio_s = "NA" if ratio is None else f"{ratio:.4f}"
        print(f"{label}, {obs_s}, {ref_val:.4f}, {ratio_s}")

    if failures:
        print("\nFAILURES:")
        for label, reason in failures:
            print(f"- {label}: {reason}")
        return 1

    print("\nPASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare run outputs to extracted DFlash baselines.")
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--category",
                        required=True,
                        choices=["math", "code", "chat", "plot"])
    parser.add_argument("--model-key",
                        default="Qwen3-8B-DFlash-b16",
                        help=("Model baseline key, ignored for --category plot. "
                              "Examples: Qwen3-8B-DFlash-b16"))
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--metric", choices=["speedup", "tau"], default="speedup")
    parser.add_argument("--min-retention", type=float, default=0.70)
    args = parser.parse_args()
    return compare(
        results_json=args.results_json,
        category=args.category,
        model_key=args.model_key,
        temperature=args.temperature,
        metric=args.metric,
        min_retention=args.min_retention,
    )


if __name__ == "__main__":
    sys.exit(main())


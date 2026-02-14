#!/usr/bin/env python3
"""Run TPU-vLLM DFlash eval and emit comparator-compatible JSON."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

CATEGORY_DATASETS = {
    "math": ["gsm8k", "math500", "aime24", "aime25"],
    "code": ["humaneval", "mbpp", "livecodebench", "swe-bench"],
    "chat": ["mt-bench", "alpaca"],
}

CANONICAL_DATASET_NAME = {
    "gsm8k": "GSM8K",
    "math500": "Math500",
    "aime24": "AIME24",
    "aime25": "AIME25",
    "humaneval": "Humaneval",
    "mbpp": "MBPP",
    "livecodebench": "LiveCodeBench",
    "swe-bench": "SWE-Bench",
    "mt-bench": "MT-Bench",
    "alpaca": "Alpaca",
}

FALLBACK_PROMPTS = {
    "gsm8k": "If 12 notebooks cost $36, what does one notebook cost?",
    "math500": "Solve: Find x if 3x + 7 = 40.",
    "aime24": "Compute the remainder when 2^20 is divided by 7.",
    "aime25": "How many positive divisors does 360 have?",
    "humaneval": "Write a Python function that returns True if a string is a palindrome.",
    "mbpp": "Write Python code to return the factorial of n recursively.",
    "livecodebench": "Write Python code that merges two sorted arrays.",
    "swe-bench": "You are given a bug report. Describe likely fix steps in Python terms.",
    "mt-bench": "Explain the difference between TCP and UDP simply.",
    "alpaca": "Give three practical tips to reduce procrastination.",
}


def _canonical_name(dataset: str) -> str:
    return CANONICAL_DATASET_NAME.get(dataset, dataset)


def _load_prompts_external(dataset: str, max_samples: int) -> list[str]:
    root = Path(__file__).resolve().parents[2]
    dflash_dir = root / "deps" / "dflash"
    if not dflash_dir.exists():
        raise FileNotFoundError(f"DFlash dir not found: {dflash_dir}")
    if str(dflash_dir) not in sys.path:
        sys.path.insert(0, str(dflash_dir))
    from model.utils import load_and_process_dataset  # type: ignore

    ds = load_and_process_dataset(dataset)
    if max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    prompts: list[str] = []
    for item in ds:
        turns = item.get("turns", None)
        if isinstance(turns, list) and turns:
            prompts.append(str(turns[0]))
    if not prompts:
        raise ValueError(f"No prompts extracted for dataset={dataset}")
    return prompts


def _load_prompts_synthetic(dataset: str, max_samples: int) -> list[str]:
    prompt = FALLBACK_PROMPTS.get(dataset, "Write a short helpful response.")
    n = max(1, max_samples if max_samples > 0 else 4)
    return [f"{prompt} (sample {i})" for i in range(n)]


def load_prompts(dataset: str, max_samples: int, prompt_source: str) -> list[str]:
    if prompt_source == "external":
        try:
            return _load_prompts_external(dataset, max_samples)
        except Exception as exc:
            print(
                f"[WARN] external prompt load failed for {dataset}: {exc}. Falling back to synthetic prompts.",
                file=sys.stderr,
            )
    return _load_prompts_synthetic(dataset, max_samples)


def _count_output_tokens(outputs: list[Any]) -> int:
    total = 0
    for out in outputs:
        candidates = getattr(out, "outputs", None)
        if not candidates:
            continue
        first = candidates[0]
        token_ids = getattr(first, "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
            continue
        text = getattr(first, "text", "")
        total += max(1, len(str(text).split()))
    return total


def _run_generate_once(
    llm: Any,
    prompts: list[str],
    sampling_params: Any,
) -> tuple[float, int]:
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0
    output_tokens = _count_output_tokens(outputs)
    return elapsed, output_tokens


def run_model(
    llm_kwargs: dict[str, Any],
    prompts: list[str],
    sampling_params: Any,
    warmup_prompts: int,
    inter_run_sleep_sec: float,
) -> dict[str, float]:
    from vllm import LLM  # Imported lazily so --help works without vLLM.

    llm = LLM(**llm_kwargs)
    try:
        if warmup_prompts > 0:
            warmup = prompts[:warmup_prompts] if prompts else ["hi"]
            _ = llm.generate(warmup, sampling_params)
        elapsed, output_tokens = _run_generate_once(llm, prompts, sampling_params)
    finally:
        del llm
        gc.collect()
        if inter_run_sleep_sec > 0:
            time.sleep(inter_run_sleep_sec)

    output_tokens = max(1, output_tokens)
    tpot = elapsed / output_tokens
    tps = output_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "elapsed_sec": float(elapsed),
        "output_tokens": float(output_tokens),
        "time_per_output_token": float(tpot),
        "tokens_per_second": float(tps),
    }


def _make_speculative_config(args: argparse.Namespace) -> dict[str, Any]:
    spec = {
        "method": args.spec_method,
        "model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
    }
    return spec


def _dataset_list_from_args(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return [d.strip() for d in args.datasets.split(",") if d.strip()]
    return list(CATEGORY_DATASETS[args.category])


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline + DFlash speculative eval and emit JSON.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--category",
                        default="math",
                        choices=sorted(CATEGORY_DATASETS.keys()))
    parser.add_argument("--datasets",
                        default="",
                        help="Comma-separated dataset ids. Overrides --category.")
    parser.add_argument("--prompt-source",
                        default="external",
                        choices=["external", "synthetic"])
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--num-speculative-tokens", type=int, default=16)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--spec-method", default="dflash")
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--inter-run-sleep-sec", type=float, default=5.0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    datasets = _dataset_list_from_args(args)
    if not datasets:
        raise ValueError("No datasets selected.")

    try:
        from vllm import SamplingParams
    except Exception as exc:
        raise RuntimeError(
            "Failed to import vLLM. Ensure vLLM TPU environment is installed and available."
        ) from exc

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    result: dict[str, Any] = {
        "meta": {
            "model": args.model,
            "draft_model": args.draft_model,
            "category": args.category,
            "datasets": datasets,
            "prompt_source": args.prompt_source,
            "temperature": args.temperature,
            "max_samples": args.max_samples,
            "max_tokens": args.max_tokens,
            "speculative_method": args.spec_method,
            "num_speculative_tokens": args.num_speculative_tokens,
            "generated_at_unix": int(time.time()),
        },
        "datasets": {},
        "average": {},
    }

    speeds: list[float] = []
    for dataset in datasets:
        prompts = load_prompts(dataset, args.max_samples, args.prompt_source)
        dname = _canonical_name(dataset)

        base_metrics = None
        if not args.skip_baseline:
            baseline_kwargs = dict(
                model=args.model,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                tensor_parallel_size=args.tensor_parallel_size,
                async_scheduling=0,
            )
            base_metrics = run_model(
                baseline_kwargs,
                prompts,
                sampling_params,
                warmup_prompts=args.warmup_prompts,
                inter_run_sleep_sec=args.inter_run_sleep_sec,
            )

        dflash_kwargs = dict(
            model=args.model,
            speculative_config=_make_speculative_config(args),
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            tensor_parallel_size=args.tensor_parallel_size,
            async_scheduling=0,
        )
        dflash_metrics = run_model(
            dflash_kwargs,
            prompts,
            sampling_params,
            warmup_prompts=args.warmup_prompts,
            inter_run_sleep_sec=args.inter_run_sleep_sec,
        )

        speedup = None
        if base_metrics is not None:
            speedup = (
                base_metrics["time_per_output_token"] /
                dflash_metrics["time_per_output_token"])
            if math.isfinite(speedup):
                speeds.append(speedup)

        result["datasets"][dname] = {
            "speedup": speedup,
            "tau": None,
            "num_prompts": len(prompts),
            "baseline": base_metrics,
            "dflash": dflash_metrics,
        }
        print(
            f"[{dname}] prompts={len(prompts)} speedup={speedup if speedup is not None else 'NA'}"
        )

    finite_speeds = _finite(speeds)
    avg_speedup = (sum(finite_speeds) / len(finite_speeds)
                   if finite_speeds else None)
    result["average"] = {
        "speedup": avg_speedup,
        "tau": None,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote results JSON: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""Run contribution-focused DFlash validation with rich logging.

This runner is designed for later analysis:
- per-prompt JSONL records (prompt, output text, timings, status)
- dataset/method summaries
- baseline-vs-spec speedups
- environment and git snapshot
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import random
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_name(dataset: str) -> str:
    return CANONICAL_DATASET_NAME.get(dataset, dataset)


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _run_cmd(cmd: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            cmd,
            cwd=str(cwd),
            stderr=subprocess.STDOUT,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _git_snapshot(repo: Path) -> dict[str, Any] | None:
    if not (repo / ".git").exists():
        return None
    commit = _run_cmd(["git", "rev-parse", "HEAD"], repo)
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
    status = _run_cmd(["git", "status", "--porcelain"], repo)
    return {
        "path": str(repo),
        "branch": branch,
        "commit": commit,
        "dirty": bool(status),
    }


def _load_prompts_external(
    repo_root: Path, dataset: str, max_samples: int
) -> list[str]:
    dflash_dir = repo_root / "deps" / "dflash"
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


def load_prompts(
    repo_root: Path, dataset: str, max_samples: int, prompt_source: str
) -> list[str]:
    if prompt_source == "external":
        try:
            return _load_prompts_external(repo_root, dataset, max_samples)
        except Exception as exc:
            print(
                f"[WARN] external prompt load failed for {dataset}: {exc}. "
                "Falling back to synthetic prompts.",
                file=sys.stderr,
            )
    return _load_prompts_synthetic(dataset, max_samples)


def _get_llm_tokenizer(llm: Any) -> Any | None:
    try:
        return llm.get_tokenizer()
    except Exception:
        return None


def _format_prompt_for_generation(
    prompt: str,
    tokenizer: Any | None,
    use_chat_template: bool,
) -> str:
    if not use_chat_template:
        return prompt
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = [{"role": "user", "content": prompt}]
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        # Some tokenizer versions may not support enable_thinking.
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            return prompt
    except Exception:
        return prompt


def _default_manifest() -> dict[str, Any]:
    return {
        "name": "default",
        "model": "Qwen/Qwen3-8B",
        "draft_model": "z-lab/Qwen3-8B-DFlash-b16",
        "eagle3_draft_model": "",
        "prompt_source": "external",
        "use_chat_template": False,
        "datasets": ["gsm8k", "math500", "aime24", "aime25"],
        "max_samples": 8,
        "methods": ["baseline", "dflash"],
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 64,
            "ignore_eos": True,
        },
        "runtime": {
            "tensor_parallel_size": 1,
            "max_model_len": 1024,
            "max_num_seqs": 1,
            "num_speculative_tokens": 8,
            "draft_tensor_parallel_size": 1,
            "warmup_prompts": 1,
            "inter_method_sleep_sec": 3.0,
            "async_scheduling": 0,
            "disable_log_stats": True,
            "seed": 0,
        },
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    defaults = _default_manifest()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    merged = defaults.copy()
    merged.update(raw)
    merged["sampling"] = {**defaults["sampling"], **raw.get("sampling", {})}
    merged["runtime"] = {**defaults["runtime"], **raw.get("runtime", {})}
    if not merged.get("datasets"):
        raise ValueError("Manifest must include non-empty 'datasets'.")
    if not merged.get("methods"):
        raise ValueError("Manifest must include non-empty 'methods'.")
    return merged


def _build_llm_kwargs(
    manifest: dict[str, Any],
    method: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    runtime = manifest["runtime"]
    kwargs: dict[str, Any] = {
        "model": manifest["model"],
        "trust_remote_code": trust_remote_code,
        "max_model_len": int(runtime["max_model_len"]),
        "max_num_seqs": int(runtime["max_num_seqs"]),
        "tensor_parallel_size": int(runtime["tensor_parallel_size"]),
        "async_scheduling": int(runtime["async_scheduling"]),
        "disable_log_stats": bool(runtime["disable_log_stats"]),
        "seed": int(runtime["seed"]),
    }
    if method != "baseline":
        draft_model = manifest["draft_model"]
        if method == "eagle3" and manifest.get("eagle3_draft_model"):
            draft_model = manifest["eagle3_draft_model"]
        kwargs["speculative_config"] = {
            "method": method,
            "model": draft_model,
            "num_speculative_tokens": int(runtime["num_speculative_tokens"]),
            "draft_tensor_parallel_size": int(runtime["draft_tensor_parallel_size"]),
        }
    return kwargs


def _count_output_tokens(output_obj: Any) -> int:
    token_ids = getattr(output_obj, "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    text = getattr(output_obj, "text", "")
    return max(1, len(str(text).split()))


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def _snapshot_spec_metrics(llm: Any) -> dict[str, Any] | None:
    """Read speculative decoding counters from vLLM metrics snapshot.

    Returns None when metric snapshots are unavailable (for example if
    disable_log_stats=True).
    """
    try:
        metrics = llm.get_metrics()
    except Exception:
        return None

    snap: dict[str, Any] = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "num_accepted_tokens_per_pos": [],
    }
    for metric in metrics:
        name = str(getattr(metric, "name", ""))
        if name == "vllm:spec_decode_num_drafts":
            snap["num_drafts"] += int(getattr(metric, "value", 0))
        elif name == "vllm:spec_decode_num_draft_tokens":
            snap["num_draft_tokens"] += int(getattr(metric, "value", 0))
        elif name == "vllm:spec_decode_num_accepted_tokens":
            snap["num_accepted_tokens"] += int(getattr(metric, "value", 0))
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values = list(getattr(metric, "values", []) or [])
            cur = snap["num_accepted_tokens_per_pos"]
            if len(values) > len(cur):
                cur.extend([0] * (len(values) - len(cur)))
            for idx, val in enumerate(values):
                cur[idx] += int(val)
    return snap


def _diff_spec_metrics(
    start: dict[str, Any] | None, end: dict[str, Any] | None
) -> dict[str, Any] | None:
    if start is None or end is None:
        return None

    num_drafts = max(0, int(end["num_drafts"]) - int(start["num_drafts"]))
    num_draft_tokens = max(
        0, int(end["num_draft_tokens"]) - int(start["num_draft_tokens"])
    )
    num_accepted_tokens = max(
        0, int(end["num_accepted_tokens"]) - int(start["num_accepted_tokens"])
    )

    start_pos = list(start.get("num_accepted_tokens_per_pos", []))
    end_pos = list(end.get("num_accepted_tokens_per_pos", []))
    max_len = max(len(start_pos), len(end_pos))
    start_pos += [0] * (max_len - len(start_pos))
    end_pos += [0] * (max_len - len(end_pos))
    accepted_per_pos = [max(0, int(e) - int(s)) for s, e in zip(start_pos, end_pos)]

    draft_acceptance_rate = (
        float(num_accepted_tokens) / float(num_draft_tokens)
        if num_draft_tokens > 0
        else None
    )
    tau = (
        1.0 + (float(num_accepted_tokens) / float(num_drafts))
        if num_drafts > 0
        else None
    )
    acceptance_rate_per_pos = (
        [float(x) / float(num_drafts) for x in accepted_per_pos]
        if num_drafts > 0
        else []
    )
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "draft_acceptance_rate": draft_acceptance_rate,
        "tau": tau,
        "acceptance_rate_per_pos": acceptance_rate_per_pos,
    }


def _records_to_summary(
    records: list[dict[str, Any]],
    method_spec_metrics: dict[str, dict[str, Any] | None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    by_method: dict[str, dict[str, Any]] = {}

    for rec in records:
        key = (rec["method"], rec["dataset"])
        if key not in grouped:
            grouped[key] = {
                "method": rec["method"],
                "dataset": rec["dataset"],
                "dataset_name": rec["dataset_name"],
                "num_prompts": 0,
                "num_ok": 0,
                "num_error": 0,
                "total_elapsed_sec": 0.0,
                "total_output_tokens": 0,
            }
        g = grouped[key]
        g["num_prompts"] += 1
        if rec["status"] == "ok":
            g["num_ok"] += 1
            g["total_elapsed_sec"] += float(rec["elapsed_sec"])
            g["total_output_tokens"] += int(rec["output_token_count"])
        else:
            g["num_error"] += 1

    rows: list[dict[str, Any]] = []
    for g in grouped.values():
        tokens = max(1, int(g["total_output_tokens"]))
        elapsed = float(g["total_elapsed_sec"])
        tpot = elapsed / tokens if tokens > 0 else None
        tps = tokens / elapsed if elapsed > 0 else 0.0
        row = {
            **g,
            "time_per_output_token": tpot,
            "tokens_per_second": tps,
            "speedup_vs_baseline": None,
        }
        rows.append(row)

    baseline_tpot_by_dataset = {
        r["dataset"]: r["time_per_output_token"]
        for r in rows
        if r["method"] == "baseline" and r["time_per_output_token"]
    }
    for row in rows:
        if row["method"] == "baseline":
            continue
        base_tpot = baseline_tpot_by_dataset.get(row["dataset"])
        cur_tpot = row["time_per_output_token"]
        if base_tpot and cur_tpot and cur_tpot > 0:
            row["speedup_vs_baseline"] = base_tpot / cur_tpot

    for row in rows:
        method = row["method"]
        if method not in by_method:
            by_method[method] = {
                "method": method,
                "num_prompts": 0,
                "num_ok": 0,
                "num_error": 0,
                "total_elapsed_sec": 0.0,
                "total_output_tokens": 0,
            }
        m = by_method[method]
        m["num_prompts"] += row["num_prompts"]
        m["num_ok"] += row["num_ok"]
        m["num_error"] += row["num_error"]
        m["total_elapsed_sec"] += row["total_elapsed_sec"]
        m["total_output_tokens"] += row["total_output_tokens"]

    for m in by_method.values():
        tokens = max(1, int(m["total_output_tokens"]))
        elapsed = float(m["total_elapsed_sec"])
        m["time_per_output_token"] = elapsed / tokens if tokens > 0 else None
        m["tokens_per_second"] = tokens / elapsed if elapsed > 0 else 0.0

    if "baseline" in by_method:
        base = by_method["baseline"].get("time_per_output_token")
        if base and base > 0:
            for m in by_method.values():
                if m["method"] == "baseline":
                    m["speedup_vs_baseline"] = 1.0
                else:
                    cur = m.get("time_per_output_token")
                    m["speedup_vs_baseline"] = (base / cur) if cur else None

    for method, m in by_method.items():
        spec = method_spec_metrics.get(method) if method_spec_metrics else None
        if spec:
            m["tau"] = spec.get("tau")
            m["draft_acceptance_rate"] = spec.get("draft_acceptance_rate")
            m["num_drafts"] = spec.get("num_drafts")
            m["num_draft_tokens"] = spec.get("num_draft_tokens")
            m["num_accepted_tokens"] = spec.get("num_accepted_tokens")
            m["acceptance_rate_per_pos"] = spec.get("acceptance_rate_per_pos")
        else:
            m["tau"] = None
            m["draft_acceptance_rate"] = None
            m["num_drafts"] = None
            m["num_draft_tokens"] = None
            m["num_accepted_tokens"] = None
            m["acceptance_rate_per_pos"] = None

    return {"methods": by_method, "datasets": rows}, rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "method",
        "dataset",
        "dataset_name",
        "num_prompts",
        "num_ok",
        "num_error",
        "total_elapsed_sec",
        "total_output_tokens",
        "time_per_output_token",
        "tokens_per_second",
        "speedup_vs_baseline",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r["dataset"], r["method"])):
            w.writerow(row)


def _build_comparator_json(summary: dict[str, Any]) -> dict[str, Any]:
    datasets_out: dict[str, Any] = {}
    dflash_speedups: list[float] = []
    dflash_tau = None
    dflash_acceptance = None
    methods = summary.get("methods", {})
    if isinstance(methods, dict):
        dflash_method = methods.get("dflash", {})
        if isinstance(dflash_method, dict):
            dflash_tau = dflash_method.get("tau")
            dflash_acceptance = dflash_method.get("draft_acceptance_rate")
    for row in summary["datasets"]:
        if row["method"] != "dflash":
            continue
        speedup = row.get("speedup_vs_baseline")
        dname = row["dataset_name"]
        datasets_out[dname] = {
            "speedup": speedup,
            "tau": None,
        }
        if speedup is not None:
            dflash_speedups.append(float(speedup))
    avg = None
    if dflash_speedups:
        avg = sum(dflash_speedups) / len(dflash_speedups)
    return {
        "datasets": datasets_out,
        "average": {"speedup": avg, "tau": dflash_tau},
        "dflash_method": {
            "tau": dflash_tau,
            "draft_acceptance_rate": dflash_acceptance,
        },
    }


def _write_report_markdown(
    run_id: str, summary: dict[str, Any], path: Path
) -> None:
    lines = []
    lines.append(f"# Contribution Run Report: {run_id}")
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append(
        "| Method | Prompts | OK | Error | TPOT | TPS | Speedup vs Baseline | Tau | Draft Acceptance |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method, vals in sorted(summary["methods"].items()):
        tpot = vals.get("time_per_output_token")
        tps = vals.get("tokens_per_second")
        speedup = vals.get("speedup_vs_baseline")
        tau = vals.get("tau")
        acceptance = vals.get("draft_acceptance_rate")
        lines.append(
            f"| {method} | {vals['num_prompts']} | {vals['num_ok']} | {vals['num_error']} | "
            f"{tpot if tpot is not None else 'NA'} | {tps if tps is not None else 'NA'} | "
            f"{speedup if speedup is not None else 'NA'} | "
            f"{tau if tau is not None else 'NA'} | "
            f"{acceptance if acceptance is not None else 'NA'} |"
        )
    lines.append("")
    lines.append("## Dataset x Method")
    lines.append("")
    lines.append("| Dataset | Method | Prompts | OK | Error | TPOT | TPS | Speedup vs Baseline |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(summary["datasets"], key=lambda r: (r["dataset"], r["method"])):
        lines.append(
            f"| {row['dataset_name']} | {row['method']} | {row['num_prompts']} | "
            f"{row['num_ok']} | {row['num_error']} | "
            f"{row['time_per_output_token'] if row['time_per_output_token'] is not None else 'NA'} | "
            f"{row['tokens_per_second'] if row['tokens_per_second'] is not None else 'NA'} | "
            f"{row['speedup_vs_baseline'] if row['speedup_vs_baseline'] is not None else 'NA'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run contribution-focused DFlash validation matrix."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    parser.add_argument("--out-root", default="", help="Root output directory.")
    parser.add_argument("--run-id", default="", help="Optional run id.")
    parser.add_argument(
        "--trust-remote-code", dest="trust_remote_code", action="store_true"
    )
    parser.add_argument(
        "--no-trust-remote-code", dest="trust_remote_code", action="store_false"
    )
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = _load_manifest(manifest_path)

    out_root = Path(args.out_root) if args.out_root else (repo_root / "verification" / "outputs" / "contribution")
    run_id = args.run_id or (
        f"{manifest['name']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = out_root / run_id
    prompt_dir = run_dir / "prompt_records"
    summary_dir = run_dir / "summaries"
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    env_snapshot = {
        "generated_at_utc": _now_utc(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "argv": sys.argv,
        "repo_root": str(repo_root),
        "manifest_path": str(manifest_path),
        "packages": {
            "vllm": _pkg_version("vllm"),
            "transformers": _pkg_version("transformers"),
            "jax": _pkg_version("jax"),
            "torch": _pkg_version("torch"),
            "flax": _pkg_version("flax"),
        },
        "git": {
            "repo_root": _git_snapshot(repo_root),
            "vllm": _git_snapshot(repo_root / "deps" / "vllm"),
            "tpu_inference": _git_snapshot(repo_root / "deps" / "tpu-inference"),
        },
    }
    _write_json(run_dir / "env_snapshot.json", env_snapshot)
    _write_json(run_dir / "run_manifest.json", manifest)

    runtime = manifest["runtime"]
    methods: list[str] = list(manifest["methods"])
    datasets: list[str] = list(manifest["datasets"])
    max_samples = int(manifest["max_samples"])
    prompt_source = str(manifest["prompt_source"])
    use_chat_template = bool(manifest.get("use_chat_template", False))
    seed = int(runtime["seed"])

    print(f"Run id: {run_id}")
    print(f"Manifest: {manifest_path}")
    print(f"Methods: {methods}")
    print(f"Datasets: {datasets}")
    print(f"Prompt source: {prompt_source}")
    print(f"Use chat template: {use_chat_template}")
    print(f"Out dir: {run_dir}")
    if args.dry_run:
        print("DRY_RUN=1 set. Exiting before model execution.")
        return 0

    random.seed(seed)
    prompt_map: dict[str, list[str]] = {}
    for ds in datasets:
        prompt_map[ds] = load_prompts(repo_root, ds, max_samples, prompt_source)

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError("Failed to import vLLM for contribution run.") from exc

    sampling = manifest["sampling"]
    sampling_params = SamplingParams(
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        top_k=int(sampling["top_k"]),
        max_tokens=int(sampling["max_tokens"]),
        ignore_eos=bool(sampling["ignore_eos"]),
    )

    records: list[dict[str, Any]] = []
    method_spec_metrics: dict[str, dict[str, Any] | None] = {}
    file_handles: dict[tuple[str, str], Any] = {}
    try:
        for method in methods:
            for ds in datasets:
                path = prompt_dir / f"{method}.{ds}.jsonl"
                file_handles[(method, ds)] = open(path, "w", encoding="utf-8")

        for method in methods:
            print(f"\n=== Method: {method} ===")
            llm_kwargs = _build_llm_kwargs(manifest, method, args.trust_remote_code)
            llm = LLM(**llm_kwargs)
            tokenizer = _get_llm_tokenizer(llm) if use_chat_template else None
            if use_chat_template and tokenizer is None:
                print(
                    f"[WARN] chat template requested but tokenizer unavailable for method={method}.",
                    file=sys.stderr,
                )
            spec_metrics_before = _snapshot_spec_metrics(llm)
            try:
                warmup_n = int(runtime["warmup_prompts"])
                if warmup_n > 0:
                    first_ds = datasets[0]
                    source_prompts = prompt_map[first_ds]
                    warmup = source_prompts[:warmup_n]
                    if not warmup:
                        warmup = ["hi"]
                    warmup_inputs = [
                        _format_prompt_for_generation(p, tokenizer, use_chat_template)
                        for p in warmup
                    ]
                    _ = llm.generate(warmup_inputs, sampling_params)

                for ds in datasets:
                    dname = _canonical_name(ds)
                    prompts = prompt_map[ds]
                    for idx, prompt in enumerate(prompts):
                        generation_prompt = _format_prompt_for_generation(
                            prompt,
                            tokenizer,
                            use_chat_template,
                        )
                        rec: dict[str, Any] = {
                            "run_id": run_id,
                            "timestamp_utc": _now_utc(),
                            "method": method,
                            "dataset": ds,
                            "dataset_name": dname,
                            "sample_idx": idx,
                            "prompt": prompt,
                            "prompt_sha256": hashlib.sha256(
                                prompt.encode("utf-8")
                            ).hexdigest(),
                            "generation_prompt_sha256": hashlib.sha256(
                                generation_prompt.encode("utf-8")
                            ).hexdigest(),
                            "used_chat_template": bool(use_chat_template),
                            "status": "error",
                            "error_message": "",
                            "elapsed_sec": None,
                            "output_token_count": 0,
                            "tokens_per_second": 0.0,
                            "output_text": "",
                        }
                        t0 = time.perf_counter()
                        try:
                            outputs = llm.generate([generation_prompt], sampling_params)
                            elapsed = time.perf_counter() - t0
                            if not outputs or not getattr(outputs[0], "outputs", None):
                                raise RuntimeError("No outputs returned by model.")
                            out0 = outputs[0].outputs[0]
                            text = str(getattr(out0, "text", ""))
                            token_count = _count_output_tokens(out0)
                            rec["status"] = "ok"
                            rec["elapsed_sec"] = float(elapsed)
                            rec["output_token_count"] = int(token_count)
                            rec["tokens_per_second"] = (
                                float(token_count) / elapsed if elapsed > 0 else 0.0
                            )
                            rec["output_text"] = text
                        except Exception as exc:
                            elapsed = time.perf_counter() - t0
                            rec["elapsed_sec"] = float(elapsed)
                            rec["error_message"] = repr(exc)

                        records.append(rec)
                        fh = file_handles[(method, ds)]
                        fh.write(_safe_json(rec) + "\n")
                        fh.flush()
            finally:
                spec_metrics_after = _snapshot_spec_metrics(llm)
                method_spec_metrics[method] = _diff_spec_metrics(
                    spec_metrics_before, spec_metrics_after
                )
                if method != "baseline" and method_spec_metrics[method] is None:
                    print(
                        "[WARN] Spec metrics unavailable for method="
                        f"{method}. Set runtime.disable_log_stats=false "
                        "to collect tau/acceptance counters.",
                        file=sys.stderr,
                    )
                del llm
                gc.collect()
                sleep_sec = float(runtime["inter_method_sleep_sec"])
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
    finally:
        for fh in file_handles.values():
            fh.close()

    summary, dataset_rows = _records_to_summary(
        records, method_spec_metrics=method_spec_metrics
    )
    _write_json(summary_dir / "overall.json", summary)
    _write_csv(dataset_rows, summary_dir / "by_dataset_method.csv")
    _write_json(summary_dir / "comparator_dflash.json", _build_comparator_json(summary))
    _write_report_markdown(run_id, summary, summary_dir / "report.md")

    # Exit non-zero only if any prompt failed.
    errors = sum(1 for r in records if r["status"] != "ok")
    if errors:
        print(f"\nCompleted with prompt-level errors: {errors}")
        return 1
    print("\nCompleted successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


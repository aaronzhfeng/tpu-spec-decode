"""Pipeline profiling for DFlash speculative decoding on TPU.

Direction 1 experiment: instruments the draft-verify-accept loop with
fine-grained timing to identify where pipeline overhead lives.

Measures:
  - Draft forward latency (draft model forward pass)
  - Verify forward latency (target model forward pass on block)
  - Acceptance logic latency (argmax comparison + acceptance computation)
  - Context update latency (aux hidden state projection + buffer update)
  - Cache management latency (KV cache crop / allocation bookkeeping)
  - Host-device transfer latency (numpy ↔ jax array conversions)
  - Total per-step overhead (sum of above)

Usage:
    python benchmarks/pipeline_profiling.py \
        --target-model Qwen/Qwen3-4B \
        --draft-model z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k --max-samples 4 --max-new-tokens 256
"""

import argparse
import functools
import json
import math
import os
import re
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import AutoTokenizer
from vllm.config import ModelConfig
from vllm.model_executor.model_loader import LoadConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import (
    init_pp_distributed_environment,
)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.models.common.model_loader import get_flax_model
from tpu_inference.runner.kv_cache import create_kv_caches
from tpu_inference.utils import get_mesh_shape_product


# ---------------------------------------------------------------------------
# Reuse from standalone_dflash.py
# ---------------------------------------------------------------------------

class StandaloneVllmConfig:
    def __init__(self, target_model, draft_model, kv_cache_dtype="auto"):
        self.model_config = ModelConfig(target_model, trust_remote_code=True)
        self.model_config.dtype = jnp.bfloat16
        self.load_config = LoadConfig(load_format="auto")
        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
        self.quant_config = None
        self.additional_config = {}
        self.speculative_config = MagicMock()
        self.speculative_config.draft_model_config = ModelConfig(
            draft_model, trust_remote_code=True)
        self.speculative_config.draft_model_config.dtype = jnp.bfloat16
        self.speculative_config.method = "dflash"


def load_and_process_dataset(data_name):
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})
    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})
    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})
    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [fmt.format(**x)]})
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    return dataset


def next_padded_size(n):
    if n <= 16:
        return 16
    p = 16
    while p < n:
        p *= 2
    return p


def pad_context(ctx):
    T = ctx.shape[0]
    T_padded = next_padded_size(T)
    if T_padded == T:
        return ctx
    pad = np.zeros((T_padded - T, ctx.shape[1]), dtype=ctx.dtype)
    return np.concatenate([ctx, pad], axis=0)


def tpu_sync():
    jax.effects_barrier()


def make_attn_metadata(input_positions, seq_len, num_query_tokens, block_tables):
    return AttentionMetadata(
        input_positions=input_positions,
        block_tables=block_tables,
        seq_lens=jnp.array([seq_len], dtype=jnp.int32),
        query_start_loc=jnp.array([0, num_query_tokens], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
    )


def create_mesh(num_devices=1):
    devices = np.array(jax.local_devices()[:num_devices])
    device_mesh = devices.reshape((1, 1, 1, num_devices))
    return Mesh(device_mesh, axis_names=("data", "attn_dp", "expert", "model"))


def load_models(config, mesh):
    rng = jax.random.PRNGKey(42)
    print("[INFO] Loading target model...")
    with jax.default_device(jax.devices()[0]):
        (target_model_fn, target_logits_fn, target_combine_fn,
         _, target_state, _, _) = get_flax_model(
            config, rng, mesh, is_draft_model=False)
    print("[INFO] Loading draft model...")
    with jax.default_device(jax.devices()[0]):
        (draft_model_fn, draft_logits_fn, draft_combine_fn,
         _, draft_state, _, _) = get_flax_model(
            config, rng, mesh, is_draft_model=True)
    target_embed = getattr(target_state.model, "embed_tokens", None)
    if target_embed is not None:
        draft_state.model.embed_tokens = target_embed
    return (
        target_model_fn, target_logits_fn, target_combine_fn, target_state,
        draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    )


def allocate_target_kv_caches(config, mesh, max_length, page_size=16):
    hf = config.model_config.hf_config
    num_kv_heads = hf.num_key_value_heads
    head_dim_orig = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    head_dim = utils.get_padded_head_dim(head_dim_orig)
    num_pages = math.ceil(max_length / page_size) + 4
    num_layers = hf.num_hidden_layers
    kv_caches = create_kv_caches(
        num_blocks=num_pages, block_size=page_size, num_kv_heads=num_kv_heads,
        head_size=head_dim, mesh=mesh, layer_names=["layer"] * num_layers,
    )
    block_tables = jnp.arange(num_pages, dtype=jnp.int32)
    return kv_caches, block_tables


def allocate_draft_kv_caches(config, mesh, max_length):
    hf = config.speculative_config.draft_model_config.hf_config
    sharding_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)
    num_heads = utils.get_padded_num_heads(hf.num_attention_heads, sharding_size)
    head_dim_orig = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    head_dim = utils.get_padded_head_dim(head_dim_orig)
    max_kv_len = next_padded_size(max_length)
    num_layers = hf.num_hidden_layers
    caches = []
    for _ in range(num_layers):
        k = jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16)
        v = jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16)
        caches.extend([k, v])
    return caches


# ---------------------------------------------------------------------------
# Timer utility
# ---------------------------------------------------------------------------

class StepTimer:
    """Fine-grained timer for one spec-decode iteration."""

    def __init__(self):
        self.timings = {}
        self._start = None
        self._label = None

    def start(self, label):
        tpu_sync()
        self._label = label
        self._start = time.perf_counter()

    def stop(self):
        tpu_sync()
        elapsed = time.perf_counter() - self._start
        self.timings[self._label] = elapsed
        self._label = None
        self._start = None

    def total(self):
        return sum(self.timings.values())


# ---------------------------------------------------------------------------
# Profiled generate loop
# ---------------------------------------------------------------------------

def profiled_dflash_generate(
    target_model_fn, target_logits_fn, target_combine_fn, target_state,
    draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    target_kv_caches, block_tables,
    draft_kv_caches,
    input_ids_np, mask_token_id, block_size, max_new_tokens,
    eos_token_id, hidden_size, max_model_len,
):
    """DFlash generate loop with per-phase timing instrumentation."""
    num_input_tokens = len(input_ids_np)
    max_length = num_input_tokens + max_new_tokens

    output_ids = np.full(max_length + block_size, mask_token_id, dtype=np.int32)
    output_ids[:num_input_tokens] = input_ids_np

    # --- Prefill (timed as a whole) ---
    t_prefill_start = time.perf_counter()
    input_ids_jax = jnp.array(input_ids_np, dtype=jnp.int32)
    positions = jnp.arange(num_input_tokens, dtype=jnp.int32)
    metadata = make_attn_metadata(positions, num_input_tokens,
                                   num_input_tokens, block_tables)
    target_kv_caches, hidden_states, aux_hidden_states = target_model_fn(
        target_state, target_kv_caches, input_ids_jax, metadata,
        None, None, None, None, None, True, True,
    )
    last_hidden = hidden_states[-1:]
    logits = target_logits_fn(target_state, last_hidden, None)
    first_token = int(jnp.argmax(logits, axis=-1)[0])
    output_ids[num_input_tokens] = first_token

    if len(aux_hidden_states) > 0:
        raw = jnp.concatenate(aux_hidden_states, axis=-1)
        projected_all = draft_combine_fn(draft_state, raw)
    else:
        projected_all = None

    tpu_sync()
    prefill_time = time.perf_counter() - t_prefill_start

    # --- Context buffer init ---
    ctx_buf = np.zeros((max_model_len, hidden_size), dtype=np.float32)
    prev_ctx_len = 0

    if projected_all is not None:
        proj_np = np.asarray(projected_all, dtype=np.float32)
        n = min(num_input_tokens, max_model_len)
        ctx_buf[:n] = proj_np[:n]

    # --- Decode loop with profiling ---
    start = num_input_tokens
    acceptance_lengths = []
    draft_cache_len = 0
    prev_seq_len = 0
    draft_prefill_done = False

    step_timings = []  # list of StepTimer.timings dicts

    while start < max_length:
        timer = StepTimer()
        seq_len = start

        # (a) Cache crop
        timer.start("cache_mgmt")
        draft_cache_len = prev_seq_len
        timer.stop()

        # (b) Context update
        timer.start("ctx_update")
        num_new = seq_len - prev_ctx_len
        if num_new > 0 and projected_all is not None:
            proj_np = np.asarray(projected_all, dtype=np.float32)
            n_copy = min(num_new, len(proj_np))
            end = min(prev_ctx_len + n_copy, max_model_len)
            ctx_buf[prev_ctx_len:end] = proj_np[:n_copy]
            new_ctx_np = proj_np[:n_copy].copy()
            actual_ctx_count = n_copy
            prev_ctx_len = seq_len
            new_ctx_np = pad_context(new_ctx_np)
        else:
            actual_ctx_count = 0
            new_ctx_np = np.zeros((16, hidden_size), dtype=np.float32)
        timer.stop()

        # (c) Build noise block
        timer.start("host_device_xfer")
        next_token = output_ids[start]
        noise_ids = np.full(block_size, mask_token_id, dtype=np.int32)
        noise_ids[0] = next_token
        noise_ids_jax = jnp.array(noise_ids, dtype=jnp.int32)

        ctx_jax = jnp.array(new_ctx_np, dtype=jnp.bfloat16)
        cache_len_arr = jnp.array([draft_cache_len], dtype=jnp.int32)
        ctx_count_arr = jnp.array([actual_ctx_count], dtype=jnp.int32)
        target_hidden = (ctx_jax, cache_len_arr, ctx_count_arr)

        dummy_positions = jnp.arange(block_size, dtype=jnp.int32) + seq_len
        dummy_metadata = make_attn_metadata(
            dummy_positions, seq_len + block_size, block_size, block_tables)
        timer.stop()

        # (d) Draft forward
        timer.start("draft_forward")
        draft_kv_caches, draft_hidden, _ = draft_model_fn(
            draft_state, draft_kv_caches, noise_ids_jax,
            target_hidden, dummy_metadata,
        )
        tpu_sync()
        timer.stop()

        # Update draft state
        draft_cache_len = draft_cache_len + actual_ctx_count + block_size
        prev_seq_len = seq_len

        # (e) Sample draft tokens
        timer.start("draft_sample")
        draft_logits = target_logits_fn(
            target_state, draft_hidden[1:block_size], None)
        draft_tokens = np.array(jnp.argmax(draft_logits, axis=-1))
        timer.stop()

        # Fill block
        output_ids[start + 1 : start + block_size] = draft_tokens
        block_ids = jnp.array(
            output_ids[start : start + block_size], dtype=jnp.int32)

        if not draft_prefill_done:
            draft_prefill_done = True
            step_timings = []  # discard warmup step
            continue

        # (f) Verify forward
        timer.start("verify_forward")
        verify_positions = jnp.arange(start, start + block_size, dtype=jnp.int32)
        verify_metadata = make_attn_metadata(
            verify_positions, start + block_size, block_size, block_tables)
        target_kv_caches, verify_hidden, aux_hidden_states = target_model_fn(
            target_state, target_kv_caches, block_ids, verify_metadata,
            None, None, None, None, None, True, True,
        )
        tpu_sync()
        timer.stop()

        # (g) Verify logits + acceptance
        timer.start("acceptance")
        verify_logits = target_logits_fn(target_state, verify_hidden, None)
        posterior = np.array(jnp.argmax(verify_logits, axis=-1))
        block_np = np.array(block_ids)
        matches = (block_np[1:] == posterior[:-1]).astype(np.int32)
        acceptance_length = int(np.cumprod(matches).sum())
        output_ids[start : start + acceptance_length + 1] = \
            block_np[:acceptance_length + 1]
        output_ids[start + acceptance_length + 1] = posterior[acceptance_length]
        timer.stop()

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1

        # (h) Project aux hidden states for next iteration
        timer.start("aux_projection")
        if len(aux_hidden_states) > 0:
            raw = jnp.concatenate(aux_hidden_states, axis=-1)
            proj = draft_combine_fn(draft_state, raw)
            projected_all = proj[:acceptance_length + 1]
        else:
            projected_all = None
        tpu_sync()
        timer.stop()

        step_timings.append(timer.timings.copy())

        # Check stop
        if eos_token_id in output_ids[num_input_tokens : start + 1]:
            break

    # Trim output
    output_ids = output_ids[:max_length]
    valid_mask = output_ids != mask_token_id
    output_ids = output_ids[valid_mask]
    out_tokens = output_ids[num_input_tokens:]
    stop_idx = np.where(out_tokens == eos_token_id)[0]
    if len(stop_idx) > 0:
        output_ids = output_ids[:num_input_tokens + stop_idx[0] + 1]
    num_output_tokens = len(output_ids) - num_input_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        prefill_time=prefill_time,
        acceptance_lengths=acceptance_lengths,
        step_timings=step_timings,
    )


# ---------------------------------------------------------------------------
# Analysis & reporting
# ---------------------------------------------------------------------------

def analyze_timings(step_timings):
    """Aggregate per-step timings into summary statistics."""
    if not step_timings:
        return {}

    phases = list(step_timings[0].keys())
    summary = {}

    for phase in phases:
        values = [s[phase] for s in step_timings if phase in s]
        if values:
            summary[phase] = {
                "mean_ms": np.mean(values) * 1000,
                "median_ms": np.median(values) * 1000,
                "std_ms": np.std(values) * 1000,
                "min_ms": np.min(values) * 1000,
                "max_ms": np.max(values) * 1000,
                "total_ms": np.sum(values) * 1000,
                "count": len(values),
            }

    # Total per-step
    step_totals = [sum(s.values()) for s in step_timings]
    summary["step_total"] = {
        "mean_ms": np.mean(step_totals) * 1000,
        "median_ms": np.median(step_totals) * 1000,
        "std_ms": np.std(step_totals) * 1000,
        "min_ms": np.min(step_totals) * 1000,
        "max_ms": np.max(step_totals) * 1000,
        "total_ms": np.sum(step_totals) * 1000,
        "count": len(step_totals),
    }

    return summary


def print_profiling_report(summary, prefill_time, acceptance_lengths):
    """Print a formatted profiling report."""
    print("\n" + "=" * 70)
    print("PIPELINE PROFILING REPORT")
    print("=" * 70)

    print(f"\nPrefill time: {prefill_time * 1000:.1f} ms")

    if not summary:
        print("No step timings collected.")
        return

    tau = np.mean(acceptance_lengths) if acceptance_lengths else 0
    print(f"Tau: {tau:.2f}")
    print(f"Steps profiled: {summary['step_total']['count']}")

    # Phase breakdown
    print(f"\n{'Phase':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'% Total':>8}")
    print("-" * 56)

    step_mean = summary["step_total"]["mean_ms"]
    ordered_phases = [
        "draft_forward", "verify_forward", "draft_sample", "acceptance",
        "aux_projection", "ctx_update", "host_device_xfer", "cache_mgmt",
    ]

    for phase in ordered_phases:
        if phase in summary:
            s = summary[phase]
            pct = (s["mean_ms"] / step_mean * 100) if step_mean > 0 else 0
            print(f"  {phase:<18} {s['mean_ms']:>7.2f} {s['median_ms']:>7.2f} "
                  f"{s['std_ms']:>7.2f} {pct:>7.1f}%")

    print("-" * 56)
    s = summary["step_total"]
    print(f"  {'TOTAL':<18} {s['mean_ms']:>7.2f} {s['median_ms']:>7.2f} "
          f"{s['std_ms']:>7.2f} {'100.0':>7}%")

    # Compute overhead = everything except draft_forward + verify_forward
    compute_phases = ["draft_forward", "verify_forward"]
    overhead_phases = [p for p in ordered_phases if p not in compute_phases]

    compute_mean = sum(summary[p]["mean_ms"] for p in compute_phases if p in summary)
    overhead_mean = sum(summary[p]["mean_ms"] for p in overhead_phases if p in summary)

    print(f"\n  Core compute (draft + verify): {compute_mean:.2f} ms "
          f"({compute_mean/step_mean*100:.1f}%)")
    print(f"  Overhead (everything else):    {overhead_mean:.2f} ms "
          f"({overhead_mean/step_mean*100:.1f}%)")

    # Bar chart
    print(f"\nPer-step time breakdown (mean):")
    max_bar = 40
    for phase in ordered_phases:
        if phase in summary:
            s = summary[phase]
            bar_len = int(s["mean_ms"] / step_mean * max_bar) if step_mean > 0 else 0
            bar = "#" * bar_len
            print(f"  {phase:<18} {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline profiling for DFlash TPU speculative decoding")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(42)
    mesh = create_mesh()
    init_pp_distributed_environment(
        ip="", rank=0, world_size=1, device=jax.devices()[0], need_pp=False)
    print(f"[INFO] JAX devices: {jax.device_count()}")

    config = StandaloneVllmConfig(args.target_model, args.draft_model)
    draft_hf = config.speculative_config.draft_model_config.hf_config
    dflash_config = getattr(draft_hf, "dflash_config", {})
    mask_token_id = dflash_config.get("mask_token_id", 0)
    block_size = args.block_size or getattr(draft_hf, "block_size", 16)
    hidden_size = draft_hf.hidden_size

    (target_model_fn, target_logits_fn, target_combine_fn, target_state,
     draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    ) = load_models(config, mesh)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    eos_token_id = tokenizer.eos_token_id or 0

    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))

    max_length = args.max_model_len
    page_size = 16
    all_step_timings = []
    all_acceptance = []
    all_prefill_times = []

    for idx in range(len(dataset)):
        instance = dataset[idx]
        prompt = instance["turns"][0]
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        input_ids = tokenizer.encode(input_text)
        input_ids_np = np.array(input_ids, dtype=np.int32)

        print(f"\n[{idx+1}/{len(dataset)}] Prompt length: {len(input_ids)}")

        target_kv, bt = allocate_target_kv_caches(config, mesh, max_length, page_size)
        draft_kv = allocate_draft_kv_caches(config, mesh, max_length)

        result = profiled_dflash_generate(
            target_model_fn, target_logits_fn, target_combine_fn, target_state,
            draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
            target_kv, bt, draft_kv,
            input_ids_np, mask_token_id, block_size, args.max_new_tokens,
            eos_token_id, hidden_size, max_length,
        )

        tau = np.mean(result.acceptance_lengths) if result.acceptance_lengths else 0
        print(f"  Tokens: {result.num_output_tokens}, tau={tau:.2f}, "
              f"prefill={result.prefill_time*1000:.1f}ms, "
              f"steps={len(result.step_timings)}")

        if idx >= args.warmup:
            all_step_timings.extend(result.step_timings)
            all_acceptance.extend(result.acceptance_lengths)
            all_prefill_times.append(result.prefill_time)

    # --- Aggregate report ---
    summary = analyze_timings(all_step_timings)
    avg_prefill = np.mean(all_prefill_times) if all_prefill_times else 0
    print_profiling_report(summary, avg_prefill, all_acceptance)

    # --- Save JSON ---
    if args.output_json:
        output = {
            "config": {
                "target_model": args.target_model,
                "draft_model": args.draft_model,
                "dataset": args.dataset,
                "block_size": block_size,
                "max_new_tokens": args.max_new_tokens,
                "warmup": args.warmup,
                "num_samples": len(dataset),
                "measured_samples": len(dataset) - args.warmup,
            },
            "summary": {k: v for k, v in summary.items()},
            "avg_prefill_ms": avg_prefill * 1000,
            "tau": float(np.mean(all_acceptance)) if all_acceptance else 0,
            "total_steps": len(all_step_timings),
            "per_step_timings": all_step_timings,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

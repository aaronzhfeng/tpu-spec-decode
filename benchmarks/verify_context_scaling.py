"""Experiment B: Does TPU verify stay flat across K at longer contexts?

Measures target model forward-pass latency as a function of BOTH:
  - K (query token count): 16, 64, 128
  - L (context length in KV cache): 64, 256, 512, 1024

This answers whether the flat-K property (Doc 33, 37) holds when attention
over the KV cache becomes a larger fraction of total forward pass time.

GPU results (Doc 42) show attention scales ~2.5x at K=128 vs K=16 at
KV=1024. If TPU attention also scales, the flat-K claim needs a
context-length caveat.

Methodology:
  - Prefill to context length L by repeating prompt tokens
  - Measure verify latency for K query tokens (fresh KV per trial)
  - tpu_sync() inside timing window (same as Doc 33/37 benchmarks)

Usage:
    python benchmarks/verify_context_scaling.py \
        --target-model Qwen/Qwen3-4B \
        --draft-model z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k \
        --trials 20 --warmup 5 \
        --output-json results/verify_context_scaling.json
"""

import argparse
import json
import math
import os
import time
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from jax.sharding import Mesh
from transformers import AutoTokenizer
from vllm.config import ModelConfig
from vllm.model_executor.model_loader import LoadConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import (
    init_pp_distributed_environment,
)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.common.model_loader import get_flax_model
from tpu_inference.runner.kv_cache import create_kv_caches
from tpu_inference.utils import get_mesh_shape_product


# ---------------------------------------------------------------------------
# Shared infra (same as amortized_verification.py)
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
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    return dataset


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
    return Mesh(devices.reshape((1, 1, 1, num_devices)),
                axis_names=("data", "attn_dp", "expert", "model"))


def load_models(config, mesh):
    rng = jax.random.PRNGKey(42)
    print("[INFO] Loading target model...")
    with jax.default_device(jax.devices()[0]):
        (target_model_fn, target_logits_fn, target_combine_fn,
         _, target_state, _, _) = get_flax_model(
            config, rng, mesh, is_draft_model=False)
    return target_model_fn, target_logits_fn, target_state


def allocate_target_kv_caches(config, mesh, max_length, page_size=16):
    hf = config.model_config.hf_config
    num_kv_heads = hf.num_key_value_heads
    head_dim = utils.get_padded_head_dim(
        getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads))
    num_pages = math.ceil(max_length / page_size) + 4
    kv_caches = create_kv_caches(
        num_blocks=num_pages, block_size=page_size, num_kv_heads=num_kv_heads,
        head_size=head_dim, mesh=mesh,
        layer_names=["layer"] * hf.num_hidden_layers)
    return kv_caches, jnp.arange(num_pages, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Experiment B: Context length × K sweep
# ---------------------------------------------------------------------------

def build_prefill_tokens(base_tokens, target_length):
    """Repeat base tokens to reach target_length."""
    if len(base_tokens) >= target_length:
        return base_tokens[:target_length]
    repeats = math.ceil(target_length / len(base_tokens))
    repeated = np.tile(base_tokens, repeats)
    return repeated[:target_length]


def run_context_k_sweep(
    target_model_fn, target_state,
    config, mesh, base_input_ids,
    context_lengths, k_values,
    num_trials, num_warmup, max_model_len,
):
    """Measure verify latency at each (context_length, K) pair."""
    page_size = 16
    results = {}

    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"  CONTEXT LENGTH: {ctx_len} tokens")
        print(f"{'='*60}")

        # Build prefill tokens of exactly ctx_len
        prefill_ids = build_prefill_tokens(base_input_ids, ctx_len)
        prefill_ids_jax = jnp.array(prefill_ids, dtype=jnp.int32)
        prefill_positions = jnp.arange(ctx_len, dtype=jnp.int32)

        results[ctx_len] = {}

        for n_query in k_values:
            print(f"\n  --- K={n_query} query tokens (context={ctx_len}) ---")

            # Create verify metadata
            dummy_ids = jnp.zeros(n_query, dtype=jnp.int32)
            seq_len_after = ctx_len + n_query
            verify_pos = jnp.arange(ctx_len, seq_len_after, dtype=jnp.int32)

            # Warmup: JIT compile for this (ctx_len, K) shape
            for w in range(num_warmup):
                target_kv_w, bt_w = allocate_target_kv_caches(
                    config, mesh, max_model_len, page_size)

                # Prefill
                prefill_meta = make_attn_metadata(
                    prefill_positions, ctx_len, ctx_len, bt_w)
                target_kv_w, _, _ = target_model_fn(
                    target_state, target_kv_w, prefill_ids_jax, prefill_meta,
                    None, None, None, None, None, True, True)
                tpu_sync()

                # Verify
                verify_meta = make_attn_metadata(
                    verify_pos, seq_len_after, n_query, bt_w)
                target_kv_w, _, _ = target_model_fn(
                    target_state, target_kv_w, dummy_ids, verify_meta,
                    None, None, None, None, None, True, True)
                tpu_sync()
                print(f"    warmup {w+1}/{num_warmup}")

            # Timed trials
            latencies = []
            for t in range(num_trials):
                # Fresh KV + prefill each trial
                target_kv_t, bt_t = allocate_target_kv_caches(
                    config, mesh, max_model_len, page_size)

                prefill_meta = make_attn_metadata(
                    prefill_positions, ctx_len, ctx_len, bt_t)
                target_kv_t, _, _ = target_model_fn(
                    target_state, target_kv_t, prefill_ids_jax, prefill_meta,
                    None, None, None, None, None, True, True)
                tpu_sync()

                # Time ONLY the verify forward pass
                verify_meta = make_attn_metadata(
                    verify_pos, seq_len_after, n_query, bt_t)

                t0 = time.perf_counter()
                target_kv_t, _, _ = target_model_fn(
                    target_state, target_kv_t, dummy_ids, verify_meta,
                    None, None, None, None, None, True, True)
                tpu_sync()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            mean_ms = float(np.mean(latencies))
            std_ms = float(np.std(latencies))
            per_token_ms = mean_ms / n_query

            results[ctx_len][n_query] = {
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "per_token_ms": per_token_ms,
                "latencies": [float(x) for x in latencies],
            }

            print(f"    K={n_query:>4}: {mean_ms:.2f} ± {std_ms:.2f} ms  "
                  f"({per_token_ms:.3f} ms/token)")

    return results


def print_summary(results, context_lengths, k_values):
    """Print comparison tables."""

    # Table 1: Raw latencies
    print(f"\n{'='*70}")
    print("VERIFY LATENCY (ms) — rows=context, cols=K")
    print(f"{'='*70}")

    header = f"  {'Context':>8} |"
    for k in k_values:
        header += f" {'K='+str(k):>14} |"
    print(header)
    print("  " + "-" * (10 + 17 * len(k_values)))

    for ctx in context_lengths:
        row = f"  {ctx:>8} |"
        for k in k_values:
            r = results[ctx][k]
            row += f" {r['mean_ms']:>8.2f} ± {r['std_ms']:.1f} |"
        print(row)

    # Table 2: K=128/K=16 ratio at each context length
    if 16 in k_values and 128 in k_values:
        print(f"\n{'='*70}")
        print("K=128 vs K=16 RATIO — the critical metric")
        print(f"{'='*70}")
        print(f"  {'Context':>8} | {'K=16 (ms)':>10} | {'K=128 (ms)':>11} | "
              f"{'Ratio':>8} | {'Verdict':>20}")
        print("  " + "-" * 70)

        for ctx in context_lengths:
            k16 = results[ctx][16]["mean_ms"]
            k128 = results[ctx][128]["mean_ms"]
            ratio = k128 / k16 if k16 > 0 else 0

            if ratio < 1.15:
                verdict = "FLAT (< 1.15x)"
            elif ratio < 1.5:
                verdict = "MILD SCALING"
            elif ratio < 2.0:
                verdict = "MODERATE SCALING"
            else:
                verdict = "LINEAR SCALING"

            print(f"  {ctx:>8} | {k16:>10.2f} | {k128:>11.2f} | "
                  f"{ratio:>7.2f}x | {verdict:>20}")

    # Table 3: Compare with GPU (Doc 42)
    gpu_attn_ratios = {64: 1.04, 256: 1.58, 512: 1.98, 1024: 2.51}

    print(f"\n{'='*70}")
    print("TPU vs GPU ATTENTION COMPARISON (K=128/K=16)")
    print(f"{'='*70}")
    print(f"  {'Context':>8} | {'TPU ratio':>10} | {'GPU ratio':>10} | {'Contrast':>12}")
    print("  " + "-" * 50)

    for ctx in context_lengths:
        if ctx in gpu_attn_ratios and 16 in k_values and 128 in k_values:
            tpu_ratio = results[ctx][128]["mean_ms"] / results[ctx][16]["mean_ms"]
            gpu_ratio = gpu_attn_ratios[ctx]
            contrast = gpu_ratio / tpu_ratio if tpu_ratio > 0 else 0

            print(f"  {ctx:>8} | {tpu_ratio:>9.2f}x | {gpu_ratio:>9.2f}x | "
                  f"{contrast:>10.1f}x worse")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: TPU verify latency vs context length and K")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=20,
                        help="Timed trials per (context, K) pair")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations per (context, K) pair")
    parser.add_argument("--context-lengths", type=str, default="64,256,512,1024",
                        help="Comma-separated context lengths to test")
    parser.add_argument("--k-values", type=str, default="16,64,128",
                        help="Comma-separated K (query token) values to test")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]

    np.random.seed(42)
    mesh = create_mesh()
    init_pp_distributed_environment(
        ip="", rank=0, world_size=1, device=jax.devices()[0], need_pp=False)

    config = StandaloneVllmConfig(args.target_model, args.draft_model)

    print(f"[INFO] JAX devices: {jax.device_count()}")
    print(f"[INFO] Context lengths: {context_lengths}")
    print(f"[INFO] K values: {k_values}")
    print(f"[INFO] Trials: {args.trials}, Warmup: {args.warmup}")

    target_model_fn, target_logits_fn, target_state = load_models(config, mesh)

    # Get base tokens from dataset for prefill padding
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    dataset = load_and_process_dataset(args.dataset)
    instance = dataset[0]
    prompt = instance["turns"][0]
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    base_input_ids = np.array(tokenizer.encode(input_text), dtype=np.int32)
    print(f"[INFO] Base prompt: {len(base_input_ids)} tokens (will tile to reach each context length)")

    # Run the sweep
    results = run_context_k_sweep(
        target_model_fn, target_state,
        config, mesh, base_input_ids,
        context_lengths, k_values,
        args.trials, args.warmup, args.max_model_len,
    )

    # Print summary tables
    print_summary(results, context_lengths, k_values)

    # Save
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)),
                    exist_ok=True)
        output = {
            "context_lengths": context_lengths,
            "k_values": k_values,
            "trials": args.trials,
            "warmup": args.warmup,
            "device": str(jax.devices()[0]),
            "results": {
                str(ctx): {
                    str(k): {key: val for key, val in v.items() if key != "latencies"}
                    for k, v in ctx_results.items()
                }
                for ctx, ctx_results in results.items()
            },
            "raw_latencies": {
                str(ctx): {
                    str(k): v["latencies"]
                    for k, v in ctx_results.items()
                }
                for ctx, ctx_results in results.items()
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

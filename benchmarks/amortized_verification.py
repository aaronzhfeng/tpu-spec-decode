"""Amortized verification: does verifying more tokens at once cost less
per token on TPU?

Measures target model forward-pass latency as a function of query token
count (16, 32, 48, 64, 96, 128).  If latency is roughly flat, larger
verification blocks get the same work done for the same cost — the MXU
128×128 tiles amortize the compute.

Part 1: Microbenchmark — raw verify latency vs query count
Part 2: Multi-block speculation — draft 2+ blocks, verify together

Usage:
    python benchmarks/amortized_verification.py \
        --target-model Qwen/Qwen3-4B \
        --draft-model z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k --max-samples 3
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
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.models.common.model_loader import get_flax_model
from tpu_inference.runner.kv_cache import create_kv_caches
from tpu_inference.utils import get_mesh_shape_product


# ---------------------------------------------------------------------------
# Shared infra (same as other benchmarks)
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


def get_dflash_aux_layer_indices(config):
    draft_hf = config.speculative_config.draft_model_config.hf_config
    dflash_config = getattr(draft_hf, "dflash_config", {})
    target_layer_ids = dflash_config.get("target_layer_ids", None)
    if target_layer_ids is not None:
        return tuple(target_layer_ids)
    num_target = config.model_config.hf_config.num_hidden_layers
    num_draft = draft_hf.num_hidden_layers
    num_selected = getattr(draft_hf, "num_target_layers", num_draft)
    if num_selected == 1:
        return (num_target // 2,)
    start = 1
    end = num_target - 3
    span = end - start
    return tuple(
        int(round(start + (i * span) / (num_selected - 1)))
        for i in range(num_selected)
    )


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
    return (target_model_fn, target_logits_fn, target_combine_fn, target_state,
            draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state)


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


def allocate_draft_kv_caches(config, mesh, max_length):
    hf = config.speculative_config.draft_model_config.hf_config
    sharding_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)
    num_heads = utils.get_padded_num_heads(hf.num_attention_heads, sharding_size)
    head_dim = utils.get_padded_head_dim(
        getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads))
    max_kv_len = max_length
    # Round to next power of 2 for padding
    p = 16
    while p < max_kv_len:
        p *= 2
    max_kv_len = p
    caches = []
    for _ in range(hf.num_hidden_layers):
        caches.extend([
            jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16),
            jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16),
        ])
    return caches


# ===================================================================
# Part 1: Verify latency microbenchmark
# ===================================================================

def run_verify_microbenchmark(
    target_model_fn, target_logits_fn, target_state,
    config, mesh, tokenizer, dataset, max_model_len,
    query_counts, num_trials, num_warmup,
):
    """Time the target model forward pass for different query token counts.

    For each query count N:
    1. Prefill a prompt to populate KV cache
    2. Warm up (JIT compile for this shape)
    3. Time num_trials forward passes with N query tokens
    """
    page_size = 16
    results = {}

    # Use the first dataset sample for all tests
    instance = dataset[0]
    prompt = instance["turns"][0]
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    input_ids = np.array(tokenizer.encode(input_text), dtype=np.int32)
    prompt_len = len(input_ids)

    print(f"\n  Prompt length: {prompt_len} tokens")
    print(f"  Query counts to test: {query_counts}")
    print(f"  Trials per count: {num_trials} (warmup: {num_warmup})")

    for n_query in query_counts:
        print(f"\n  --- Testing {n_query} query tokens ---")

        # Allocate fresh KV cache for each test
        target_kv, bt = allocate_target_kv_caches(config, mesh, max_model_len, page_size)

        # Prefill
        input_ids_jax = jnp.array(input_ids, dtype=jnp.int32)
        positions = jnp.arange(prompt_len, dtype=jnp.int32)
        metadata = make_attn_metadata(positions, prompt_len, prompt_len, bt)
        target_kv, hidden, _ = target_model_fn(
            target_state, target_kv, input_ids_jax, metadata,
            None, None, None, None, None, True, True)
        tpu_sync()

        # Create dummy query tokens (simulate verification of N tokens)
        dummy_ids = jnp.zeros(n_query, dtype=jnp.int32)
        seq_len_after = prompt_len + n_query
        verify_pos = jnp.arange(prompt_len, seq_len_after, dtype=jnp.int32)
        verify_meta = make_attn_metadata(verify_pos, seq_len_after, n_query, bt)

        # Warmup: JIT compile for this shape
        for w in range(num_warmup):
            # Need fresh KV for each warmup since verify mutates it
            target_kv_w, _ = allocate_target_kv_caches(config, mesh, max_model_len, page_size)
            # Re-prefill
            target_kv_w, _, _ = target_model_fn(
                target_state, target_kv_w, input_ids_jax, metadata,
                None, None, None, None, None, True, True)
            tpu_sync()
            # Verify
            target_kv_w, _, _ = target_model_fn(
                target_state, target_kv_w, dummy_ids, verify_meta,
                None, None, None, None, None, True, True)
            tpu_sync()
            print(f"    warmup {w+1}/{num_warmup} done")

        # Timed trials
        latencies = []
        for t in range(num_trials):
            # Fresh KV + prefill for each trial
            target_kv_t, _ = allocate_target_kv_caches(config, mesh, max_model_len, page_size)
            target_kv_t, _, _ = target_model_fn(
                target_state, target_kv_t, input_ids_jax, metadata,
                None, None, None, None, None, True, True)
            tpu_sync()

            # Time the verify forward pass
            t0 = time.perf_counter()
            target_kv_t, verify_hidden, _ = target_model_fn(
                target_state, target_kv_t, dummy_ids, verify_meta,
                None, None, None, None, None, True, True)
            tpu_sync()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        mean_ms = np.mean(latencies)
        std_ms = np.std(latencies)
        per_token_ms = mean_ms / n_query
        print(f"    {n_query:>4} tokens: {mean_ms:.2f} ± {std_ms:.2f} ms  "
              f"({per_token_ms:.3f} ms/token)")

        results[n_query] = {
            "mean_ms": float(mean_ms),
            "std_ms": float(std_ms),
            "per_token_ms": float(per_token_ms),
            "latencies": [float(x) for x in latencies],
        }

    return results


# ===================================================================
# Part 2: Multi-block speculation end-to-end
# ===================================================================

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
    return np.concatenate([ctx, np.zeros((T_padded - T, ctx.shape[1]), dtype=ctx.dtype)])


def run_multiblock_experiment(
    target_model_fn, target_logits_fn, target_combine_fn, target_state,
    draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    config, mesh, tokenizer, dataset, max_model_len, block_size,
    mask_token_id, hidden_size, max_new_tokens, warmup_samples,
    dflash_aux_layers, num_blocks_list,
):
    """Run speculative decoding with 1-block vs multi-block verification.

    For num_blocks=2: draft block 1, speculatively draft block 2
    (assuming block 1 fully accepted), verify both in single forward pass.
    """
    page_size = 16
    eos_token_id = tokenizer.eos_token_id or 0

    all_results = {}

    for num_blocks in num_blocks_list:
        verify_width = block_size * num_blocks
        print(f"\n  === {num_blocks}-block speculation "
              f"(verify width: {verify_width} tokens) ===")

        step_times = []
        acceptance_lengths = []
        total_tokens_generated = 0

        for idx in range(len(dataset)):
            instance = dataset[idx]
            prompt = instance["turns"][0]
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            input_ids = np.array(tokenizer.encode(input_text), dtype=np.int32)

            target_kv, bt = allocate_target_kv_caches(
                config, mesh, max_model_len, page_size)
            draft_kv = allocate_draft_kv_caches(config, mesh, max_model_len)

            # Prefill
            input_ids_jax = jnp.array(input_ids, dtype=jnp.int32)
            positions = jnp.arange(len(input_ids), dtype=jnp.int32)
            metadata = make_attn_metadata(
                positions, len(input_ids), len(input_ids), bt)
            target_kv, hidden_states, all_aux = target_model_fn(
                target_state, target_kv, input_ids_jax, metadata,
                None, None, None, None, None, True, True)

            # all_aux already contains only the DFlash aux layers (in order)
            dflash_aux = list(all_aux)
            logits = target_logits_fn(target_state, hidden_states[-1:], None)
            first_token = int(jnp.argmax(logits, axis=-1)[0])

            output_ids = np.full(
                len(input_ids) + max_new_tokens + verify_width * 2,
                mask_token_id, dtype=np.int32)
            output_ids[:len(input_ids)] = input_ids
            output_ids[len(input_ids)] = first_token

            if len(dflash_aux) > 0:
                raw = jnp.concatenate(dflash_aux, axis=-1)
                projected_all = draft_combine_fn(draft_state, raw)
            else:
                projected_all = None
            tpu_sync()

            ctx_buf = np.zeros((max_model_len, hidden_size), dtype=np.float32)
            prev_ctx_len = 0
            if projected_all is not None:
                proj_np = np.asarray(projected_all, dtype=np.float32)
                n = min(len(input_ids), max_model_len)
                ctx_buf[:n] = proj_np[:n]

            start = len(input_ids)
            max_length = len(input_ids) + max_new_tokens
            draft_cache_len = 0
            prev_seq_len = 0
            draft_prefill_done = False
            step_count = 0

            while start < max_length:
                t_step_start = time.perf_counter()
                seq_len = start
                draft_cache_len = prev_seq_len

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

                # --- Draft block 1 ---
                next_token = output_ids[start]
                noise_ids = np.full(block_size, mask_token_id, dtype=np.int32)
                noise_ids[0] = next_token
                noise_ids_jax = jnp.array(noise_ids, dtype=jnp.int32)

                ctx_jax = jnp.array(new_ctx_np, dtype=jnp.bfloat16)
                target_hidden = (ctx_jax,
                                 jnp.array([draft_cache_len], dtype=jnp.int32),
                                 jnp.array([actual_ctx_count], dtype=jnp.int32))
                dummy_pos = jnp.arange(block_size, dtype=jnp.int32) + seq_len
                dummy_meta = make_attn_metadata(
                    dummy_pos, seq_len + block_size, block_size, bt)

                draft_kv, draft_hidden, _ = draft_model_fn(
                    draft_state, draft_kv, noise_ids_jax,
                    target_hidden, dummy_meta)
                dc_len_after_b1 = draft_cache_len + actual_ctx_count + block_size

                # Block 1 draft tokens
                draft_logits_b1 = target_logits_fn(
                    target_state, draft_hidden[1:block_size], None)
                draft_tokens_b1 = np.array(jnp.argmax(draft_logits_b1, axis=-1))
                output_ids[start + 1: start + block_size] = draft_tokens_b1

                # --- Draft additional blocks (speculative, assuming previous accepted) ---
                all_block_draft_tokens = [draft_tokens_b1]

                for blk in range(1, num_blocks):
                    blk_start = start + blk * block_size
                    if blk_start >= max_length:
                        break
                    blk_next_token = output_ids[blk_start]
                    blk_noise = np.full(block_size, mask_token_id, dtype=np.int32)
                    blk_noise[0] = blk_next_token
                    blk_noise_jax = jnp.array(blk_noise, dtype=jnp.int32)

                    # No new context for speculative blocks — use empty
                    blk_ctx = jnp.zeros((16, hidden_size), dtype=jnp.bfloat16)
                    blk_hidden = (blk_ctx,
                                  jnp.array([dc_len_after_b1 + (blk - 1) * block_size],
                                            dtype=jnp.int32),
                                  jnp.array([0], dtype=jnp.int32))
                    blk_pos = jnp.arange(block_size, dtype=jnp.int32) + blk_start
                    blk_meta = make_attn_metadata(
                        blk_pos, blk_start + block_size, block_size, bt)

                    draft_kv, blk_draft_hidden, _ = draft_model_fn(
                        draft_state, draft_kv, blk_noise_jax,
                        blk_hidden, blk_meta)

                    blk_logits = target_logits_fn(
                        target_state, blk_draft_hidden[1:block_size], None)
                    blk_draft = np.array(jnp.argmax(blk_logits, axis=-1))
                    output_ids[blk_start + 1: blk_start + block_size] = blk_draft
                    all_block_draft_tokens.append(blk_draft)

                actual_num_blocks = 1 + len(all_block_draft_tokens) - 1
                total_verify_tokens = actual_num_blocks * block_size

                if not draft_prefill_done:
                    draft_prefill_done = True
                    prev_seq_len = seq_len
                    tpu_sync()
                    continue

                # --- Verify all blocks in one forward pass ---
                block_ids = jnp.array(
                    output_ids[start: start + total_verify_tokens],
                    dtype=jnp.int32)
                verify_pos = jnp.arange(
                    start, start + total_verify_tokens, dtype=jnp.int32)
                verify_meta = make_attn_metadata(
                    verify_pos, start + total_verify_tokens,
                    total_verify_tokens, bt)
                target_kv, verify_hidden, all_aux = target_model_fn(
                    target_state, target_kv, block_ids, verify_meta,
                    None, None, None, None, None, True, True)

                # Compute acceptance across all blocks
                verify_logits = target_logits_fn(
                    target_state, verify_hidden, None)
                full_posterior = np.array(jnp.argmax(verify_logits, axis=-1))

                block_np = np.array(block_ids)
                matches = (block_np[1:] == full_posterior[:-1]).astype(np.int32)
                acceptance_length = int(np.cumprod(matches).sum())

                tpu_sync()
                t_step_end = time.perf_counter()

                # Update state
                output_ids[start: start + acceptance_length + 1] = \
                    block_np[:acceptance_length + 1]
                output_ids[start + acceptance_length + 1] = \
                    full_posterior[acceptance_length]

                # DFlash context update (all_aux already is DFlash aux layers)
                dflash_aux = list(all_aux)
                if len(dflash_aux) > 0:
                    raw = jnp.concatenate(dflash_aux, axis=-1)
                    proj = draft_combine_fn(draft_state, raw)
                    projected_all = proj[:acceptance_length + 1]
                else:
                    projected_all = None

                if idx >= warmup_samples:
                    step_times.append((t_step_end - t_step_start) * 1000)
                    acceptance_lengths.append(acceptance_length + 1)

                start += acceptance_length + 1
                prev_seq_len = seq_len
                step_count += 1
                total_tokens_generated += acceptance_length + 1

                if eos_token_id in output_ids[len(input_ids): start + 1]:
                    break

            tau = np.mean(acceptance_lengths[-step_count:]) if step_count else 0
            print(f"    [{idx+1}/{len(dataset)}] steps={step_count}, "
                  f"tau={tau:.2f}")

        avg_step_ms = np.mean(step_times) if step_times else 0
        avg_tau = np.mean(acceptance_lengths) if acceptance_lengths else 0
        throughput = avg_tau / avg_step_ms * 1000 if avg_step_ms > 0 else 0

        print(f"\n    {num_blocks}-block: tau={avg_tau:.2f}, "
              f"step={avg_step_ms:.2f}ms, "
              f"throughput={throughput:.1f} tok/s")

        all_results[num_blocks] = {
            "num_blocks": num_blocks,
            "verify_width": verify_width,
            "avg_tau": float(avg_tau),
            "avg_step_ms": float(avg_step_ms),
            "throughput_tok_s": float(throughput),
            "total_steps": len(step_times),
            "step_times": [float(x) for x in step_times],
            "acceptance_lengths": [int(x) for x in acceptance_lengths],
        }

    return all_results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Amortized verification")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--micro-trials", type=int, default=10)
    parser.add_argument("--micro-warmup", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(42)
    mesh = create_mesh()
    init_pp_distributed_environment(
        ip="", rank=0, world_size=1, device=jax.devices()[0], need_pp=False)

    config = StandaloneVllmConfig(args.target_model, args.draft_model)
    draft_hf = config.speculative_config.draft_model_config.hf_config
    dflash_config = getattr(draft_hf, "dflash_config", {})
    mask_token_id = dflash_config.get("mask_token_id", 0)
    block_size = getattr(draft_hf, "block_size", 16)
    hidden_size = draft_hf.hidden_size

    print(f"[INFO] JAX devices: {jax.device_count()}")
    print(f"[INFO] Block size: {block_size}")

    dflash_aux_layers = get_dflash_aux_layer_indices(config)
    print(f"[INFO] DFlash aux layers: {dflash_aux_layers}")

    (target_model_fn, target_logits_fn, target_combine_fn, target_state,
     draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    ) = load_models(config, mesh)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))

    print(f"[INFO] Dataset: {args.dataset}, samples: {len(dataset)}")

    # ==================================================================
    # Part 1: Microbenchmark — verify latency vs query count
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 1: VERIFY LATENCY MICROBENCHMARK")
    print("=" * 70)

    query_counts = [16, 32, 48, 64, 96, 128]
    micro_results = run_verify_microbenchmark(
        target_model_fn, target_logits_fn, target_state,
        config, mesh, tokenizer, dataset, args.max_model_len,
        query_counts, args.micro_trials, args.micro_warmup)

    # Summary table
    print("\n  " + "=" * 65)
    print(f"  {'Tokens':>7} | {'Latency (ms)':>14} | {'Per-token (ms)':>15} | "
          f"{'vs 16-tok':>10} | {'Amort. ratio':>12}")
    print("  " + "-" * 65)

    base_ms = micro_results[16]["mean_ms"] if 16 in micro_results else 1
    for n in query_counts:
        r = micro_results[n]
        ratio = r["mean_ms"] / base_ms
        amort = ratio / (n / 16)  # <1 means amortization works
        print(f"  {n:>7} | {r['mean_ms']:>10.2f} ± {r['std_ms']:.1f} | "
              f"{r['per_token_ms']:>15.3f} | {ratio:>9.2f}x | {amort:>12.3f}")

    # ==================================================================
    # Part 2: Multi-block speculation end-to-end
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 2: MULTI-BLOCK SPECULATION")
    print("=" * 70)

    num_blocks_list = [1, 2]
    multiblock_results = run_multiblock_experiment(
        target_model_fn, target_logits_fn, target_combine_fn, target_state,
        draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
        config, mesh, tokenizer, dataset, args.max_model_len, block_size,
        mask_token_id, hidden_size, args.max_new_tokens, args.warmup_samples,
        dflash_aux_layers, num_blocks_list)

    # Summary
    print("\n  " + "=" * 65)
    print(f"  {'Blocks':>7} | {'Width':>6} | {'Tau':>6} | "
          f"{'Step (ms)':>10} | {'Tok/s':>8}")
    print("  " + "-" * 65)
    for nb in num_blocks_list:
        r = multiblock_results[nb]
        print(f"  {nb:>7} | {r['verify_width']:>6} | {r['avg_tau']:>6.2f} | "
              f"{r['avg_step_ms']:>10.2f} | {r['throughput_tok_s']:>8.1f}")

    # Save
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)),
                    exist_ok=True)
        output = {
            "microbenchmark": micro_results,
            "multiblock": {str(k): v for k, v in multiblock_results.items()},
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

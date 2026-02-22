"""Iterative refinement drafting for DFlash on TPU.

Direction 2 experiment: instead of a single draft forward pass, feed the
draft model's predictions back as input for k refinement steps before
verifying with the target model.

Hypothesis: 2-3 refinement steps improve draft quality (tau) because each
step lets block positions attend to each other's updated predictions via
non-causal attention, improving cross-position coherence. On TPU, XLA can
fuse consecutive passes, and the block tensors are already in HBM, so
the marginal latency per refinement step may be small relative to the
quality gain.

Mechanism:
  Step 0 (standard):  [tok, mask, mask, ...] -> draft -> [t1, t2, ..., t15]
  Step 1 (refine):    [tok, t1, t2, ..., t14] -> draft -> [t1', t2', ..., t15']
  Step 2 (refine):    [tok, t1', t2', ..., t14'] -> draft -> [t1'', ...]
  Then verify [tok, t1'', t2'', ..., t15''] with target model.

Key: before each refinement pass, we rewrite the draft KV cache to drop
the previous noise block entries (reset cache_len to pre-noise value), so
the draft model re-processes the block from scratch with updated token
embeddings.

Usage:
    python benchmarks/iterative_refinement.py \
        --target-model Qwen/Qwen3-4B \
        --draft-model z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k --max-samples 4 --max-new-tokens 256 \
        --refinement-steps 0 1 2 3
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
# Shared infrastructure (from standalone_dflash.py)
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
# Iterative refinement generate loop
# ---------------------------------------------------------------------------

def refinement_dflash_generate(
    target_model_fn, target_logits_fn, target_combine_fn, target_state,
    draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    target_kv_caches, block_tables,
    draft_kv_caches,
    input_ids_np, mask_token_id, block_size, max_new_tokens,
    eos_token_id, hidden_size, max_model_len,
    refinement_steps=0,
):
    """DFlash generate loop with iterative refinement.

    Args:
        refinement_steps: Number of additional passes through the draft model
            after the initial prediction. 0 = standard DFlash (no refinement).
    """
    num_input_tokens = len(input_ids_np)
    max_length = num_input_tokens + max_new_tokens

    output_ids = np.full(max_length + block_size, mask_token_id, dtype=np.int32)
    output_ids[:num_input_tokens] = input_ids_np

    # --- Prefill ---
    prefill_start = time.perf_counter()
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
    time_to_first_token = time.perf_counter() - prefill_start

    # --- Context buffer init ---
    ctx_buf = np.zeros((max_model_len, hidden_size), dtype=np.float32)
    prev_ctx_len = 0

    if projected_all is not None:
        proj_np = np.asarray(projected_all, dtype=np.float32)
        n = min(num_input_tokens, max_model_len)
        ctx_buf[:n] = proj_np[:n]

    # --- Decode loop ---
    decode_start = time.perf_counter()
    start = num_input_tokens
    acceptance_lengths = []
    draft_cache_len = 0
    prev_seq_len = 0
    draft_prefill_done = False

    # Per-refinement-step timing
    draft_times = []      # time for initial draft pass
    refine_times = []     # time for each refinement pass (list of lists)
    verify_times = []

    while start < max_length:
        seq_len = start

        # (a) Cache crop
        draft_cache_len = prev_seq_len

        # (b) Context update
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

        # (c) Build initial noise block: [next_token, mask, mask, ..., mask]
        next_token = output_ids[start]
        noise_ids = np.full(block_size, mask_token_id, dtype=np.int32)
        noise_ids[0] = next_token

        ctx_jax = jnp.array(new_ctx_np, dtype=jnp.bfloat16)
        cache_len_arr = jnp.array([draft_cache_len], dtype=jnp.int32)
        ctx_count_arr = jnp.array([actual_ctx_count], dtype=jnp.int32)
        target_hidden = (ctx_jax, cache_len_arr, ctx_count_arr)

        dummy_positions = jnp.arange(block_size, dtype=jnp.int32) + seq_len
        dummy_metadata = make_attn_metadata(
            dummy_positions, seq_len + block_size, block_size, block_tables)

        # ============================================================
        # INITIAL DRAFT PASS (step 0)
        # ============================================================
        noise_ids_jax = jnp.array(noise_ids, dtype=jnp.int32)

        tpu_sync()
        t_draft_start = time.perf_counter()

        draft_kv_caches, draft_hidden, _ = draft_model_fn(
            draft_state, draft_kv_caches, noise_ids_jax,
            target_hidden, dummy_metadata,
        )

        # Sample draft tokens
        draft_logits = target_logits_fn(
            target_state, draft_hidden[1:block_size], None)
        draft_tokens = np.array(jnp.argmax(draft_logits, axis=-1))

        tpu_sync()
        t_draft_elapsed = time.perf_counter() - t_draft_start

        # Save the cache_len BEFORE noise was written, so we can reset
        # to this point for refinement passes.
        pre_noise_cache_len = draft_cache_len

        # After initial pass, the draft model wrote context + noise to cache.
        # For the standard path (no refinement), we'd continue to verify.
        # For refinement, we reset and re-run with updated tokens.

        step_refine_times = []

        # ============================================================
        # REFINEMENT PASSES (steps 1..k)
        # ============================================================
        for refine_step in range(refinement_steps):
            # Build refined input: [next_token, draft_t1, draft_t2, ..., draft_t14]
            # (replace masks with current best predictions)
            refined_ids = np.full(block_size, mask_token_id, dtype=np.int32)
            refined_ids[0] = next_token
            # Fill positions 1..block_size-1 with current draft predictions
            n_fill = min(len(draft_tokens), block_size - 1)
            refined_ids[1:1 + n_fill] = draft_tokens[:n_fill]
            refined_ids_jax = jnp.array(refined_ids, dtype=jnp.int32)

            # Reset draft KV cache to pre-noise state.
            # We do this by setting cache_len back to the pre-noise value.
            # The dynamic_update_slice in the next forward pass will overwrite
            # the stale noise K/V entries.
            #
            # For refinement, we pass actual_ctx_count=0 because context was
            # already written in the initial pass and is still valid in the cache.
            # We only need to rewrite the noise portion.
            refine_cache_len = jnp.array(
                [pre_noise_cache_len + actual_ctx_count], dtype=jnp.int32)
            refine_ctx_count = jnp.array([0], dtype=jnp.int32)
            # Empty context (no new context tokens for refinement)
            empty_ctx = jnp.zeros((16, hidden_size), dtype=jnp.bfloat16)
            refine_hidden = (empty_ctx, refine_cache_len, refine_ctx_count)

            tpu_sync()
            t_refine_start = time.perf_counter()

            draft_kv_caches, draft_hidden, _ = draft_model_fn(
                draft_state, draft_kv_caches, refined_ids_jax,
                refine_hidden, dummy_metadata,
            )

            # Re-sample draft tokens from refined hidden states
            draft_logits = target_logits_fn(
                target_state, draft_hidden[1:block_size], None)
            draft_tokens = np.array(jnp.argmax(draft_logits, axis=-1))

            tpu_sync()
            t_refine_elapsed = time.perf_counter() - t_refine_start
            step_refine_times.append(t_refine_elapsed)

        # Update draft state for next iteration
        draft_cache_len = pre_noise_cache_len + actual_ctx_count + block_size
        prev_seq_len = seq_len

        # Fill block with final draft tokens
        output_ids[start + 1 : start + block_size] = draft_tokens
        block_ids = jnp.array(
            output_ids[start : start + block_size], dtype=jnp.int32)

        if not draft_prefill_done:
            draft_prefill_done = True
            tpu_sync()
            decode_start = time.perf_counter()
            continue

        # (f) Verify with target model
        tpu_sync()
        t_verify_start = time.perf_counter()

        verify_positions = jnp.arange(start, start + block_size, dtype=jnp.int32)
        verify_metadata = make_attn_metadata(
            verify_positions, start + block_size, block_size, block_tables)
        target_kv_caches, verify_hidden, aux_hidden_states = target_model_fn(
            target_state, target_kv_caches, block_ids, verify_metadata,
            None, None, None, None, None, True, True,
        )
        verify_logits = target_logits_fn(target_state, verify_hidden, None)
        posterior = np.array(jnp.argmax(verify_logits, axis=-1))

        tpu_sync()
        t_verify_elapsed = time.perf_counter() - t_verify_start

        # (g) Accept
        block_np = np.array(block_ids)
        matches = (block_np[1:] == posterior[:-1]).astype(np.int32)
        acceptance_length = int(np.cumprod(matches).sum())

        output_ids[start : start + acceptance_length + 1] = \
            block_np[:acceptance_length + 1]
        output_ids[start + acceptance_length + 1] = posterior[acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1

        draft_times.append(t_draft_elapsed)
        refine_times.append(step_refine_times)
        verify_times.append(t_verify_elapsed)

        # (h) Update context
        if len(aux_hidden_states) > 0:
            raw = jnp.concatenate(aux_hidden_states, axis=-1)
            proj = draft_combine_fn(draft_state, raw)
            projected_all = proj[:acceptance_length + 1]
        else:
            projected_all = None

        if eos_token_id in output_ids[num_input_tokens : start + 1]:
            break

    tpu_sync()
    total_decode_time = time.perf_counter() - decode_start

    # Trim output
    output_ids = output_ids[:max_length]
    valid_mask = output_ids != mask_token_id
    output_ids = output_ids[valid_mask]
    out_tokens = output_ids[num_input_tokens:]
    stop_idx = np.where(out_tokens == eos_token_id)[0]
    if len(stop_idx) > 0:
        output_ids = output_ids[:num_input_tokens + stop_idx[0] + 1]
    num_output_tokens = len(output_ids) - num_input_tokens
    time_per_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_token,
        total_decode_time=total_decode_time,
        acceptance_lengths=acceptance_lengths,
        draft_times=draft_times,
        refine_times=refine_times,
        verify_times=verify_times,
        refinement_steps=refinement_steps,
    )


# ---------------------------------------------------------------------------
# Baseline (no speculation)
# ---------------------------------------------------------------------------

def baseline_generate(
    target_model_fn, target_logits_fn, target_state,
    target_kv_caches, block_tables,
    input_ids_np, max_new_tokens, eos_token_id, mask_token_id,
):
    num_input_tokens = len(input_ids_np)
    max_length = num_input_tokens + max_new_tokens

    output_ids = np.full(max_length, mask_token_id, dtype=np.int32)
    output_ids[:num_input_tokens] = input_ids_np

    input_ids_jax = jnp.array(input_ids_np, dtype=jnp.int32)
    positions = jnp.arange(num_input_tokens, dtype=jnp.int32)
    metadata = make_attn_metadata(positions, num_input_tokens,
                                   num_input_tokens, block_tables)
    target_kv_caches, hidden_states, _ = target_model_fn(
        target_state, target_kv_caches, input_ids_jax, metadata,
        None, None, None, None, None, True, True,
    )
    last_hidden = hidden_states[-1:]
    logits = target_logits_fn(target_state, last_hidden, None)
    next_token = int(jnp.argmax(logits, axis=-1)[0])
    output_ids[num_input_tokens] = next_token
    tpu_sync()

    decode_start = time.perf_counter()
    for pos in range(num_input_tokens + 1, max_length):
        token_jax = jnp.array([next_token], dtype=jnp.int32)
        positions = jnp.array([pos - 1], dtype=jnp.int32)
        metadata = make_attn_metadata(positions, pos, 1, block_tables)
        target_kv_caches, hidden_states, _ = target_model_fn(
            target_state, target_kv_caches, token_jax, metadata,
            None, None, None, None, None, True, True,
        )
        logits = target_logits_fn(target_state, hidden_states, None)
        next_token = int(jnp.argmax(logits, axis=-1)[0])
        output_ids[pos] = next_token
        if next_token == eos_token_id:
            break

    tpu_sync()
    total_decode_time = time.perf_counter() - decode_start
    valid_mask = output_ids != mask_token_id
    output_ids = output_ids[valid_mask]
    num_output_tokens = len(output_ids) - num_input_tokens
    time_per_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_output_tokens=num_output_tokens,
        time_per_output_token=time_per_token,
        total_decode_time=total_decode_time,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Iterative refinement DFlash benchmark on TPU")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--refinement-steps", type=int, nargs="+",
                        default=[0, 1, 2, 3],
                        help="Number of refinement steps to test (space-separated)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run autoregressive baseline for speedup calculation")
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

    print(f"[INFO] Block size: {block_size}, mask_token_id: {mask_token_id}")
    print(f"[INFO] Refinement steps to test: {args.refinement_steps}")

    (target_model_fn, target_logits_fn, target_combine_fn, target_state,
     draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    ) = load_models(config, mesh)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    eos_token_id = tokenizer.eos_token_id or 0

    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))
    print(f"[INFO] Dataset: {args.dataset}, samples: {len(dataset)}")

    max_length = args.max_model_len
    page_size = 16

    # Prepare prompts
    prompts = []
    for idx in range(len(dataset)):
        instance = dataset[idx]
        prompt = instance["turns"][0]
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        input_ids = tokenizer.encode(input_text)
        prompts.append(np.array(input_ids, dtype=np.int32))

    # --- Run baseline if requested ---
    baseline_tpot = None
    if args.run_baseline:
        print("\n" + "=" * 60)
        print("BASELINE (autoregressive, no speculation)")
        print("=" * 60)
        baseline_tpots = []
        for idx, input_ids_np in enumerate(prompts):
            print(f"  [{idx+1}/{len(prompts)}] len={len(input_ids_np)}", end=" ")
            target_kv, bt = allocate_target_kv_caches(
                config, mesh, max_length, page_size)
            bl = baseline_generate(
                target_model_fn, target_logits_fn, target_state,
                target_kv, bt, input_ids_np,
                args.max_new_tokens, eos_token_id, mask_token_id,
            )
            print(f"-> {bl.num_output_tokens} tokens, "
                  f"TPOT={bl.time_per_output_token*1000:.1f}ms")
            if idx >= args.warmup:
                baseline_tpots.append(bl.time_per_output_token)
        baseline_tpot = np.mean(baseline_tpots) if baseline_tpots else None
        if baseline_tpot:
            print(f"\n  Baseline avg TPOT: {baseline_tpot*1000:.2f} ms "
                  f"({1/baseline_tpot:.1f} TPS)")

    # --- Run each refinement level ---
    results_by_k = {}

    for k in args.refinement_steps:
        print(f"\n{'='*60}")
        print(f"REFINEMENT k={k} {'(standard DFlash)' if k == 0 else ''}")
        print(f"{'='*60}")

        sample_results = []
        for idx, input_ids_np in enumerate(prompts):
            print(f"  [{idx+1}/{len(prompts)}] len={len(input_ids_np)}", end=" ")

            target_kv, bt = allocate_target_kv_caches(
                config, mesh, max_length, page_size)
            draft_kv = allocate_draft_kv_caches(config, mesh, max_length)

            result = refinement_dflash_generate(
                target_model_fn, target_logits_fn, target_combine_fn, target_state,
                draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
                target_kv, bt, draft_kv,
                input_ids_np, mask_token_id, block_size, args.max_new_tokens,
                eos_token_id, hidden_size, max_length,
                refinement_steps=k,
            )

            tau = np.mean(result.acceptance_lengths) if result.acceptance_lengths else 0
            print(f"-> {result.num_output_tokens} tokens, "
                  f"TPOT={result.time_per_output_token*1000:.1f}ms, tau={tau:.2f}")

            sample_results.append(result)

        # Aggregate (skip warmup)
        measured = sample_results[args.warmup:]
        if not measured:
            print(f"  No measured samples for k={k}")
            continue

        tpots = [r.time_per_output_token for r in measured]
        all_acc = []
        for r in measured:
            all_acc.extend(r.acceptance_lengths)
        tau = np.mean(all_acc) if all_acc else 0
        avg_tpot = np.mean(tpots)
        tps = 1.0 / avg_tpot if avg_tpot > 0 else 0

        # Per-position acceptance rates
        pos_counts = np.zeros(block_size)
        total_drafts = len(all_acc)
        for a in all_acc:
            for p in range(min(int(a), block_size)):
                pos_counts[p] += 1
        pos_rates = pos_counts / max(total_drafts, 1)

        # Draft/refine/verify timing
        all_draft_times = []
        all_refine_times = [[] for _ in range(k)]
        all_verify_times = []
        for r in measured:
            all_draft_times.extend(r.draft_times)
            all_verify_times.extend(r.verify_times)
            for step_refines in r.refine_times:
                for ri, rt in enumerate(step_refines):
                    if ri < k:
                        all_refine_times[ri].append(rt)

        results_by_k[k] = {
            "tau": float(tau),
            "avg_tpot_ms": avg_tpot * 1000,
            "tps": tps,
            "speedup": (baseline_tpot / avg_tpot) if baseline_tpot else None,
            "total_drafts": total_drafts,
            "pos_rates": [float(r) for r in pos_rates],
            "avg_draft_ms": np.mean(all_draft_times) * 1000 if all_draft_times else 0,
            "avg_verify_ms": np.mean(all_verify_times) * 1000 if all_verify_times else 0,
            "avg_refine_ms": [
                np.mean(rt) * 1000 if rt else 0 for rt in all_refine_times
            ],
        }

        print(f"\n  k={k}: tau={tau:.2f}, TPOT={avg_tpot*1000:.2f}ms, TPS={tps:.1f}")
        if baseline_tpot:
            print(f"  Speedup: {baseline_tpot/avg_tpot:.2f}x")
        print(f"  Avg draft: {results_by_k[k]['avg_draft_ms']:.2f}ms")
        for ri, rt_ms in enumerate(results_by_k[k]["avg_refine_ms"]):
            print(f"  Avg refine step {ri+1}: {rt_ms:.2f}ms")
        print(f"  Avg verify: {results_by_k[k]['avg_verify_ms']:.2f}ms")
        print(f"  Per-position acceptance: "
              f"{[f'{r:.3f}' for r in pos_rates[:6]]}")

    # --- Comparison table ---
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'k':>3} {'tau':>6} {'TPOT(ms)':>10} {'TPS':>8} {'Speedup':>8} "
          f"{'Draft(ms)':>10} {'Refine(ms)':>11} {'Verify(ms)':>11}")
    print("-" * 70)
    for k in sorted(results_by_k.keys()):
        r = results_by_k[k]
        refine_total = sum(r["avg_refine_ms"])
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
        print(f"{k:>3} {r['tau']:>6.2f} {r['avg_tpot_ms']:>10.2f} "
              f"{r['tps']:>8.1f} {speedup_str:>8} "
              f"{r['avg_draft_ms']:>10.2f} {refine_total:>11.2f} "
              f"{r['avg_verify_ms']:>11.2f}")

    # Key insight: is refinement worth it?
    if 0 in results_by_k and len(results_by_k) > 1:
        base_tau = results_by_k[0]["tau"]
        base_tpot = results_by_k[0]["avg_tpot_ms"]
        print(f"\n  Relative to k=0 (standard DFlash, tau={base_tau:.2f}):")
        for k in sorted(results_by_k.keys()):
            if k == 0:
                continue
            r = results_by_k[k]
            tau_delta = r["tau"] - base_tau
            tau_pct = (tau_delta / base_tau * 100) if base_tau > 0 else 0
            tpot_delta = r["avg_tpot_ms"] - base_tpot
            tpot_pct = (tpot_delta / base_tpot * 100) if base_tpot > 0 else 0
            net = "BETTER" if (r["tps"] > results_by_k[0]["tps"]) else "WORSE"
            print(f"    k={k}: tau {'+' if tau_delta >= 0 else ''}{tau_delta:.2f} "
                  f"({'+' if tau_pct >= 0 else ''}{tau_pct:.1f}%), "
                  f"TPOT {'+' if tpot_delta >= 0 else ''}{tpot_delta:.2f}ms "
                  f"({'+' if tpot_pct >= 0 else ''}{tpot_pct:.1f}%) "
                  f"-> net TPS: {net}")

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
                "num_samples": len(prompts),
                "measured_samples": len(prompts) - args.warmup,
                "refinement_steps_tested": args.refinement_steps,
            },
            "baseline_tpot_ms": baseline_tpot * 1000 if baseline_tpot else None,
            "results_by_k": results_by_k,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

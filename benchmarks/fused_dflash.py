"""Fused DFlash speculative decoding benchmark on TPU.

Direction 1 optimization: eliminates host-device roundtrips by fusing
operations into larger JIT-compiled functions.

The profiling (Doc 27) showed that 82.8% of step time is overhead:
  - 47% in acceptance (target_logits_fn + argmax + compare, separate calls)
  - 14% in draft_sample (target_logits_fn + argmax, separate calls)
  - 10% in host_device_xfer (numpy ↔ jax array conversions)

This benchmark fuses these into two JIT-compiled device functions:
  1. fused_draft: draft_model_fn → logits → argmax → block assembly (all on device)
  2. fused_verify: target_model_fn → logits → argmax → acceptance (all on device)

The entire draft→verify→accept path runs with only ONE host sync per step
(to read acceptance_length and bonus_token for the next iteration).

Usage:
    python benchmarks/fused_dflash.py \
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
# Shared infrastructure
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
# Fused JIT-compiled functions
# ---------------------------------------------------------------------------

def make_fused_draft_fn(draft_model_fn, target_logits_fn, block_size):
    """Create a fused draft function: draft_forward + logits + argmax + block assembly.

    Returns draft_token_ids and full block_ids on device — no host roundtrip.
    """

    @jax.jit
    def fused_draft(
        draft_state, target_state, draft_kv_caches,
        noise_ids, target_hidden, draft_attn_metadata,
    ):
        # 1. Draft model forward
        draft_kv_caches, draft_hidden, _ = draft_model_fn(
            draft_state, draft_kv_caches, noise_ids,
            target_hidden, draft_attn_metadata,
        )

        # 2. Compute logits using target LM head (on draft hidden states)
        draft_logits = target_logits_fn(
            target_state, draft_hidden[1:block_size], None)

        # 3. Greedy sample (argmax) — stays on device
        draft_token_ids = jnp.argmax(draft_logits, axis=-1).astype(jnp.int32)

        # 4. Assemble full block: [first_token, draft_t1, ..., draft_t(bs-1)]
        block_ids = jnp.concatenate([noise_ids[:1], draft_token_ids])

        return draft_kv_caches, draft_hidden, draft_token_ids, block_ids

    return fused_draft


def make_fused_verify_fn(target_model_fn, target_logits_fn, draft_combine_fn):
    """Create a fused verify function: verify_forward + logits + argmax + acceptance.

    Returns acceptance_length, bonus_token, and projected aux hidden states —
    only ONE host sync needed to read the scalar acceptance_length.
    """

    @jax.jit
    def fused_verify(
        target_state, draft_state, target_kv_caches,
        block_ids, verify_attn_metadata,
    ):
        # 1. Target model forward (verify block)
        target_kv_caches, verify_hidden, aux_hidden_states = target_model_fn(
            target_state, target_kv_caches, block_ids, verify_attn_metadata,
            None, None, None, None, None, True, True,
        )

        # 2. Compute verify logits + argmax — stays on device
        verify_logits = target_logits_fn(target_state, verify_hidden, None)
        posterior = jnp.argmax(verify_logits, axis=-1).astype(jnp.int32)

        # 3. Compute acceptance length on device
        # matches[i] = 1 if block_ids[i+1] == posterior[i]
        matches = (block_ids[1:] == posterior[:-1]).astype(jnp.int32)
        # Consecutive matches via cumprod
        cummatches = jnp.cumprod(matches)
        acceptance_length = jnp.sum(cummatches).astype(jnp.int32)

        # 4. Bonus token = posterior[acceptance_length]
        bonus_token = posterior[acceptance_length]

        # 5. Project aux hidden states for next iteration's context
        raw = jnp.concatenate(aux_hidden_states, axis=-1)
        projected = draft_combine_fn(draft_state, raw)

        return (target_kv_caches, acceptance_length, bonus_token,
                block_ids, projected)

    return fused_verify


# ---------------------------------------------------------------------------
# Fused generate loop
# ---------------------------------------------------------------------------

def fused_dflash_generate(
    # Models
    target_model_fn, target_logits_fn, target_combine_fn, target_state,
    draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    # Fused functions
    fused_draft, fused_verify,
    # Caches
    target_kv_caches, block_tables,
    draft_kv_caches,
    # Config
    input_ids_np, mask_token_id, block_size, max_new_tokens,
    eos_token_id, hidden_size, max_model_len,
):
    """Fused DFlash generate loop with minimal host-device roundtrips."""
    num_input_tokens = len(input_ids_np)
    max_length = num_input_tokens + max_new_tokens

    output_ids = np.full(max_length + block_size, mask_token_id, dtype=np.int32)
    output_ids[:num_input_tokens] = input_ids_np

    # --- Prefill (same as unfused) ---
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

    # --- Fused decode loop ---
    decode_start = time.perf_counter()
    start = num_input_tokens
    acceptance_lengths = []
    draft_cache_len = 0
    prev_seq_len = 0
    draft_prefill_done = False

    # Per-step timing (fused)
    step_times = []

    while start < max_length:
        step_start = time.perf_counter()
        seq_len = start

        # (a) Cache crop
        draft_cache_len = prev_seq_len

        # (b) Context update (host — cheap, 0.27ms per profiling)
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

        # (c) Build inputs (one upload to device)
        next_token = output_ids[start]
        noise_ids = np.full(block_size, mask_token_id, dtype=np.int32)
        noise_ids[0] = next_token
        noise_ids_jax = jnp.array(noise_ids, dtype=jnp.int32)

        ctx_jax = jnp.array(new_ctx_np, dtype=jnp.bfloat16)
        cache_len_arr = jnp.array([draft_cache_len], dtype=jnp.int32)
        ctx_count_arr = jnp.array([actual_ctx_count], dtype=jnp.int32)
        target_hidden = (ctx_jax, cache_len_arr, ctx_count_arr)

        dummy_positions = jnp.arange(block_size, dtype=jnp.int32) + seq_len
        draft_metadata = make_attn_metadata(
            dummy_positions, seq_len + block_size, block_size, block_tables)

        # ============================================================
        # FUSED DRAFT: draft_forward + logits + argmax + block assembly
        # (one JIT call, no host sync in between)
        # ============================================================
        draft_kv_caches, draft_hidden, draft_token_ids, block_ids = fused_draft(
            draft_state, target_state, draft_kv_caches,
            noise_ids_jax, target_hidden, draft_metadata,
        )

        # Update draft state
        draft_cache_len = draft_cache_len + actual_ctx_count + block_size
        prev_seq_len = seq_len

        if not draft_prefill_done:
            draft_prefill_done = True
            tpu_sync()
            decode_start = time.perf_counter()
            continue

        # ============================================================
        # FUSED VERIFY: verify_forward + logits + argmax + acceptance + aux_proj
        # (one JIT call, one host sync at the end to read results)
        # ============================================================
        verify_positions = jnp.arange(start, start + block_size, dtype=jnp.int32)
        verify_metadata = make_attn_metadata(
            verify_positions, start + block_size, block_size, block_tables)

        (target_kv_caches, acceptance_length_jax, bonus_token_jax,
         block_ids_verified, projected_all) = fused_verify(
            target_state, draft_state, target_kv_caches,
            block_ids, verify_metadata,
        )

        # ============================================================
        # SINGLE HOST SYNC: read acceptance_length + bonus_token
        # (the only mandatory host roundtrip per step)
        # ============================================================
        tpu_sync()
        acceptance_length = int(jax.device_get(acceptance_length_jax))
        bonus_token = int(jax.device_get(bonus_token_jax))
        block_np = np.array(jax.device_get(block_ids_verified))

        # Update output_ids on host
        output_ids[start : start + acceptance_length + 1] = \
            block_np[:acceptance_length + 1]
        output_ids[start + acceptance_length + 1] = bonus_token

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1

        # Trim projected to accepted tokens only
        projected_all = projected_all[:acceptance_length + 1]

        step_elapsed = time.perf_counter() - step_start
        step_times.append(step_elapsed)

        # Check stop
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
        step_times=step_times,
    )


# ---------------------------------------------------------------------------
# Unfused generate loop (for comparison — same as standalone_dflash.py)
# ---------------------------------------------------------------------------

def unfused_dflash_generate(
    target_model_fn, target_logits_fn, target_combine_fn, target_state,
    draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    target_kv_caches, block_tables,
    draft_kv_caches,
    input_ids_np, mask_token_id, block_size, max_new_tokens,
    eos_token_id, hidden_size, max_model_len,
):
    """Standard unfused DFlash generate loop (baseline for comparison)."""
    num_input_tokens = len(input_ids_np)
    max_length = num_input_tokens + max_new_tokens

    output_ids = np.full(max_length + block_size, mask_token_id, dtype=np.int32)
    output_ids[:num_input_tokens] = input_ids_np

    # Prefill
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

    ctx_buf = np.zeros((max_model_len, hidden_size), dtype=np.float32)
    prev_ctx_len = 0
    if projected_all is not None:
        proj_np = np.asarray(projected_all, dtype=np.float32)
        n = min(num_input_tokens, max_model_len)
        ctx_buf[:n] = proj_np[:n]

    decode_start = time.perf_counter()
    start = num_input_tokens
    acceptance_lengths = []
    draft_cache_len = 0
    prev_seq_len = 0
    draft_prefill_done = False
    step_times = []

    while start < max_length:
        step_start = time.perf_counter()
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

        # Unfused draft
        draft_kv_caches, draft_hidden, _ = draft_model_fn(
            draft_state, draft_kv_caches, noise_ids_jax,
            target_hidden, dummy_metadata,
        )
        draft_cache_len = draft_cache_len + actual_ctx_count + block_size
        prev_seq_len = seq_len

        draft_logits = target_logits_fn(
            target_state, draft_hidden[1:block_size], None)
        draft_tokens = np.array(jnp.argmax(draft_logits, axis=-1))

        output_ids[start + 1 : start + block_size] = draft_tokens
        block_ids = jnp.array(
            output_ids[start : start + block_size], dtype=jnp.int32)

        if not draft_prefill_done:
            draft_prefill_done = True
            tpu_sync()
            decode_start = time.perf_counter()
            continue

        # Unfused verify
        verify_positions = jnp.arange(start, start + block_size, dtype=jnp.int32)
        verify_metadata = make_attn_metadata(
            verify_positions, start + block_size, block_size, block_tables)

        target_kv_caches, verify_hidden, aux_hidden_states = target_model_fn(
            target_state, target_kv_caches, block_ids, verify_metadata,
            None, None, None, None, None, True, True,
        )

        verify_logits = target_logits_fn(target_state, verify_hidden, None)
        posterior = np.array(jnp.argmax(verify_logits, axis=-1))

        block_np = np.array(block_ids)
        matches = (block_np[1:] == posterior[:-1]).astype(np.int32)
        acceptance_length = int(np.cumprod(matches).sum())

        output_ids[start : start + acceptance_length + 1] = \
            block_np[:acceptance_length + 1]
        output_ids[start + acceptance_length + 1] = posterior[acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1

        if len(aux_hidden_states) > 0:
            raw = jnp.concatenate(aux_hidden_states, axis=-1)
            proj = draft_combine_fn(draft_state, raw)
            projected_all = proj[:acceptance_length + 1]
        else:
            projected_all = None

        step_elapsed = time.perf_counter() - step_start
        step_times.append(step_elapsed)

        if eos_token_id in output_ids[num_input_tokens : start + 1]:
            break

    tpu_sync()
    total_decode_time = time.perf_counter() - decode_start

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
        time_per_output_token=time_per_token,
        total_decode_time=total_decode_time,
        acceptance_lengths=acceptance_lengths,
        step_times=step_times,
    )


# ---------------------------------------------------------------------------
# Baseline autoregressive
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
        description="Fused DFlash benchmark — Direction 1 optimization")
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

    print(f"[INFO] Block size: {block_size}, mask_token_id: {mask_token_id}")

    (target_model_fn, target_logits_fn, target_combine_fn, target_state,
     draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    ) = load_models(config, mesh)

    # Build fused functions
    print("[INFO] Building fused JIT functions...")
    fused_draft = make_fused_draft_fn(draft_model_fn, target_logits_fn, block_size)
    fused_verify = make_fused_verify_fn(target_model_fn, target_logits_fn, draft_combine_fn)
    print("[INFO] Fused functions ready.")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    eos_token_id = tokenizer.eos_token_id or 0

    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))
    print(f"[INFO] Dataset: {args.dataset}, samples: {len(dataset)}")

    max_length = args.max_model_len
    page_size = 16

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

    # ========================
    # Run baseline
    # ========================
    print("\n" + "=" * 60)
    print("BASELINE (autoregressive)")
    print("=" * 60)
    baseline_tpots = []
    for idx, inp in enumerate(prompts):
        print(f"  [{idx+1}/{len(prompts)}] len={len(inp)}", end=" ")
        tkv, bt = allocate_target_kv_caches(config, mesh, max_length, page_size)
        bl = baseline_generate(
            target_model_fn, target_logits_fn, target_state,
            tkv, bt, inp, args.max_new_tokens, eos_token_id, mask_token_id)
        print(f"-> {bl.num_output_tokens} tokens, TPOT={bl.time_per_output_token*1000:.1f}ms")
        if idx >= args.warmup:
            baseline_tpots.append(bl.time_per_output_token)
    baseline_tpot = np.mean(baseline_tpots) if baseline_tpots else None

    # ========================
    # Run unfused DFlash
    # ========================
    print("\n" + "=" * 60)
    print("UNFUSED DFLASH (standard)")
    print("=" * 60)
    unfused_results = []
    for idx, inp in enumerate(prompts):
        print(f"  [{idx+1}/{len(prompts)}] len={len(inp)}", end=" ")
        tkv, bt = allocate_target_kv_caches(config, mesh, max_length, page_size)
        dkv = allocate_draft_kv_caches(config, mesh, max_length)
        r = unfused_dflash_generate(
            target_model_fn, target_logits_fn, target_combine_fn, target_state,
            draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
            tkv, bt, dkv, inp, mask_token_id, block_size,
            args.max_new_tokens, eos_token_id, hidden_size, max_length)
        tau = np.mean(r.acceptance_lengths) if r.acceptance_lengths else 0
        print(f"-> {r.num_output_tokens} tokens, TPOT={r.time_per_output_token*1000:.1f}ms, tau={tau:.2f}")
        unfused_results.append(r)

    # ========================
    # Run fused DFlash
    # ========================
    print("\n" + "=" * 60)
    print("FUSED DFLASH (optimized)")
    print("=" * 60)
    fused_results = []
    for idx, inp in enumerate(prompts):
        print(f"  [{idx+1}/{len(prompts)}] len={len(inp)}", end=" ")
        tkv, bt = allocate_target_kv_caches(config, mesh, max_length, page_size)
        dkv = allocate_draft_kv_caches(config, mesh, max_length)
        r = fused_dflash_generate(
            target_model_fn, target_logits_fn, target_combine_fn, target_state,
            draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
            fused_draft, fused_verify,
            tkv, bt, dkv, inp, mask_token_id, block_size,
            args.max_new_tokens, eos_token_id, hidden_size, max_length)
        tau = np.mean(r.acceptance_lengths) if r.acceptance_lengths else 0
        print(f"-> {r.num_output_tokens} tokens, TPOT={r.time_per_output_token*1000:.1f}ms, tau={tau:.2f}")
        fused_results.append(r)

    # ========================
    # Comparison
    # ========================
    unfused_measured = unfused_results[args.warmup:]
    fused_measured = fused_results[args.warmup:]

    def aggregate(results):
        tpots = [r.time_per_output_token for r in results]
        all_acc = []
        for r in results:
            all_acc.extend(r.acceptance_lengths)
        tau = np.mean(all_acc) if all_acc else 0
        avg_tpot = np.mean(tpots) if tpots else 0
        tps = 1.0 / avg_tpot if avg_tpot > 0 else 0
        # Step times
        all_steps = []
        for r in results:
            all_steps.extend(r.step_times)
        avg_step = np.mean(all_steps) * 1000 if all_steps else 0
        median_step = np.median(all_steps) * 1000 if all_steps else 0
        return {
            "tau": float(tau),
            "avg_tpot_ms": avg_tpot * 1000,
            "tps": tps,
            "avg_step_ms": avg_step,
            "median_step_ms": median_step,
            "total_steps": len(all_steps),
        }

    unfused_stats = aggregate(unfused_measured)
    fused_stats = aggregate(fused_measured)

    print("\n" + "=" * 70)
    print("COMPARISON: FUSED vs UNFUSED")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Baseline':>10} {'Unfused':>10} {'Fused':>10} {'Improvement':>12}")
    print("-" * 70)

    bl_tpot_str = f"{baseline_tpot*1000:.2f}" if baseline_tpot else "N/A"
    print(f"{'TPOT (ms)':<25} {bl_tpot_str:>10} "
          f"{unfused_stats['avg_tpot_ms']:>10.2f} {fused_stats['avg_tpot_ms']:>10.2f} ", end="")
    if unfused_stats['avg_tpot_ms'] > 0:
        imp = (unfused_stats['avg_tpot_ms'] - fused_stats['avg_tpot_ms']) / unfused_stats['avg_tpot_ms'] * 100
        print(f"{imp:>+10.1f}%")
    else:
        print()

    bl_tps_str = f"{1/baseline_tpot:.1f}" if baseline_tpot else "N/A"
    print(f"{'TPS':<25} {bl_tps_str:>10} "
          f"{unfused_stats['tps']:>10.1f} {fused_stats['tps']:>10.1f} ", end="")
    if unfused_stats['tps'] > 0:
        imp = (fused_stats['tps'] - unfused_stats['tps']) / unfused_stats['tps'] * 100
        print(f"{imp:>+10.1f}%")
    else:
        print()

    print(f"{'Tau':<25} {'—':>10} "
          f"{unfused_stats['tau']:>10.2f} {fused_stats['tau']:>10.2f}")

    print(f"{'Avg step (ms)':<25} {'—':>10} "
          f"{unfused_stats['avg_step_ms']:>10.2f} {fused_stats['avg_step_ms']:>10.2f} ", end="")
    if unfused_stats['avg_step_ms'] > 0:
        imp = (unfused_stats['avg_step_ms'] - fused_stats['avg_step_ms']) / unfused_stats['avg_step_ms'] * 100
        print(f"{imp:>+10.1f}%")
    else:
        print()

    print(f"{'Median step (ms)':<25} {'—':>10} "
          f"{unfused_stats['median_step_ms']:>10.2f} {fused_stats['median_step_ms']:>10.2f} ", end="")
    if unfused_stats['median_step_ms'] > 0:
        imp = (unfused_stats['median_step_ms'] - fused_stats['median_step_ms']) / unfused_stats['median_step_ms'] * 100
        print(f"{imp:>+10.1f}%")
    else:
        print()

    if baseline_tpot:
        unfused_speedup = baseline_tpot / (unfused_stats['avg_tpot_ms'] / 1000) if unfused_stats['avg_tpot_ms'] > 0 else 0
        fused_speedup = baseline_tpot / (fused_stats['avg_tpot_ms'] / 1000) if fused_stats['avg_tpot_ms'] > 0 else 0
        print(f"{'Speedup vs baseline':<25} {'1.00x':>10} "
              f"{unfused_speedup:>9.2f}x {fused_speedup:>9.2f}x")

    # Output quality check
    if unfused_measured and fused_measured:
        n_match = 0
        n_total = min(len(unfused_measured), len(fused_measured))
        for i in range(n_total):
            u_out = unfused_measured[i].output_ids
            f_out = fused_measured[i].output_ids
            min_len = min(len(u_out), len(f_out))
            if min_len > 0 and np.array_equal(u_out[:min_len], f_out[:min_len]):
                n_match += 1
        print(f"\n  Output quality: {n_match}/{n_total} samples match between fused and unfused")

    # Save JSON
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
            },
            "baseline_tpot_ms": baseline_tpot * 1000 if baseline_tpot else None,
            "unfused": unfused_stats,
            "fused": fused_stats,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

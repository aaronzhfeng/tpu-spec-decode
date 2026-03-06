"""Standalone JAX/TPU DFlash speculative decoding benchmark.

Replicates the GPU benchmark (zhongyan_dev/dflash/benchmark.py) in JAX
without using vLLM's serving pipeline. This provides an apples-to-apples
comparison with the paper's reported numbers.

Usage:
    python benchmarks/standalone_dflash.py \
        --target-model Qwen/Qwen3-4B \
        --draft-model z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k --max-samples 8 --max-new-tokens 256
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
# Config mock — minimal VllmConfig for model loading (no vLLM engine)
# ---------------------------------------------------------------------------

class StandaloneVllmConfig:
    """Minimal VllmConfig mock sufficient for get_flax_model()."""

    def __init__(
        self,
        target_model: str,
        draft_model: str,
        kv_cache_dtype: str = "auto",
    ):
        self.model_config = ModelConfig(
            target_model, trust_remote_code=True)
        self.model_config.dtype = jnp.bfloat16
        self.load_config = LoadConfig(load_format="auto")
        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
        self.quant_config = None
        self.additional_config = {}

        # Speculative config for draft model
        self.speculative_config = MagicMock()
        self.speculative_config.draft_model_config = ModelConfig(
            draft_model, trust_remote_code=True)
        self.speculative_config.draft_model_config.dtype = jnp.bfloat16
        self.speculative_config.method = "dflash"


# ---------------------------------------------------------------------------
# Dataset loading (adapted from zhongyan_dev/dflash/model/utils.py)
# ---------------------------------------------------------------------------

def load_and_process_dataset(data_name: str):
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {"formatted_input": (
            f"{x['instruction']}\n\nInput:\n{x['input']}" if x['input']
            else x['instruction'])})
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})
    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = ("Write a solution to the following problem and make sure "
                      "that it passes the tests:\n```python\n{prompt}\n```")
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = ("Problem Statement:\n{problem_statement}\n"
                      "Please fix the issue described above.")
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    elif data_name == "livecodebench":
        from datasets import Features, Sequence, Value
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        urls = [base + fn for fn in [
            "test.jsonl", "test2.jsonl", "test3.jsonl",
            "test4.jsonl", "test5.jsonl", "test6.jsonl"]]
        dataset = load_dataset("json", data_files={"test": urls})["test"]
        def _format_lcb(doc):
            sys_prompt = (
                "You are an expert Python programmer. You will be given a "
                "question (problem specification) and will generate a correct "
                "Python program that matches the specification and passes all "
                "tests. You will NOT return anything except for the program")
            q = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                fmt = "### Format: Use the following code structure:"
                code = f"```python\n{doc['starter_code']}\n```"
            else:
                fmt = "### Format: Write your code in the following format:"
                code = "```python\n# YOUR CODE HERE\n```"
            footer = "### Answer: (use the provided format with backticks)"
            return f"{sys_prompt}\n\n{q}\n\n{fmt}\n{code}\n\n{footer}"
        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [_format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features)
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    return dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def next_padded_size(n: int) -> int:
    """Round n up to the next power-of-two (min 16)."""
    if n <= 16:
        return 16
    p = 16
    while p < n:
        p *= 2
    return p


def pad_context(ctx: np.ndarray) -> np.ndarray:
    """Pad (T, D) array to (T_padded, D) with zero padding."""
    T = ctx.shape[0]
    T_padded = next_padded_size(T)
    if T_padded == T:
        return ctx
    pad = np.zeros((T_padded - T, ctx.shape[1]), dtype=ctx.dtype)
    return np.concatenate([ctx, pad], axis=0)


def tpu_sync():
    """Synchronize TPU for timing."""
    jax.effects_barrier()


def make_attn_metadata(
    input_positions: jnp.ndarray,
    seq_len: int,
    num_query_tokens: int,
    block_tables: jnp.ndarray,
) -> AttentionMetadata:
    """Build AttentionMetadata for a single-sequence chunked prefill."""
    return AttentionMetadata(
        input_positions=input_positions,
        block_tables=block_tables,
        seq_lens=jnp.array([seq_len], dtype=jnp.int32),
        query_start_loc=jnp.array([0, num_query_tokens], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Mesh setup
# ---------------------------------------------------------------------------

def create_mesh(num_devices: int = 1) -> Mesh:
    """Create a JAX mesh.

    For the standalone benchmark we default to 1 device — matching the GPU
    standalone benchmark (1 GPU).  This also avoids the Pallas shard_map
    requirement that arises when flash_attention is auto-partitioned.
    """
    devices = np.array(jax.local_devices()[:num_devices])
    device_mesh = devices.reshape((1, 1, 1, num_devices))
    return Mesh(device_mesh, axis_names=("data", "attn_dp", "expert", "model"))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(config: StandaloneVllmConfig, mesh: Mesh):
    """Load target and draft models via get_flax_model."""
    rng = jax.random.PRNGKey(42)

    def _unpack_flax_result(flax_result):
        """Unpack get_flax_model result (handles 7-tuple old and 8-tuple new)."""
        model_fn = flax_result[0]
        logits_fn = flax_result[1]
        # Old 7-tuple: model_fn, logits, combine, multimodal, state, lora, model
        # New 8-tuple: model_fn, logits, _not_support, combine, multimodal, state, lora, model
        if len(flax_result) == 8:
            combine_fn = flax_result[3]
        else:
            combine_fn = flax_result[2]
        state = flax_result[-3]
        return model_fn, logits_fn, combine_fn, state

    print("[INFO] Loading target model...")
    with jax.default_device(jax.devices()[0]), jax.set_mesh(mesh):
        flax_result = get_flax_model(
            config, rng, mesh, is_draft_model=False)
        target_model_fn, target_logits_fn, target_combine_fn, target_state = \
            _unpack_flax_result(flax_result)

    print("[INFO] Loading draft model...")
    with jax.default_device(jax.devices()[0]), jax.set_mesh(mesh):
        flax_result = get_flax_model(
            config, rng, mesh, is_draft_model=True)
        draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state = \
            _unpack_flax_result(flax_result)

    # Share target embedding with draft model
    target_embed = getattr(target_state.model, "embed_tokens", None)
    if target_embed is not None:
        draft_state.model.embed_tokens = target_embed
        print("[INFO] Shared target embedding with draft model.")

    return (
        target_model_fn, target_logits_fn, target_combine_fn, target_state,
        draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    )


# ---------------------------------------------------------------------------
# KV cache allocation
# ---------------------------------------------------------------------------

def allocate_target_kv_caches(config: StandaloneVllmConfig, mesh: Mesh,
                               max_length: int, page_size: int = 16):
    """Allocate paged KV caches for the target model."""
    hf = config.model_config.hf_config
    num_kv_heads = hf.num_key_value_heads
    head_dim_orig = getattr(hf, "head_dim",
                            hf.hidden_size // hf.num_attention_heads)
    head_dim = utils.get_padded_head_dim(head_dim_orig)
    num_pages = math.ceil(max_length / page_size) + 4
    num_layers = hf.num_hidden_layers
    kv_caches = create_kv_caches(
        num_blocks=num_pages,
        block_size=page_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        mesh=mesh,
        layer_names=["layer"] * num_layers,
    )
    block_tables = jnp.arange(num_pages, dtype=jnp.int32)
    return kv_caches, block_tables


def allocate_draft_kv_caches(config: StandaloneVllmConfig, mesh: Mesh,
                              max_length: int):
    """Allocate static on-device KV caches for the draft model."""
    hf = config.speculative_config.draft_model_config.hf_config
    sharding_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)
    num_heads = utils.get_padded_num_heads(
        hf.num_attention_heads, sharding_size)
    head_dim_orig = getattr(hf, "head_dim",
                            hf.hidden_size // hf.num_attention_heads)
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
# Speculative decoding generate loop
# ---------------------------------------------------------------------------

def dflash_generate(
    # Models
    target_model_fn, target_logits_fn, target_combine_fn, target_state,
    draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    # Caches (will be modified)
    target_kv_caches, block_tables,
    draft_kv_caches,
    # Config
    input_ids_np: np.ndarray,
    mask_token_id: int,
    block_size: int,
    max_new_tokens: int,
    eos_token_id: int,
    hidden_size: int,
    max_model_len: int,
) -> SimpleNamespace:
    """Standalone DFlash speculative decoding generate loop.

    Matches the DFlashProposer interface: the draft model receives the full
    accumulated context tensor (not a tuple) and derives positions from
    attention_metadata.input_positions.
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

    print(f"  [DEBUG] aux_hidden_states type={type(aux_hidden_states)}, "
          f"len={len(aux_hidden_states) if hasattr(aux_hidden_states, '__len__') else 'N/A'}")
    if hasattr(aux_hidden_states, '__len__') and len(aux_hidden_states) > 0:
        print(f"  [DEBUG] aux[0] shape={aux_hidden_states[0].shape}")
    if len(aux_hidden_states) > 0:
        raw = jnp.concatenate(aux_hidden_states, axis=-1)
        projected_all = draft_combine_fn(draft_state, raw)
        print(f"  [DEBUG] projected_all shape={projected_all.shape}")
    else:
        projected_all = None
        print(f"  [DEBUG] projected_all is None — no aux hidden states!")

    tpu_sync()
    time_to_first_token = time.perf_counter() - prefill_start

    # --- Context buffer init ---
    # Store projected prefill features, but prev_ctx_len stays 0 because
    # the draft model hasn't consumed any context yet.
    ctx_buf = np.zeros((max_model_len, hidden_size), dtype=np.float32)
    prev_ctx_len = 0  # how much context the draft model has been fed

    if projected_all is not None:
        proj_np = np.asarray(projected_all, dtype=np.float32)
        n = min(num_input_tokens, max_model_len)
        ctx_buf[:n] = proj_np[:n]
        # Note: prev_ctx_len stays 0 — draft hasn't seen this yet

    # --- Decode loop ---
    decode_start = time.perf_counter()
    start = num_input_tokens
    acceptance_lengths = []
    draft_cache_len = 0
    prev_seq_len = 0
    draft_prefill_done = False

    while start < max_length:
        seq_len = start

        # (a) Crop draft cache to prev_seq_len (DynamicCache.crop semantics)
        draft_cache_len = prev_seq_len

        # (b) Compute new context features for the draft model.
        # Only pass NEW features the draft hasn't seen yet.
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

        # (c) Build noise block: [next_token, mask, mask, ..., mask]
        next_token = output_ids[start]
        noise_ids = np.full(block_size, mask_token_id, dtype=np.int32)
        noise_ids[0] = next_token
        noise_ids_jax = jnp.array(noise_ids, dtype=jnp.int32)

        # (d) Pack target_hidden_states as 3-tuple for Phase 3 model
        ctx_jax = jnp.array(new_ctx_np, dtype=jnp.bfloat16)
        cache_len_arr = jnp.array([draft_cache_len], dtype=jnp.int32)
        ctx_count_arr = jnp.array([actual_ctx_count], dtype=jnp.int32)
        target_hidden = (ctx_jax, cache_len_arr, ctx_count_arr)

        # (e) Draft forward
        noise_positions = jnp.arange(block_size, dtype=jnp.int32) + seq_len
        draft_metadata = make_attn_metadata(
            noise_positions, seq_len + block_size, block_size, block_tables)

        draft_kv_caches, draft_hidden, _ = draft_model_fn(
            draft_state, draft_kv_caches, noise_ids_jax,
            target_hidden, draft_metadata,
        )

        # DEBUG: Print draft hidden stats on first iteration
        if len(acceptance_lengths) == 0:
            dh = jnp.asarray(draft_hidden, dtype=jnp.float32)
            print(f"  [DEBUG] draft_hidden shape={dh.shape}, "
                  f"mean={float(dh.mean()):.6f}, std={float(dh.std()):.6f}, "
                  f"min={float(dh.min()):.6f}, max={float(dh.max()):.6f}")
            print(f"  [DEBUG] ctx shape={new_ctx_np.shape}, "
                  f"cache_len={draft_cache_len}, ctx_count={actual_ctx_count}")

        # Update draft KV cache tracking
        draft_cache_len = draft_cache_len + actual_ctx_count + block_size
        prev_seq_len = seq_len

        # (d) Sample draft tokens using target LM head
        draft_logits = target_logits_fn(
            target_state, draft_hidden[1:block_size], None)
        draft_tokens = np.array(jnp.argmax(draft_logits, axis=-1))

        # DEBUG: Print draft token info on first iteration
        if len(acceptance_lengths) == 0:
            print(f"  [DEBUG] draft_logits shape={draft_logits.shape}, "
                  f"argmax first 5={draft_tokens[:5]}")

        output_ids[start + 1 : start + block_size] = draft_tokens
        block_ids = jnp.array(
            output_ids[start : start + block_size], dtype=jnp.int32)

        if not draft_prefill_done:
            draft_prefill_done = True
            tpu_sync()
            decode_start = time.perf_counter()

        # (e) Verify with target model
        verify_positions = jnp.arange(start, start + block_size, dtype=jnp.int32)
        verify_metadata = make_attn_metadata(
            verify_positions, start + block_size, block_size, block_tables)

        target_kv_caches, verify_hidden, aux_hidden_states = target_model_fn(
            target_state, target_kv_caches, block_ids, verify_metadata,
            None, None, None, None, None, True, True,
        )

        verify_logits = target_logits_fn(target_state, verify_hidden, None)
        posterior = np.array(jnp.argmax(verify_logits, axis=-1))

        # (f) Accept: consecutive matches from position 1
        block_np = np.array(block_ids)
        matches = (block_np[1:] == posterior[:-1]).astype(np.int32)
        acceptance_length = int(np.cumprod(matches).sum())

        output_ids[start : start + acceptance_length + 1] = \
            block_np[:acceptance_length + 1]
        output_ids[start + acceptance_length + 1] = posterior[acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1

        # (g) Update context for next iteration
        if len(aux_hidden_states) > 0:
            raw = jnp.concatenate(aux_hidden_states, axis=-1)
            proj = draft_combine_fn(draft_state, raw)
            # Trim to accepted portion only
            projected_all = proj[:acceptance_length + 1]
        else:
            projected_all = None

        # (h) Check stop
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
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_token,
        acceptance_lengths=acceptance_lengths,
    )


# ---------------------------------------------------------------------------
# Baseline generate (block_size=1, no speculation)
# ---------------------------------------------------------------------------

def baseline_generate(
    target_model_fn, target_logits_fn, target_state,
    target_kv_caches, block_tables,
    input_ids_np: np.ndarray,
    max_new_tokens: int,
    eos_token_id: int,
    mask_token_id: int,
) -> SimpleNamespace:
    """Simple autoregressive generation without speculation."""
    num_input_tokens = len(input_ids_np)
    max_length = num_input_tokens + max_new_tokens

    output_ids = np.full(max_length, mask_token_id, dtype=np.int32)
    output_ids[:num_input_tokens] = input_ids_np

    # Prefill
    prefill_start = time.perf_counter()
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
    time_to_first_token = time.perf_counter() - prefill_start

    # Decode
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
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_token,
        acceptance_lengths=[],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone JAX/TPU DFlash benchmark")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None,
                        help="Block size (default: from draft config)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Only temperature=0 (greedy) supported")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup samples to exclude from metrics")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save results as JSON")
    args = parser.parse_args()

    assert args.temperature < 1e-5, "Only greedy decoding (temperature=0) supported"

    # Setup
    np.random.seed(42)
    mesh = create_mesh()
    init_pp_distributed_environment(
        ip="", rank=0, world_size=1,
        device=jax.devices()[0], need_pp=False,
    )
    print(f"[INFO] JAX devices: {jax.device_count()}")

    # Config
    config = StandaloneVllmConfig(args.target_model, args.draft_model)
    draft_hf = config.speculative_config.draft_model_config.hf_config
    dflash_config = getattr(draft_hf, "dflash_config", {})
    mask_token_id = dflash_config.get("mask_token_id", 0)
    block_size = args.block_size or getattr(draft_hf, "block_size", 16)
    hidden_size = draft_hf.hidden_size

    print(f"[INFO] Block size: {block_size}, mask_token_id: {mask_token_id}")
    print(f"[INFO] Draft hidden_size: {hidden_size}")

    # Load models
    (target_model_fn, target_logits_fn, target_combine_fn, target_state,
     draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    ) = load_models(config, mesh)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    eos_token_id = tokenizer.eos_token_id or 0

    # Dataset
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))
    print(f"[INFO] Dataset: {args.dataset}, samples: {len(dataset)}")

    # Run benchmarks
    max_length = args.max_model_len
    page_size = 16
    responses = []
    quality_results = []

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

        response = {}

        # --- Baseline (block_size=1) ---
        target_kv_caches_b, block_tables_b = allocate_target_kv_caches(
            config, mesh, max_length, page_size)
        response[1] = baseline_generate(
            target_model_fn, target_logits_fn, target_state,
            target_kv_caches_b, block_tables_b,
            input_ids_np, args.max_new_tokens, eos_token_id, mask_token_id,
        )
        print(f"  Baseline: {response[1].num_output_tokens} tokens, "
              f"TPOT={response[1].time_per_output_token*1000:.1f}ms")

        # --- DFlash speculative ---
        target_kv_caches_s, block_tables_s = allocate_target_kv_caches(
            config, mesh, max_length, page_size)
        draft_kv_caches_s = allocate_draft_kv_caches(
            config, mesh, max_length)
        response[block_size] = dflash_generate(
            target_model_fn, target_logits_fn, target_combine_fn, target_state,
            draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
            target_kv_caches_s, block_tables_s,
            draft_kv_caches_s,
            input_ids_np, mask_token_id, block_size,
            args.max_new_tokens, eos_token_id, hidden_size, max_length,
        )
        r = response[block_size]
        tau = np.mean(r.acceptance_lengths) if r.acceptance_lengths else 0
        print(f"  DFlash:   {r.num_output_tokens} tokens, "
              f"TPOT={r.time_per_output_token*1000:.1f}ms, tau={tau:.2f}")

        # --- Output quality check ---
        n_in = len(input_ids_np)
        bl_out = response[1].output_ids[n_in:]
        df_out = r.output_ids[n_in:]
        min_len = min(len(bl_out), len(df_out))
        if min_len > 0 and np.array_equal(bl_out[:min_len], df_out[:min_len]):
            print(f"  Quality:  MATCH (first {min_len} output tokens identical)")
            quality_results.append({
                "sample": idx, "match": True, "min_len": int(min_len),
                "mismatches": [],
            })
        else:
            # Find ALL mismatch positions (not just the first)
            mismatches = []
            for i in range(min_len):
                if bl_out[i] != df_out[i]:
                    mismatches.append({
                        "pos": i,
                        "baseline_id": int(bl_out[i]),
                        "dflash_id": int(df_out[i]),
                        "baseline_text": tokenizer.decode([int(bl_out[i])]),
                        "dflash_text": tokenizer.decode([int(df_out[i])]),
                    })
            first = mismatches[0]
            print(f"  Quality:  MISMATCH — {len(mismatches)} divergent tokens "
                  f"(first at output pos {first['pos']}: "
                  f"baseline='{first['baseline_text']}' vs "
                  f"dflash='{first['dflash_text']}')")
            # Show decoded text window around first mismatch
            ctx_start = max(0, first["pos"] - 5)
            ctx_end = min(min_len, first["pos"] + 10)
            bl_window = tokenizer.decode(bl_out[ctx_start:ctx_end].tolist())
            df_window = tokenizer.decode(df_out[ctx_start:ctx_end].tolist())
            print(f"    Baseline [{ctx_start}:{ctx_end}]: ...{bl_window}...")
            print(f"    DFlash   [{ctx_start}:{ctx_end}]: ...{df_window}...")
            quality_results.append({
                "sample": idx, "match": False, "min_len": int(min_len),
                "mismatches": mismatches,
            })

        # Check if final \boxed{} answers match (for math datasets)
        bl_text = tokenizer.decode(bl_out.tolist())
        df_text = tokenizer.decode(df_out.tolist())
        bl_boxes = re.findall(r'\\boxed\{([^}]*)\}', bl_text)
        df_boxes = re.findall(r'\\boxed\{([^}]*)\}', df_text)
        bl_ans = bl_boxes[-1].strip() if bl_boxes else None
        df_ans = df_boxes[-1].strip() if df_boxes else None
        if bl_ans is not None or df_ans is not None:
            if bl_ans == df_ans:
                print(f"  Answer:   SAME (\\boxed{{{bl_ans}}})")
            else:
                print(f"  Answer:   DIFFER (baseline=\\boxed{{{bl_ans}}}, "
                      f"dflash=\\boxed{{{df_ans}}})")
        quality_results[-1]["baseline_answer"] = bl_ans
        quality_results[-1]["dflash_answer"] = df_ans
        quality_results[-1]["answer_match"] = (bl_ans == df_ans)

        responses.append(response)

    # --- Summary ---
    warmup = min(args.warmup, len(responses))
    measured = responses[warmup:]

    print("\n" + "=" * 60)
    print("RESULTS")
    if warmup > 0:
        print(f"(excluding first {warmup} warmup sample(s))")
    print("=" * 60)

    if not measured:
        print("No measured samples (all were warmup). Increase --max-samples.")
        return

    t1_list = [r[1].time_per_output_token for r in measured]
    tb_list = [r[block_size].time_per_output_token for r in measured]
    t1 = np.mean(t1_list)
    tb = np.mean(tb_list)
    speedup = t1 / tb if tb > 0 else 0

    all_acceptance = []
    for r in measured:
        all_acceptance.extend(r[block_size].acceptance_lengths)
    tau = np.mean(all_acceptance) if all_acceptance else 0

    # Per-position acceptance
    pos_counts = np.zeros(block_size)
    total_drafts = len(all_acceptance)
    for a in all_acceptance:
        for p in range(min(int(a), block_size)):
            pos_counts[p] += 1
    pos_rates = pos_counts / max(total_drafts, 1)

    print(f"\nSamples:        {len(measured)} (+ {warmup} warmup)")
    print(f"Baseline TPOT:  {t1*1000:.2f} ms ({1/t1:.1f} TPS)")
    print(f"DFlash TPOT:    {tb*1000:.2f} ms ({1/tb:.1f} TPS)")
    print(f"Speedup:        {speedup:.2f}x")
    print(f"Tau (avg acceptance length): {tau:.2f}")
    print(f"Total drafts: {total_drafts}")

    print("\nPer-position acceptance rate:")
    for i, rate in enumerate(pos_rates):
        bar = "#" * int(rate * 40)
        print(f"  pos {i:2d}: {rate:.3f} {bar}")

    histogram = [sum(1 for a in all_acceptance if a == b) / max(total_drafts, 1)
                 for b in range(block_size + 1)]
    print(f"\nAcceptance length histogram: "
          f"{[f'{x*100:.1f}%' for x in histogram]}")

    # --- Quality summary ---
    n_match = sum(1 for q in quality_results if q["match"])
    n_total = len(quality_results)
    n_ans_total = sum(1 for q in quality_results
                      if q.get("baseline_answer") is not None
                      or q.get("dflash_answer") is not None)
    n_ans_match = sum(1 for q in quality_results
                      if q.get("answer_match")
                      and (q.get("baseline_answer") is not None
                           or q.get("dflash_answer") is not None))
    print(f"\nOutput quality: {n_match}/{n_total} samples match baseline exactly")
    if n_ans_total > 0:
        print(f"Final answer:   {n_ans_match}/{n_ans_total} samples have same "
              f"\\boxed{{}} answer")
    if n_match < n_total:
        print("  (Mismatches are expected with bf16 speculative decoding — "
              "batch-16 verify vs single-token baseline can produce different "
              "floating-point accumulation, flipping argmax at decision boundaries.)")
        for q in quality_results:
            if not q["match"]:
                mm = q["mismatches"]
                ans_note = ""
                if q.get("baseline_answer") is not None or q.get("dflash_answer") is not None:
                    ans_note = (f", answer={'SAME' if q.get('answer_match') else 'DIFFER'}"
                                f" ({q.get('baseline_answer')} vs {q.get('dflash_answer')})")
                print(f"  Sample {q['sample']}: {len(mm)} mismatched tokens "
                      f"out of {q['min_len']} (first at pos {mm[0]['pos']}{ans_note})")

    # --- Save JSON ---
    if args.output_json:
        per_sample = []
        for idx, r in enumerate(responses):
            bl = r[1]
            df = r[block_size]
            s_tau = float(np.mean(df.acceptance_lengths)) if df.acceptance_lengths else 0
            sample_data = {
                "sample_index": idx,
                "is_warmup": idx < warmup,
                "num_input_tokens": int(bl.num_input_tokens),
                "baseline_num_output_tokens": int(bl.num_output_tokens),
                "baseline_tpot_ms": bl.time_per_output_token * 1000,
                "baseline_tps": 1.0 / bl.time_per_output_token if bl.time_per_output_token > 0 else 0,
                "dflash_num_output_tokens": int(df.num_output_tokens),
                "dflash_tpot_ms": df.time_per_output_token * 1000,
                "dflash_tps": 1.0 / df.time_per_output_token if df.time_per_output_token > 0 else 0,
                "tau": s_tau,
                "num_drafts": len(df.acceptance_lengths),
                "acceptance_lengths": [int(a) for a in df.acceptance_lengths],
                "baseline_text": tokenizer.decode(bl.output_ids[bl.num_input_tokens:].tolist(), skip_special_tokens=True),
                "dflash_text": tokenizer.decode(df.output_ids[df.num_input_tokens:].tolist(), skip_special_tokens=True),
                "prompt_text": tokenizer.decode(bl.output_ids[:bl.num_input_tokens].tolist(), skip_special_tokens=True),
            }
            # Add quality data if available
            if idx < len(quality_results):
                q = quality_results[idx]
                sample_data["quality_match"] = q["match"]
                sample_data["quality_mismatches"] = q["mismatches"]
                sample_data["baseline_answer"] = q.get("baseline_answer")
                sample_data["dflash_answer"] = q.get("dflash_answer")
                sample_data["answer_match"] = q.get("answer_match")
            per_sample.append(sample_data)

        result = {
            "config": {
                "target_model": args.target_model,
                "draft_model": args.draft_model,
                "dataset": args.dataset,
                "block_size": block_size,
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "temperature": args.temperature,
                "warmup_samples": warmup,
                "measured_samples": len(measured),
                "total_samples": len(responses),
                "num_devices": 1,
            },
            "summary": {
                "baseline_tpot_ms": t1 * 1000,
                "baseline_tps": 1.0 / t1 if t1 > 0 else 0,
                "dflash_tpot_ms": tb * 1000,
                "dflash_tps": 1.0 / tb if tb > 0 else 0,
                "speedup": speedup,
                "tau": float(tau),
                "total_drafts": total_drafts,
                "acceptance_rate_per_pos": [float(r) for r in pos_rates],
                "acceptance_histogram": [float(h) for h in histogram],
            },
            "quality": {
                "total_samples": n_total,
                "exact_matches": n_match,
                "match_rate": n_match / max(n_total, 1),
                "answer_matches": n_ans_match,
                "answer_total": n_ans_total,
            },
            "per_sample": per_sample,
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

"""Ablation study: isolate optimization targets for DFlash on TPU.

Tests three hypotheses by measuring step time under different conditions:

  Hypothesis 1 (LM Head): The two LM head matmuls (hidden @ vocab_matrix)
  are the main optimization target (~30% of step time).
  -> Test: microbenchmark the raw matmul, then run DFlash with/without
     the draft LM head call.

  Hypothesis 2 (Host Loop): Host-side Python orchestration adds latency
  that jax.lax.while_loop could eliminate.
  -> Test: measure pure host loop overhead with dummy device ops.

  Hypothesis 3 (vLLM Pipeline): The tau gap (6.67 standalone vs 4.48 vLLM)
  comes from vLLM-specific overhead, not the spec decode loop itself.
  -> Test: compare standalone step time vs what we'd expect from vLLM's
     tau, to bound where the gap originates.

Usage:
    python benchmarks/ablation_study.py \
        --target-model Qwen/Qwen3-4B \
        --draft-model z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k --max-samples 3 --max-new-tokens 128
"""

import argparse
import json
import math
import os
import time
from types import SimpleNamespace
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
# Shared infra (minimal)
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
    max_kv_len = next_padded_size(max_length)
    caches = []
    for _ in range(hf.num_hidden_layers):
        caches.extend([
            jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16),
            jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16),
        ])
    return caches


# ===================================================================
# TEST 1: LM Head Microbenchmark
# ===================================================================

def test_lm_head_microbenchmark(target_state, target_logits_fn):
    """Time the raw LM head matmul at different batch sizes."""
    print("\n" + "=" * 60)
    print("TEST 1: LM Head Microbenchmark")
    print("=" * 60)

    embed_weight = target_state.model.embed_tokens.weight.value
    vocab_size = embed_weight.shape[0]
    hidden_size = embed_weight.shape[1]
    print(f"  Vocab: {vocab_size}, Hidden: {hidden_size}")
    print(f"  Matmul: ({{}}, {hidden_size}) @ ({hidden_size}, {vocab_size})")

    results = {}
    for batch in [1, 4, 8, 15, 16, 32]:
        fake_hidden = jnp.ones((batch, hidden_size), dtype=jnp.bfloat16)

        # Warmup
        for _ in range(3):
            logits = target_logits_fn(target_state, fake_hidden, None)
            _ = jnp.argmax(logits, axis=-1)
        tpu_sync()

        # Measure
        times = []
        for _ in range(20):
            tpu_sync()
            t0 = time.perf_counter()
            logits = target_logits_fn(target_state, fake_hidden, None)
            token_ids = jnp.argmax(logits, axis=-1)
            tpu_sync()
            times.append(time.perf_counter() - t0)

        avg = np.mean(times) * 1000
        std = np.std(times) * 1000
        results[batch] = {"avg_ms": avg, "std_ms": std}
        print(f"  batch={batch:>3}: {avg:.2f} +/- {std:.2f} ms")

    # Also test just argmax without logits
    fake_logits = jnp.ones((16, vocab_size), dtype=jnp.bfloat16)
    for _ in range(3):
        _ = jnp.argmax(fake_logits, axis=-1)
    tpu_sync()
    times = []
    for _ in range(20):
        tpu_sync()
        t0 = time.perf_counter()
        _ = jnp.argmax(fake_logits, axis=-1)
        tpu_sync()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times) * 1000
    print(f"  argmax only (16, {vocab_size}): {avg:.2f} ms")
    results["argmax_only"] = {"avg_ms": avg}

    # Test just the matmul without argmax
    fake_hidden = jnp.ones((16, hidden_size), dtype=jnp.bfloat16)
    for _ in range(3):
        _ = jnp.dot(fake_hidden, embed_weight.T)
    tpu_sync()
    times = []
    for _ in range(20):
        tpu_sync()
        t0 = time.perf_counter()
        _ = jnp.dot(fake_hidden, embed_weight.T)
        tpu_sync()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times) * 1000
    print(f"  matmul only (16, {hidden_size}) @ ({hidden_size}, {vocab_size}): {avg:.2f} ms")
    results["matmul_only"] = {"avg_ms": avg}

    return results


# ===================================================================
# TEST 2: Host Loop Overhead
# ===================================================================

def test_host_loop_overhead():
    """Measure pure Python loop + tiny device op overhead per iteration."""
    print("\n" + "=" * 60)
    print("TEST 2: Host Loop Overhead")
    print("=" * 60)

    # Test 1: Empty Python loop
    n_iters = 1000
    t0 = time.perf_counter()
    x = 0
    for i in range(n_iters):
        x += 1  # trivial work
    empty_loop = (time.perf_counter() - t0) / n_iters * 1000
    print(f"  Empty Python loop:    {empty_loop:.4f} ms/iter")

    # Test 2: Loop with small jnp operations (simulates the host-side work)
    a = jnp.ones(16, dtype=jnp.int32)
    tpu_sync()

    t0 = time.perf_counter()
    for i in range(n_iters):
        b = jnp.array([i], dtype=jnp.int32)
        c = jnp.arange(16, dtype=jnp.int32) + b[0]
        d = jnp.concatenate([a[:1], c[1:]])
    tpu_sync()
    small_ops = (time.perf_counter() - t0) / n_iters * 1000
    print(f"  Small jnp ops/iter:   {small_ops:.4f} ms/iter")

    # Test 3: Loop with jnp.array from numpy (host→device transfer)
    np_arr = np.ones((16, 2560), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(200):
        j = jnp.array(np_arr, dtype=jnp.bfloat16)
    tpu_sync()
    h2d = (time.perf_counter() - t0) / 200 * 1000
    print(f"  H→D transfer (16,2560) bf16: {h2d:.4f} ms/iter")

    # Test 4: Loop with device→host transfer
    dev_arr = jnp.ones(16, dtype=jnp.int32)
    t0 = time.perf_counter()
    for i in range(200):
        _ = np.array(dev_arr)
    d2h = (time.perf_counter() - t0) / 200 * 1000
    print(f"  D→H transfer (16,) int32:    {d2h:.4f} ms/iter")

    # Test 5: numpy context buffer ops (simulating the host-side work in spec decode)
    ctx_buf = np.zeros((4096, 2560), dtype=np.float32)
    proj = np.random.randn(16, 2560).astype(np.float32)
    t0 = time.perf_counter()
    for i in range(1000):
        pos = (i * 16) % 4000
        ctx_buf[pos:pos+16] = proj
        new_ctx = proj.copy()
        padded = pad_context(new_ctx)
    numpy_work = (time.perf_counter() - t0) / 1000 * 1000
    print(f"  Numpy ctx buffer ops: {numpy_work:.4f} ms/iter")

    # Test 6: Full simulated host iteration (all host-side work, no model calls)
    noise = np.full(16, 0, dtype=np.int32)
    t0 = time.perf_counter()
    for i in range(500):
        # Build noise block
        noise[0] = i % 100
        noise_jax = jnp.array(noise, dtype=jnp.int32)
        # Build context
        ctx_jax = jnp.array(np.zeros((16, 2560), dtype=np.float32), dtype=jnp.bfloat16)
        cache_len = jnp.array([i], dtype=jnp.int32)
        ctx_count = jnp.array([8], dtype=jnp.int32)
        # Build metadata
        positions = jnp.arange(16, dtype=jnp.int32) + i
        metadata = AttentionMetadata(
            input_positions=positions,
            block_tables=jnp.arange(260, dtype=jnp.int32),
            seq_lens=jnp.array([i + 16], dtype=jnp.int32),
            query_start_loc=jnp.array([0, 16], dtype=jnp.int32),
            request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        )
        # Simulate acceptance
        block_ids = jnp.arange(16, dtype=jnp.int32)
        posterior = jnp.arange(16, dtype=jnp.int32)
        matches = (block_ids[1:] == posterior[:-1])
        acc_len = int(jnp.sum(jnp.cumprod(matches.astype(jnp.int32))))
    tpu_sync()
    full_host = (time.perf_counter() - t0) / 500 * 1000
    print(f"  Full host iteration:  {full_host:.4f} ms/iter")

    return {
        "empty_loop_ms": empty_loop,
        "small_ops_ms": small_ops,
        "h2d_transfer_ms": h2d,
        "d2h_transfer_ms": d2h,
        "numpy_work_ms": numpy_work,
        "full_host_iter_ms": full_host,
    }


# ===================================================================
# TEST 3: DFlash with/without Draft LM Head
# ===================================================================

def test_skip_draft_lm_head(
    target_model_fn, target_logits_fn, target_combine_fn, target_state,
    draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    config, mesh, tokenizer, dataset, max_model_len, block_size,
    mask_token_id, hidden_size, max_new_tokens, warmup,
):
    """Compare step time with real vs random draft tokens.

    'Random draft' skips the draft LM head entirely, giving us the cost
    of everything EXCEPT the draft LM head matmul+argmax.
    """
    print("\n" + "=" * 60)
    print("TEST 3: DFlash With vs Without Draft LM Head")
    print("=" * 60)

    eos_token_id = tokenizer.eos_token_id or 0
    page_size = 16
    results = {}

    for mode in ["full", "skip_draft_lm_head"]:
        print(f"\n  Mode: {mode}")
        all_step_times = []
        all_acc = []

        for idx in range(len(dataset)):
            instance = dataset[idx]
            prompt = instance["turns"][0]
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            input_ids = np.array(tokenizer.encode(input_text), dtype=np.int32)

            target_kv, bt = allocate_target_kv_caches(config, mesh, max_model_len, page_size)
            draft_kv = allocate_draft_kv_caches(config, mesh, max_model_len)

            # Prefill
            input_ids_jax = jnp.array(input_ids, dtype=jnp.int32)
            positions = jnp.arange(len(input_ids), dtype=jnp.int32)
            metadata = make_attn_metadata(positions, len(input_ids), len(input_ids), bt)
            target_kv, hidden_states, aux_hidden_states = target_model_fn(
                target_state, target_kv, input_ids_jax, metadata,
                None, None, None, None, None, True, True)
            logits = target_logits_fn(target_state, hidden_states[-1:], None)
            first_token = int(jnp.argmax(logits, axis=-1)[0])

            output_ids = np.full(len(input_ids) + max_new_tokens + block_size,
                                 mask_token_id, dtype=np.int32)
            output_ids[:len(input_ids)] = input_ids
            output_ids[len(input_ids)] = first_token

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
                n = min(len(input_ids), max_model_len)
                ctx_buf[:n] = proj_np[:n]

            start = len(input_ids)
            max_length = len(input_ids) + max_new_tokens
            draft_cache_len = 0
            prev_seq_len = 0
            draft_prefill_done = False
            step_times = []
            acceptance_lengths = []

            while start < max_length:
                t_step = time.perf_counter()
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
                target_hidden = (ctx_jax,
                                 jnp.array([draft_cache_len], dtype=jnp.int32),
                                 jnp.array([actual_ctx_count], dtype=jnp.int32))
                dummy_pos = jnp.arange(block_size, dtype=jnp.int32) + seq_len
                dummy_meta = make_attn_metadata(
                    dummy_pos, seq_len + block_size, block_size, bt)

                # Draft forward
                draft_kv, draft_hidden, _ = draft_model_fn(
                    draft_state, draft_kv, noise_ids_jax, target_hidden, dummy_meta)
                draft_cache_len = draft_cache_len + actual_ctx_count + block_size
                prev_seq_len = seq_len

                if mode == "full":
                    # Full: compute draft logits + argmax
                    draft_logits = target_logits_fn(
                        target_state, draft_hidden[1:block_size], None)
                    draft_tokens = np.array(jnp.argmax(draft_logits, axis=-1))
                else:
                    # Skip: use random tokens (no LM head call)
                    draft_tokens = np.random.randint(0, 1000, size=block_size - 1,
                                                      ).astype(np.int32)

                output_ids[start + 1 : start + block_size] = draft_tokens
                block_ids = jnp.array(
                    output_ids[start : start + block_size], dtype=jnp.int32)

                if not draft_prefill_done:
                    draft_prefill_done = True
                    tpu_sync()
                    continue

                # Verify
                verify_pos = jnp.arange(start, start + block_size, dtype=jnp.int32)
                verify_meta = make_attn_metadata(
                    verify_pos, start + block_size, block_size, bt)
                target_kv, verify_hidden, aux_hidden_states = target_model_fn(
                    target_state, target_kv, block_ids, verify_meta,
                    None, None, None, None, None, True, True)

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

                tpu_sync()
                step_times.append(time.perf_counter() - t_step)

                if eos_token_id in output_ids[len(input_ids) : start + 1]:
                    break

            if idx >= warmup:
                all_step_times.extend(step_times)
                all_acc.extend(acceptance_lengths)

            tau = np.mean(acceptance_lengths) if acceptance_lengths else 0
            print(f"    [{idx+1}/{len(dataset)}] steps={len(step_times)}, "
                  f"tau={tau:.2f}, avg_step={np.mean(step_times)*1000:.1f}ms")

        avg_step = np.mean(all_step_times) * 1000 if all_step_times else 0
        tau = np.mean(all_acc) if all_acc else 0
        results[mode] = {
            "avg_step_ms": avg_step,
            "median_step_ms": np.median(all_step_times) * 1000 if all_step_times else 0,
            "tau": tau,
            "total_steps": len(all_step_times),
        }
        print(f"  {mode}: avg_step={avg_step:.2f}ms, tau={tau:.2f}")

    # Compare
    if "full" in results and "skip_draft_lm_head" in results:
        diff = results["full"]["avg_step_ms"] - results["skip_draft_lm_head"]["avg_step_ms"]
        pct = diff / results["full"]["avg_step_ms"] * 100 if results["full"]["avg_step_ms"] > 0 else 0
        print(f"\n  Draft LM head cost: {diff:.2f} ms/step ({pct:.1f}% of step)")
        results["draft_lm_head_cost_ms"] = diff

    return results


# ===================================================================
# TEST 4: Component timing (no sync barriers between, natural pipelining)
# ===================================================================

def test_component_overlap(
    target_model_fn, target_logits_fn, target_state,
    draft_model_fn, draft_combine_fn, draft_state,
    config, mesh, block_size, mask_token_id, hidden_size,
):
    """Measure individual components WITH natural JAX pipelining.

    Unlike the profiling benchmark (Doc 27), we do NOT insert sync barriers
    between sub-operations. Instead we measure:
      A: draft_forward alone
      B: draft_forward + draft_logits
      C: verify_forward alone
      D: verify_forward + verify_logits + argmax
      E: full step (A+B+C+D+acceptance)

    The differences tell us what actually overlaps.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Component Overlap Analysis")
    print("=" * 60)

    max_len = 2048
    page_size = 16
    target_kv, bt = allocate_target_kv_caches(config, mesh, max_len, page_size)
    draft_kv = allocate_draft_kv_caches(config, mesh, max_len)

    # Dummy inputs
    noise_ids = jnp.full(block_size, mask_token_id, dtype=jnp.int32)
    noise_ids = noise_ids.at[0].set(42)
    ctx = jnp.zeros((16, hidden_size), dtype=jnp.bfloat16)
    cache_len = jnp.array([0], dtype=jnp.int32)
    ctx_count = jnp.array([8], dtype=jnp.int32)
    target_hidden = (ctx, cache_len, ctx_count)
    dummy_pos = jnp.arange(block_size, dtype=jnp.int32)
    draft_meta = make_attn_metadata(dummy_pos, block_size, block_size, bt)

    # Need a prefilled target KV first
    fake_input = jnp.ones(64, dtype=jnp.int32)
    prefill_pos = jnp.arange(64, dtype=jnp.int32)
    prefill_meta = make_attn_metadata(prefill_pos, 64, 64, bt)
    target_kv, _, aux = target_model_fn(
        target_state, target_kv, fake_input, prefill_meta,
        None, None, None, None, None, True, True)
    tpu_sync()

    verify_pos = jnp.arange(64, 64 + block_size, dtype=jnp.int32)
    verify_meta = make_attn_metadata(verify_pos, 64 + block_size, block_size, bt)
    block_ids = jnp.ones(block_size, dtype=jnp.int32)

    n_warmup = 5
    n_measure = 30
    results = {}

    # A: draft_forward only
    # NOTE: draft_model_fn donates KV caches, so we must chain outputs back.
    draft_kv_a = allocate_draft_kv_caches(config, mesh, max_len)
    for _ in range(n_warmup):
        draft_kv_a, dh, _ = draft_model_fn(
            draft_state, draft_kv_a, noise_ids, target_hidden, draft_meta)
    tpu_sync()
    times = []
    for _ in range(n_measure):
        tpu_sync()
        t0 = time.perf_counter()
        draft_kv_a, dh, _ = draft_model_fn(
            draft_state, draft_kv_a, noise_ids, target_hidden, draft_meta)
        tpu_sync()
        times.append(time.perf_counter() - t0)
    results["A_draft_forward"] = np.mean(times) * 1000
    print(f"  A) draft_forward:                {results['A_draft_forward']:.2f} ms")

    # B: draft_forward + draft_logits + argmax
    draft_kv_b = allocate_draft_kv_caches(config, mesh, max_len)
    for _ in range(n_warmup):
        draft_kv_b, dh, _ = draft_model_fn(
            draft_state, draft_kv_b, noise_ids, target_hidden, draft_meta)
        dl = target_logits_fn(target_state, dh[1:block_size], None)
        dt = jnp.argmax(dl, axis=-1)
    tpu_sync()
    times = []
    for _ in range(n_measure):
        tpu_sync()
        t0 = time.perf_counter()
        draft_kv_b, dh, _ = draft_model_fn(
            draft_state, draft_kv_b, noise_ids, target_hidden, draft_meta)
        dl = target_logits_fn(target_state, dh[1:block_size], None)
        dt = jnp.argmax(dl, axis=-1)
        tpu_sync()
        times.append(time.perf_counter() - t0)
    results["B_draft_full"] = np.mean(times) * 1000
    results["B_draft_lm_head"] = results["B_draft_full"] - results["A_draft_forward"]
    print(f"  B) draft + logits + argmax:      {results['B_draft_full']:.2f} ms "
          f"(LM head adds {results['B_draft_lm_head']:.2f} ms)")

    # C: verify_forward only
    # NOTE: target_model_fn also donates KV caches, chain outputs back.
    target_kv_c = target_kv  # start from prefilled cache
    for _ in range(n_warmup):
        target_kv_c, vh, aux2 = target_model_fn(
            target_state, target_kv_c, block_ids, verify_meta,
            None, None, None, None, None, True, True)
    tpu_sync()
    times = []
    for _ in range(n_measure):
        tpu_sync()
        t0 = time.perf_counter()
        target_kv_c, vh, aux2 = target_model_fn(
            target_state, target_kv_c, block_ids, verify_meta,
            None, None, None, None, None, True, True)
        tpu_sync()
        times.append(time.perf_counter() - t0)
    results["C_verify_forward"] = np.mean(times) * 1000
    print(f"  C) verify_forward:               {results['C_verify_forward']:.2f} ms")

    # D: verify_forward + logits + argmax
    for _ in range(n_warmup):
        target_kv_c, vh, aux2 = target_model_fn(
            target_state, target_kv_c, block_ids, verify_meta,
            None, None, None, None, None, True, True)
        vl = target_logits_fn(target_state, vh, None)
        vt = jnp.argmax(vl, axis=-1)
    tpu_sync()
    times = []
    for _ in range(n_measure):
        tpu_sync()
        t0 = time.perf_counter()
        target_kv_c, vh, aux2 = target_model_fn(
            target_state, target_kv_c, block_ids, verify_meta,
            None, None, None, None, None, True, True)
        vl = target_logits_fn(target_state, vh, None)
        vt = jnp.argmax(vl, axis=-1)
        tpu_sync()
        times.append(time.perf_counter() - t0)
    results["D_verify_full"] = np.mean(times) * 1000
    results["D_verify_lm_head"] = results["D_verify_full"] - results["C_verify_forward"]
    print(f"  D) verify + logits + argmax:     {results['D_verify_full']:.2f} ms "
          f"(LM head adds {results['D_verify_lm_head']:.2f} ms)")

    # E: aux projection
    if len(aux2) > 0:
        raw = jnp.concatenate(aux2, axis=-1)
        for _ in range(n_warmup):
            p = draft_combine_fn(draft_state, raw)
        tpu_sync()
        times = []
        for _ in range(n_measure):
            tpu_sync()
            t0 = time.perf_counter()
            p = draft_combine_fn(draft_state, raw)
            tpu_sync()
            times.append(time.perf_counter() - t0)
        results["E_aux_proj"] = np.mean(times) * 1000
        print(f"  E) aux_projection:               {results['E_aux_proj']:.2f} ms")

    # Sum
    total_parts = (results.get("B_draft_full", 0) + results.get("D_verify_full", 0)
                   + results.get("E_aux_proj", 0))
    print(f"\n  Sum of parts: {total_parts:.2f} ms")
    print(f"  Actual full step (from earlier): ~16-17 ms")
    print(f"  This shows how much pipelining JAX already does")

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation study for DFlash TPU")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
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

    (target_model_fn, target_logits_fn, target_combine_fn, target_state,
     draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    ) = load_models(config, mesh)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))

    all_results = {}

    # Test 1: LM Head Microbenchmark
    all_results["lm_head"] = test_lm_head_microbenchmark(target_state, target_logits_fn)

    # Test 2: Host Loop Overhead
    all_results["host_loop"] = test_host_loop_overhead()

    # Test 3: With/without Draft LM Head
    all_results["skip_lm_head"] = test_skip_draft_lm_head(
        target_model_fn, target_logits_fn, target_combine_fn, target_state,
        draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
        config, mesh, tokenizer, dataset, args.max_model_len, block_size,
        mask_token_id, hidden_size, args.max_new_tokens, args.warmup)

    # Test 4: Component Overlap
    all_results["components"] = test_component_overlap(
        target_model_fn, target_logits_fn, target_state,
        draft_model_fn, draft_combine_fn, draft_state,
        config, mesh, block_size, mask_token_id, hidden_size)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    lm = all_results["lm_head"]
    host = all_results["host_loop"]
    skip = all_results["skip_lm_head"]
    comp = all_results["components"]

    print(f"\n  LM Head (batch=15, logits+argmax): {lm.get(15, lm.get(16, {})).get('avg_ms', 0):.2f} ms")
    print(f"  LM Head (matmul only):             {lm.get('matmul_only', {}).get('avg_ms', 0):.2f} ms")
    print(f"  LM Head (argmax only):             {lm.get('argmax_only', {}).get('avg_ms', 0):.2f} ms")
    print(f"  Host loop overhead:                {host['full_host_iter_ms']:.2f} ms/iter")
    print(f"  Draft LM head cost (in loop):      {skip.get('draft_lm_head_cost_ms', 0):.2f} ms/step")
    print(f"  Draft forward (isolated):          {comp.get('A_draft_forward', 0):.2f} ms")
    print(f"  Verify forward (isolated):         {comp.get('C_verify_forward', 0):.2f} ms")
    print(f"  Verify + LM head (isolated):       {comp.get('D_verify_full', 0):.2f} ms")

    print(f"\n  Verdict:")
    draft_lm_cost = skip.get('draft_lm_head_cost_ms', 0)
    full_step = skip.get("full", {}).get("avg_step_ms", 17)
    if draft_lm_cost > 0:
        pct = draft_lm_cost / full_step * 100
        print(f"    LM Head optimization ceiling: {pct:.1f}% of step time")
    print(f"    Host loop overhead: {host['full_host_iter_ms']:.2f} ms "
          f"({host['full_host_iter_ms']/full_step*100:.1f}% of step)")
    print(f"    jax.lax.while_loop ceiling: ~{host['full_host_iter_ms']:.1f} ms savings/step")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        # Convert numpy values to native Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        with open(args.output_json, "w") as f:
            json.dump(convert(all_results), f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

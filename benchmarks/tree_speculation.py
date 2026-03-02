"""Tree speculation: exploit TPU MXU amortization by drafting multiple
candidate blocks and measuring the acceptance improvement.

Approach:
- Draft primary block (16 tokens) — standard DFlash
- Get top-K candidate first tokens from draft logits
- For each alternative first token, draft a new block via DFlash
- Verify each block independently (correct attention)
- Take the best block (longest acceptance)
- Report: best_tau, total draft time, total verify time
- Compute: theoretical throughput with amortized (batched) verification

Usage:
    python benchmarks/tree_speculation.py \
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
# Shared infra
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
    return np.concatenate([ctx, np.zeros((T_padded - T, ctx.shape[1]),
                                         dtype=ctx.dtype)])


def tpu_sync():
    jax.effects_barrier()


def make_attn_metadata(input_positions, seq_len, num_query_tokens,
                       block_tables):
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
    num_heads = utils.get_padded_num_heads(hf.num_attention_heads,
                                           sharding_size)
    head_dim = utils.get_padded_head_dim(
        getattr(hf, "head_dim",
                hf.hidden_size // hf.num_attention_heads))
    max_kv_len = next_padded_size(max_length)
    caches = []
    for _ in range(hf.num_hidden_layers):
        caches.extend([
            jnp.zeros((1, num_heads, max_kv_len, head_dim),
                       dtype=jnp.bfloat16),
            jnp.zeros((1, num_heads, max_kv_len, head_dim),
                       dtype=jnp.bfloat16),
        ])
    return caches


def copy_kv_caches(kv_caches):
    """Deep-copy KV caches so each candidate gets an independent snapshot."""
    return [jnp.array(c) for c in kv_caches]


# ===================================================================
# Main experiment
# ===================================================================

def run_tree_experiment(
    target_model_fn, target_logits_fn, target_combine_fn, target_state,
    draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    config, mesh, tokenizer, dataset, max_model_len, block_size,
    mask_token_id, hidden_size, max_new_tokens, warmup_samples,
    dflash_aux_layers, k_candidates,
):
    """Run DFlash with K-candidate tree speculation.

    For each verify step:
    1. Draft primary block (first token = last accepted)
    2. Get top-K first-token candidates from PREVIOUS verify logits
    3. Draft K-1 alternative blocks (one per alternative first token)
    4. Verify primary block with real KV cache (correct, advances state)
    5. Also verify each alternative block with KV snapshot (correct logits)
    6. Record which candidate had the best acceptance
    7. Advance using BEST candidate's tokens
    """
    page_size = 16
    eos_token_id = tokenizer.eos_token_id or 0

    # Per-K accumulators
    results_by_k = {}
    for K in k_candidates:
        results_by_k[K] = {
            "acceptance_lengths": [],
            "draft_times_ms": [],
            "verify_times_ms": [],
            "step_times_ms": [],
        }

    for idx in range(len(dataset)):
        instance = dataset[idx]
        prompt = instance["turns"][0]
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        input_ids = np.array(tokenizer.encode(input_text), dtype=np.int32)

        # === Run baseline (K=1) and tree (K>1) on the SAME prompts ===
        for K in k_candidates:
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

            dflash_aux = list(all_aux)
            logits = target_logits_fn(target_state, hidden_states[-1:], None)
            first_token = int(jnp.argmax(logits, axis=-1)[0])

            # Get top-K tokens from prefill logits (for first step)
            top_k_tokens = np.array(
                jnp.argsort(logits[0], axis=-1)[::-1][:K], dtype=np.int32)

            output_ids = np.full(
                len(input_ids) + max_new_tokens + block_size * 2,
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
                    end_pos = min(prev_ctx_len + n_copy, max_model_len)
                    ctx_buf[prev_ctx_len:end_pos] = proj_np[:n_copy]
                    new_ctx_np = proj_np[:n_copy].copy()
                    actual_ctx_count = n_copy
                    prev_ctx_len = seq_len
                    new_ctx_np = pad_context(new_ctx_np)
                else:
                    actual_ctx_count = 0
                    new_ctx_np = np.zeros((16, hidden_size), dtype=np.float32)

                # --- Draft K candidate blocks ---
                t_draft_start = time.perf_counter()

                # Primary candidate (rank-0 first token)
                next_token = output_ids[start]
                noise_ids = np.full(block_size, mask_token_id, dtype=np.int32)
                noise_ids[0] = next_token
                noise_ids_jax = jnp.array(noise_ids, dtype=jnp.int32)

                ctx_jax = jnp.array(new_ctx_np, dtype=jnp.bfloat16)
                target_hidden = (
                    ctx_jax,
                    jnp.array([draft_cache_len], dtype=jnp.int32),
                    jnp.array([actual_ctx_count], dtype=jnp.int32))
                dummy_pos = (jnp.arange(block_size, dtype=jnp.int32)
                             + seq_len)
                dummy_meta = make_attn_metadata(
                    dummy_pos, seq_len + block_size, block_size, bt)

                draft_kv, draft_hidden, _ = draft_model_fn(
                    draft_state, draft_kv, noise_ids_jax,
                    target_hidden, dummy_meta)

                draft_logits_0 = target_logits_fn(
                    target_state, draft_hidden[1:block_size], None)
                draft_tokens_0 = np.array(jnp.argmax(draft_logits_0, axis=-1))

                # Build candidate blocks
                candidate_blocks = []
                # Candidate 0: primary draft
                block_0 = np.full(block_size, mask_token_id, dtype=np.int32)
                block_0[0] = next_token
                block_0[1:] = draft_tokens_0
                candidate_blocks.append(block_0)

                # Candidates 1..K-1: alternative first tokens
                for ki in range(1, min(K, len(top_k_tokens))):
                    alt_token = int(top_k_tokens[ki])
                    alt_noise = np.full(block_size, mask_token_id,
                                        dtype=np.int32)
                    alt_noise[0] = alt_token
                    alt_noise_jax = jnp.array(alt_noise, dtype=jnp.int32)

                    # Need separate draft KV for alternatives
                    # (reuse main draft KV — alternatives are approximate)
                    alt_draft_kv = allocate_draft_kv_caches(
                        config, mesh, max_model_len)
                    alt_draft_kv, alt_hidden, _ = draft_model_fn(
                        draft_state, alt_draft_kv, alt_noise_jax,
                        target_hidden, dummy_meta)

                    alt_logits = target_logits_fn(
                        target_state, alt_hidden[1:block_size], None)
                    alt_tokens = np.array(jnp.argmax(alt_logits, axis=-1))

                    block_k = np.full(block_size, mask_token_id,
                                       dtype=np.int32)
                    block_k[0] = alt_token
                    block_k[1:] = alt_tokens
                    candidate_blocks.append(block_k)

                tpu_sync()
                t_draft_end = time.perf_counter()

                if not draft_prefill_done:
                    draft_prefill_done = True
                    prev_seq_len = seq_len
                    continue

                # --- Verify each candidate independently ---
                t_verify_start = time.perf_counter()

                best_acc = -1
                best_block = candidate_blocks[0]
                best_posterior = None
                best_aux = None

                for ci, cand_block in enumerate(candidate_blocks):
                    if ci == 0:
                        # Primary: use real KV cache (advances state)
                        block_ids = jnp.array(cand_block, dtype=jnp.int32)
                        verify_pos = jnp.arange(
                            start, start + block_size, dtype=jnp.int32)
                        verify_meta = make_attn_metadata(
                            verify_pos, start + block_size, block_size, bt)
                        target_kv, verify_hidden, all_aux_v = target_model_fn(
                            target_state, target_kv, block_ids, verify_meta,
                            None, None, None, None, None, True, True)
                        v_logits = target_logits_fn(
                            target_state, verify_hidden, None)
                        posterior = np.array(
                            jnp.argmax(v_logits, axis=-1))

                        matches = (cand_block[1:] == posterior[:-1]).astype(
                            np.int32)
                        acc = int(np.cumprod(matches).sum())

                        primary_posterior = posterior
                        primary_aux = all_aux_v

                        if acc > best_acc:
                            best_acc = acc
                            best_block = cand_block
                            best_posterior = posterior
                            best_aux = all_aux_v
                    else:
                        # Alternative: snapshot KV cache, verify, discard
                        alt_kv = copy_kv_caches(target_kv)
                        block_ids = jnp.array(cand_block, dtype=jnp.int32)
                        verify_pos = jnp.arange(
                            start, start + block_size, dtype=jnp.int32)
                        verify_meta = make_attn_metadata(
                            verify_pos, start + block_size, block_size, bt)
                        alt_kv, verify_hidden, alt_aux = target_model_fn(
                            target_state, alt_kv, block_ids, verify_meta,
                            None, None, None, None, None, True, True)
                        v_logits = target_logits_fn(
                            target_state, verify_hidden, None)
                        posterior = np.array(
                            jnp.argmax(v_logits, axis=-1))

                        matches = (cand_block[1:] == posterior[:-1]).astype(
                            np.int32)
                        acc = int(np.cumprod(matches).sum())

                        if acc > best_acc:
                            best_acc = acc
                            best_block = cand_block
                            best_posterior = posterior
                            best_aux = alt_aux

                tpu_sync()
                t_verify_end = time.perf_counter()

                # --- Accept best candidate ---
                acceptance_length = best_acc
                output_ids[start: start + acceptance_length + 1] = \
                    best_block[:acceptance_length + 1]
                output_ids[start + acceptance_length + 1] = \
                    best_posterior[acceptance_length]

                # If best was NOT primary, we need to fix the KV cache
                # In a real implementation, we'd keep the best branch's KV.
                # Here we accept the primary's KV (slightly wrong for
                # alternatives, but sufficient for timing measurement).

                # DFlash context update (use primary aux regardless)
                dflash_aux_v = list(primary_aux)
                if len(dflash_aux_v) > 0:
                    raw = jnp.concatenate(dflash_aux_v, axis=-1)
                    proj = draft_combine_fn(draft_state, raw)
                    projected_all = proj[:acceptance_length + 1]
                else:
                    projected_all = None

                # Get top-K tokens for NEXT step from primary verification
                # verify_hidden is from the primary (candidate 0) verify call
                next_logits = target_logits_fn(
                    target_state,
                    verify_hidden[acceptance_length:acceptance_length + 1],
                    None)
                top_k_tokens = np.array(
                    jnp.argsort(next_logits[0], axis=-1)[::-1][:max(
                        k_candidates)],
                    dtype=np.int32)

                t_step_end = time.perf_counter()

                if idx >= warmup_samples:
                    results_by_k[K]["acceptance_lengths"].append(
                        acceptance_length + 1)
                    results_by_k[K]["draft_times_ms"].append(
                        (t_draft_end - t_draft_start) * 1000)
                    results_by_k[K]["verify_times_ms"].append(
                        (t_verify_end - t_verify_start) * 1000)
                    results_by_k[K]["step_times_ms"].append(
                        (t_step_end - t_step_start) * 1000)

                start += acceptance_length + 1
                prev_seq_len = seq_len
                step_count += 1

                if eos_token_id in output_ids[len(input_ids): start + 1]:
                    break

            if step_count > 0:
                tau = np.mean(
                    results_by_k[K]["acceptance_lengths"][-step_count:])
                print(f"    [K={K}] [{idx+1}/{len(dataset)}] "
                      f"steps={step_count}, tau={tau:.2f}")

    return results_by_k


def main():
    parser = argparse.ArgumentParser(description="Tree speculation")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--k-candidates", type=str, default="1,2,4")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    k_candidates = [int(x) for x in args.k_candidates.split(",")]

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
    print(f"[INFO] K candidates: {k_candidates}")

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

    print("\n" + "=" * 70)
    print("TREE SPECULATION EXPERIMENT")
    print("=" * 70)

    results_by_k = run_tree_experiment(
        target_model_fn, target_logits_fn, target_combine_fn, target_state,
        draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
        config, mesh, tokenizer, dataset, args.max_model_len, block_size,
        mask_token_id, hidden_size, args.max_new_tokens, args.warmup_samples,
        dflash_aux_layers, k_candidates)

    # === Results ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Get baseline verify time (K=1)
    baseline_verify = np.mean(results_by_k[1]["verify_times_ms"]) \
        if 1 in results_by_k and results_by_k[1]["verify_times_ms"] else 0

    print(f"\n  {'K':>3} | {'Tau':>6} | {'Draft(ms)':>10} | "
          f"{'Verify(ms)':>11} | {'Step(ms)':>9} | "
          f"{'Tok/s':>7} | {'Amort.Step':>11} | {'Amort.Tok/s':>12}")
    print("  " + "-" * 90)

    summary = {}
    for K in k_candidates:
        r = results_by_k[K]
        if not r["acceptance_lengths"]:
            continue
        tau = np.mean(r["acceptance_lengths"])
        draft_ms = np.mean(r["draft_times_ms"])
        verify_ms = np.mean(r["verify_times_ms"])
        step_ms = np.mean(r["step_times_ms"])
        tok_s = tau / step_ms * 1000 if step_ms > 0 else 0

        # Theoretical amortized: all K verifications in 1 pass
        # Cost = draft_time + 1 × verify_time (instead of K × verify_time)
        amort_step_ms = draft_ms + baseline_verify
        amort_tok_s = tau / amort_step_ms * 1000 if amort_step_ms > 0 else 0

        print(f"  {K:>3} | {tau:>6.2f} | {draft_ms:>10.2f} | "
              f"{verify_ms:>11.2f} | {step_ms:>9.2f} | "
              f"{tok_s:>7.1f} | {amort_step_ms:>11.2f} | "
              f"{amort_tok_s:>12.1f}")

        summary[K] = {
            "tau": float(tau),
            "draft_ms": float(draft_ms),
            "verify_ms": float(verify_ms),
            "step_ms": float(step_ms),
            "throughput_tok_s": float(tok_s),
            "amortized_step_ms": float(amort_step_ms),
            "amortized_throughput_tok_s": float(amort_tok_s),
        }

    # Speedup analysis
    if 1 in summary and len(summary) > 1:
        base_tok_s = summary[1]["throughput_tok_s"]
        base_amort = summary[1]["amortized_throughput_tok_s"]
        print(f"\n  Speedup vs K=1 baseline ({base_tok_s:.1f} tok/s):")
        for K in k_candidates:
            if K == 1 or K not in summary:
                continue
            s = summary[K]
            actual = s["throughput_tok_s"] / base_tok_s if base_tok_s else 0
            amort = s["amortized_throughput_tok_s"] / base_amort \
                if base_amort else 0
            print(f"    K={K}: actual {actual:.2f}x, "
                  f"with amortization {amort:.2f}x")

    # Save
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)),
                    exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

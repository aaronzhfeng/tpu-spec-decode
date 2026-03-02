"""Layer-truncated verification: how many target model layers does
speculative decoding verification actually need?

Captures hidden states at every layer during verification, computes
logits at each truncation point, and measures simulated acceptance
rate (tau) as if verification used only the first N layers.

Usage:
    python benchmarks/layer_truncation.py \
        --target-model Qwen/Qwen3-4B \
        --draft-model z-lab/Qwen3-4B-DFlash-b16 \
        --dataset gsm8k --max-samples 4 --max-new-tokens 256
"""

# ---------------------------------------------------------------
# IMPORTANT: Monkey-patch MUST happen before model loading.
# We override Qwen3Model.__init__ to capture hidden states at ALL
# layers (not just the DFlash-specific ones).
# ---------------------------------------------------------------
from tpu_inference.models.jax import qwen3 as _qwen3_module

_original_qwen3_model_init = _qwen3_module.Qwen3Model.__init__


def _patched_qwen3_model_init(self, *args, **kwargs):
    _original_qwen3_model_init(self, *args, **kwargs)
    num_layers = self.end_layer - self.start_layer
    self.aux_hidden_state_layers = tuple(range(num_layers))
    self.capture_aux_after_layer = True


_qwen3_module.Qwen3Model.__init__ = _patched_qwen3_model_init

# ---------------------------------------------------------------
# Now proceed with normal imports
# ---------------------------------------------------------------
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


def get_dflash_aux_layer_indices(config):
    """Compute which layers DFlash needs, matching _get_dflash_aux_layers."""
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
    max_kv_len = next_padded_size(max_length)
    caches = []
    for _ in range(hf.num_hidden_layers):
        caches.extend([
            jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16),
            jnp.zeros((1, num_heads, max_kv_len, head_dim), dtype=jnp.bfloat16),
        ])
    return caches


def rms_norm(x, weight, eps=1e-6):
    """Apply RMSNorm manually to intermediate hidden states."""
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps) * weight


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment(
    target_model_fn, target_logits_fn, target_combine_fn, target_state,
    draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    config, mesh, tokenizer, dataset, max_model_len, block_size,
    mask_token_id, hidden_size, max_new_tokens, warmup,
    dflash_aux_layers, norm_weight, rms_eps,
):
    num_target_layers = config.model_config.hf_config.num_hidden_layers
    truncation_points = [6, 12, 18, 24, 30, 31, 32, 33, 34, 35, num_target_layers]
    truncation_points = [n for n in truncation_points if n <= num_target_layers]

    print(f"\n  Target model: {num_target_layers} layers")
    print(f"  DFlash aux layers: {dflash_aux_layers}")
    print(f"  Truncation points: {truncation_points}")
    print(f"  RMS norm eps: {rms_eps}")

    eos_token_id = tokenizer.eos_token_id or 0
    page_size = 16

    # Per-truncation-point accumulators
    all_matches = {n: [] for n in truncation_points}
    all_simulated_acc = {n: [] for n in truncation_points}
    all_real_acc = []
    total_positions = 0

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
        target_kv, hidden_states, all_aux = target_model_fn(
            target_state, target_kv, input_ids_jax, metadata,
            None, None, None, None, None, True, True)

        # Extract DFlash-specific aux from the full list
        dflash_aux = [all_aux[i] for i in dflash_aux_layers]

        logits = target_logits_fn(target_state, hidden_states[-1:], None)
        first_token = int(jnp.argmax(logits, axis=-1)[0])

        output_ids = np.full(len(input_ids) + max_new_tokens + block_size,
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
        acceptance_lengths = []

        while start < max_length:
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

            # Draft logits + tokens
            draft_logits = target_logits_fn(
                target_state, draft_hidden[1:block_size], None)
            draft_tokens = np.array(jnp.argmax(draft_logits, axis=-1))

            output_ids[start + 1 : start + block_size] = draft_tokens
            block_ids = jnp.array(
                output_ids[start : start + block_size], dtype=jnp.int32)

            if not draft_prefill_done:
                draft_prefill_done = True
                tpu_sync()
                continue

            # ============================================================
            # VERIFY: full forward pass (returns ALL 36 intermediate states)
            # ============================================================
            verify_pos = jnp.arange(start, start + block_size, dtype=jnp.int32)
            verify_meta = make_attn_metadata(
                verify_pos, start + block_size, block_size, bt)
            target_kv, verify_hidden, all_aux = target_model_fn(
                target_state, target_kv, block_ids, verify_meta,
                None, None, None, None, None, True, True)

            # Full-model logits and acceptance (ground truth)
            verify_logits = target_logits_fn(target_state, verify_hidden, None)
            full_posterior = np.array(jnp.argmax(verify_logits, axis=-1))

            block_np = np.array(block_ids)
            matches = (block_np[1:] == full_posterior[:-1]).astype(np.int32)
            acceptance_length = int(np.cumprod(matches).sum())

            # ============================================================
            # ANALYSIS: simulate acceptance at each truncation point
            # ============================================================
            if idx >= warmup:
                total_positions += block_size

                for n in truncation_points:
                    if n >= num_target_layers:
                        # Full model — use ground truth
                        layer_posterior = full_posterior
                    else:
                        # Truncated: apply norm to layer-N hidden state, compute logits
                        layer_hidden = all_aux[n - 1]  # 0-indexed: layer n is at index n-1
                        normed = rms_norm(layer_hidden, norm_weight, rms_eps)
                        layer_logits = target_logits_fn(target_state, normed, None)
                        layer_posterior = np.array(jnp.argmax(layer_logits, axis=-1))

                    # Token match rate
                    match_count = int(np.sum(layer_posterior == full_posterior))
                    all_matches[n].append(match_count)

                    # Simulated acceptance using truncated logits
                    sim_matches = (block_np[1:] == layer_posterior[:-1]).astype(np.int32)
                    sim_acc = int(np.cumprod(sim_matches).sum()) + 1
                    all_simulated_acc[n].append(sim_acc)

                all_real_acc.append(acceptance_length + 1)

            # Update state using full-model results (ground truth path)
            output_ids[start : start + acceptance_length + 1] = \
                block_np[:acceptance_length + 1]
            output_ids[start + acceptance_length + 1] = full_posterior[acceptance_length]

            acceptance_lengths.append(acceptance_length + 1)
            start += acceptance_length + 1

            # DFlash context update (extract DFlash-specific aux)
            dflash_aux = [all_aux[i] for i in dflash_aux_layers]
            if len(dflash_aux) > 0:
                raw = jnp.concatenate(dflash_aux, axis=-1)
                proj = draft_combine_fn(draft_state, raw)
                projected_all = proj[:acceptance_length + 1]
            else:
                projected_all = None

            tpu_sync()
            step_count += 1

            if eos_token_id in output_ids[len(input_ids) : start + 1]:
                break

        tau = np.mean(acceptance_lengths) if acceptance_lengths else 0
        print(f"  [{idx+1}/{len(dataset)}] steps={step_count}, tau={tau:.2f}")

    return truncation_points, all_matches, all_simulated_acc, all_real_acc, total_positions


def main():
    parser = argparse.ArgumentParser(description="Layer-truncated verification")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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

    num_target_layers = config.model_config.hf_config.num_hidden_layers
    print(f"[INFO] JAX devices: {jax.device_count()}")
    print(f"[INFO] Target layers: {num_target_layers}, block_size: {block_size}")

    # Compute DFlash aux layer indices
    dflash_aux_layers = get_dflash_aux_layer_indices(config)
    print(f"[INFO] DFlash aux layers: {dflash_aux_layers}")

    (target_model_fn, target_logits_fn, target_combine_fn, target_state,
     draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
    ) = load_models(config, mesh)

    # Extract norm weight for manual RMSNorm application
    norm_weight = target_state.model.norm.weight.value
    rms_eps = config.model_config.hf_config.rms_norm_eps
    print(f"[INFO] Norm weight shape: {norm_weight.shape}, eps: {rms_eps}")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))

    print(f"[INFO] Dataset: {args.dataset}, samples: {len(dataset)}")

    print("\n" + "=" * 70)
    print("LAYER-TRUNCATED VERIFICATION EXPERIMENT")
    print("=" * 70)

    truncation_points, all_matches, all_simulated_acc, all_real_acc, total_positions = \
        run_experiment(
            target_model_fn, target_logits_fn, target_combine_fn, target_state,
            draft_model_fn, draft_logits_fn, draft_combine_fn, draft_state,
            config, mesh, tokenizer, dataset, args.max_model_len, block_size,
            mask_token_id, hidden_size, args.max_new_tokens, args.warmup,
            dflash_aux_layers, norm_weight, rms_eps)

    # ============================================================
    # Results
    # ============================================================
    num_target_layers = config.model_config.hf_config.num_hidden_layers
    real_tau = np.mean(all_real_acc) if all_real_acc else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Full model tau (ground truth): {real_tau:.2f}")
    print(f"  Total verify steps analyzed: {len(all_real_acc)}")
    print(f"  Total token positions analyzed: {total_positions}")

    print(f"\n  {'Layers':>6} | {'Token Match':>12} | {'Sim. Tau':>9} | {'vs Full':>8} | {'Compute Saved':>14}")
    print("  " + "-" * 60)

    results = {}
    for n in truncation_points:
        match_pct = np.sum(all_matches[n]) / total_positions * 100 if total_positions > 0 else 0
        sim_tau = np.mean(all_simulated_acc[n]) if all_simulated_acc[n] else 0
        tau_pct = sim_tau / real_tau * 100 if real_tau > 0 else 0
        compute_saved = (num_target_layers - n) / num_target_layers * 100

        print(f"  {n:>6} | {match_pct:>10.1f}% | {sim_tau:>9.2f} | {tau_pct:>6.1f}% | {compute_saved:>12.1f}%")
        results[n] = {
            "token_match_pct": float(match_pct),
            "simulated_tau": float(sim_tau),
            "tau_retention_pct": float(tau_pct),
            "compute_saved_pct": float(compute_saved),
        }

    # Per-layer stability analysis
    print(f"\n  Per-layer analysis:")
    print(f"    If tau stays above 90% of full at N=24:")
    if 24 in results:
        r24 = results[24]["tau_retention_pct"]
        print(f"      -> {r24:.1f}% tau retention at 33% compute savings")
        if r24 >= 90:
            print(f"      -> HEADROOM EXISTS: early exit could save ~33% of verification time")
        else:
            print(f"      -> LIMITED HEADROOM: verification needs most/all layers")

    # Save results
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        output = {
            "num_target_layers": num_target_layers,
            "full_tau": float(real_tau),
            "total_steps": len(all_real_acc),
            "total_positions": total_positions,
            "truncation_results": results,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

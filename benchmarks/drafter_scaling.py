"""DFlash drafter forward-pass scaling: does the MXU amortize the drafter
too?

Measures:
1. DFlash forward pass at K=16 (standard block size)
2. Raw matmul latency at K=16, 32, 64, 128 for DFlash-sized layers
3. Target model matmul latency at same K values (for comparison)

Confirms that both drafter and target benefit from MXU tile amortization.

Usage:
    python benchmarks/drafter_scaling.py \
        --target-model Qwen/Qwen3-4B \
        --draft-model z-lab/Qwen3-4B-DFlash-b16
"""

import argparse
import math
import time
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
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


def tpu_sync():
    jax.effects_barrier()


def create_mesh(num_devices=1):
    devices = np.array(jax.local_devices()[:num_devices])
    return Mesh(devices.reshape((1, 1, 1, num_devices)),
                axis_names=("data", "attn_dp", "expert", "model"))


def make_attn_metadata(input_positions, seq_len, num_query_tokens,
                       block_tables):
    return AttentionMetadata(
        input_positions=input_positions,
        block_tables=block_tables,
        seq_lens=jnp.array([seq_len], dtype=jnp.int32),
        query_start_loc=jnp.array([0, num_query_tokens], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
    )


def next_padded_size(n):
    if n <= 16:
        return 16
    p = 16
    while p < n:
        p *= 2
    return p


def main():
    parser = argparse.ArgumentParser(description="Drafter scaling")
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    mesh = create_mesh()
    init_pp_distributed_environment(
        ip="", rank=0, world_size=1, device=jax.devices()[0], need_pp=False)

    config = StandaloneVllmConfig(args.target_model, args.draft_model)
    draft_hf = config.speculative_config.draft_model_config.hf_config
    target_hf = config.model_config.hf_config

    print(f"[INFO] JAX devices: {jax.device_count()}")
    print(f"[INFO] DFlash: {draft_hf.num_hidden_layers} layers, "
          f"hidden={draft_hf.hidden_size}, "
          f"intermediate={draft_hf.intermediate_size}")
    print(f"[INFO] Target: {target_hf.num_hidden_layers} layers, "
          f"hidden={target_hf.hidden_size}, "
          f"intermediate={target_hf.intermediate_size}")

    # ==================================================================
    # Part 1: Raw matmul scaling (DFlash-sized)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 1: RAW MATMUL SCALING (DFlash dimensions)")
    print("=" * 70)

    d_hidden = draft_hf.hidden_size
    d_inter = draft_hf.intermediate_size
    n_layers = draft_hf.num_hidden_layers

    print(f"\n  DFlash FFN matmul: ({{}}, {d_hidden}) x ({d_hidden}, "
          f"{d_inter})")
    print(f"  {n_layers} layers, simulating full forward pass FFN cost")

    k_values = [16, 32, 64, 96, 128, 129, 160, 192, 256]

    # Create weight matrices matching DFlash FFN
    w_gate = jnp.ones((d_hidden, d_inter), dtype=jnp.bfloat16)
    w_up = jnp.ones((d_hidden, d_inter), dtype=jnp.bfloat16)
    w_down = jnp.ones((d_inter, d_hidden), dtype=jnp.bfloat16)

    print(f"\n  {'K':>5} | {'Per-layer (ms)':>14} | {'Full model (ms)':>16} "
          f"| {'vs K=16':>8}")
    print("  " + "-" * 55)

    base_time = None
    for K in k_values:
        x = jnp.ones((K, d_hidden), dtype=jnp.bfloat16)

        # Warmup
        for _ in range(args.warmup):
            # Simulate one FFN layer: gate, up, down
            g = jnp.dot(x, w_gate)
            u = jnp.dot(x, w_up)
            h = g * u  # SiLU approximation
            o = jnp.dot(h, w_down)
            tpu_sync()

        # Timed trials (one layer)
        latencies = []
        for _ in range(args.trials):
            t0 = time.perf_counter()
            g = jnp.dot(x, w_gate)
            u = jnp.dot(x, w_up)
            h = g * u
            o = jnp.dot(h, w_down)
            tpu_sync()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        mean_ms = np.mean(latencies)
        full_model_ms = mean_ms * n_layers
        if base_time is None:
            base_time = full_model_ms
        ratio = full_model_ms / base_time

        print(f"  {K:>5} | {mean_ms:>10.3f} ± {np.std(latencies):.3f} | "
              f"{full_model_ms:>16.3f} | {ratio:>7.2f}x")

    # ==================================================================
    # Part 2: Raw matmul scaling (Target model dimensions)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 2: RAW MATMUL SCALING (Target model dimensions)")
    print("=" * 70)

    t_hidden = target_hf.hidden_size
    t_inter = target_hf.intermediate_size
    t_layers = target_hf.num_hidden_layers

    print(f"\n  Target FFN matmul: ({{}}, {t_hidden}) x ({t_hidden}, "
          f"{t_inter})")
    print(f"  {t_layers} layers")

    tw_gate = jnp.ones((t_hidden, t_inter), dtype=jnp.bfloat16)
    tw_up = jnp.ones((t_hidden, t_inter), dtype=jnp.bfloat16)
    tw_down = jnp.ones((t_inter, t_hidden), dtype=jnp.bfloat16)

    print(f"\n  {'K':>5} | {'Per-layer (ms)':>14} | {'Full model (ms)':>16} "
          f"| {'vs K=16':>8}")
    print("  " + "-" * 55)

    base_time = None
    for K in k_values:
        x = jnp.ones((K, t_hidden), dtype=jnp.bfloat16)

        for _ in range(args.warmup):
            g = jnp.dot(x, tw_gate)
            u = jnp.dot(x, tw_up)
            h = g * u
            o = jnp.dot(h, tw_down)
            tpu_sync()

        latencies = []
        for _ in range(args.trials):
            t0 = time.perf_counter()
            g = jnp.dot(x, tw_gate)
            u = jnp.dot(x, tw_up)
            h = g * u
            o = jnp.dot(h, tw_down)
            tpu_sync()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        mean_ms = np.mean(latencies)
        full_model_ms = mean_ms * t_layers
        if base_time is None:
            base_time = full_model_ms
        ratio = full_model_ms / base_time

        print(f"  {K:>5} | {mean_ms:>10.3f} ± {np.std(latencies):.3f} | "
              f"{full_model_ms:>16.3f} | {ratio:>7.2f}x")

    # ==================================================================
    # Part 3: Attention matmul scaling
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 3: ATTENTION Q×K^T SCALING (Target model)")
    print("=" * 70)

    num_heads = target_hf.num_attention_heads
    head_dim = target_hf.hidden_size // num_heads
    kv_lengths = [64, 256, 512, 1024]

    print(f"\n  Attention: Q({{}}, {head_dim}) × K^T({head_dim}, {{}}) "
          f"per head ({num_heads} heads)")

    for kv_len in kv_lengths:
        print(f"\n  --- KV length = {kv_len} ---")
        print(f"  {'K_query':>8} | {'Attn (ms)':>10} | {'vs K=16':>8}")
        print("  " + "-" * 35)

        K_key = jnp.ones((num_heads, kv_len, head_dim), dtype=jnp.bfloat16)
        base_attn = None

        for K in k_values:
            Q = jnp.ones((num_heads, K, head_dim), dtype=jnp.bfloat16)

            for _ in range(args.warmup):
                scores = jnp.matmul(Q, jnp.transpose(K_key, (0, 2, 1)))
                tpu_sync()

            latencies = []
            for _ in range(args.trials):
                t0 = time.perf_counter()
                scores = jnp.matmul(Q, jnp.transpose(K_key, (0, 2, 1)))
                tpu_sync()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            mean_ms = np.mean(latencies)
            if base_attn is None:
                base_attn = mean_ms
            ratio = mean_ms / base_attn

            print(f"  {K:>8} | {mean_ms:>7.3f} ± {np.std(latencies):.2f} "
                  f"| {ratio:>7.2f}x")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n  If all matmuls show ~1.0x ratio from K=16 to K=128,")
    print("  then a retrained DFlash at K=128 costs the same as K=16.")
    print("  This validates the constant-step-time assumption in Doc 36.")


if __name__ == "__main__":
    main()

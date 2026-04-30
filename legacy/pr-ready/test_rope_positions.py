"""Test whether different context vs noise RoPE positions produce different outputs.

This script directly tests dflash_concat_attention with real RoPE applied,
comparing:
  A) Same positions for context and noise keys (current bug)
  B) Different positions (context at cache_len offset, noise at block positions)

If the outputs differ, the RoPE fix matters for correctness.
"""

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.dflash_attention_interface import dflash_concat_attention
from tpu_inference.layers.jax.rope_interface import apply_rope


def run_test():
    # Simulate: 1 request, 4 tokens (block_size=4), head_dim=64, 1 head
    T = 4
    num_heads = 1
    head_dim = 64
    rope_theta = 10000.0

    jax_key = jax.random.PRNGKey(42)
    keys = jax.random.split(jax_key, 5)

    q_raw = jax.random.normal(keys[0], (T, num_heads, head_dim), dtype=jnp.float32)
    k_ctx_raw = jax.random.normal(keys[1], (T, num_heads, head_dim), dtype=jnp.float32)
    k_noise_raw = jax.random.normal(keys[2], (T, num_heads, head_dim), dtype=jnp.float32)
    v_ctx = jax.random.normal(keys[3], (T, num_heads, head_dim), dtype=jnp.float32)
    v_noise = jax.random.normal(keys[4], (T, num_heads, head_dim), dtype=jnp.float32)

    md = AttentionMetadata(
        input_positions=jnp.arange(T, dtype=jnp.int32),
        block_tables=None,
        seq_lens=jnp.array([T], dtype=jnp.int32),
        query_start_loc=jnp.array([0, T], dtype=jnp.int32),
        request_distribution=None,
    )

    sm_scale = head_dim ** -0.5

    # === Case A: Same positions (current bug) ===
    noise_positions = jnp.arange(T, dtype=jnp.int32) + 10  # positions [10,11,12,13]
    ctx_positions_same = noise_positions  # SAME as noise

    q_a = apply_rope(q_raw, noise_positions, head_dim, rope_theta, None)
    k_ctx_a = apply_rope(k_ctx_raw, ctx_positions_same, head_dim, rope_theta, None)
    k_noise_a = apply_rope(k_noise_raw, noise_positions, head_dim, rope_theta, None)

    md_a = AttentionMetadata(
        input_positions=noise_positions,
        block_tables=None,
        seq_lens=jnp.array([T], dtype=jnp.int32),
        query_start_loc=jnp.array([0, T], dtype=jnp.int32),
        request_distribution=None,
    )

    out_a = dflash_concat_attention(
        q_a, k_ctx_a, k_noise_a, v_ctx, v_noise,
        md_a, max_query_len=T, sm_scale=sm_scale,
    )

    # === Case B: Different positions (fix) ===
    cache_len = 6
    ctx_positions_diff = jnp.arange(T, dtype=jnp.int32) + cache_len  # [6,7,8,9]
    # noise_positions stays [10,11,12,13]

    q_b = apply_rope(q_raw, noise_positions, head_dim, rope_theta, None)
    k_ctx_b = apply_rope(k_ctx_raw, ctx_positions_diff, head_dim, rope_theta, None)
    k_noise_b = apply_rope(k_noise_raw, noise_positions, head_dim, rope_theta, None)

    out_b = dflash_concat_attention(
        q_b, k_ctx_b, k_noise_b, v_ctx, v_noise,
        md_a, max_query_len=T, sm_scale=sm_scale,
    )

    # === Compare ===
    out_a_np = np.asarray(out_a)
    out_b_np = np.asarray(out_b)

    max_diff = np.max(np.abs(out_a_np - out_b_np))
    mean_diff = np.mean(np.abs(out_a_np - out_b_np))

    print(f"Output shape: {out_a_np.shape}")
    print(f"Max absolute difference:  {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Outputs are {'DIFFERENT' if max_diff > 1e-5 else 'SAME'}")

    if max_diff > 1e-5:
        # Show how attention weights shift
        print(f"\nCase A output (same positions):\n{out_a_np[:, 0, :4]}")
        print(f"\nCase B output (different positions):\n{out_b_np[:, 0, :4]}")


if __name__ == "__main__":
    run_test()

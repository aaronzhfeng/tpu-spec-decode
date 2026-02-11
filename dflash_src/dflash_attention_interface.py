# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DFlash concat attention: concatenate context and noise K/V on token axis, dense (causal=False)."""

from typing import Optional

import jax
import jax.numpy as jnp

from tpu_inference.layers.common.attention_metadata import AttentionMetadata


def _attention_one_request(
    q: jnp.ndarray,
    k_ctx: jnp.ndarray,
    k_noise: jnp.ndarray,
    v_ctx: jnp.ndarray,
    v_noise: jnp.ndarray,
    sm_scale: float,
) -> jnp.ndarray:
    """Single-request DFlash concat attention. K/V = concat(context, noise) on token axis."""
    # q (L, Hq, D), k_* (L, Hk, D), v_* (L, Hk, D)
    k = jnp.concatenate([k_ctx, k_noise], axis=0)  # (2*L, Hk, D)
    v = jnp.concatenate([v_ctx, v_noise], axis=0)   # (2*L, Hk, D)
    n_heads_q, n_heads_kv = q.shape[1], k.shape[1]
    if n_heads_q != n_heads_kv:
        n_rep = n_heads_q // n_heads_kv
        k = jnp.repeat(k, n_rep, axis=1)  # (2*L, Hq, D)
        v = jnp.repeat(v, n_rep, axis=1)
    # scores (L, 2*L), causal=False
    scores = jnp.einsum("qhd,khd->qk", q, k) * sm_scale
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("qk,khd->qhd", attn, v)
    return out


def dflash_concat_attention(
    q: jnp.ndarray,
    k_ctx: jnp.ndarray,
    k_noise: jnp.ndarray,
    v_ctx: jnp.ndarray,
    v_noise: jnp.ndarray,
    attention_metadata: AttentionMetadata,
    max_query_len: int,
    sm_scale: float,
) -> jnp.ndarray:
    """DFlash concat attention: concat context and noise K/V on token axis, dense (non-causal).

    Per-request slicing is done via attention_metadata.query_start_loc.
    Returns attention outputs for query positions only (same shape as q).
    """
    query_start_loc: Optional[jax.Array] = getattr(
        attention_metadata, "query_start_loc", None
    )
    if query_start_loc is None or query_start_loc.shape[0] <= 1:
        # Single request or no slicing
        return _attention_one_request(q, k_ctx, k_noise, v_ctx, v_noise, sm_scale)

    num_reqs = query_start_loc.shape[0] - 1
    outputs = []
    for i in range(num_reqs):
        start = query_start_loc[i]
        end = query_start_loc[i + 1]
        length = end - start
        q_i = jax.lax.dynamic_slice(q, (start, 0, 0), (length, q.shape[1], q.shape[2]))
        k_ctx_i = jax.lax.dynamic_slice(k_ctx, (start, 0, 0), (length, k_ctx.shape[1], k_ctx.shape[2]))
        k_noise_i = jax.lax.dynamic_slice(k_noise, (start, 0, 0), (length, k_noise.shape[1], k_noise.shape[2]))
        v_ctx_i = jax.lax.dynamic_slice(v_ctx, (start, 0, 0), (length, v_ctx.shape[1], v_ctx.shape[2]))
        v_noise_i = jax.lax.dynamic_slice(v_noise, (start, 0, 0), (length, v_noise.shape[1], v_noise.shape[2]))
        out_i = _attention_one_request(
            q_i, k_ctx_i, k_noise_i, v_ctx_i, v_noise_i, sm_scale
        )
        outputs.append(out_i)
    return jnp.concatenate(outputs, axis=0)

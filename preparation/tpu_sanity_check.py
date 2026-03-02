#!/usr/bin/env python3
"""TPU sanity check -- works with JAX or PyTorch/XLA.

Tries JAX first (Path A: bare-metal venv). Falls back to PyTorch/XLA
(Path B: Docker image). Runs a small matmul on TPU to confirm execution.
"""

import sys


def check_jax() -> int:
    """JAX-based TPU check (Path A)."""
    try:
        import jax
    except ImportError:
        return -1  # JAX not installed, try torch

    print(f"JAX version: {jax.__version__}")
    devices = jax.devices()
    print(f"Device count: {len(devices)}")
    for d in devices:
        print(f"  {d}")

    tpu_count = sum(1 for d in devices if "tpu" in str(d).lower())
    if tpu_count == 0:
        print("ERROR: JAX found no TPU devices")
        print("  Check /dev/vfio/ exists (v5p) or /dev/accel* (v4)")
        print("  Try: PJRT_DEVICE=TPU python3 -c \"import jax; print(jax.devices())\"")
        return 1

    print(f"TPU chips: {tpu_count}")

    # Matmul on TPU (multiple of 128 to exercise MXU)
    import jax.numpy as jnp
    x = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
    y = x @ x.T
    jax.effects_barrier()
    print(f"Matmul OK: shape={y.shape}, dtype={y.dtype}, device={y.devices()}")

    # Dot-product attention pattern (QK^T * V)
    q = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 16, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(1), (1, 2, 16, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(2), (1, 2, 16, 128), dtype=jnp.bfloat16)
    attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) / (128 ** 0.5)
    out = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(attn_weights, axis=-1), v)
    jax.effects_barrier()
    print(f"Attention OK: shape={out.shape}, dtype={out.dtype}")

    print("TPU sanity check passed (JAX).")
    return 0


def check_torch_xla() -> int:
    """PyTorch/XLA-based TPU check (Path B: Docker)."""
    try:
        import torch
    except Exception as exc:
        print(f"ERROR: torch not installed: {exc}")
        return 1

    try:
        import torch_xla.core.xla_model as xm
    except Exception as exc:
        print(f"ERROR: torch_xla not installed or failed to import: {exc}")
        return 1

    device = xm.xla_device()
    print(f"Device: {device}")

    # Small matmul (multiple of 128 to exercise MXU)
    x = torch.randn(1024, 1024, device=device)
    y = x @ x.T
    xm.mark_step()
    print(f"Matmul OK: {y.shape} on {y.device}")

    # SDPA (closest to DFlash attention)
    q = torch.randn(1, 2, 16, 128, device=device)
    k = torch.randn(1, 2, 16, 128, device=device)
    v = torch.randn(1, 2, 16, 128, device=device)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    xm.mark_step()
    print(f"SDPA OK: {out.shape} on {out.device}")

    print("TPU sanity check passed (PyTorch/XLA).")
    return 0


def main() -> int:
    # Try JAX first (Path A), fall back to PyTorch/XLA (Path B)
    rc = check_jax()
    if rc == -1:
        print("JAX not found, trying PyTorch/XLA...")
        return check_torch_xla()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

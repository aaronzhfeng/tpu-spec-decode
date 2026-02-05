#!/usr/bin/env python3
"""TPU sanity check for PyTorch/XLA.

Runs a small matmul and SDPA to confirm XLA execution.
"""

import sys


def main() -> int:
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

    print("TPU sanity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

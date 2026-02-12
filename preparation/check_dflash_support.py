#!/usr/bin/env python3
"""Verify that the currently imported vLLM supports DFlash.

This script is intended to be run after setting PYTHONPATH so that local
repo copies of `vllm` and `tpu-inference` are preferred over site-packages.
"""

from __future__ import annotations

import os
import sys
from typing import Any, get_args


def _flatten_literal_args(tp: Any) -> list[str]:
    values: list[str] = []
    for item in get_args(tp):
        if isinstance(item, str):
            values.append(item)
        else:
            values.extend(_flatten_literal_args(item))
    return values


def main() -> int:
    print(f"python_executable={sys.executable}")
    print(f"python_version={sys.version.split()[0]}")
    print(f"cwd={os.getcwd()}")
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")

    try:
        import vllm
        from vllm.config.speculative import SpeculativeMethod
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] Failed to import vllm and SpeculativeMethod: {exc}")
        return 2

    print(f"vllm_path={getattr(vllm, '__file__', '<unknown>')}")

    methods = sorted(set(_flatten_literal_args(SpeculativeMethod)))
    print("supported_speculative_methods=" + ",".join(methods))

    dflash_supported = "dflash" in methods
    print(f"dflash_supported={dflash_supported}")
    if not dflash_supported:
        print(
            "[ERROR] DFlash is NOT supported by the currently imported vLLM."
        )
        return 3

    # Best-effort import check for tpu-inference DFlash path.
    try:
        from tpu_inference.spec_decode.jax.dflash import DFlashProposer  # noqa: F401

        print("tpu_inference_dflash_import=ok")
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not import tpu_inference DFlash proposer: {exc}")

    print("[OK] DFlash support check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


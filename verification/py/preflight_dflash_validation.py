#!/usr/bin/env python3
"""Preflight checks for DFlash validation workflows."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _module_status(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _print_item(label: str, ok: bool, details: str = "") -> None:
    state = "OK" if ok else "MISSING"
    if details:
        print(f"[{state}] {label}: {details}")
    else:
        print(f"[{state}] {label}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local preflight checks for DFlash validation.")
    parser.add_argument("--require-pytest",
                        action="store_true",
                        help="Fail if pytest is missing.")
    parser.add_argument("--require-vllm",
                        action="store_true",
                        help="Fail if vLLM is missing.")
    parser.add_argument("--require-external-dflash",
                        action="store_true",
                        help="Fail if external/dflash directory is missing.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    expected_paths = [
        ("repo root", root),
        ("nested tpu-inference repo", root / "deps" / "tpu-inference"),
        ("verification scripts", root / "verification" / "sh"),
        ("verification python", root / "verification" / "py"),
    ]
    if args.require_external_dflash:
        expected_paths.append(("dflash repo (datasets)", root / "deps" / "dflash"))

    missing_required = False
    print("== Path Checks ==")
    for label, path in expected_paths:
        ok = path.exists()
        _print_item(label, ok, str(path))
        missing_required = missing_required or not ok

    module_checks = [
        ("pytest", args.require_pytest),
        ("vllm", args.require_vllm),
        ("jax", False),
        ("flax", False),
        ("transformers", False),
    ]

    print("\n== Python Module Checks ==")
    for module_name, required in module_checks:
        ok = _module_status(module_name)
        tag = "required" if required else "optional"
        _print_item(module_name, ok, tag)
        if required and not ok:
            missing_required = True

    print("\n== Environment Snapshot ==")
    print(f"python={sys.executable}")
    print(f"python_version={sys.version.split()[0]}")
    print(f"cwd={os.getcwd()}")

    if missing_required:
        print("\nPreflight: FAIL")
        return 1

    print("\nPreflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

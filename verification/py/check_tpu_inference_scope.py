#!/usr/bin/env python3
"""Check that modified files in nested tpu-inference repo stay in allowlist."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path


def _read_allowlist(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    patterns = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _git_status_paths(repo_dir: Path) -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(repo_dir), "status", "--porcelain"],
        text=True,
    )
    paths: list[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        path_part = line[3:]
        if " -> " in path_part:
            _, path_part = path_part.split(" -> ", 1)
        path_part = path_part.strip()
        if path_part:
            paths.append(path_part)
    return sorted(set(paths))


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate change scope in nested tpu-inference repo.")
    parser.add_argument("--repo-dir",
                        default="deps/tpu-inference",
                        help="Path to nested repo.")
    parser.add_argument(
        "--allowlist",
        default="verification/config/tpu_inference_dflash_allowlist.txt",
        help="Allowlist file with glob patterns relative to nested repo root.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    repo_dir = (root / args.repo_dir).resolve()
    allowlist_path = (root / args.allowlist).resolve()

    if not (repo_dir / ".git").exists():
        print(f"ERROR: nested repo not found or missing .git: {repo_dir}")
        return 2
    if not allowlist_path.exists():
        print(f"ERROR: allowlist file not found: {allowlist_path}")
        return 2

    patterns = _read_allowlist(allowlist_path)
    changed_paths = _git_status_paths(repo_dir)

    print(f"Nested repo: {repo_dir}")
    print(f"Allowlist: {allowlist_path}")
    if not changed_paths:
        print("No changes detected in nested tpu-inference repo. PASS")
        return 0

    violations = [p for p in changed_paths if not _matches_any(p, patterns)]
    print("Changed files:")
    for p in changed_paths:
        print(f"- {p}")

    if violations:
        print("\nScope violations (outside allowlist):")
        for p in violations:
            print(f"- {p}")
        return 1

    print("\nPASS: all changed files are within allowlist scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


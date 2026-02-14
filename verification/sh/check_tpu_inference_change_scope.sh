#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${ROOT_DIR}/verification/py/check_tpu_inference_scope.py" \
  --repo-dir "deps/tpu-inference" \
  --allowlist "verification/config/tpu_inference_dflash_allowlist.txt"


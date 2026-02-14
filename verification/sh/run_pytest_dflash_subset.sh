#!/usr/bin/env bash
set -euo pipefail

# Runs the DFlash-focused unit/integration pytest subset.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/deps/tpu-inference:${PYTHONPATH:-}"

PYTEST_BIN="${PYTEST_BIN:-python -m pytest}"
PYTEST_FLAGS="${PYTEST_FLAGS:--q}"

TESTS=(
  "deps/tpu-inference/tests/spec_decode/test_dflash.py"
  "deps/tpu-inference/tests/models/jax/test_qwen3_dflash.py"
  "deps/tpu-inference/tests/models/jax/test_qwen3_dflash_attention.py"
  "deps/tpu-inference/tests/runner/test_speculative_decoding_manager.py"
  "deps/tpu-inference/tests/runner/test_kv_cache_manager.py"
  "deps/tpu-inference/tests/runner/test_tpu_runner.py"
)

echo "Running DFlash pytest subset"
echo "pytest_bin=${PYTEST_BIN}"
printf 'tests:\n'
printf '  - %s\n' "${TESTS[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; exiting before execution."
  exit 0
fi

# shellcheck disable=SC2086
${PYTEST_BIN} ${PYTEST_FLAGS} "${TESTS[@]}"

#!/usr/bin/env bash
set -euo pipefail

# One-shot validation matrix runner for DFlash integration.
# Stages:
#  1) preflight
#  2) scope guard
#  3) python syntax checks
#  4) pytest subset
#  5) TPU smoke
#  6) TPU eval + baseline comparison
#  7) optional external reference benchmark

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STRICT_PRECHECK="${STRICT_PRECHECK:-1}"
RUN_PYTEST="${RUN_PYTEST:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_EXTERNAL="${RUN_EXTERNAL:-0}"

echo "DFlash validation matrix"
echo "root=${ROOT_DIR}"
echo "strict_precheck=${STRICT_PRECHECK} run_pytest=${RUN_PYTEST} run_smoke=${RUN_SMOKE} run_eval=${RUN_EVAL} run_external=${RUN_EXTERNAL}"

echo
echo "== Stage 1: preflight =="
if [[ "${STRICT_PRECHECK}" == "1" ]]; then
  python verification/py/preflight_dflash_validation.py \
    --require-pytest \
    --require-vllm \
    --require-external-dflash
else
  python verification/py/preflight_dflash_validation.py
fi

echo
echo "== Stage 2: scope guard =="
bash verification/sh/check_tpu_inference_change_scope.sh

echo
echo "== Stage 3: syntax checks =="
python -m py_compile \
  verification/py/preflight_dflash_validation.py \
  verification/py/run_tpu_dflash_eval.py \
  verification/py/compare_results_to_baseline.py \
  verification/py/dflash_reported_baselines.py \
  deps/tpu-inference/tpu_inference/layers/common/dflash_attention_interface.py \
  deps/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py \
  deps/tpu-inference/tests/models/jax/test_qwen3_dflash_attention.py

if [[ "${RUN_PYTEST}" == "1" ]]; then
  echo
  echo "== Stage 4: pytest subset =="
  bash verification/sh/run_pytest_dflash_subset.sh
else
  echo
  echo "== Stage 4: pytest subset (skipped) =="
fi

if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo
  echo "== Stage 5: TPU smoke =="
  bash verification/sh/run_tpu_inference_dflash_smoke.sh
else
  echo
  echo "== Stage 5: TPU smoke (skipped) =="
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo
  echo "== Stage 6: TPU eval =="
  bash verification/sh/run_tpu_inference_dflash_eval.sh
else
  echo
  echo "== Stage 6: TPU eval (skipped) =="
fi

if [[ "${RUN_EXTERNAL}" == "1" ]]; then
  echo
  echo "== Stage 7: external reference benchmark =="
  bash verification/sh/run_external_dflash_reference.sh
else
  echo
  echo "== Stage 7: external reference benchmark (skipped) =="
fi

echo
echo "Validation matrix completed."

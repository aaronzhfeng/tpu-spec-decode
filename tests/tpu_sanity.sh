#!/usr/bin/env bash
# tests/tpu_sanity.sh — Verify TPU hardware works (matmul + SDPA attention).
#
# Runs a small PyTorch/XLA sanity check to confirm the TPU device is
# accessible and basic operations (matmul, scaled_dot_product_attention)
# execute correctly.
#
# Usage:
#   bash tests/tpu_sanity.sh          # Run inside Docker (default)
#   bash tests/tpu_sanity.sh host     # Run on host directly

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

MODE="${1:-docker}"

echo "==========================================="
echo "  TPU Sanity Check (mode=${MODE})"
echo "==========================================="
echo ""

if [[ "${MODE}" == "docker" ]]; then
  docker_exec_python "preparation/tpu_sanity_check.py"
else
  export PYTHONPATH="${VLLM_DIR}:${TPU_INFERENCE_DIR}:${PYTHONPATH:-}"
  py="$(resolve_python || echo python3)"
  "${py}" "${REPO_ROOT}/preparation/tpu_sanity_check.py"
fi

ok "TPU sanity check complete."

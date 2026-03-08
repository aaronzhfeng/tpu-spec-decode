#!/usr/bin/env bash
# tests/smoke.sh — Quick DFlash end-to-end smoke test.
#
# Runs DFlash speculative decoding on a few prompts to verify the full
# pipeline works: model load, draft model load, proposal, verification,
# and token generation.
#
# Usage:
#   bash tests/smoke.sh                # Greedy smoke (default)
#   bash tests/smoke.sh greedy         # Greedy (temperature=0)
#   bash tests/smoke.sh stochastic     # Stochastic (temperature=1)
#
# Env overrides:
#   DRY_RUN=1 bash tests/smoke.sh      # Validate config without running models
#   FLAX_VERSION=0.12.4 bash tests/smoke.sh   # v5p (Docker default 0.11.1 is for v4)

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

VARIANT="${1:-greedy}"

case "${VARIANT}" in
  greedy)      CONFIG="${CONFIGS_DIR}/smoke_greedy.json" ;;
  stochastic)  CONFIG="${CONFIGS_DIR}/smoke_stochastic.json" ;;
  *)           die "Unknown variant '${VARIANT}'. Use: greedy | stochastic" ;;
esac

RUN_ID="$(make_run_id "smoke_${VARIANT}")"
DRY_RUN="${DRY_RUN:-0}"

echo "==========================================="
echo "  DFlash Smoke Test (${VARIANT})"
echo "==========================================="
echo "  config:  ${CONFIG}"
echo "  run_id:  ${RUN_ID}"
echo "  output:  ${HOST_OUTPUT_DIR}/${RUN_ID}"
echo "  dry_run: ${DRY_RUN}"
echo "==========================================="
echo ""

require_dir "tpu-inference" "${TPU_INFERENCE_DIR}"
require_dir "vllm"          "${VLLM_DIR}"

# The manifest path is relative to /workspace/tpu-spec-decode inside Docker.
MANIFEST_REL="tests/configs/smoke_${VARIANT}.json"

DRY_FLAG=""
[[ "${DRY_RUN}" == "1" ]] && DRY_FLAG="--dry-run"

docker_exec "
  OUT_ROOT=/output \
  MANIFEST=${MANIFEST_REL} \
  RUN_ID=${RUN_ID} \
  DRY_RUN=${DRY_RUN} \
  TRUST_REMOTE_CODE=1 \
  bash verification/contribution/sh/run_contribution_matrix.sh
"

echo ""
ok "Smoke test (${VARIANT}) complete."
info "Artifacts: ${HOST_OUTPUT_DIR}/${RUN_ID}"

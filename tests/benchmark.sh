#!/usr/bin/env bash
# tests/benchmark.sh — Run DFlash performance benchmarks.
#
# Measures DFlash vs baseline across dataset categories. Produces per-prompt
# JSONL records, dataset summaries, speedup comparisons, and acceptance
# metrics (tau, draft_acceptance_rate).
#
# Usage:
#   bash tests/benchmark.sh              # Math benchmarks (default)
#   bash tests/benchmark.sh math         # Math: GSM8K, Math500, AIME24, AIME25
#   bash tests/benchmark.sh code         # Code: Humaneval, MBPP, LiveCodeBench, SWE-Bench
#   bash tests/benchmark.sh chat         # Chat: MT-Bench, Alpaca
#   bash tests/benchmark.sh full         # All datasets
#   bash tests/benchmark.sh CONFIG_PATH  # Custom manifest JSON path
#
# Env overrides:
#   DRY_RUN=1 bash tests/benchmark.sh    # Validate config only
#   MAX_SAMPLES=4 bash tests/benchmark.sh  # Override sample count

source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/docker_run.sh"

SUITE="${1:-math}"

case "${SUITE}" in
  math)  CONFIG="${CONFIGS_DIR}/benchmark_math.json"; LABEL="math" ;;
  code)  CONFIG="${CONFIGS_DIR}/benchmark_code.json"; LABEL="code" ;;
  chat)  CONFIG="${CONFIGS_DIR}/benchmark_chat.json"; LABEL="chat" ;;
  full)  CONFIG="${CONFIGS_DIR}/benchmark_full.json"; LABEL="full" ;;
  *)
    # Treat as a custom config path.
    if [[ -f "${SUITE}" ]]; then
      CONFIG="${SUITE}"
      LABEL="custom"
    elif [[ -f "${CONFIGS_DIR}/${SUITE}" ]]; then
      CONFIG="${CONFIGS_DIR}/${SUITE}"
      LABEL="custom"
    else
      die "Unknown suite '${SUITE}'. Use: math | code | chat | full | <path-to-config.json>"
    fi
    ;;
esac

RUN_ID="$(make_run_id "bench_${LABEL}")"
DRY_RUN="${DRY_RUN:-0}"

echo "==========================================="
echo "  DFlash Benchmark (${LABEL})"
echo "==========================================="
echo "  config:  ${CONFIG}"
echo "  run_id:  ${RUN_ID}"
echo "  output:  ${HOST_OUTPUT_DIR}/${RUN_ID}"
echo "  dry_run: ${DRY_RUN}"
echo "==========================================="
echo ""

require_dir "tpu-inference" "${TPU_INFERENCE_DIR}"
require_dir "vllm"          "${VLLM_DIR}"

# Convert config path to be relative to repo root for Docker mount.
MANIFEST_REL="${CONFIG#${REPO_ROOT}/}"

docker_exec "
  OUT_ROOT=/output \
  MANIFEST=${MANIFEST_REL} \
  RUN_ID=${RUN_ID} \
  DRY_RUN=${DRY_RUN} \
  TRUST_REMOTE_CODE=1 \
  bash verification/contribution/sh/run_contribution_matrix.sh
"

echo ""
ok "Benchmark (${LABEL}) complete."
info "Artifacts: ${HOST_OUTPUT_DIR}/${RUN_ID}"
info "View report: cat ${HOST_OUTPUT_DIR}/${RUN_ID}/summaries/report.md"

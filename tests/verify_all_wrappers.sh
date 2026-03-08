#!/usr/bin/env bash
# tests/verify_all_wrappers.sh — Run every test wrapper to verify repo works on v4 or v5p.
#
# Assumes a fresh clone of tpu-spec-decode (on this branch). Runs clone_repos.sh
# to bootstrap sub-repos, then executes each shell wrapper in sequence.
#
# Usage:
#   bash tests/verify_all_wrappers.sh v4
#   bash tests/verify_all_wrappers.sh v5p
#
# Options:
#   --skip-prep       Skip clone_repos (repos already present)
#   --quick           Use minimal samples (MAX_SAMPLES=1-2) for faster run
#   --no-cleanup      Do not run cleanup.sh at the end
#
# Environment:
#   REPO_ROOT         tpu-spec-decode root (auto-detected)
#   CLONE_SKIP_EXT=1  Pass --skip-ext to clone_repos (skips external/dflash; preflight may fail)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TESTS_DIR="${SCRIPT_DIR}"

# Use user-writable output dir so cleanup can remove it (Docker may create root-owned files in /dev/shm/dflash-test-outputs)
export HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-/dev/shm/${USER:-user}-dflash-outputs}"

# Parse args
HW=""
SKIP_PREP=0
QUICK=0
RUN_CLEANUP=1
for arg in "$@"; do
  case "${arg}" in
    v4|v5p)   HW="${arg}" ;;
    --skip-prep) SKIP_PREP=1 ;;
    --quick)  QUICK=1 ;;
    --no-cleanup) RUN_CLEANUP=0 ;;
    *)        echo "[ERROR] Unknown arg: ${arg}. Use: v4 | v5p [--skip-prep] [--quick] [--no-cleanup]" >&2; exit 1 ;;
  esac
done

if [[ -z "${HW}" ]]; then
  echo "Usage: bash tests/verify_all_wrappers.sh v4 | v5p [--skip-prep] [--quick] [--no-cleanup]"
  echo ""
  echo "  v4 | v5p    Required. Target TPU hardware (run_standalone_all_v5p.sh only on v5p)."
  echo "  --skip-prep Skip clone_repos.sh (use if repos already cloned)."
  echo "  --quick     Use minimal samples for faster verification."
  echo "  --no-cleanup  Do not run cleanup.sh at the end."
  exit 1
fi

# Quick-mode env overrides
export MAX_SAMPLES="${MAX_SAMPLES:-$([ "${QUICK}" = "1" ] && echo 1 || echo 3)}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export TRIALS="${TRIALS:-$([ "${QUICK}" = "1" ] && echo 5 || echo 20)}"
export WARMUP="${WARMUP:-$([ "${QUICK}" = "1" ] && echo 2 || echo 5)}"

PASSED=0
FAILED=0
SKIPPED=0
LOG_DIR="${REPO_ROOT}/.verify_wrappers_log"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/verify_${HW}_$(date +%Y%m%d_%H%M%S).log"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RUN_LOG}"; }
ok()   { log "[PASS] $*"; PASSED=$((PASSED + 1)); }
fail() { log "[FAIL] $*"; FAILED=$((FAILED + 1)); }
skip() { log "[SKIP] $*"; SKIPPED=$((SKIPPED + 1)); }

run_wrapper() {
  local name="$1"
  shift
  local args=("$@")
  log "Running: ${name} ${args[*]:-}"
  if (cd "${REPO_ROOT}" && bash "${TESTS_DIR}/${name}" "${args[@]}" >> "${RUN_LOG}" 2>&1); then
    ok "${name}"
  else
    fail "${name} (see ${RUN_LOG})"
  fi
}

echo "=============================================="
echo "  Verify All Wrappers — TPU ${HW}"
echo "  Started: $(date)"
echo "  Quick:   ${QUICK} | Prep: $([ "${SKIP_PREP}" = "1" ] && echo skip || echo run)"
echo "  Log:     ${RUN_LOG}"
echo "=============================================="
echo ""

# ── Step 0: Preparation (fresh clone) ─────────────────────────────────────────
if [[ "${SKIP_PREP}" != "1" ]]; then
  log "Step 0: Cloning repos..."
  # Remove tpu-inference/vllm if they exist but are not git repos (leftover from failed clone)
  for d in "${REPO_ROOT}/tpu-inference" "${REPO_ROOT}/vllm"; do
    if [[ -d "${d}" && ! -d "${d}/.git" ]]; then
      log "Removing non-git dir: ${d}"
      rm -rf "${d}"
    fi
  done
  EXT_FLAG=""
  [[ "${CLONE_SKIP_EXT:-0}" == "1" ]] && EXT_FLAG="--skip-ext"
  if (cd "${REPO_ROOT}" && bash preparation/clone_repos.sh --skip-pr ${EXT_FLAG} >> "${RUN_LOG}" 2>&1); then
    ok "clone_repos.sh"
  else
    fail "clone_repos.sh"
  fi
  echo ""
else
  skip "clone_repos (--skip-prep)"
fi

# ── TPU wrappers ──────────────────────────────────────────────────────────────
# v5p: run only non-vLLM wrappers (vLLM pipeline has no successful runs on v5p; run vLLM on v4)
# v4: run all wrappers
log "Running TPU wrappers..."

run_wrapper "tpu_sanity.sh"
run_wrapper "standalone_benchmark.sh"

if [[ "${HW}" != "v5p" ]]; then
  # vLLM pipeline wrappers (skip on v5p — ecosystem failure)
  run_wrapper "preflight.sh"
  run_wrapper "smoke.sh" "greedy"
  run_wrapper "benchmark.sh" "math"
  run_wrapper "compare.sh" "latest"
  run_wrapper "verify_context_scaling.sh"
  run_wrapper "drafter_scaling.sh"
  run_wrapper "amortized_verification.sh"
  run_wrapper "layer_truncation.sh"
  run_wrapper "tree_speculation.sh"
  run_wrapper "pipeline_profiling.sh"
  run_wrapper "ablation_study.sh"
  run_wrapper "iterative_refinement.sh"
  run_wrapper "fused_benchmark.sh"
else
  skip "preflight, smoke, benchmark, compare, verify_*, drafter_*, etc. (v5p: vLLM pipeline not supported)"
fi

# ── v5p-only wrapper ──────────────────────────────────────────────────────────
if [[ "${HW}" == "v5p" ]]; then
  run_wrapper "run_standalone_all_v5p.sh"
else
  skip "run_standalone_all_v5p.sh (v5p only, HW=${HW})"
fi

# ── Host-only wrappers ────────────────────────────────────────────────────────
if [[ "${RUN_CLEANUP}" == "1" ]]; then
  # cleanup.sh: run with "outputs" (non-interactive) to verify script works
  run_wrapper "cleanup.sh" "outputs"
else
  skip "cleanup.sh (--no-cleanup)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Verify All Wrappers — Summary"
echo "=============================================="
echo "  Passed:  ${PASSED}"
echo "  Failed:  ${FAILED}"
echo "  Skipped: ${SKIPPED}"
echo "  Log:     ${RUN_LOG}"
echo "=============================================="

if [[ ${FAILED} -gt 0 ]]; then
  exit 1
fi

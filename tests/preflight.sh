#!/usr/bin/env bash
# tests/preflight.sh — Verify environment is ready for DFlash testing.
#
# Checks: repos cloned, Docker image available, vLLM accepts "dflash" method.
#
# Usage:
#   bash tests/preflight.sh          # Run checks inside Docker (default)
#   bash tests/preflight.sh host     # Run checks on host directly
#   bash tests/preflight.sh docker   # Explicitly use Docker

source "$(dirname "$0")/lib/common.sh"

MODE="${1:-docker}"
PASSED=0
FAILED=0

check() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    ok "${label}"
    ((PASSED++))
  else
    fail "${label}"
    ((FAILED++))
  fi
}

echo "==========================================="
echo "  DFlash Preflight Checks (mode=${MODE})"
echo "==========================================="
echo ""

# ── Repo layout ──────────────────────────────────────────────────────────────
info "Checking repo layout..."
check "tpu-inference dir exists"     test -d "${TPU_INFERENCE_DIR}"
check "vllm dir exists"             test -d "${VLLM_DIR}"
check "external/dflash dir exists"  test -d "${EXTERNAL_DFLASH_DIR}"
check "tests/configs dir exists"    test -d "${CONFIGS_DIR}"

# ── Docker ───────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "docker" ]]; then
  info "Checking Docker..."
  check "Docker available" bash -c "docker info >/dev/null 2>&1 || sudo docker info >/dev/null 2>&1"

  dcmd="$(docker_cmd)"
  if ${dcmd} image inspect "${DOCKER_IMAGE}" &>/dev/null; then
    ok "Docker image ${DOCKER_IMAGE} available"
    ((PASSED++))
  else
    warn "Docker image ${DOCKER_IMAGE} not found locally. Will pull on first test run."
  fi
fi

# ── vLLM DFlash support check ───────────────────────────────────────────────
info "Checking vLLM DFlash support..."
if [[ "${MODE}" == "docker" ]]; then
  source "$(dirname "$0")/lib/docker_run.sh"
  if docker_exec "python3 /workspace/tpu-spec-decode/preparation/check_dflash_support.py" 2>&1 | grep -q "dflash_supported=True"; then
    ok "vLLM accepts 'dflash' speculative method"
    ((PASSED++))
  else
    fail "vLLM does NOT accept 'dflash' speculative method"
    ((FAILED++))
  fi
else
  export PYTHONPATH="${VLLM_DIR}:${TPU_INFERENCE_DIR}:${PYTHONPATH:-}"
  py="$(resolve_python || echo python3)"
  if "${py}" "${REPO_ROOT}/preparation/check_dflash_support.py" 2>&1 | grep -q "dflash_supported=True"; then
    ok "vLLM accepts 'dflash' speculative method"
    ((PASSED++))
  else
    fail "vLLM does NOT accept 'dflash' speculative method"
    ((FAILED++))
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "==========================================="
if [[ ${FAILED} -eq 0 ]]; then
  ok "All ${PASSED} checks passed."
else
  fail "${FAILED} check(s) failed, ${PASSED} passed."
  exit 1
fi

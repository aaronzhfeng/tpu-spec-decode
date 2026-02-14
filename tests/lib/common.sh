#!/usr/bin/env bash
# tests/lib/common.sh — Shared helpers for all test scripts.
# Source this file; do not execute directly.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
# REPO_ROOT is the tpu-spec-decode root directory.
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${TESTS_DIR}/.." && pwd)"
CONFIGS_DIR="${TESTS_DIR}/configs"

# Bootstrap dependencies live under deps/ (override via env if needed).
DEPS_DIR="${DEPS_DIR:-${REPO_ROOT}/deps}"
TPU_INFERENCE_DIR="${TPU_INFERENCE_DIR:-${DEPS_DIR}/tpu-inference}"
VLLM_DIR="${VLLM_DIR:-${DEPS_DIR}/vllm}"
EXTERNAL_DFLASH_DIR="${EXTERNAL_DFLASH_DIR:-${DEPS_DIR}/dflash}"

# ── Docker ───────────────────────────────────────────────────────────────────
DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-tpu:latest}"

docker_cmd() {
  # Returns "docker" or "sudo docker" depending on permissions.
  if docker info >/dev/null 2>&1; then
    echo "docker"
  else
    echo "sudo docker"
  fi
}

# ── Storage ──────────────────────────────────────────────────────────────────
HOST_HF_CACHE="${HOST_HF_CACHE:-/dev/shm/hf-cache}"
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-/dev/shm/dflash-test-outputs}"
HOST_TPU_LOG_DIR="${HOST_TPU_LOG_DIR:-/dev/shm/tpu-logs}"
TMPFS_SIZE="${TMPFS_SIZE:-80g}"

ensure_writable_dir() {
  # Usage: ensure_writable_dir VAR_NAME FALLBACK_PATH
  local var_name="$1"
  local fallback="$2"
  local dir="${!var_name}"

  if mkdir -p "${dir}" 2>/dev/null && [[ -w "${dir}" ]]; then
    return 0
  fi
  echo "[WARN] ${dir} not writable, falling back to ${fallback}"
  mkdir -p "${fallback}"
  if [[ ! -d "${fallback}" || ! -w "${fallback}" ]]; then
    echo "[ERROR] Fallback ${fallback} not writable either." >&2
    exit 1
  fi
  printf -v "${var_name}" "%s" "${fallback}"
}

# ── Python ───────────────────────────────────────────────────────────────────
resolve_python() {
  # Find a Python 3.11+ interpreter. Returns path or fails.
  local check='import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)'

  if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" &>/dev/null; then
    if "${PYTHON_BIN}" -c "${check}" 2>/dev/null; then
      echo "${PYTHON_BIN}"
      return 0
    fi
  fi
  for candidate in python3.11 python3 python; do
    if command -v "${candidate}" &>/dev/null && "${candidate}" -c "${check}" 2>/dev/null; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

# ── Output helpers ───────────────────────────────────────────────────────────
_GREEN='\033[0;32m'
_RED='\033[0;31m'
_YELLOW='\033[0;33m'
_CYAN='\033[0;36m'
_RESET='\033[0m'

info()  { echo -e "${_CYAN}[INFO]${_RESET}  $*"; }
ok()    { echo -e "${_GREEN}[OK]${_RESET}    $*"; }
warn()  { echo -e "${_YELLOW}[WARN]${_RESET}  $*" >&2; }
fail()  { echo -e "${_RED}[FAIL]${_RESET}  $*" >&2; }
die()   { fail "$@"; exit 1; }

# ── Run ID ───────────────────────────────────────────────────────────────────
make_run_id() {
  # Usage: make_run_id PREFIX  →  "PREFIX_20260212_143000"
  local prefix="${1:-run}"
  echo "${prefix}_$(date +%Y%m%d_%H%M%S)"
}

# ── Pre-check helpers ────────────────────────────────────────────────────────
require_dir() {
  local label="$1" dir="$2"
  if [[ ! -d "${dir}" ]]; then
    die "${label} not found at ${dir}. Run: bash preparation/clone_repos.sh"
  fi
}

require_docker_image() {
  local dcmd
  dcmd="$(docker_cmd)"
  if ! ${dcmd} image inspect "${DOCKER_IMAGE}" &>/dev/null; then
    info "Pulling Docker image ${DOCKER_IMAGE} ..."
    ${dcmd} pull "${DOCKER_IMAGE}"
  fi
}

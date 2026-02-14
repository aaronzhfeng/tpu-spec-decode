#!/usr/bin/env bash
#
# Bootstrap the three repos into deps/ (same results, simple layout):
#   deps/tpu-inference   (TPU runtime + DFlash integration)
#   deps/vllm           (vLLM with TPU/speculative support)
#   deps/dflash         (datasets + load_and_process_dataset; preflight expects this)
#
# Override repo or branch via env vars.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -----------------------------------------------------------------------------
# Repos and branches (override via env to use your own forks/branches)
# -----------------------------------------------------------------------------
TPU_INFERENCE_REPO="${TPU_INFERENCE_REPO:-https://github.com/aaronzhfeng/tpu-inference.git}"
TPU_INFERENCE_BRANCH="${TPU_INFERENCE_BRANCH:-dflash-integration}"

VLLM_REPO="${VLLM_REPO:-https://github.com/aaronzhfeng/vllm.git}"
VLLM_BRANCH="${VLLM_BRANCH:-zhongyan_dev}"

DFLASH_REPO="${DFLASH_REPO:-https://github.com/Zhongyan0721/dflash}"
DFLASH_BRANCH="${DFLASH_BRANCH:-main}"

# Paths (all under deps/)
DEPS_DIR="${DEPS_DIR:-${REPO_ROOT}/deps}"
TPU_INFERENCE_DIR="${TPU_INFERENCE_DIR:-${DEPS_DIR}/tpu-inference}"
VLLM_DIR="${VLLM_DIR:-${DEPS_DIR}/vllm}"
EXTERNAL_DFLASH_DIR="${EXTERNAL_DFLASH_DIR:-${DEPS_DIR}/dflash}"

SYNC_MAIN_BRANCH="${SYNC_MAIN_BRANCH:-main}"
ALLOW_MAIN_BRANCH="${ALLOW_MAIN_BRANCH:-0}"

require_non_main_branch() {
  local repo_name="$1"
  local branch="$2"
  if [[ "${ALLOW_MAIN_BRANCH}" != "1" && "${branch}" == "main" ]]; then
    echo "[ERROR] ${repo_name} branch is 'main', but this workflow expects a sub-branch." >&2
    echo "        Set the branch env var for that repo, or ALLOW_MAIN_BRANCH=1 to allow main." >&2
    exit 1
  fi
}

ensure_remote_branch_exists() {
  local repo_dir="$1"
  local branch="$2"
  if ! git -C "${repo_dir}" rev-parse --verify "origin/${branch}" >/dev/null 2>&1; then
    echo "[ERROR] Branch origin/${branch} not found in ${repo_dir}" >&2
    exit 1
  fi
}

clone_or_checkout_branch() {
  local repo_dir="$1"
  local remote_url="$2"
  local branch="$3"

  if [[ -e "${repo_dir}/.git" ]]; then
    echo "[INFO] Repo exists: ${repo_dir}"
    git -C "${repo_dir}" fetch origin
    git -C "${repo_dir}" fetch origin "${SYNC_MAIN_BRANCH}" >/dev/null 2>&1 || true
    ensure_remote_branch_exists "${repo_dir}" "${branch}"
    if git -C "${repo_dir}" rev-parse --verify "refs/heads/${branch}" >/dev/null 2>&1; then
      git -C "${repo_dir}" checkout "${branch}"
    else
      git -C "${repo_dir}" checkout -b "${branch}" "origin/${branch}"
    fi
    git -C "${repo_dir}" branch --set-upstream-to="origin/${branch}" "${branch}" >/dev/null 2>&1 || true
    return 0
  fi

  if [[ -e "${repo_dir}" ]]; then
    echo "[ERROR] Path exists but is not a git repo: ${repo_dir}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${repo_dir}")"
  echo "[INFO] Cloning ${remote_url} -> ${repo_dir} (branch=${branch})"
  git clone --branch "${branch}" --single-branch "${remote_url}" "${repo_dir}"
  git -C "${repo_dir}" fetch origin "${SYNC_MAIN_BRANCH}" >/dev/null 2>&1 || true
}

# DFlash upstream only has main; skip branch check when using main
[[ "${TPU_INFERENCE_BRANCH}" != "main" ]] && require_non_main_branch "TPU_INFERENCE" "${TPU_INFERENCE_BRANCH}"
[[ "${VLLM_BRANCH}" != "main" ]] && require_non_main_branch "VLLM" "${VLLM_BRANCH}"
[[ "${DFLASH_BRANCH}" != "main" ]] && require_non_main_branch "DFLASH" "${DFLASH_BRANCH}"

echo "==> tpu-inference (tests and benchmarks use this)"
clone_or_checkout_branch "${TPU_INFERENCE_DIR}" "${TPU_INFERENCE_REPO}" "${TPU_INFERENCE_BRANCH}"

echo ""
echo "==> vllm (at repo root, no symlink)"
clone_or_checkout_branch "${VLLM_DIR}" "${VLLM_REPO}" "${VLLM_BRANCH}"

echo ""
echo "==> external/dflash (datasets and preflight)"
clone_or_checkout_branch "${EXTERNAL_DFLASH_DIR}" "${DFLASH_REPO}" "${DFLASH_BRANCH}"

echo ""
echo "Done. All dependencies under deps/:"
echo "  deps/tpu-inference  ${TPU_INFERENCE_DIR}"
echo "  deps/vllm          ${VLLM_DIR}"
echo "  deps/dflash       ${EXTERNAL_DFLASH_DIR}"

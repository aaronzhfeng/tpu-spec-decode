#!/usr/bin/env bash
#
# Bootstrap repos for current workflow:
# 1) Root-level tpu-inference on your DFlash branch.
# 2) zhongyan_dev/{dflash,tpu-inference,vllm}.
#
# NOTE: This script intentionally does NOT clone forked/ anymore.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROOT_TPU_INF_DIR="${ROOT_TPU_INF_DIR:-${REPO_ROOT}/tpu-inference}"
ROOT_TPU_INF_REPO="${ROOT_TPU_INF_REPO:-https://github.com/aaronzhfeng/tpu-inference.git}"
ROOT_TPU_INF_BRANCH="${ROOT_TPU_INF_BRANCH:-dflash-integration}"

ZHONGYAN_DIR="${ZHONGYAN_DIR:-${REPO_ROOT}/zhongyan_dev}"
ZHONGYAN_DFLASH_DIR="${ZHONGYAN_DFLASH_DIR:-${ZHONGYAN_DIR}/dflash}"
ZHONGYAN_DFLASH_REPO="${ZHONGYAN_DFLASH_REPO:-https://github.com/Zhongyan0721/dflash}"
ZHONGYAN_DFLASH_BRANCH="${ZHONGYAN_DFLASH_BRANCH:-zhongyan_dev}"
ZHONGYAN_TPU_INF_DIR="${ZHONGYAN_TPU_INF_DIR:-${ZHONGYAN_DIR}/tpu-inference}"
ZHONGYAN_TPU_INF_REPO="${ZHONGYAN_TPU_INF_REPO:-${ROOT_TPU_INF_REPO}}"
ZHONGYAN_TPU_INF_BRANCH="${ZHONGYAN_TPU_INF_BRANCH:-zhongyan_dev}"
ZHONGYAN_VLLM_DIR="${ZHONGYAN_VLLM_DIR:-${ZHONGYAN_DIR}/vllm}"
ZHONGYAN_VLLM_REPO="${ZHONGYAN_VLLM_REPO:-https://github.com/aaronzhfeng/vllm.git}"
ZHONGYAN_VLLM_BRANCH="${ZHONGYAN_VLLM_BRANCH:-zhongyan_dev}"
SYNC_MAIN_BRANCH="${SYNC_MAIN_BRANCH:-main}"
ALLOW_MAIN_BRANCH="${ALLOW_MAIN_BRANCH:-0}"

require_non_main_branch() {
  local repo_name="$1"
  local branch="$2"
  if [[ "${ALLOW_MAIN_BRANCH}" != "1" && "${branch}" == "main" ]]; then
    echo "[ERROR] ${repo_name} branch is 'main', but this workflow expects a sub-branch." >&2
    echo "        Set ${repo_name} branch env var to your working branch." >&2
    echo "        Use ALLOW_MAIN_BRANCH=1 only if you intentionally want main." >&2
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

require_non_main_branch "ROOT_TPU_INF" "${ROOT_TPU_INF_BRANCH}"
require_non_main_branch "ZHONGYAN_DFLASH" "${ZHONGYAN_DFLASH_BRANCH}"
require_non_main_branch "ZHONGYAN_TPU_INF" "${ZHONGYAN_TPU_INF_BRANCH}"
require_non_main_branch "ZHONGYAN_VLLM" "${ZHONGYAN_VLLM_BRANCH}"

echo "==> Bootstrap root repo"
clone_or_checkout_branch "${ROOT_TPU_INF_DIR}" "${ROOT_TPU_INF_REPO}" "${ROOT_TPU_INF_BRANCH}"

echo ""
echo "==> Bootstrap zhongyan_dev repos"
mkdir -p "${ZHONGYAN_DIR}"

# Standalone clones.
clone_or_checkout_branch "${ZHONGYAN_DFLASH_DIR}" "${ZHONGYAN_DFLASH_REPO}" "${ZHONGYAN_DFLASH_BRANCH}"
clone_or_checkout_branch "${ZHONGYAN_TPU_INF_DIR}" "${ZHONGYAN_TPU_INF_REPO}" "${ZHONGYAN_TPU_INF_BRANCH}"
clone_or_checkout_branch "${ZHONGYAN_VLLM_DIR}" "${ZHONGYAN_VLLM_REPO}" "${ZHONGYAN_VLLM_BRANCH}"

echo ""
echo "Done."
echo "Root TPU inference: ${ROOT_TPU_INF_DIR} (branch=${ROOT_TPU_INF_BRANCH})"
echo "Zhongyan repos:"
echo "  - ${ZHONGYAN_DFLASH_DIR}"
echo "  - ${ZHONGYAN_TPU_INF_DIR}"
echo "  - ${ZHONGYAN_VLLM_DIR}"
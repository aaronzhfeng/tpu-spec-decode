#!/usr/bin/env bash
#
# Bootstrap all repos needed for the tpu-spec-decode workspace.
#
# Groups:
#   1) Root-level working repos: tpu-inference, vllm
#   2) zhongyan_dev/{dflash, tpu-inference, vllm}
#   3) PR-ready: 3 tpu-inference clones (main, dflash working, PR branch)
#   4) External references (upstream read-only)
#   5) dflash-wide (standalone GPU training repo)
#   6) Brainstorm repos
#   7) Slides theme
#
# Usage:
#   bash preparation/clone_repos.sh              # All groups
#   bash preparation/clone_repos.sh --skip-pr    # Skip PR-ready setup
#   bash preparation/clone_repos.sh --skip-ext   # Skip external references
#
# Environment overrides:
#   ROOT_TPU_INF_BRANCH=dflash-integration bash preparation/clone_repos.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Flags ---
SKIP_PR=0
SKIP_EXTERNAL=0
for arg in "$@"; do
  case "${arg}" in
    --skip-pr)  SKIP_PR=1 ;;
    --skip-ext) SKIP_EXTERNAL=1 ;;
    *)          echo "[WARN] Unknown flag: ${arg}" ;;
  esac
done

# =====================================================================
# Config: Root-level working repos
# =====================================================================
ROOT_TPU_INF_DIR="${ROOT_TPU_INF_DIR:-${REPO_ROOT}/tpu-inference}"
ROOT_TPU_INF_REPO="${ROOT_TPU_INF_REPO:-https://github.com/aaronzhfeng/tpu-inference.git}"
ROOT_TPU_INF_BRANCH="${ROOT_TPU_INF_BRANCH:-dflash-integration}"

ROOT_VLLM_DIR="${ROOT_VLLM_DIR:-${REPO_ROOT}/vllm}"
ROOT_VLLM_REPO="${ROOT_VLLM_REPO:-https://github.com/aaronzhfeng/vllm.git}"
ROOT_VLLM_BRANCH="${ROOT_VLLM_BRANCH:-dflash-speculative-config}"
# vLLM LKG commit required by upstream tpu-inference main
VLLM_LKG_COMMIT="${VLLM_LKG_COMMIT:-05972ea7e5f81250cc4ceaae8a174cfffe7755ac}"

# =====================================================================
# Config: zhongyan_dev repos
# =====================================================================
ZHONGYAN_DIR="${ZHONGYAN_DIR:-${REPO_ROOT}/zhongyan_dev}"
ZHONGYAN_DFLASH_DIR="${ZHONGYAN_DFLASH_DIR:-${ZHONGYAN_DIR}/dflash}"
ZHONGYAN_DFLASH_REPO="${ZHONGYAN_DFLASH_REPO:-https://github.com/Zhongyan0721/dflash}"
ZHONGYAN_DFLASH_BRANCH="${ZHONGYAN_DFLASH_BRANCH:-main}"  # Zhongyan0721/dflash only has main
ZHONGYAN_TPU_INF_DIR="${ZHONGYAN_TPU_INF_DIR:-${ZHONGYAN_DIR}/tpu-inference}"
ZHONGYAN_TPU_INF_REPO="${ZHONGYAN_TPU_INF_REPO:-${ROOT_TPU_INF_REPO}}"
ZHONGYAN_TPU_INF_BRANCH="${ZHONGYAN_TPU_INF_BRANCH:-zhongyan_dev}"
ZHONGYAN_VLLM_DIR="${ZHONGYAN_VLLM_DIR:-${ZHONGYAN_DIR}/vllm}"
ZHONGYAN_VLLM_REPO="${ZHONGYAN_VLLM_REPO:-https://github.com/aaronzhfeng/vllm.git}"
ZHONGYAN_VLLM_BRANCH="${ZHONGYAN_VLLM_BRANCH:-zhongyan_dev}"

# =====================================================================
# Config: PR-ready (3 tpu-inference clones)
# =====================================================================
PR_DIR="${PR_DIR:-${REPO_ROOT}/pr-ready}"
PR_TPU_INF_REPO="${PR_TPU_INF_REPO:-${ROOT_TPU_INF_REPO}}"
UPSTREAM_TPU_INF_REPO="${UPSTREAM_TPU_INF_REPO:-https://github.com/vllm-project/tpu-inference.git}"

# =====================================================================
# Config: External references
# =====================================================================
EXTERNAL_DIR="${EXTERNAL_DIR:-${REPO_ROOT}/external}"

# =====================================================================
# Config: dflash-wide
# =====================================================================
DFLASH_WIDE_DIR="${DFLASH_WIDE_DIR:-${REPO_ROOT}/dflash-wide}"
DFLASH_WIDE_REPO="${DFLASH_WIDE_REPO:-https://github.com/aaronzhfeng/dflash-wide.git}"
DFLASH_WIDE_BRANCH="${DFLASH_WIDE_BRANCH:-main}"

# =====================================================================
# Config: Brainstorm repos
# =====================================================================
BRAINSTORM_00_DIR="${BRAINSTORM_00_DIR:-${REPO_ROOT}/brainstorm-00-core}"
BRAINSTORM_00_REPO="${BRAINSTORM_00_REPO:-https://github.com/aaronzhfeng/brainstorm-00-core.git}"
BRAINSTORM_20_DIR="${BRAINSTORM_20_DIR:-${REPO_ROOT}/brainstorm-20-spec-decode-diffusion}"
BRAINSTORM_20_REPO="${BRAINSTORM_20_REPO:-https://github.com/aaronzhfeng/brainstorm-20-spec-decode-diffusion.git}"

# =====================================================================
# Config: Slides theme
# =====================================================================
SLIDES_MTHEME_DIR="${SLIDES_MTHEME_DIR:-${REPO_ROOT}/slides/mtheme}"
SLIDES_MTHEME_REPO="${SLIDES_MTHEME_REPO:-https://github.com/matze/mtheme.git}"

# =====================================================================
# Shared config
# =====================================================================
SYNC_MAIN_BRANCH="${SYNC_MAIN_BRANCH:-main}"
ALLOW_MAIN_BRANCH="${ALLOW_MAIN_BRANCH:-0}"

# =====================================================================
# Helpers
# =====================================================================
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

clone_if_missing() {
  local repo_dir="$1"
  local remote_url="$2"
  local branch="${3:-main}"

  if [[ -e "${repo_dir}/.git" ]]; then
    echo "[INFO] Exists: ${repo_dir}"
    return 0
  fi
  if [[ -e "${repo_dir}" ]]; then
    echo "[WARN] Path exists but not a git repo: ${repo_dir} — skipping"
    return 0
  fi
  mkdir -p "$(dirname "${repo_dir}")"
  echo "[INFO] Cloning ${remote_url} -> ${repo_dir} (branch=${branch})"
  git clone --branch "${branch}" --single-branch "${remote_url}" "${repo_dir}" 2>/dev/null || \
    git clone "${remote_url}" "${repo_dir}"
}

echo "========================================"
echo "  Clone Repos — tpu-spec-decode"
echo "  Root: ${REPO_ROOT}"
echo "========================================"

# =====================================================================
# Group 1: Root-level working repos
# =====================================================================
echo ""
echo "==> [1/7] Root working repos"

require_non_main_branch "ROOT_TPU_INF" "${ROOT_TPU_INF_BRANCH}"
clone_or_checkout_branch "${ROOT_TPU_INF_DIR}" "${ROOT_TPU_INF_REPO}" "${ROOT_TPU_INF_BRANCH}"

require_non_main_branch "ROOT_VLLM" "${ROOT_VLLM_BRANCH}"
clone_or_checkout_branch "${ROOT_VLLM_DIR}" "${ROOT_VLLM_REPO}" "${ROOT_VLLM_BRANCH}"

# Create vllm-lkg branch at the known-good commit for upstream compatibility
if ! git -C "${ROOT_VLLM_DIR}" rev-parse --verify "refs/heads/vllm-lkg" >/dev/null 2>&1; then
  echo "[INFO] Creating vllm-lkg branch at ${VLLM_LKG_COMMIT}"
  git -C "${ROOT_VLLM_DIR}" fetch origin "${VLLM_LKG_COMMIT}" 2>/dev/null || true
  git -C "${ROOT_VLLM_DIR}" branch vllm-lkg "${VLLM_LKG_COMMIT}" 2>/dev/null || true
fi

# =====================================================================
# Group 2: zhongyan_dev repos
# =====================================================================
echo ""
echo "==> [2/7] zhongyan_dev repos"
mkdir -p "${ZHONGYAN_DIR}"

# Zhongyan's dflash fork only has main — allow it
ALLOW_MAIN_BRANCH=1 clone_or_checkout_branch "${ZHONGYAN_DFLASH_DIR}" "${ZHONGYAN_DFLASH_REPO}" "${ZHONGYAN_DFLASH_BRANCH}" 2>/dev/null || \
  clone_if_missing "${ZHONGYAN_DFLASH_DIR}" "${ZHONGYAN_DFLASH_REPO}" "${ZHONGYAN_DFLASH_BRANCH}"

require_non_main_branch "ZHONGYAN_TPU_INF" "${ZHONGYAN_TPU_INF_BRANCH}"
require_non_main_branch "ZHONGYAN_VLLM" "${ZHONGYAN_VLLM_BRANCH}"
clone_or_checkout_branch "${ZHONGYAN_TPU_INF_DIR}" "${ZHONGYAN_TPU_INF_REPO}" "${ZHONGYAN_TPU_INF_BRANCH}"
clone_or_checkout_branch "${ZHONGYAN_VLLM_DIR}" "${ZHONGYAN_VLLM_REPO}" "${ZHONGYAN_VLLM_BRANCH}"

# =====================================================================
# Group 3: PR-ready (3 tpu-inference clones)
# =====================================================================
if [[ "${SKIP_PR}" != "1" ]]; then
  echo ""
  echo "==> [3/7] PR-ready setup"
  mkdir -p "${PR_DIR}"

  # main — synced with upstream
  clone_if_missing "${PR_DIR}/main" "${PR_TPU_INF_REPO}" "main"
  if ! git -C "${PR_DIR}/main" remote get-url upstream >/dev/null 2>&1; then
    git -C "${PR_DIR}/main" remote add upstream "${UPSTREAM_TPU_INF_REPO}"
  fi

  # dflash — working branch copy
  clone_if_missing "${PR_DIR}/dflash" "${PR_TPU_INF_REPO}" "dflash-integration"

  # pr — clean PR branch (based on upstream/main with DFlash commits)
  clone_if_missing "${PR_DIR}/pr" "${PR_TPU_INF_REPO}" "main"
  if ! git -C "${PR_DIR}/pr" remote get-url upstream >/dev/null 2>&1; then
    git -C "${PR_DIR}/pr" remote add upstream "${UPSTREAM_TPU_INF_REPO}"
  fi
  # Create pr/dflash branch if it doesn't exist locally
  if ! git -C "${PR_DIR}/pr" rev-parse --verify "refs/heads/pr/dflash" >/dev/null 2>&1; then
    echo "[INFO] Fetching upstream and creating pr/dflash branch"
    git -C "${PR_DIR}/pr" fetch upstream >/dev/null 2>&1 || true
    git -C "${PR_DIR}/pr" checkout -b pr/dflash upstream/main 2>/dev/null || true
  else
    git -C "${PR_DIR}/pr" checkout pr/dflash 2>/dev/null || true
  fi
else
  echo ""
  echo "==> [3/7] PR-ready setup — SKIPPED (--skip-pr)"
fi

# =====================================================================
# Group 4: External references (upstream, read-only)
# =====================================================================
if [[ "${SKIP_EXTERNAL}" != "1" ]]; then
  echo ""
  echo "==> [4/7] External references"
  mkdir -p "${EXTERNAL_DIR}"

  clone_if_missing "${EXTERNAL_DIR}/tpu-inference"          "https://github.com/vllm-project/tpu-inference.git"
  clone_if_missing "${EXTERNAL_DIR}/dflash"                 "https://github.com/z-lab/dflash.git"
  clone_if_missing "${EXTERNAL_DIR}/speculators"            "https://github.com/vllm-project/speculators.git"
  clone_if_missing "${EXTERNAL_DIR}/failfast"               "https://github.com/ruipeterpan/failfast.git"
  clone_if_missing "${EXTERNAL_DIR}/LookaheadReasoning"     "https://github.com/hao-ai-lab/LookaheadReasoning.git"
  clone_if_missing "${EXTERNAL_DIR}/Lookahead-Reasoning-TPU" "https://github.com/ConstBob/Lookahead-Reasoning.git" "tpu"
  clone_if_missing "${EXTERNAL_DIR}/prompt-lookup-decoding" "https://github.com/apoorvumang/prompt-lookup-decoding.git"
else
  echo ""
  echo "==> [4/7] External references — SKIPPED (--skip-ext)"
fi

# =====================================================================
# Group 5: dflash-wide (standalone GPU training repo)
# =====================================================================
echo ""
echo "==> [5/7] dflash-wide"
clone_if_missing "${DFLASH_WIDE_DIR}" "${DFLASH_WIDE_REPO}" "${DFLASH_WIDE_BRANCH}"

# =====================================================================
# Group 6: Brainstorm repos
# =====================================================================
echo ""
echo "==> [6/7] Brainstorm repos"
clone_if_missing "${BRAINSTORM_00_DIR}" "${BRAINSTORM_00_REPO}" "main"
clone_if_missing "${BRAINSTORM_20_DIR}" "${BRAINSTORM_20_REPO}" "main"

# =====================================================================
# Group 7: Slides theme
# =====================================================================
echo ""
echo "==> [7/7] Slides theme"
clone_if_missing "${SLIDES_MTHEME_DIR}" "${SLIDES_MTHEME_REPO}" "master"

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "========================================"
echo "  Done. Repository layout:"
echo "========================================"
echo ""
echo "Working repos:"
echo "  tpu-inference/      ${ROOT_TPU_INF_BRANCH}"
echo "  vllm/               ${ROOT_VLLM_BRANCH} (+ vllm-lkg at ${VLLM_LKG_COMMIT:0:8})"
echo ""
echo "zhongyan_dev:"
echo "  dflash/             ${ZHONGYAN_DFLASH_BRANCH} (Zhongyan0721/dflash)"
echo "  tpu-inference/      ${ZHONGYAN_TPU_INF_BRANCH}"
echo "  vllm/               ${ZHONGYAN_VLLM_BRANCH}"
echo ""
if [[ "${SKIP_PR}" != "1" ]]; then
echo "PR-ready:"
echo "  pr-ready/main/      main (synced with upstream)"
echo "  pr-ready/dflash/    dflash-integration"
echo "  pr-ready/pr/        pr/dflash (clean PR branch)"
echo ""
fi
echo "dflash-wide/          ${DFLASH_WIDE_BRANCH}"
echo ""
echo "Brainstorm:"
echo "  brainstorm-00-core/                    main"
echo "  brainstorm-20-spec-decode-diffusion/   main"
echo ""
echo "Slides:"
echo "  slides/mtheme/      master (matze/mtheme)"
if [[ "${SKIP_EXTERNAL}" != "1" ]]; then
echo ""
echo "External (read-only):"
echo "  tpu-inference, dflash, speculators, failfast,"
echo "  LookaheadReasoning, Lookahead-Reasoning-TPU (tpu branch),"
echo "  prompt-lookup-decoding"
fi

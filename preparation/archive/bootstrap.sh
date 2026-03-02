#!/usr/bin/env bash
# preparation/bootstrap.sh
#
# Entry-point for a completely fresh tpu-ubuntu2204-base TPU VM.
# Run this ONCE right after SSH-ing into a new node.
#
# It will:
#   1. Clone tpu-spec-decode (the parent repo)
#   2. Clone all sub-repos on their correct branches
#   3. Install all Python dependencies (JAX / TPU v5 stack)
#   4. Run a JAX smoke check
#
# Usage — paste this block into the fresh terminal:
#
#   export ROOT_TPU_INF_BRANCH=dflash-integration
#   export ZHONGYAN_DFLASH_BRANCH=zhongyan_dev
#   export ZHONGYAN_TPU_INF_BRANCH=zhongyan_dev
#   export ZHONGYAN_VLLM_BRANCH=zhongyan_dev
#   export HF_TOKEN=hf_xxxxxxxxxxxx          # optional
#   bash <(curl -fsSL https://raw.githubusercontent.com/aaronzhfeng/tpu-spec-decode/main/preparation/bootstrap.sh)
#
# Or clone manually first and run:
#   git clone https://github.com/aaronzhfeng/tpu-spec-decode.git
#   cd tpu-spec-decode
#   bash preparation/bootstrap.sh
#
# Branch defaults (override with env vars above):
#   ROOT_TPU_INF_BRANCH   = dflash-integration
#   ZHONGYAN_DFLASH_BRANCH  = zhongyan_dev
#   ZHONGYAN_TPU_INF_BRANCH = zhongyan_dev
#   ZHONGYAN_VLLM_BRANCH    = zhongyan_dev

set -euo pipefail

PARENT_REPO_URL="${PARENT_REPO_URL:-https://github.com/aaronzhfeng/tpu-spec-decode.git}"
PARENT_REPO_BRANCH="${PARENT_REPO_BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/tpu-spec-decode}"

# -------------------------------------------------------------------
# Step 1: Clone or locate the parent repo
# -------------------------------------------------------------------
echo ""
echo "==> [1/3] Parent repo"

if [[ "$(pwd)" == "${INSTALL_DIR}" ]] || [[ -f "$(pwd)/preparation/bootstrap.sh" ]]; then
  # Already inside the repo
  INSTALL_DIR="$(pwd)"
  echo "     Already in repo: ${INSTALL_DIR}"
elif [[ -d "${INSTALL_DIR}/.git" ]]; then
  echo "     Repo exists at ${INSTALL_DIR}, pulling latest"
  git -C "${INSTALL_DIR}" pull --ff-only
else
  echo "     Cloning ${PARENT_REPO_URL} -> ${INSTALL_DIR}"
  git clone --branch "${PARENT_REPO_BRANCH}" "${PARENT_REPO_URL}" "${INSTALL_DIR}"
fi

cd "${INSTALL_DIR}"

# -------------------------------------------------------------------
# Step 2: Sub-repos + Python environment (delegates to setup_tpu_v5.sh)
# -------------------------------------------------------------------
echo ""
echo "==> [2/3] Delegating to preparation/setup_tpu_v5.sh"
echo "     ROOT_TPU_INF_BRANCH   = ${ROOT_TPU_INF_BRANCH:-dflash-integration}"
echo "     ZHONGYAN_DFLASH_BRANCH  = ${ZHONGYAN_DFLASH_BRANCH:-zhongyan_dev}"
echo "     ZHONGYAN_TPU_INF_BRANCH = ${ZHONGYAN_TPU_INF_BRANCH:-zhongyan_dev}"
echo "     ZHONGYAN_VLLM_BRANCH    = ${ZHONGYAN_VLLM_BRANCH:-zhongyan_dev}"
echo ""

ROOT_TPU_INF_BRANCH="${ROOT_TPU_INF_BRANCH:-dflash-integration}" \
ZHONGYAN_DFLASH_BRANCH="${ZHONGYAN_DFLASH_BRANCH:-zhongyan_dev}" \
ZHONGYAN_TPU_INF_BRANCH="${ZHONGYAN_TPU_INF_BRANCH:-zhongyan_dev}" \
ZHONGYAN_VLLM_BRANCH="${ZHONGYAN_VLLM_BRANCH:-zhongyan_dev}" \
HF_TOKEN="${HF_TOKEN:-}" \
bash "${INSTALL_DIR}/preparation/setup_tpu_v5.sh"

# -------------------------------------------------------------------
# Step 3: Summary
# -------------------------------------------------------------------
echo ""
echo "==> [3/3] Bootstrap complete"
echo ""
echo "  Repo      : ${INSTALL_DIR}"
echo "  Sub-repos :"
echo "    tpu-inference/  (branch: ${ROOT_TPU_INF_BRANCH:-dflash-integration})"
echo "    zhongyan_dev/dflash         (branch: ${ZHONGYAN_DFLASH_BRANCH:-zhongyan_dev})"
echo "    zhongyan_dev/tpu-inference  (branch: ${ZHONGYAN_TPU_INF_BRANCH:-zhongyan_dev})"
echo "    zhongyan_dev/vllm           (branch: ${ZHONGYAN_VLLM_BRANCH:-zhongyan_dev})"
echo ""
echo "  Run preflight check:"
echo "    cd ${INSTALL_DIR}"
echo "    bash preparation/check_dflash_support.sh host"

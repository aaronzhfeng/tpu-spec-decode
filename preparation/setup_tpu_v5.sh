#!/usr/bin/env bash
# preparation/setup_tpu_v5.sh
#
# One-shot environment setup for a fresh tpu-ubuntu2204-base TPU v5 VM.
# Run this once after SSH-ing into a new node.
#
# Usage:
#   bash preparation/setup_tpu_v5.sh
#
# Optional env overrides:
#   ROOT_TPU_INF_BRANCH=dflash-integration \
#   ZHONGYAN_DFLASH_BRANCH=zhongyan_dev \
#   bash preparation/setup_tpu_v5.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "========================================"
echo "  TPU v5 Environment Setup"
echo "  Repo root: ${REPO_ROOT}"
echo "========================================"

# -------------------------------------------------------------------
# 1. System packages
# -------------------------------------------------------------------
echo ""
echo "==> [1/5] Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y git docker.io jq curl python3-pip python3-venv
sudo usermod -aG docker "$USER"

# -------------------------------------------------------------------
# 2. Clone sub-repos
# -------------------------------------------------------------------
echo ""
echo "==> [2/5] Cloning sub-repos"
cd "${REPO_ROOT}"

# Initialize git submodules (vllm, brainstorm-00-core) if not already done
echo "     Initializing git submodules..."
git submodule update --init --recursive 2>/dev/null || true

# Clone branch-pinned standalone repos (tpu-inference, zhongyan_dev/*)
ROOT_TPU_INF_BRANCH="${ROOT_TPU_INF_BRANCH:-dflash-integration}" \
ZHONGYAN_DFLASH_BRANCH="${ZHONGYAN_DFLASH_BRANCH:-zhongyan_dev}" \
ZHONGYAN_TPU_INF_BRANCH="${ZHONGYAN_TPU_INF_BRANCH:-zhongyan_dev}" \
ZHONGYAN_VLLM_BRANCH="${ZHONGYAN_VLLM_BRANCH:-zhongyan_dev}" \
bash "${SCRIPT_DIR}/clone_repos.sh"

# -------------------------------------------------------------------
# 3. Install Python dependencies (JAX + tpu-inference stack)
# -------------------------------------------------------------------
echo ""
echo "==> [3/5] Installing Python dependencies (JAX / TPU v5)"

# Core tpu-inference stack
pip install --upgrade pip
pip install -r "${REPO_ROOT}/tpu-inference/requirements.txt"
pip install -r "${SCRIPT_DIR}/requirements_v5.txt"
pip install -r "${REPO_ROOT}/tpu-inference/requirements_benchmarking.txt"

# vLLM (install from the local fork)
pip install -e "${REPO_ROOT}/vllm" --no-build-isolation

# tpu-inference itself
pip install -e "${REPO_ROOT}/tpu-inference" --no-build-isolation

# -------------------------------------------------------------------
# 4. HuggingFace setup
# -------------------------------------------------------------------
echo ""
echo "==> [4/5] HuggingFace token (optional)"
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
  echo "     HF token applied."
else
  echo "     HF_TOKEN not set — skip. Run: huggingface-cli login"
fi

# -------------------------------------------------------------------
# 5. Smoke check
# -------------------------------------------------------------------
echo ""
echo "==> [5/5] Smoke check: JAX TPU device visibility"
python3 - <<'PYEOF'
import jax
devices = jax.devices()
print(f"  JAX version : {jax.__version__}")
print(f"  Devices     : {len(devices)}")
for d in devices:
    print(f"    {d}")
if not any("tpu" in str(d).lower() for d in devices):
    print("  [WARN] No TPU devices found — are you on a TPU VM?")
else:
    print("  [OK] TPU devices visible")
PYEOF

echo ""
echo "========================================"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    bash preparation/check_dflash_support.sh host"
echo "    bash preparation/run_dflash_acceptance_smoke.sh"
echo ""
echo "  NOTE: Log out and back in (or run 'newgrp docker')"
echo "  for Docker group membership to take effect."
echo "========================================"

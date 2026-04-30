#!/usr/bin/env bash
# preparation/bootstrap.sh
#
# Entry point for a fresh tpu-ubuntu2204-base TPU VM. Run this once after
# SSH-ing into a new node. Idempotent: safe to re-run.
#
# Tested on TPU v6e (primary) and v5p. v4 paths preserved but not validated;
# the original v4-era setup script remains at preparation/setup_tpu_v5.sh
# for reference (no longer invoked by this bootstrap).
#
# Usage:
#   git clone --recurse-submodules https://github.com/aaronzhfeng/tpu-spec-decode.git
#   cd tpu-spec-decode
#   bash preparation/bootstrap.sh
#
# Optional env overrides:
#   HF_TOKEN=hf_xxxxxxxx  bash preparation/bootstrap.sh   # gated checkpoints (Llama)
#   SKIP_PIP=1            bash preparation/bootstrap.sh   # already-installed env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

info()  { echo "[INFO]  $*"; }
ok()    { echo "[OK]    $*"; }
warn()  { echo "[WARN]  $*"; }

echo "========================================"
echo "  tpu-spec-decode bootstrap"
echo "  Repo root: ${REPO_ROOT}"
echo "========================================"

# -------------------------------------------------------------------
# Step 1: Initialize the tpu-inference submodule
# -------------------------------------------------------------------
echo ""
info "[1/3] Initializing tpu-inference submodule"

if [[ ! -f "tpu-inference/setup.py" ]]; then
  git submodule update --init --recursive tpu-inference
fi

if [[ ! -f "tpu-inference/setup.py" ]]; then
  echo "[ERROR] tpu-inference submodule did not populate. Try:" >&2
  echo "        git submodule update --init --recursive" >&2
  exit 1
fi
ok "tpu-inference at $(git -C tpu-inference rev-parse --short HEAD) ($(git -C tpu-inference rev-parse --abbrev-ref HEAD))"

# -------------------------------------------------------------------
# Step 2: Verify test path symlinks (idempotent)
# -------------------------------------------------------------------
echo ""
info "[2/3] Verifying test path symlinks"

for link_target in models spec_decode; do
  if [[ ! -e "tests/${link_target}" ]]; then
    ln -sfn "../tpu-inference/tests/${link_target}" "tests/${link_target}"
    ok "created tests/${link_target}"
  else
    ok "tests/${link_target} present"
  fi
done

# -------------------------------------------------------------------
# Step 3: Install Python dependencies
# -------------------------------------------------------------------
echo ""
info "[3/3] Installing Python dependencies"

if [[ "${SKIP_PIP:-0}" == "1" ]]; then
  warn "SKIP_PIP=1 set, skipping pip install (assuming env is already configured)"
else
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e tpu-inference/
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  ok "HF_TOKEN set; gated checkpoints (e.g. Llama-3.1-8B-Instruct) will resolve"
fi

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Bootstrap complete"
echo "========================================"
cat <<EOF

Next steps (run from ${REPO_ROOT}):

  # Unit tests (no TPU run; resolve via the symlinks into tpu-inference/)
  pytest tests/spec_decode/test_dflash.py
  pytest tests/models/jax/test_qwen3_dflash.py
  pytest tests/models/jax/test_qwen3_dflash_attention.py

  # Standalone benchmark (single dataset, ~5 min on v6e)
  DATASET=math500 bash tests/standalone_benchmark.sh \\
      --max-samples 8 --max-new-tokens 256

  # Full vLLM pipeline benchmark
  bash tests/benchmark.sh math

EOF

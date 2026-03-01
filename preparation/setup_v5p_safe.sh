#!/usr/bin/env bash
# preparation/setup_v5p_safe.sh
#
# Safe, incremental setup for TPU v5p-8 on tpu-ubuntu2204-base.
#
# Key differences from the old setup_tpu_v5.sh:
#   - Uses a venv (not global pip)
#   - Pins JAX < 0.7 (Python 3.10 compatibility)
#   - No apt-get update/upgrade
#   - Verifies SSH works between steps
#   - Each step can be re-run independently
#
# Usage:
#   bash preparation/setup_v5p_safe.sh
#
# Or run individual steps:
#   bash preparation/setup_v5p_safe.sh step1
#   bash preparation/setup_v5p_safe.sh step3
#
# See V5P_SETUP_MANUAL.md for alternatives when things go wrong.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${HOME}/venv}"
STEP="${1:-all}"

info()  { echo "[INFO]  $*"; }
ok()    { echo "[OK]    $*"; }
warn()  { echo "[WARN]  $*"; }
fail()  { echo "[FAIL]  $*" >&2; exit 1; }

check_ssh() {
    # Verify sshd is still running - if this fails, we've broken the node
    if systemctl is-active --quiet ssh; then
        ok "SSH daemon is running"
    else
        fail "SSH daemon is NOT running - stop immediately and investigate"
    fi
}

# ---------------------------------------------------------------
# Step 1: Docker group (requires re-login to take effect)
# ---------------------------------------------------------------
step1() {
    info "Step 1: Adding user to docker group"
    if groups | grep -q docker; then
        ok "Already in docker group"
    else
        sudo usermod -aG docker "$USER"
        warn "Added to docker group - log out and back in for it to take effect"
    fi
    check_ssh
}

# ---------------------------------------------------------------
# Step 2: Create Python venv
# ---------------------------------------------------------------
step2() {
    info "Step 2: Creating Python venv at ${VENV_DIR}"
    if [[ -d "${VENV_DIR}" ]]; then
        ok "Venv already exists at ${VENV_DIR}"
    else
        python3 -m venv "${VENV_DIR}"
        ok "Venv created"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip --quiet
    ok "pip upgraded inside venv: $(pip --version)"
    check_ssh
}

# ---------------------------------------------------------------
# Step 3: Install JAX for v5p (pinned < 0.7 for Python 3.10)
# ---------------------------------------------------------------
step3() {
    info "Step 3: Installing JAX (pinned < 0.7 for Python 3.10)"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    pip install "jax[tpu]<0.7" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

    ok "JAX installed: $(python3 -c 'import jax; print(jax.__version__)')"
    check_ssh
}

# ---------------------------------------------------------------
# Step 4: Verify JAX sees TPU devices
# ---------------------------------------------------------------
step4() {
    info "Step 4: Verifying JAX TPU device visibility"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    python3 -c "
import jax
devices = jax.devices()
print(f'  JAX version:  {jax.__version__}')
print(f'  Device count: {len(devices)}')
for d in devices:
    print(f'    {d}')
tpu_count = sum(1 for d in devices if 'tpu' in str(d).lower())
if tpu_count == 0:
    raise RuntimeError('No TPU devices found - check /dev/vfio/ and libtpu')
print(f'  TPU chips: {tpu_count}')
"
    ok "JAX sees TPU devices"
    check_ssh
}

# ---------------------------------------------------------------
# Step 5: Clone repos
# ---------------------------------------------------------------
step5() {
    info "Step 5: Cloning repos"
    cd "${REPO_ROOT}"

    ROOT_TPU_INF_BRANCH="${ROOT_TPU_INF_BRANCH:-dflash-integration}" \
    ZHONGYAN_DFLASH_BRANCH="${ZHONGYAN_DFLASH_BRANCH:-zhongyan_dev}" \
    ZHONGYAN_TPU_INF_BRANCH="${ZHONGYAN_TPU_INF_BRANCH:-zhongyan_dev}" \
    ZHONGYAN_VLLM_BRANCH="${ZHONGYAN_VLLM_BRANCH:-zhongyan_dev}" \
    bash "${SCRIPT_DIR}/clone_repos.sh"

    ok "Repos cloned"
    check_ssh
}

# ---------------------------------------------------------------
# Step 6: Install tpu-inference
# ---------------------------------------------------------------
step6() {
    info "Step 6: Installing tpu-inference dependencies"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    cd "${REPO_ROOT}"

    if [[ -f tpu-inference/requirements.txt ]]; then
        pip install -r tpu-inference/requirements.txt
    fi
    if [[ -f tpu-inference/requirements_benchmarking.txt ]]; then
        pip install -r tpu-inference/requirements_benchmarking.txt
    fi
    pip install -e tpu-inference --no-build-isolation

    ok "tpu-inference installed"
    check_ssh
}

# ---------------------------------------------------------------
# Step 7: Install vLLM (heavyweight - compiles C extensions)
# ---------------------------------------------------------------
step7() {
    info "Step 7: Installing vLLM (this may take several minutes)"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    cd "${REPO_ROOT}"

    pip install -e vllm --no-build-isolation

    ok "vLLM installed"
    check_ssh
}

# ---------------------------------------------------------------
# Step 8: Re-pin JAX if requirements overwrote it
# ---------------------------------------------------------------
step8() {
    info "Step 8: Verifying JAX version (re-pin if overwritten)"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    JAX_VER=$(python3 -c "import jax; print(jax.__version__)")
    JAX_MAJOR_MINOR=$(echo "${JAX_VER}" | cut -d. -f1,2)

    # Check if JAX got upgraded past 0.6.x
    if python3 -c "
import sys
v = '${JAX_MAJOR_MINOR}'.split('.')
major, minor = int(v[0]), int(v[1])
if major > 0 or (major == 0 and minor >= 7):
    print('JAX ${JAX_VER} is too new for Python 3.10 - re-pinning')
    sys.exit(1)
"; then
        ok "JAX version ${JAX_VER} is compatible with Python 3.10"
    else
        warn "JAX was upgraded to ${JAX_VER} - re-pinning to < 0.7"
        pip install "jax[tpu]<0.7" --force-reinstall \
            -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ok "JAX re-pinned: $(python3 -c 'import jax; print(jax.__version__)')"
    fi

    # Re-verify TPU visibility
    python3 -c "import jax; assert jax.device_count() > 0, 'No devices'; print(f'OK: {jax.device_count()} devices')"
    check_ssh
}

# ---------------------------------------------------------------
# Step 9: Verify DFlash imports and run smoke tests
# ---------------------------------------------------------------
step9() {
    info "Step 9: Smoke tests"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    cd "${REPO_ROOT}"

    python3 -c "
from tpu_inference.spec_decode.jax.dflash import DFlashProposer
print('  DFlash import: OK')
"

    bash preparation/check_dflash_support.sh host

    ok "All smoke tests passed"
    check_ssh
}

# ---------------------------------------------------------------
# Run steps
# ---------------------------------------------------------------
case "${STEP}" in
    all)
        step1; step2; step3; step4; step5; step6; step7; step8; step9
        echo ""
        echo "========================================"
        echo "  Setup complete."
        echo ""
        echo "  Activate venv: source ~/venv/bin/activate"
        echo ""
        echo "  Run DFlash benchmark:"
        echo "    cd ${REPO_ROOT}"
        echo "    bash preparation/run_dflash_acceptance_smoke.sh host"
        echo "========================================"
        ;;
    step[1-9])
        "${STEP}"
        ;;
    *)
        echo "Usage: $0 [all|step1|step2|...|step9]"
        exit 1
        ;;
esac

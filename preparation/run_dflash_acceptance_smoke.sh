#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-tpu:latest}"
MANIFEST="${MANIFEST:-verification/contribution/manifests/dflash_acceptance_smoke.json}"
RUN_ID="${RUN_ID:-dflash_acceptance_smoke_$(date +%Y%m%d_%H%M%S)}"
HOST_HF_CACHE="${HOST_HF_CACHE:-/dev/shm/hf-cache}"
HOST_OUT_ROOT="${HOST_OUT_ROOT:-/dev/shm/contrib-out}"
TMPFS_TMP_SIZE="${TMPFS_TMP_SIZE:-80g}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
HOST_TPU_LOG_DIR="${HOST_TPU_LOG_DIR:-/dev/shm/tpu-logs}"

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  bash "${ROOT_DIR}/preparation/check_dflash_support.sh" docker
fi

resolve_writable_dir() {
  local var_name="$1"
  local fallback_dir="$2"
  local dir="${!var_name}"

  if ! mkdir -p "${dir}" >/dev/null 2>&1; then
    echo "[WARN] Could not create '${dir}'."
  fi
  if [[ -d "${dir}" ]]; then
    chmod 777 "${dir}" >/dev/null 2>&1 || true
  fi
  if [[ -d "${dir}" && -w "${dir}" ]]; then
    return 0
  fi

  echo "[WARN] '${dir}' is not writable. Falling back to '${fallback_dir}'."
  mkdir -p "${fallback_dir}"
  chmod 700 "${fallback_dir}" >/dev/null 2>&1 || true
  if [[ ! -d "${fallback_dir}" || ! -w "${fallback_dir}" ]]; then
    echo "[ERROR] Fallback directory '${fallback_dir}' is not writable." >&2
    exit 1
  fi
  printf -v "${var_name}" "%s" "${fallback_dir}"
}

resolve_writable_dir HOST_HF_CACHE "/dev/shm/${USER}-hf-cache"
resolve_writable_dir HOST_OUT_ROOT "/dev/shm/${USER}-contrib-out"
resolve_writable_dir HOST_TPU_LOG_DIR "/dev/shm/${USER}-tpu-logs"

if docker info >/dev/null 2>&1; then
  DOCKER_CMD=(docker)
else
  DOCKER_CMD=(sudo docker)
fi

echo "Running DFlash acceptance smoke"
echo "root=${ROOT_DIR}"
echo "manifest=${MANIFEST}"
echo "run_id=${RUN_ID}"
echo "host_hf_cache=${HOST_HF_CACHE}"
echo "host_out_root=${HOST_OUT_ROOT}"
echo "host_tpu_log_dir=${HOST_TPU_LOG_DIR}"
echo "docker_image=${DOCKER_IMAGE}"

"${DOCKER_CMD[@]}" run --rm --pull=never \
  --privileged --network host --ipc host \
  --tmpfs "/tmp:rw,size=${TMPFS_TMP_SIZE}" \
  -e HF_HOME=/hf-cache \
  -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub \
  -e XDG_CACHE_HOME=/hf-cache/xdg \
  -e JAX_COMPILATION_CACHE_DIR=/hf-cache/jax \
  -e TMPDIR=/tmp \
  -e TPU_LOG_DIR=/tmp/tpu_logs \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONPATH=/workspace/tpu-spec-decode/deps/vllm:/workspace/tpu-spec-decode/deps/tpu-inference \
  -v "${HOST_HF_CACHE}:/hf-cache" \
  -v "${HOST_OUT_ROOT}:/out" \
  -v "${HOST_TPU_LOG_DIR}:/tmp/tpu_logs" \
  -v "${ROOT_DIR}:/workspace/tpu-spec-decode" \
  -w /workspace/tpu-spec-decode \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    OUT_ROOT=/out \
    MANIFEST=\"${MANIFEST}\" \
    RUN_ID=\"${RUN_ID}\" \
    bash verification/contribution/sh/run_contribution_matrix.sh
  "

echo "Done. Artifacts: ${HOST_OUT_ROOT}/${RUN_ID}"


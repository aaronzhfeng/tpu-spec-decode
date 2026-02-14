#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-docker}" # docker|host
PYTHON_BIN="${PYTHON_BIN:-python3}"
DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-tpu:latest}"
HOST_TPU_LOG_DIR="${HOST_TPU_LOG_DIR:-/tmp/tpu_logs}"

resolve_writable_dir() {
  local var_name="$1"
  local fallback_dir="$2"
  local dir="${!var_name}"

  if ! mkdir -p "${dir}" >/dev/null 2>&1; then
    echo "[WARN] Could not create '${dir}'."
  fi
  if [[ -d "${dir}" ]]; then
    chmod 700 "${dir}" >/dev/null 2>&1 || true
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

run_host_check() {
  resolve_writable_dir HOST_TPU_LOG_DIR "/dev/shm/${USER}-tpu-logs"
  export TPU_LOG_DIR="${HOST_TPU_LOG_DIR}"
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="${ROOT_DIR}/deps/vllm:${ROOT_DIR}/deps/tpu-inference:${PYTHONPATH:-}"
  "${PYTHON_BIN}" "${ROOT_DIR}/preparation/check_dflash_support.py"
}

run_docker_check() {
  resolve_writable_dir HOST_TPU_LOG_DIR "/dev/shm/${USER}-tpu-logs"
  local -a docker_cmd
  if docker info >/dev/null 2>&1; then
    docker_cmd=(docker)
  else
    docker_cmd=(sudo docker)
  fi

  "${docker_cmd[@]}" run --rm --pull=never \
    --privileged --network host --ipc host \
    -v "${ROOT_DIR}:/workspace/tpu-spec-decode" \
    -v "${HOST_TPU_LOG_DIR}:/tmp/tpu_logs" \
    -e TPU_LOG_DIR=/tmp/tpu_logs \
    -w /workspace/tpu-spec-decode \
    "${DOCKER_IMAGE}" \
    bash -lc '
      set -euo pipefail
      export PYTHONNOUSERSITE=1
      export PYTHONPATH=/workspace/tpu-spec-decode/deps/vllm:/workspace/tpu-spec-decode/deps/tpu-inference:${PYTHONPATH:-}
      python3 /workspace/tpu-spec-decode/preparation/check_dflash_support.py
    '
}

case "${MODE}" in
  host)
    echo "mode=host"
    run_host_check
    ;;
  docker)
    echo "mode=docker image=${DOCKER_IMAGE}"
    run_docker_check
    ;;
  *)
    echo "Usage: $0 [docker|host]" >&2
    exit 1
    ;;
esac


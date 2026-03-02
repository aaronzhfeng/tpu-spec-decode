#!/usr/bin/env bash
set -euo pipefail

# DFlash acceptance smoke test.
#
# Modes:
#   host   — run directly in venv (Path A: bare-metal pip install)
#   docker — run inside Docker container (Path B: Docker-based)
#
# Usage:
#   bash preparation/run_dflash_acceptance_smoke.sh host
#   bash preparation/run_dflash_acceptance_smoke.sh docker

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-docker}"
DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-tpu:latest}"
MANIFEST="${MANIFEST:-verification/contribution/manifests/dflash_acceptance_smoke.json}"
RUN_ID="${RUN_ID:-dflash_acceptance_smoke_$(date +%Y%m%d_%H%M%S)}"
HOST_HF_CACHE="${HOST_HF_CACHE:-/dev/shm/hf-cache}"
HOST_OUT_ROOT="${HOST_OUT_ROOT:-/dev/shm/contrib-out}"
TMPFS_TMP_SIZE="${TMPFS_TMP_SIZE:-80g}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
HOST_TPU_LOG_DIR="${HOST_TPU_LOG_DIR:-/dev/shm/tpu-logs}"

# Benchmark defaults (host mode)
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

# ---------------------------------------------------------------
# Host mode (Path A: bare-metal venv with JAX)
# ---------------------------------------------------------------
run_host() {
  resolve_writable_dir HOST_HF_CACHE "/dev/shm/${USER}-hf-cache"
  resolve_writable_dir HOST_OUT_ROOT "/dev/shm/${USER}-contrib-out"
  resolve_writable_dir HOST_TPU_LOG_DIR "/dev/shm/${USER}-tpu-logs"

  if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    bash "${ROOT_DIR}/preparation/check_dflash_support.sh" host
  fi

  export HF_HOME="${HOST_HF_CACHE}"
  export HUGGINGFACE_HUB_CACHE="${HOST_HF_CACHE}/hub"
  export TPU_LOG_DIR="${HOST_TPU_LOG_DIR}"
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="${ROOT_DIR}/vllm:${ROOT_DIR}/tpu-inference:${PYTHONPATH:-}"

  local output_json="${HOST_OUT_ROOT}/${RUN_ID}/results.json"
  mkdir -p "$(dirname "${output_json}")"

  echo "Running DFlash acceptance smoke (host mode)"
  echo "  root=${ROOT_DIR}"
  echo "  target_model=${TARGET_MODEL}"
  echo "  draft_model=${DRAFT_MODEL}"
  echo "  dataset=${DATASET}"
  echo "  max_samples=${MAX_SAMPLES}"
  echo "  max_new_tokens=${MAX_NEW_TOKENS}"
  echo "  max_model_len=${MAX_MODEL_LEN}"
  echo "  hf_cache=${HOST_HF_CACHE}"
  echo "  output=${output_json}"

  "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/standalone_dflash.py" \
    --target-model "${TARGET_MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --dataset "${DATASET}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --output-json "${output_json}"

  echo "Done. Artifacts: ${HOST_OUT_ROOT}/${RUN_ID}"
}

# ---------------------------------------------------------------
# Docker mode (Path B: Docker container)
# ---------------------------------------------------------------
run_docker() {
  resolve_writable_dir HOST_HF_CACHE "/dev/shm/${USER}-hf-cache"
  resolve_writable_dir HOST_OUT_ROOT "/dev/shm/${USER}-contrib-out"
  resolve_writable_dir HOST_TPU_LOG_DIR "/dev/shm/${USER}-tpu-logs"

  if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    bash "${ROOT_DIR}/preparation/check_dflash_support.sh" docker
  fi

  if docker info >/dev/null 2>&1; then
    DOCKER_CMD=(docker)
  else
    DOCKER_CMD=(sudo docker)
  fi

  echo "Running DFlash acceptance smoke (docker mode)"
  echo "  root=${ROOT_DIR}"
  echo "  manifest=${MANIFEST}"
  echo "  run_id=${RUN_ID}"
  echo "  host_hf_cache=${HOST_HF_CACHE}"
  echo "  host_out_root=${HOST_OUT_ROOT}"
  echo "  host_tpu_log_dir=${HOST_TPU_LOG_DIR}"
  echo "  docker_image=${DOCKER_IMAGE}"

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
    -e PYTHONPATH=/workspace/tpu-spec-decode/vllm:/workspace/tpu-spec-decode/tpu-inference \
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
}

# ---------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------
case "${MODE}" in
  host)
    run_host
    ;;
  docker)
    run_docker
    ;;
  *)
    echo "Usage: $0 [host|docker]" >&2
    echo "  host   — run directly (Path A: bare-metal venv)" >&2
    echo "  docker — run in container (Path B: Docker)" >&2
    exit 1
    ;;
esac

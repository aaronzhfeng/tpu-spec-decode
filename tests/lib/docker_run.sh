#!/usr/bin/env bash
# tests/lib/docker_run.sh — Run a command inside the vLLM TPU Docker container.
#
# Usage (from another script that already sourced common.sh):
#   source tests/lib/docker_run.sh
#   docker_exec "python3 my_script.py --flag"
#   docker_exec_script tests/lib/some_script.py
#
# All repo paths, HF cache, and output dirs are automatically mounted.
#
# PYTHONPATH: Use TPU_INFERENCE_DIR and VLLM_DIR (default: root tpu-inference, vllm).
# For v5p with pr-ready: export TPU_INFERENCE_DIR=.../pr-ready/pr VLLM_DIR=.../pr-ready/vllm-lkg
#
# Flax: v4 uses 0.11.1 (default); v5p uses 0.12.4. Set FLAX_VERSION before running.

# Guard: common.sh must already be sourced.
: "${REPO_ROOT:?source tests/lib/common.sh first}"

_build_docker_args() {
  # Build the common Docker run arguments as an array.
  # Caller reads DOCKER_RUN_ARGS after this returns.
  ensure_writable_dir HOST_HF_CACHE    "/dev/shm/${USER:-user}-hf-cache"
  ensure_writable_dir HOST_OUTPUT_DIR   "/dev/shm/${USER:-user}-dflash-outputs"
  ensure_writable_dir HOST_TPU_LOG_DIR  "/dev/shm/${USER:-user}-tpu-logs"

  # Convert host paths to container paths (repo mounted at /workspace/tpu-spec-decode)
  local rel_tpu rel_vllm
  rel_tpu="${TPU_INFERENCE_DIR#${REPO_ROOT}/}"
  rel_vllm="${VLLM_DIR#${REPO_ROOT}/}"
  local py_path="/workspace/tpu-spec-decode/${rel_tpu}:/workspace/tpu-spec-decode/${rel_vllm}"

  DOCKER_RUN_ARGS=(
    --rm
    --privileged
    --network host
    --ipc host
    --tmpfs "/tmp:rw,size=${TMPFS_SIZE}"

    # HuggingFace / JAX caches
    -e HF_HOME=/hf-cache
    -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub
    -e XDG_CACHE_HOME=/hf-cache/xdg
    -e JAX_COMPILATION_CACHE_DIR=/hf-cache/jax
    -e TMPDIR=/tmp
    -e TPU_LOG_DIR=/tmp/tpu_logs
    -e PYTHONNOUSERSITE=1
    -e PYTHONPATH="${py_path}"
    ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"}

    # Mounts
    -v "${HOST_HF_CACHE}:/hf-cache"
    -v "${HOST_OUTPUT_DIR}:/output"
    -v "${HOST_TPU_LOG_DIR}:/tmp/tpu_logs"
    -v "${REPO_ROOT}:/workspace/tpu-spec-decode"

    -w /workspace/tpu-spec-decode
  )
}

docker_exec() {
  # Run an arbitrary bash command string inside the container.
  # Usage: docker_exec "python3 script.py --arg"
  local cmd_string="$1"
  local dcmd
  dcmd="$(docker_cmd)"

  _build_docker_args
  require_docker_image

  # Flax: v4 default 0.11.1 (<0.12); v5p needs 0.12.4 (set FLAX_VERSION=0.12.4)
  local flv="${FLAX_VERSION:-0.11.1}"
  local flax_pin="pip install \"flax==${flv}\" --quiet 2>/dev/null || true; "

  ${dcmd} run "${DOCKER_RUN_ARGS[@]}" "${DOCKER_IMAGE}" \
    bash -lc "set -euo pipefail; ${flax_pin}${cmd_string}"
}

docker_exec_script() {
  # Run a script file (relative to repo root) inside the container.
  # Usage: docker_exec_script path/to/script.sh [args...]
  local script="$1"
  shift
  local args_str=""
  for a in "$@"; do args_str+=" $(printf '%q' "$a")"; done

  docker_exec "bash /workspace/tpu-spec-decode/${script}${args_str}"
}

docker_exec_python() {
  # Run a Python script (relative to repo root) inside the container.
  # Usage: docker_exec_python path/to/script.py [args...]
  local script="$1"
  shift
  local args_str=""
  for a in "$@"; do args_str+=" $(printf '%q' "$a")"; done

  docker_exec "python3 /workspace/tpu-spec-decode/${script}${args_str}"
}

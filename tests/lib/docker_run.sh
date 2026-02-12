#!/usr/bin/env bash
# tests/lib/docker_run.sh — Run a command inside the vLLM TPU Docker container.
#
# Usage (from another script that already sourced common.sh):
#   source tests/lib/docker_run.sh
#   docker_exec "python3 my_script.py --flag"
#   docker_exec_script tests/lib/some_script.py
#
# All repo paths, HF cache, and output dirs are automatically mounted.

# Guard: common.sh must already be sourced.
: "${REPO_ROOT:?source tests/lib/common.sh first}"

_build_docker_args() {
  # Build the common Docker run arguments as an array.
  # Caller reads DOCKER_RUN_ARGS after this returns.
  ensure_writable_dir HOST_HF_CACHE    "/dev/shm/${USER:-user}-hf-cache"
  ensure_writable_dir HOST_OUTPUT_DIR   "/dev/shm/${USER:-user}-dflash-outputs"
  ensure_writable_dir HOST_TPU_LOG_DIR  "/dev/shm/${USER:-user}-tpu-logs"

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
    -e PYTHONPATH=/workspace/tpu-spec-decode/vllm:/workspace/tpu-spec-decode/tpu-inference
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

  ${dcmd} run "${DOCKER_RUN_ARGS[@]}" "${DOCKER_IMAGE}" \
    bash -lc "set -euo pipefail; ${cmd_string}"
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

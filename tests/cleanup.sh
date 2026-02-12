#!/usr/bin/env bash
# tests/cleanup.sh — Clean test outputs and caches.
#
# Usage:
#   bash tests/cleanup.sh              # Interactive: show sizes, ask before deleting
#   bash tests/cleanup.sh outputs      # Delete test outputs only
#   bash tests/cleanup.sh cache        # Delete HF/JAX cache only
#   bash tests/cleanup.sh docker       # Prune Docker images/containers
#   bash tests/cleanup.sh all          # Delete everything (no prompt)

source "$(dirname "$0")/lib/common.sh"

TARGET="${1:-}"

show_sizes() {
  info "Current disk usage:"
  echo ""
  for dir in "${HOST_OUTPUT_DIR}" "${HOST_HF_CACHE}" "${HOST_TPU_LOG_DIR}"; do
    if [[ -d "${dir}" ]]; then
      size="$(du -sh "${dir}" 2>/dev/null | cut -f1)"
      echo "  ${dir}: ${size}"
    else
      echo "  ${dir}: (not found)"
    fi
  done
  echo ""
  df -h / 2>/dev/null | head -2
  echo ""
}

clean_outputs() {
  if [[ -d "${HOST_OUTPUT_DIR}" ]]; then
    info "Removing test outputs: ${HOST_OUTPUT_DIR}"
    rm -rf "${HOST_OUTPUT_DIR}"
    ok "Test outputs removed."
  else
    info "No test outputs to clean."
  fi
}

clean_cache() {
  if [[ -d "${HOST_HF_CACHE}" ]]; then
    info "Removing HF/JAX cache: ${HOST_HF_CACHE}"
    rm -rf "${HOST_HF_CACHE}"
    ok "Cache removed."
  else
    info "No cache to clean."
  fi
  if [[ -d "${HOST_TPU_LOG_DIR}" ]]; then
    rm -rf "${HOST_TPU_LOG_DIR}"
    ok "TPU logs removed."
  fi
}

clean_docker() {
  local dcmd
  dcmd="$(docker_cmd)"
  info "Pruning Docker system..."
  ${dcmd} system prune -af
  ok "Docker pruned."
}

case "${TARGET}" in
  outputs)
    clean_outputs
    ;;
  cache)
    clean_cache
    ;;
  docker)
    clean_docker
    ;;
  all)
    clean_outputs
    clean_cache
    clean_docker
    ;;
  "")
    # Interactive mode.
    show_sizes
    echo "What to clean?"
    echo "  1) Test outputs only"
    echo "  2) HF/JAX cache only"
    echo "  3) Docker prune"
    echo "  4) All of the above"
    echo "  q) Quit"
    echo ""
    read -rp "Choice [1/2/3/4/q]: " choice
    case "${choice}" in
      1) clean_outputs ;;
      2) clean_cache ;;
      3) clean_docker ;;
      4) clean_outputs; clean_cache; clean_docker ;;
      q|Q) echo "No changes." ;;
      *) die "Invalid choice." ;;
    esac
    ;;
  *)
    die "Unknown target '${TARGET}'. Use: outputs | cache | docker | all"
    ;;
esac

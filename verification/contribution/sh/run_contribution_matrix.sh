#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

is_py311_or_newer() {
  local py_bin="$1"
  "${py_bin}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      echo "[ERROR] PYTHON_BIN='${PYTHON_BIN}' is not executable." >&2
      return 1
    fi
    if ! is_py311_or_newer "${PYTHON_BIN}"; then
      echo "[ERROR] PYTHON_BIN='${PYTHON_BIN}' must be Python 3.11+." >&2
      return 1
    fi
    echo "${PYTHON_BIN}"
    return 0
  fi

  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1 && is_py311_or_newer python3; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1 && is_py311_or_newer python; then
    echo "python"
    return 0
  fi
  return 1
}

if ! PYTHON_BIN_RESOLVED="$(resolve_python_bin)"; then
  echo "[ERROR] Contribution runner requires Python 3.11+ (host parser support for fused_moe kernels)." >&2
  echo "        Install python3.11 or run inside the supported Docker environment." >&2
  exit 1
fi

MANIFEST="${MANIFEST:-verification/contribution/manifests/default.json}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/verification/outputs/contribution}"
RUN_ID="${RUN_ID:-contrib_$(date +%Y%m%d_%H%M%S)}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUT_ROOT}/${RUN_ID}"

CMD=(
  "${PYTHON_BIN_RESOLVED}" "${ROOT_DIR}/verification/contribution/py/run_matrix.py"
  --manifest "${MANIFEST}"
  --out-root "${OUT_ROOT}"
  --run-id "${RUN_ID}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust-remote-code)
else
  CMD+=(--no-trust-remote-code)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "Contribution validation runner"
echo "root=${ROOT_DIR}"
echo "manifest=${MANIFEST}"
echo "out_root=${OUT_ROOT}"
echo "run_id=${RUN_ID}"
echo "python_bin=${PYTHON_BIN_RESOLVED}"
echo "trust_remote_code=${TRUST_REMOTE_CODE}"
echo "dry_run=${DRY_RUN}"
echo "cmd=${CMD[*]}"

"${CMD[@]}" | tee "${OUT_ROOT}/${RUN_ID}/runner.log"

echo "Done. Artifacts: ${OUT_ROOT}/${RUN_ID}"


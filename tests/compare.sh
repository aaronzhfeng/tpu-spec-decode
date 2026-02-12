#!/usr/bin/env bash
# tests/compare.sh — Compare benchmark results against reported DFlash baselines.
#
# Reads the comparator JSON from a benchmark run and checks speedup/tau
# against the reported DFlash numbers (GPU B200 baselines).
#
# Usage:
#   bash tests/compare.sh RUN_ID                  # Compare a specific run
#   bash tests/compare.sh /path/to/overall.json   # Compare from a JSON file
#   bash tests/compare.sh latest                   # Compare most recent run
#
# The comparison prints per-dataset speedup vs reported, and overall pass/fail.

source "$(dirname "$0")/lib/common.sh"

TARGET="${1:-latest}"

# ── Resolve the comparator JSON ──────────────────────────────────────────────
find_comparator_json() {
  local target="$1"

  # Direct file path.
  if [[ -f "${target}" ]]; then
    echo "${target}"
    return 0
  fi

  # Run ID under output directory.
  local by_id="${HOST_OUTPUT_DIR}/${target}/summaries/comparator_dflash.json"
  if [[ -f "${by_id}" ]]; then
    echo "${by_id}"
    return 0
  fi

  # "latest" — find most recent run.
  if [[ "${target}" == "latest" ]]; then
    local latest
    latest="$(ls -1td "${HOST_OUTPUT_DIR}"/*/summaries/comparator_dflash.json 2>/dev/null | head -1)"
    if [[ -n "${latest}" ]]; then
      echo "${latest}"
      return 0
    fi
  fi

  return 1
}

COMPARATOR_JSON="$(find_comparator_json "${TARGET}")" || \
  die "Could not find comparator JSON for '${TARGET}'.
  Provide a run ID, file path, or 'latest'.
  Output dir: ${HOST_OUTPUT_DIR}"

echo "==========================================="
echo "  DFlash Baseline Comparison"
echo "==========================================="
echo "  source: ${COMPARATOR_JSON}"
echo "==========================================="
echo ""

# ── Reported GPU baselines (DFlash paper, Qwen3-4B-DFlash-b16, temp=0) ──────
# Format: dataset speedup tau
declare -A BASELINE_SPEEDUP
BASELINE_SPEEDUP[GSM8K]=5.17
BASELINE_SPEEDUP[Math500]=6.19
BASELINE_SPEEDUP[AIME24]=6.00
BASELINE_SPEEDUP[AIME25]=5.79
BASELINE_SPEEDUP[Humaneval]=5.26
BASELINE_SPEEDUP[MBPP]=4.87
BASELINE_SPEEDUP[LiveCodeBench]=5.41
BASELINE_SPEEDUP[SWE-Bench]=2.97
BASELINE_SPEEDUP[MT-Bench]=2.87
BASELINE_SPEEDUP[Alpaca]=2.23

# Minimum retention threshold (what fraction of GPU speedup we consider passing).
THRESHOLD="${THRESHOLD:-0.70}"

# ── Parse and compare ────────────────────────────────────────────────────────
# We use Python for JSON parsing since jq may not be installed.
PY="$(resolve_python 2>/dev/null || echo python3)"

RESULTS="$(${PY} -c "
import json, sys
with open('${COMPARATOR_JSON}') as f:
    data = json.load(f)
datasets = data.get('datasets', {})
avg = data.get('average', {})
dflash = data.get('dflash_method', {})
for name, vals in sorted(datasets.items()):
    speedup = vals.get('speedup')
    print(f'DATASET {name} {speedup}')
print(f'AVERAGE {avg.get(\"speedup\", \"None\")}')
print(f'TAU {dflash.get(\"tau\", \"None\")}')
print(f'ACCEPTANCE {dflash.get(\"draft_acceptance_rate\", \"None\")}')
" 2>&1)"

PASS=0
TOTAL=0
echo "Dataset            | TPU Speedup | GPU Baseline | Retention | Status"
echo "-------------------|-------------|--------------|-----------|-------"

while IFS= read -r line; do
  type="$(echo "${line}" | awk '{print $1}')"
  if [[ "${type}" == "DATASET" ]]; then
    ds="$(echo "${line}" | awk '{print $2}')"
    tpu_speedup="$(echo "${line}" | awk '{print $3}')"
    gpu_baseline="${BASELINE_SPEEDUP[${ds}]:-}"

    if [[ -z "${gpu_baseline}" || "${tpu_speedup}" == "None" ]]; then
      printf "%-18s | %-11s | %-12s | %-9s | %s\n" "${ds}" "${tpu_speedup:-N/A}" "${gpu_baseline:-N/A}" "N/A" "SKIP"
      continue
    fi

    retention="$(${PY} -c "
s=${tpu_speedup}; b=${gpu_baseline}
print(f'{s/b:.2%}' if b > 0 else 'N/A')
")"
    pass_check="$(${PY} -c "
s=${tpu_speedup}; b=${gpu_baseline}; t=${THRESHOLD}
print('PASS' if s >= b * t else 'FAIL')
")"
    ((TOTAL++))
    if [[ "${pass_check}" == "PASS" ]]; then
      ((PASS++))
      printf "%-18s | %-11s | %-12s | %-9s | \033[0;32mPASS\033[0m\n" "${ds}" "${tpu_speedup}" "${gpu_baseline}" "${retention}"
    else
      printf "%-18s | %-11s | %-12s | %-9s | \033[0;31mFAIL\033[0m\n" "${ds}" "${tpu_speedup}" "${gpu_baseline}" "${retention}"
    fi
  fi
done <<< "${RESULTS}"

echo ""

# Print overall metrics.
AVG_SPEEDUP="$(echo "${RESULTS}" | grep "^AVERAGE" | awk '{print $2}')"
TAU="$(echo "${RESULTS}" | grep "^TAU" | awk '{print $2}')"
ACCEPTANCE="$(echo "${RESULTS}" | grep "^ACCEPTANCE" | awk '{print $2}')"

info "Average speedup: ${AVG_SPEEDUP}"
info "Tau (mean accepted length): ${TAU}"
info "Draft acceptance rate: ${ACCEPTANCE}"
info "Threshold: ${THRESHOLD} of GPU baseline"
echo ""

if [[ ${TOTAL} -eq 0 ]]; then
  warn "No datasets to compare."
elif [[ ${PASS} -eq ${TOTAL} ]]; then
  ok "All ${TOTAL} datasets meet the ${THRESHOLD} retention threshold."
else
  fail "${PASS}/${TOTAL} datasets passed. $(( TOTAL - PASS )) below threshold."
  exit 1
fi

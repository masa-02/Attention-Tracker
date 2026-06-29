#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
DATASET_NAME="${2:-deepset/prompt-injections}"
SEED="${3:-0}"
OUTPUT_FILE="${4:-analysis_runtime_configs_dataset.txt}"
RUN_PREFIX="${5:-}"
PHASE="${6:-phase1}"

require_int "${SEED}" "seed"

if [[ "${PHASE}" != "phase1" && "${PHASE}" != "phase2" ]]; then
    echo "Error: phase must be phase1 or phase2, got '${PHASE}'." >&2
    exit 2
fi

: > "${OUTPUT_FILE}"

PHASE_ARGS=()
if [[ "${PHASE}" == "phase2" ]]; then
    PHASE_ARGS+=(--attn-summary --phase2)
fi

CONFIG=""
while IFS= read -r CONFIG || [[ -n "${CONFIG}" ]]; do
    CONFIG="$(trim_manifest_line "${CONFIG}")"
    if [[ -z "${CONFIG}" ]]; then
        continue
    fi

    NAME="$(config_stem "${CONFIG}")"
    if [[ -n "${RUN_PREFIX}" ]]; then
        RUN_ID="${RUN_PREFIX}-${NAME}-seed${SEED}-${PHASE}"
    else
        RUN_ID="${NAME}-seed${SEED}-${PHASE}"
    fi

    echo "===== ${PHASE} evaluation: ${CONFIG} =====" >> "${OUTPUT_FILE}"
    if ! has_important_heads "${CONFIG}" >> "${OUTPUT_FILE}" 2>&1; then
        echo "Skip ${CONFIG}: important_heads is empty. Run head selection with --update_config first." >> "${OUTPUT_FILE}"
        continue
    fi

    if ! uv run python -u run_dataset.py \
        --config "${CONFIG}" \
        --dataset_name "${DATASET_NAME}" \
        --seed "${SEED}" \
        --audit-log \
        "${PHASE_ARGS[@]}" \
        --run-id "${RUN_ID}" >> "${OUTPUT_FILE}" 2>&1; then
        echo "Skip ${CONFIG}: ${PHASE} dataset evaluation failed. $(skip_hint)" >> "${OUTPUT_FILE}"
        cleanup_hf_cache_for_config "${CONFIG}" "failure" "${OUTPUT_FILE}"
        continue
    fi
    cleanup_hf_cache_for_config "${CONFIG}" "success" "${OUTPUT_FILE}"
done < "${LIST_FILE}"

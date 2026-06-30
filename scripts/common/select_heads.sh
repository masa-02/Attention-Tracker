#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
DATASET="${2:-llm}"
NUM_DATA="${3:-30}"
SELECT_K="${4:-4}"
OUTPUT_FILE="${5:-analysis_runtime_configs_heads.txt}"
RUN_PREFIX="${6:-}"
SUCCESS_FILE="${7:-}"
PHASE="${8:-phase1}"
REUSE_EXISTING="${9:-false}"

require_int "${NUM_DATA}" "num_data"
require_int "${SELECT_K}" "select_k"

if [[ "${PHASE}" != "phase1" && "${PHASE}" != "phase2" ]]; then
    echo "Error: phase must be phase1 or phase2, got '${PHASE}'." >&2
    exit 2
fi

: > "${OUTPUT_FILE}"
if [[ -n "${SUCCESS_FILE}" ]]; then
    : > "${SUCCESS_FILE}"
fi

PHASE_ARGS=()
if [[ "${PHASE}" == "phase2" ]]; then
    PHASE_ARGS+=(--phase2)
fi

CONFIG=""
while IFS= read -r CONFIG || [[ -n "${CONFIG}" ]]; do
    CONFIG="$(trim_manifest_line "${CONFIG}")"
    if [[ -z "${CONFIG}" ]]; then
        continue
    fi

    NAME="$(config_stem "${CONFIG}")"
    if [[ -n "${RUN_PREFIX}" ]]; then
        RUN_ID="${RUN_PREFIX}-${NAME}-head-selection"
    else
        RUN_ID="${NAME}-head-selection"
    fi

    echo "===== ${PHASE} head selection: ${CONFIG} =====" >> "${OUTPUT_FILE}"
    if [[ "${REUSE_EXISTING}" == "true" ]] && has_important_heads "${CONFIG}" >> "${OUTPUT_FILE}" 2>&1; then
        echo "Reuse existing important_heads: ${CONFIG}" >> "${OUTPUT_FILE}"
        if [[ -n "${SUCCESS_FILE}" ]]; then
            echo "${CONFIG}" >> "${SUCCESS_FILE}"
        fi
        continue
    fi

    if uv run python -u select_head.py \
        --config "${CONFIG}" \
        --dataset "${DATASET}" \
        --num_data "${NUM_DATA}" \
        --select_k "${SELECT_K}" \
        --update_config \
        --audit-log \
        "${PHASE_ARGS[@]}" \
        --run-id "${RUN_ID}" >> "${OUTPUT_FILE}" 2>&1; then
        :
    else
        EXIT_CODE="$?"
        echo "Skip ${CONFIG}: ${PHASE} head selection failed with exit code ${EXIT_CODE}. $(skip_hint)" >> "${OUTPUT_FILE}"
        cleanup_hf_cache_for_config "${CONFIG}" "failure" "${OUTPUT_FILE}"
        continue
    fi

    if has_important_heads "${CONFIG}" >> "${OUTPUT_FILE}" 2>&1; then
        if [[ -n "${SUCCESS_FILE}" ]]; then
            echo "${CONFIG}" >> "${SUCCESS_FILE}"
        fi
        cleanup_hf_cache_for_config "${CONFIG}" "success" "${OUTPUT_FILE}"
    else
        echo "Skip ${CONFIG}: head selection completed but important_heads is still empty." >> "${OUTPUT_FILE}"
        cleanup_hf_cache_for_config "${CONFIG}" "failure" "${OUTPUT_FILE}"
    fi
done < "${LIST_FILE}"

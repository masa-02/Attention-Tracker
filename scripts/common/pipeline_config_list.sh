#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

PHASE="${1:-phase1}"
LIST_FILE="${2:-configs/runtime/manifests/pilot.txt}"
RUN_PREFIX="${3:-${PHASE}-list}"
HEAD_DATASET="${4:-llm}"
HEAD_NUM_DATA="${5:-30}"
SELECT_K="${6:-4}"
EVAL_DATASET="${7:-deepset/prompt-injections}"
SEED="${8:-0}"
REUSE_EXISTING="${9:-false}"

require_int "${HEAD_NUM_DATA}" "head_num_data"
require_int "${SELECT_K}" "select_k"
require_int "${SEED}" "seed"

if [[ "${PHASE}" != "phase1" && "${PHASE}" != "phase2" ]]; then
    echo "Error: phase must be phase1 or phase2, got '${PHASE}'." >&2
    exit 2
fi

# Group pipelines are intended for constrained experiment machines. Stream each
# config end-to-end and delete that model's HF cache by default after success or
# failure. Override with ATTN_TRACKER_CLEAN_HF_CACHE=never|on_failure|on_success|always.
export ATTN_TRACKER_CLEAN_HF_CACHE="${ATTN_TRACKER_CLEAN_HF_CACHE:-always}"

SELECT_LOG="analysis_${RUN_PREFIX}_${PHASE}_heads.txt"
EVAL_LOG="analysis_${RUN_PREFIX}_${PHASE}_dataset.txt"
SUCCESS_LIST="analysis_${RUN_PREFIX}_${PHASE}_selected_configs.txt"

: > "${SELECT_LOG}"
: > "${EVAL_LOG}"
: > "${SUCCESS_LIST}"

CURRENT_CONFIG=""
CURRENT_LOG="${SELECT_LOG}"
cleanup_current_config_on_exit() {
    if [[ -n "${CURRENT_CONFIG}" ]]; then
        cleanup_hf_cache_for_config "${CURRENT_CONFIG}" "interrupted" "${CURRENT_LOG}"
    fi
}
trap cleanup_current_config_on_exit EXIT INT TERM

SELECT_PHASE_ARGS=()
EVAL_PHASE_ARGS=()
if [[ "${PHASE}" == "phase2" ]]; then
    SELECT_PHASE_ARGS+=(--phase2)
    EVAL_PHASE_ARGS+=(--attn-summary --phase2)
fi

CONFIG=""
while IFS= read -r CONFIG || [[ -n "${CONFIG}" ]]; do
    CONFIG="$(trim_manifest_line "${CONFIG}")"
    if [[ -z "${CONFIG}" ]]; then
        continue
    fi

    NAME="$(config_stem "${CONFIG}")"
    SELECT_RUN_ID="${RUN_PREFIX}-${NAME}-head-selection"
    EVAL_RUN_ID="${RUN_PREFIX}-${NAME}-seed${SEED}-${PHASE}"
    SELECT_OK="false"
    CURRENT_CONFIG="${CONFIG}"
    CURRENT_LOG="${SELECT_LOG}"

    echo "===== ${PHASE} head selection: ${CONFIG} =====" >> "${SELECT_LOG}"
    if [[ "${REUSE_EXISTING}" == "true" ]] && has_important_heads "${CONFIG}" >> "${SELECT_LOG}" 2>&1; then
        echo "Reuse existing important_heads: ${CONFIG}" >> "${SELECT_LOG}"
        SELECT_OK="true"
    elif uv run python -u select_head.py \
        --config "${CONFIG}" \
        --dataset "${HEAD_DATASET}" \
        --num_data "${HEAD_NUM_DATA}" \
        --select_k "${SELECT_K}" \
        --update_config \
        --audit-log \
        "${SELECT_PHASE_ARGS[@]}" \
        --run-id "${SELECT_RUN_ID}" >> "${SELECT_LOG}" 2>&1; then
        if has_important_heads "${CONFIG}" >> "${SELECT_LOG}" 2>&1; then
            SELECT_OK="true"
        else
            echo "Skip ${CONFIG}: head selection completed but important_heads is still empty." >> "${SELECT_LOG}"
        fi
    else
        echo "Skip ${CONFIG}: ${PHASE} head selection failed. $(skip_hint)" >> "${SELECT_LOG}"
    fi

    if [[ "${SELECT_OK}" != "true" ]]; then
        cleanup_hf_cache_for_config "${CONFIG}" "failure" "${SELECT_LOG}"
        CURRENT_CONFIG=""
        continue
    fi

    echo "${CONFIG}" >> "${SUCCESS_LIST}"

    echo "===== ${PHASE} evaluation: ${CONFIG} =====" >> "${EVAL_LOG}"
    CURRENT_LOG="${EVAL_LOG}"
    if uv run python -u run_dataset.py \
        --config "${CONFIG}" \
        --dataset_name "${EVAL_DATASET}" \
        --seed "${SEED}" \
        --audit-log \
        "${EVAL_PHASE_ARGS[@]}" \
        --run-id "${EVAL_RUN_ID}" >> "${EVAL_LOG}" 2>&1; then
        cleanup_hf_cache_for_config "${CONFIG}" "success" "${EVAL_LOG}"
    else
        echo "Skip ${CONFIG}: ${PHASE} dataset evaluation failed. $(skip_hint)" >> "${EVAL_LOG}"
        cleanup_hf_cache_for_config "${CONFIG}" "failure" "${EVAL_LOG}"
    fi
    CURRENT_CONFIG=""
done < "${LIST_FILE}"

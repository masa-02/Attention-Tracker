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

SELECT_LOG="analysis_${RUN_PREFIX}_${PHASE}_heads.txt"
EVAL_LOG="analysis_${RUN_PREFIX}_${PHASE}_dataset.txt"
SUCCESS_LIST="analysis_${RUN_PREFIX}_${PHASE}_selected_configs.txt"

"${SCRIPT_DIR}/select_heads.sh" \
    "${LIST_FILE}" \
    "${HEAD_DATASET}" \
    "${HEAD_NUM_DATA}" \
    "${SELECT_K}" \
    "${SELECT_LOG}" \
    "${RUN_PREFIX}" \
    "${SUCCESS_LIST}" \
    "${PHASE}" \
    "${REUSE_EXISTING}"

"${SCRIPT_DIR}/run_dataset.sh" \
    "${SUCCESS_LIST}" \
    "${EVAL_DATASET}" \
    "${SEED}" \
    "${EVAL_LOG}" \
    "${RUN_PREFIX}" \
    "${PHASE}"

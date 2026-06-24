#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

LIST_FILE="${LIST_FILE:-scripts/tmp/corllm_unexecuted_configs.txt}"
RUN_PREFIX="${1:-corllm-unexecuted}"
HEAD_DATASET="${2:-llm}"
HEAD_NUM_DATA="${3:-30}"
SELECT_K="${4:-4}"
EVAL_DATASET="${5:-deepset/prompt-injections}"
SEED="${6:-0}"

./scripts/select_config_list_heads.sh \
    "${LIST_FILE}" \
    "${HEAD_DATASET}" \
    "${HEAD_NUM_DATA}" \
    "${SELECT_K}" \
    "analysis_${RUN_PREFIX}_heads.txt" \
    "${RUN_PREFIX}" \
    "analysis_${RUN_PREFIX}_selected_configs.txt"

./scripts/run_config_list_dataset.sh \
    "analysis_${RUN_PREFIX}_selected_configs.txt" \
    "${EVAL_DATASET}" \
    "${SEED}" \
    "analysis_${RUN_PREFIX}_dataset.txt" \
    "${RUN_PREFIX}"

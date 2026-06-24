set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
RUN_PREFIX="${2:-gate1-list}"
HEAD_DATASET="${3:-llm}"
HEAD_NUM_DATA="${4:-30}"
SELECT_K="${5:-4}"
EVAL_DATASET="${6:-deepset/prompt-injections}"
SEED="${7:-0}"

SELECT_LOG="analysis_${RUN_PREFIX}_heads.txt"
EVAL_LOG="analysis_${RUN_PREFIX}_dataset.txt"
SUCCESS_LIST="analysis_${RUN_PREFIX}_selected_configs.txt"

./scripts/select_config_list_heads.sh \
    "${LIST_FILE}" \
    "${HEAD_DATASET}" \
    "${HEAD_NUM_DATA}" \
    "${SELECT_K}" \
    "${SELECT_LOG}" \
    "${RUN_PREFIX}" \
    "${SUCCESS_LIST}"

./scripts/run_config_list_dataset.sh \
    "${SUCCESS_LIST}" \
    "${EVAL_DATASET}" \
    "${SEED}" \
    "${EVAL_LOG}" \
    "${RUN_PREFIX}"

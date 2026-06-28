set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
RUN_PREFIX="${2:-phase2-list}"
HEAD_DATASET="${3:-llm}"
HEAD_NUM_DATA="${4:-30}"
SELECT_K="${5:-4}"
EVAL_DATASET="${6:-deepset/prompt-injections}"
SEED="${7:-0}"

if ! [[ "${HEAD_NUM_DATA}" =~ ^[0-9]+$ ]]; then
    echo "Error: head_num_data must be an integer, got '${HEAD_NUM_DATA}'." >&2
    exit 2
fi

if ! [[ "${SELECT_K}" =~ ^[0-9]+$ ]]; then
    echo "Error: select_k must be an integer, got '${SELECT_K}'." >&2
    exit 2
fi

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be an integer, got '${SEED}'." >&2
    exit 2
fi

SELECT_LOG="analysis_${RUN_PREFIX}_phase2_heads.txt"
EVAL_LOG="analysis_${RUN_PREFIX}_phase2_dataset.txt"
SUCCESS_LIST="analysis_${RUN_PREFIX}_phase2_selected_configs.txt"

: > "${SELECT_LOG}"
: > "${EVAL_LOG}"
: > "${SUCCESS_LIST}"

CONFIG=""
while IFS= read -r CONFIG || [[ -n "${CONFIG}" ]]; do
    CONFIG="${CONFIG%%#*}"
    CONFIG="${CONFIG//$'\r'/}"
    CONFIG="${CONFIG#"${CONFIG%%[![:space:]]*}"}"
    CONFIG="${CONFIG%"${CONFIG##*[![:space:]]}"}"
    if [[ -z "${CONFIG}" ]]; then
        continue
    fi

    NAME="$(basename "${CONFIG}")"
    NAME="${NAME%.*}"
    RUN_ID="${RUN_PREFIX}-${NAME}-head-selection"

    echo "===== Phase2 head selection: ${CONFIG} =====" >> "${SELECT_LOG}"
    if ! uv run python -u select_head.py \
        --config "${CONFIG}" \
        --dataset "${HEAD_DATASET}" \
        --num_data "${HEAD_NUM_DATA}" \
        --select_k "${SELECT_K}" \
        --update_config \
        --audit-log \
        --phase2 \
        --run-id "${RUN_ID}" >> "${SELECT_LOG}" 2>&1; then
        echo "Skip ${CONFIG}: Phase2 head selection failed. Check GPU memory, HF access, model id availability, span mapping, or attention support." >> "${SELECT_LOG}"
        continue
    fi

    echo "${CONFIG}" >> "${SUCCESS_LIST}"
done < "${LIST_FILE}"

while IFS= read -r CONFIG || [[ -n "${CONFIG}" ]]; do
    CONFIG="${CONFIG%%#*}"
    CONFIG="${CONFIG//$'\r'/}"
    CONFIG="${CONFIG#"${CONFIG%%[![:space:]]*}"}"
    CONFIG="${CONFIG%"${CONFIG##*[![:space:]]}"}"
    if [[ -z "${CONFIG}" ]]; then
        continue
    fi

    NAME="$(basename "${CONFIG}")"
    NAME="${NAME%.*}"
    RUN_ID="${RUN_PREFIX}-${NAME}-seed${SEED}-phase2"

    echo "===== Phase2 evaluation: ${CONFIG} =====" >> "${EVAL_LOG}"
    if ! uv run python -u run_dataset.py \
        --config "${CONFIG}" \
        --dataset_name "${EVAL_DATASET}" \
        --seed "${SEED}" \
        --audit-log \
        --attn-summary \
        --phase2 \
        --run-id "${RUN_ID}" >> "${EVAL_LOG}" 2>&1; then
        echo "Skip ${CONFIG}: Phase2 evaluation failed. Check GPU memory, HF access, model id availability, strict span mapping, or dataset availability." >> "${EVAL_LOG}"
        continue
    fi
done < "${SUCCESS_LIST}"


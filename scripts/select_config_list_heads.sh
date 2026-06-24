set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
DATASET="${2:-llm}"
NUM_DATA="${3:-30}"
SELECT_K="${4:-4}"
OUTPUT_FILE="${5:-analysis_runtime_configs.txt}"
RUN_PREFIX="${6:-}"
SUCCESS_FILE="${7:-}"

: > "${OUTPUT_FILE}"
if [[ -n "${SUCCESS_FILE}" ]]; then
    : > "${SUCCESS_FILE}"
fi

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
    if [[ -n "${RUN_PREFIX}" ]]; then
        RUN_ID="${RUN_PREFIX}-${NAME}-head-selection"
    else
        RUN_ID="${NAME}-head-selection"
    fi

    echo "===== ${CONFIG} =====" >> "${OUTPUT_FILE}"
    if ! uv run python -u select_head.py \
        --config "${CONFIG}" \
        --dataset "${DATASET}" \
        --num_data "${NUM_DATA}" \
        --select_k "${SELECT_K}" \
        --update_config \
        --audit-log \
        --run-id "${RUN_ID}" >> "${OUTPUT_FILE}" 2>&1; then
        echo "Skip ${CONFIG}: head selection failed. Check GPU memory, HF access, model id availability, or model-specific attention support." >> "${OUTPUT_FILE}"
        continue
    fi

    if [[ -n "${SUCCESS_FILE}" ]]; then
        echo "${CONFIG}" >> "${SUCCESS_FILE}"
    fi
done < "${LIST_FILE}"

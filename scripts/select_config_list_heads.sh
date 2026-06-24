set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
DATASET="${2:-llm}"
NUM_DATA="${3:-30}"
SELECT_K="${4:-4}"
OUTPUT_FILE="${5:-analysis_runtime_configs.txt}"
RUN_PREFIX="${6:-}"
SUCCESS_FILE="${7:-}"

if ! [[ "${NUM_DATA}" =~ ^[0-9]+$ ]]; then
    echo "Error: num_data must be an integer, got '${NUM_DATA}'." >&2
    echo "Usage: $0 [list_file] [head_dataset] [num_data] [select_k] [output_file] [run_prefix] [success_file]" >&2
    exit 2
fi

if ! [[ "${SELECT_K}" =~ ^[0-9]+$ ]]; then
    echo "Error: select_k must be an integer, got '${SELECT_K}'." >&2
    echo "Usage: $0 [list_file] [head_dataset] [num_data] [select_k] [output_file] [run_prefix] [success_file]" >&2
    echo "Example: $0 configs/runtime/manifests/core.txt llm 30 4 analysis_core_heads.txt core-gate1 analysis_core_selected.txt" >&2
    exit 2
fi

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

set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
RUN_PREFIX="${2:-gate1-list}"
HEAD_DATASET="${3:-llm}"
HEAD_NUM_DATA="${4:-30}"
SELECT_K="${5:-4}"
EVAL_DATASET="${6:-deepset/prompt-injections}"
SEED="${7:-0}"

usage() {
    echo "Usage: $0 [list_file] [run_prefix] [head_dataset] [head_num_data] [select_k] [eval_dataset] [seed]" >&2
    echo "Example: $0 configs/runtime/manifests/core.txt core-gate1 llm 30 4 deepset/prompt-injections 0" >&2
}

if ! [[ "${HEAD_NUM_DATA}" =~ ^[0-9]+$ ]]; then
    echo "Error: head_num_data must be an integer, got '${HEAD_NUM_DATA}'." >&2
    usage
    exit 2
fi

if ! [[ "${SELECT_K}" =~ ^[0-9]+$ ]]; then
    echo "Error: select_k must be an integer, got '${SELECT_K}'." >&2
    usage
    exit 2
fi

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be an integer, got '${SEED}'." >&2
    usage
    exit 2
fi

SELECT_LOG="analysis_${RUN_PREFIX}_heads.txt"
EVAL_LOG="analysis_${RUN_PREFIX}_dataset.txt"
SUCCESS_LIST="analysis_${RUN_PREFIX}_selected_configs.txt"

: > "${SELECT_LOG}"
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

    echo "===== ${CONFIG} =====" >> "${SELECT_LOG}"
    if uv run python -c "import sys; from utils import load_runtime_config; cfg, _ = load_runtime_config(config=sys.argv[1]); raise SystemExit(0 if cfg.get('params', {}).get('important_heads') else 1)" "${CONFIG}" >> "${SELECT_LOG}" 2>&1; then
        echo "Reuse existing important_heads: ${CONFIG}" >> "${SELECT_LOG}"
        echo "${CONFIG}" >> "${SUCCESS_LIST}"
        continue
    fi

    echo "Run head selection: ${CONFIG}" >> "${SELECT_LOG}"
    if ! uv run python -u select_head.py \
        --config "${CONFIG}" \
        --dataset "${HEAD_DATASET}" \
        --num_data "${HEAD_NUM_DATA}" \
        --select_k "${SELECT_K}" \
        --update_config \
        --audit-log \
        --run-id "${RUN_ID}" >> "${SELECT_LOG}" 2>&1; then
        echo "Skip ${CONFIG}: head selection failed. Check GPU memory, HF access, model id availability, or model-specific attention support." >> "${SELECT_LOG}"
        continue
    fi

    if uv run python -c "import sys; from utils import load_runtime_config; cfg, _ = load_runtime_config(config=sys.argv[1]); raise SystemExit(0 if cfg.get('params', {}).get('important_heads') else 1)" "${CONFIG}" >> "${SELECT_LOG}" 2>&1; then
        echo "${CONFIG}" >> "${SUCCESS_LIST}"
    else
        echo "Skip ${CONFIG}: head selection completed but important_heads is still empty." >> "${SELECT_LOG}"
    fi
done < "${LIST_FILE}"

./scripts/run_config_list_dataset.sh \
    "${SUCCESS_LIST}" \
    "${EVAL_DATASET}" \
    "${SEED}" \
    "${EVAL_LOG}" \
    "${RUN_PREFIX}"

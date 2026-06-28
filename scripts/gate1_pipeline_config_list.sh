set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
RUN_PREFIX="${2:-phase1-list}"
HEAD_DATASET="${3:-llm}"
HEAD_NUM_DATA="${4:-30}"
SELECT_K="${5:-4}"
EVAL_DATASET="${6:-deepset/prompt-injections}"
SEED="${7:-0}"

if ! [[ "${HEAD_NUM_DATA}" =~ ^[0-9]+$ ]]; then
    echo "Error: head_num_data must be an integer, got '${HEAD_NUM_DATA}'." >&2
    echo "Usage: $0 [list_file] [run_prefix] [head_dataset] [head_num_data] [select_k] [eval_dataset] [seed]" >&2
    exit 2
fi

if ! [[ "${SELECT_K}" =~ ^[0-9]+$ ]]; then
    echo "Error: select_k must be an integer, got '${SELECT_K}'." >&2
    echo "Usage: $0 [list_file] [run_prefix] [head_dataset] [head_num_data] [select_k] [eval_dataset] [seed]" >&2
    echo "Example: $0 configs/runtime/manifests/core.txt core-phase1 llm 30 4 deepset/prompt-injections 0" >&2
    exit 2
fi

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be an integer, got '${SEED}'." >&2
    echo "Usage: $0 [list_file] [run_prefix] [head_dataset] [head_num_data] [select_k] [eval_dataset] [seed]" >&2
    exit 2
fi

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


set -euo pipefail

CONFIG="${1:-configs/runtime/qwen2.5-7b-instruct.yml}"
RUN_PREFIX="${2:-qwen-phase1-smoke}"
NUM_DATA="${3:-10}"
SELECT_K="${4:-4}"
SEED="${5:-0}"

uv run python select_head.py \
    --config "${CONFIG}" \
    --dataset "llm" \
    --num_data "${NUM_DATA}" \
    --select_k "${SELECT_K}" \
    --update_config \
    --audit-log \
    --run-id "${RUN_PREFIX}-head-selection"

uv run python run_dataset.py \
    --config "${CONFIG}" \
    --dataset_name "deepset/prompt-injections" \
    --seed "${SEED}" \
    --audit-log \
    --run-id "${RUN_PREFIX}-dataset"


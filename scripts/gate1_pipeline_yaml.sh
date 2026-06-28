set -euo pipefail

CONFIG="${1:-configs/runtime/qwen2.5-7b-instruct.yml}"
RUN_PREFIX="${2:-phase1}"
HEAD_DATASET="${3:-llm}"
HEAD_NUM_DATA="${4:-30}"
SELECT_K="${5:-4}"
EVAL_DATASET="${6:-deepset/prompt-injections}"
SEED="${7:-0}"
PLOT_DATASET="${8:-deepset}"
PLOT_NUM_DATA="${9:-30}"

uv run python select_head.py \
    --config "${CONFIG}" \
    --dataset "${HEAD_DATASET}" \
    --num_data "${HEAD_NUM_DATA}" \
    --select_k "${SELECT_K}" \
    --update_config \
    --audit-log \
    --run-id "${RUN_PREFIX}-head-selection"

uv run python run_dataset.py \
    --config "${CONFIG}" \
    --dataset_name "${EVAL_DATASET}" \
    --seed "${SEED}" \
    --audit-log \
    --run-id "${RUN_PREFIX}-dataset"

uv run python render/plot_attention_maps.py \
    --config "${CONFIG}" \
    --dataset "${PLOT_DATASET}" \
    --num_data "${PLOT_NUM_DATA}" \
    --output_dir "render/outputs/${RUN_PREFIX}"


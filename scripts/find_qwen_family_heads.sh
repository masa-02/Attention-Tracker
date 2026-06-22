MODELS=(
    "qwen2_5_1_5b-attn"
    "qwen2_5_3b-attn"
    "qwen2_5_7b-attn"
    "qwen2_5_14b-attn"
    "qwen2_5_32b-attn"
    "qwen3_4b-attn"
    "qwen3_8b-attn"
    "qwen3_14b-attn"
    "qwen3_32b-attn"
    # "qwen3_5_4b-attn"
    # "qwen3_5_9b-attn"
    # "qwen3_5_27b-attn"
)

DATASET="${1:-llm}"
NUM_DATA="${2:-30}"
OUTPUT_FILE="${3:-analysis_qwen_family.txt}"
SELECT_K="${4:-4}"

for MODEL in "${MODELS[@]}"; do
    echo "===== ${MODEL} =====" >> "${OUTPUT_FILE}"
    if ! uv run python3 -u select_head.py \
        --model_name "${MODEL}" \
        --num_data "${NUM_DATA}" \
        --dataset "${DATASET}" \
        --select_k "${SELECT_K}" \
        --update_config >> "${OUTPUT_FILE}" 2>> "${OUTPUT_FILE}"; then
        echo "Skip ${MODEL}: failed during head selection. This may be caused by insufficient GPU memory, gated model access, or an unavailable model id." >> "${OUTPUT_FILE}"
        continue
    fi
done

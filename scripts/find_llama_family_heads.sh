MODELS=(
    "llama3_8b-attn"
    # "llama3_70b-attn"
    "llama3_1_8b-attn"
    "llama3_2_1b-attn"
    "llama3_2_3b-attn"
)

DATASET="${1:-llm}"
NUM_DATA="${2:-30}"
OUTPUT_FILE="${3:-analysis_llama_family.txt}"
SELECT_K="${4:-4}"

for MODEL in "${MODELS[@]}"; do
    echo "===== ${MODEL} =====" >> "${OUTPUT_FILE}"
    if ! uv run python3 -u select_head.py \
        --model_name "${MODEL}" \
        --num_data "${NUM_DATA}" \
        --dataset "${DATASET}" \
        --select_k "${SELECT_K}" \
        --update_config >> "${OUTPUT_FILE}" 2>> "${OUTPUT_FILE}"; then
        echo "Skip ${MODEL}: failed during head selection. This may be caused by insufficient GPU memory or gated model access." >> "${OUTPUT_FILE}"
        continue
    fi
done

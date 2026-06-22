MODELS=(
    "qwen2_5_1_5b-attn"
    "qwen2_5_3b-attn"
    "qwen2_5_7b-attn"
    "qwen2_5_14b-attn"
    "qwen2_5_32b-attn"
    "qwen3_32b-attn"
    "qwen3_14b-attn"
    "qwen3_8b-attn"
    "qwen3_4b-attn"
    # "qwen3_5_4b-attn"
    # "qwen3_5_9b-attn"
    # "qwen3_5_27b-attn"
)

DATASET="${1:-deepset}"
NUM_DATA="${2:-100}"
OUTPUT_DIR="${3:-render/outputs/qwen_family}"

uv run python render/plot_attention_maps.py \
    --model_name "${MODELS[@]}" \
    --dataset "${DATASET}" \
    --num_data "${NUM_DATA}" \
    --output_dir "${OUTPUT_DIR}"

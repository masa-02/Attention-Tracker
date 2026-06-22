MODELS=(
    "llama3_8b-attn"
    # "llama3_70b-attn"
    "llama3_1_8b-attn"
    "llama3_2_1b-attn"
    "llama3_2_3b-attn"
)

DATASET="${1:-deepset}"
NUM_DATA="${2:-100}"
OUTPUT_DIR="${3:-render/outputs/llama_family}"

uv run python render/plot_attention_maps.py \
    --model_name "${MODELS[@]}" \
    --dataset "${DATASET}" \
    --num_data "${NUM_DATA}" \
    --output_dir "${OUTPUT_DIR}"

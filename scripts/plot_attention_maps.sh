MODEL_NAME="${1:-all}"
DATASET="${2:-deepset}"
NUM_DATA="${3:-100}"
OUTPUT_DIR="${4:-render/outputs}"

uv run python render/plot_attention_maps.py \
    --model_name ${MODEL_NAME} \
    --dataset "${DATASET}" \
    --num_data "${NUM_DATA}" \
    --output_dir "${OUTPUT_DIR}"

set -euo pipefail

MODELS=( "granite3_8b-attn" "qwen2-attn" "llama3_8b-attn" "phi3-attn" "mistral_7b-attn" "gemma2_9b-attn" )
DATASET="${1:-llm}"
NUM_DATA="${2:-30}"
OUTPUT_FILE="${3:-analysis.txt}"
SELECT_K="${4:-4}"

: > "${OUTPUT_FILE}"
for MODEL in "${MODELS[@]}"; do
    echo "===== ${MODEL} =====" >> "${OUTPUT_FILE}"
    uv run python3 -u select_head.py \
        --model_name "${MODEL}" \
        --num_data "${NUM_DATA}" \
        --dataset "${DATASET}" \
        --select_k "${SELECT_K}" \
        --update_config \
        --audit-log \
        --run-id "${MODEL}-head-selection" >> "${OUTPUT_FILE}" 2>> "${OUTPUT_FILE}"
done

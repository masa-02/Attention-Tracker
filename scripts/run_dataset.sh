set -euo pipefail

MODELS=( "granite3_8b-attn" "llama3_8b-attn" "mistral_7b-attn" "gemma2_9b-attn"  "phi3-attn" "qwen2-attn" )
DATASET_NAME="${1:-deepset/prompt-injections}"
SEED="${2:-0}"

for MODEL in "${MODELS[@]}"; do
    uv run python run_dataset.py \
        --model_name "${MODEL}" \
        --dataset_name "${DATASET_NAME}" \
        --seed "${SEED}" \
        --audit-log \
        --run-id "${MODEL}-${SEED}-phase1"
done


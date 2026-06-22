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

DATASET_NAME="${1:-deepset/prompt-injections}"
SEED="${2:-0}"

for MODEL in "${MODELS[@]}"; do
    CONFIG_PATH="configs/model_configs/${MODEL}_config.json"
    if grep -q '"important_heads": \[\]' "${CONFIG_PATH}"; then
        echo "Skip ${MODEL}: important_heads is empty. Run find_qwen_family_heads.sh and update ${CONFIG_PATH} first."
        continue
    fi

    if ! uv run python run_dataset.py \
        --model_name "${MODEL}" \
        --dataset_name "${DATASET_NAME}" \
        --seed "${SEED}"; then
        echo "Skip ${MODEL}: failed during dataset evaluation. This may be caused by insufficient GPU memory, gated model access, or an unavailable model id."
        continue
    fi
done

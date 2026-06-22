MODELS=(
    "llama3_8b-attn"
    # "llama3_70b-attn"
    "llama3_1_8b-attn"
    "llama3_2_1b-attn"
    "llama3_2_3b-attn"
)

DATASET_NAME="${1:-deepset/prompt-injections}"
SEED="${2:-0}"

for MODEL in "${MODELS[@]}"; do
    CONFIG_PATH="configs/model_configs/${MODEL}_config.json"
    if grep -q '"important_heads": \[\]' "${CONFIG_PATH}"; then
        echo "Skip ${MODEL}: important_heads is empty. Run find_llama_family_heads.sh and update ${CONFIG_PATH} first."
        continue
    fi

    if ! uv run python run_dataset.py \
        --model_name "${MODEL}" \
        --dataset_name "${DATASET_NAME}" \
        --seed "${SEED}"; then
        echo "Skip ${MODEL}: failed during dataset evaluation. This may be caused by insufficient GPU memory or gated model access."
        continue
    fi
done

set -euo pipefail

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
DATASET_NAME="${2:-deepset/prompt-injections}"
SEED="${3:-0}"
OUTPUT_FILE="${4:-run_runtime_configs.txt}"
RUN_PREFIX="${5:-}"

: > "${OUTPUT_FILE}"

CONFIG=""
while IFS= read -r CONFIG || [[ -n "${CONFIG}" ]]; do
    CONFIG="${CONFIG%%#*}"
    CONFIG="${CONFIG//$'\r'/}"
    CONFIG="${CONFIG#"${CONFIG%%[![:space:]]*}"}"
    CONFIG="${CONFIG%"${CONFIG##*[![:space:]]}"}"
    if [[ -z "${CONFIG}" ]]; then
        continue
    fi

    NAME="$(basename "${CONFIG}")"
    NAME="${NAME%.*}"
    if [[ -n "${RUN_PREFIX}" ]]; then
        RUN_ID="${RUN_PREFIX}-${NAME}-seed${SEED}-phase1"
    else
        RUN_ID="${NAME}-seed${SEED}-phase1"
    fi

    echo "===== ${CONFIG} =====" >> "${OUTPUT_FILE}"
    if ! uv run python -c "import sys; from utils import load_runtime_config; cfg, _ = load_runtime_config(config=sys.argv[1]); raise SystemExit(0 if cfg.get('params', {}).get('important_heads') else 1)" "${CONFIG}" >> "${OUTPUT_FILE}" 2>&1; then
        echo "Skip ${CONFIG}: important_heads is empty. Run head selection with --update_config first." >> "${OUTPUT_FILE}"
        continue
    fi

    if ! uv run python -u run_dataset.py \
        --config "${CONFIG}" \
        --dataset_name "${DATASET_NAME}" \
        --seed "${SEED}" \
        --audit-log \
        --run-id "${RUN_ID}" >> "${OUTPUT_FILE}" 2>&1; then
        echo "Skip ${CONFIG}: dataset evaluation failed. Run head selection first or check GPU memory, HF access, and model id availability." >> "${OUTPUT_FILE}"
        continue
    fi
done < "${LIST_FILE}"


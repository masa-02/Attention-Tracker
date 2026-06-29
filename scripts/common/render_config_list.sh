#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

LIST_FILE="${1:-configs/runtime/manifests/pilot.txt}"
DATASET="${2:-deepset}"
NUM_DATA="${3:-100}"
OUTPUT_DIR="${4:-render/outputs}"
OUTPUT_FILE="${5:-analysis_render_attention_maps.txt}"

require_int "${NUM_DATA}" "num_data"

: > "${OUTPUT_FILE}"

CONFIG=""
while IFS= read -r CONFIG || [[ -n "${CONFIG}" ]]; do
    CONFIG="$(trim_manifest_line "${CONFIG}")"
    if [[ -z "${CONFIG}" ]]; then
        continue
    fi

    NAME="$(config_stem "${CONFIG}")"
    echo "===== render attention maps: ${CONFIG} =====" >> "${OUTPUT_FILE}"
    if ! uv run python -u render/plot_attention_maps.py \
        --config "${CONFIG}" \
        --dataset "${DATASET}" \
        --num_data "${NUM_DATA}" \
        --output_dir "${OUTPUT_DIR}/${NAME}" >> "${OUTPUT_FILE}" 2>&1; then
        echo "Skip ${CONFIG}: attention-map rendering failed. $(skip_hint)" >> "${OUTPUT_FILE}"
        continue
    fi
done < "${LIST_FILE}"

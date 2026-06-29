#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-deepset}"
NUM_DATA="${2:-100}"
OUTPUT_DIR="${3:-render/outputs/llama}"
OUTPUT_FILE="${4:-analysis_llama_render.txt}"

"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common/render_config_list.sh" \
    configs/runtime/manifests/llama.txt \
    "${DATASET}" \
    "${NUM_DATA}" \
    "${OUTPUT_DIR}" \
    "${OUTPUT_FILE}"

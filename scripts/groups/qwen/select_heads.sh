#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-llm}"
NUM_DATA="${2:-30}"
OUTPUT_FILE="${3:-analysis_qwen_phase1_heads.txt}"
SELECT_K="${4:-4}"
RUN_PREFIX="${5:-qwen-phase1}"
PHASE="${6:-phase1}"
REUSE_EXISTING="${7:-false}"

"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common/select_heads.sh" \
    configs/runtime/manifests/qwen.txt \
    "${DATASET}" \
    "${NUM_DATA}" \
    "${SELECT_K}" \
    "${OUTPUT_FILE}" \
    "${RUN_PREFIX}" \
    "" \
    "${PHASE}" \
    "${REUSE_EXISTING}"

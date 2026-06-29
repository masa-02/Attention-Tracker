#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${1:-deepset/prompt-injections}"
SEED="${2:-0}"
OUTPUT_FILE="${3:-analysis_llama_phase1_dataset.txt}"
RUN_PREFIX="${4:-llama-phase1}"
PHASE="${5:-phase1}"

"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common/run_dataset.sh" \
    configs/runtime/manifests/llama.txt \
    "${DATASET_NAME}" \
    "${SEED}" \
    "${OUTPUT_FILE}" \
    "${RUN_PREFIX}" \
    "${PHASE}"

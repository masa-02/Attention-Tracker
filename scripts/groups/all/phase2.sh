#!/usr/bin/env bash
set -euo pipefail

RUN_PREFIX="${1:-all-phase2}"
if [[ "$#" -gt 0 ]]; then
    shift
fi

"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common/pipeline_config_list.sh" \
    phase2 \
    configs/runtime/manifests/all.txt \
    "${RUN_PREFIX}" \
    "$@"

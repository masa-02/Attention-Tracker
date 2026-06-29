#!/usr/bin/env bash
set -euo pipefail

RUN_PREFIX="${1:-moonlight-phase1}"
if [[ "$#" -gt 0 ]]; then
    shift
fi

"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/common/pipeline_config_list.sh" \
    phase1 \
    configs/runtime/manifests/moonlight.txt \
    "${RUN_PREFIX}" \
    "$@"

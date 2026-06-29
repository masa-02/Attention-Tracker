#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

trim_manifest_line() {
    local line="$1"
    line="${line%%#*}"
    line="${line//$'\r'/}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    printf '%s' "${line}"
}

config_stem() {
    local config="$1"
    local name
    name="$(basename "${config}")"
    printf '%s' "${name%.*}"
}

require_int() {
    local value="$1"
    local name="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "Error: ${name} must be an integer, got '${value}'." >&2
        exit 2
    fi
}

has_important_heads() {
    local config="$1"
    uv run python -c "import sys; from utils import load_runtime_config; cfg, _ = load_runtime_config(config=sys.argv[1]); raise SystemExit(0 if cfg.get('params', {}).get('important_heads') else 1)" "${config}"
}

skip_hint() {
    printf '%s' "Check GPU memory, HF access, model id availability, span mapping, dataset availability, and model-specific attention support."
}

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
    printf '%s' "Check GPU memory, HF access, HF download/cache state, model id availability, span mapping, dataset availability, and model-specific attention support."
}

cleanup_hf_cache_for_config() {
    local config="$1"
    local reason="$2"
    local output_file="$3"
    local policy="${ATTN_TRACKER_CLEAN_HF_CACHE:-never}"
    local should_cleanup="false"

    case "${policy}" in
        always|true|1|yes)
            should_cleanup="true"
            ;;
        on_success)
            if [[ "${reason}" == "success" ]]; then
                should_cleanup="true"
            fi
            ;;
        on_failure)
            if [[ "${reason}" == "failure" ]]; then
                should_cleanup="true"
            fi
            ;;
        never|false|0|no|"")
            should_cleanup="false"
            ;;
        *)
            echo "Warning: unknown ATTN_TRACKER_CLEAN_HF_CACHE='${policy}', skip cache cleanup." >> "${output_file}"
            should_cleanup="false"
            ;;
    esac

    if [[ "${should_cleanup}" != "true" ]]; then
        return 0
    fi

    local cleanup_args=(--config "${config}")
    if [[ "${ATTN_TRACKER_CLEAN_HF_CACHE_DRY_RUN:-0}" == "1" ]]; then
        cleanup_args+=(--dry-run)
    fi

    echo "===== HF cache cleanup (${reason}): ${config} =====" >> "${output_file}"
    if ! uv run python -u scripts/common/cleanup_hf_cache.py "${cleanup_args[@]}" >> "${output_file}" 2>&1; then
        echo "Warning: HF cache cleanup failed for ${config}" >> "${output_file}"
    fi
}

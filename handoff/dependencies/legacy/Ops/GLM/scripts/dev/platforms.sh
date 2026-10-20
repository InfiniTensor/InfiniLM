#!/usr/bin/env bash

INFINI_OPS_SUPPORTED_PLATFORMS=(
    cpu
    nvidia
    iluvatar
    hygon
    thead
    metax
    moore
    cambricon
    ascend
)

infini_ops_supported_platforms_usage() {
    local IFS='|'
    printf '%s|auto' "${INFINI_OPS_SUPPORTED_PLATFORMS[*]}"
}

infini_ops_supported_platforms_csv() {
    local IFS=', '
    printf '%s' "${INFINI_OPS_SUPPORTED_PLATFORMS[*]}"
}

infini_ops_is_supported_platform() {
    local candidate="${1:-}"
    local platform

    for platform in "${INFINI_OPS_SUPPORTED_PLATFORMS[@]}"; do
        if [[ "${platform}" == "${candidate}" ]]; then
            return 0
        fi
    done

    return 1
}

_infini_ops_has_glob_match() {
    compgen -G "$1" > /dev/null
}

_infini_ops_find_hygon_cuda_root() {
    local dtk_root="$1"
    local candidate
    local versioned_candidates=()

    for candidate in \
        "${dtk_root}/cuda" \
        "${dtk_root}/cuda/cuda"; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    shopt -s nullglob
    versioned_candidates=("${dtk_root}"/cuda/cuda-*)
    shopt -u nullglob

    for candidate in "${versioned_candidates[@]}"; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

_infini_ops_detect_nvidia() {
    _infini_ops_has_glob_match "${INFINI_OPS_NVIDIA_DEVICE_GLOB:-/dev/nvidia*}"
}

_infini_ops_detect_iluvatar() {
    _infini_ops_has_glob_match "${INFINI_OPS_ILUVATAR_DEVICE_GLOB:-/dev/iluvatar*}"
}

_infini_ops_detect_hygon() {
    local dtk_root="${DTK_ROOT:-${INFINI_OPS_HYGON_DTK_ROOT:-/opt/dtk}}"

    _infini_ops_find_hygon_cuda_root "${dtk_root}" > /dev/null
}

_infini_ops_detect_thead() {
    local cuda_root="${CUDA_SDK:-/usr/local/PPU_SDK/CUDA_SDK}"

    [[ -x "${cuda_root}/bin/nvcc" ]]
}

_infini_ops_detect_metax() {
    if [[ -n "${MACA_PATH:-}" ]]; then
        return 0
    fi

    grep -hqs 9999 ${INFINI_OPS_METAX_PCI_VENDOR_GLOB:-/sys/bus/pci/devices/*/vendor} 2> /dev/null
}

_infini_ops_detect_moore() {
    [[ -n "${MUSA_ROOT:-}" || -n "${MUSA_HOME:-}" || -n "${MUSA_PATH:-}" ]]
}

_infini_ops_detect_cambricon() {
    [[ -n "${NEUWARE_HOME:-}" ]]
}

_infini_ops_detect_ascend() {
    [[ -n "${ASCEND_HOME_PATH:-}" ]] || \
        _infini_ops_has_glob_match "${INFINI_OPS_ASCEND_DEVICE_GLOB:-/dev/davinci0}"
}

infini_ops_detect_platforms() {
    local detected=()

    if _infini_ops_detect_nvidia; then
        detected+=("nvidia")
    fi

    if _infini_ops_detect_iluvatar; then
        detected+=("iluvatar")
    fi

    if _infini_ops_detect_hygon; then
        detected+=("hygon")
    fi

    if _infini_ops_detect_thead; then
        detected+=("thead")
    fi

    if _infini_ops_detect_metax; then
        detected+=("metax")
    fi

    if _infini_ops_detect_cambricon; then
        detected+=("cambricon")
    fi

    if _infini_ops_detect_moore; then
        detected+=("moore")
    fi

    if _infini_ops_detect_ascend; then
        detected+=("ascend")
    fi

    if ((${#detected[@]} > 0)); then
        printf '%s\n' "${detected[@]}"
    fi
}

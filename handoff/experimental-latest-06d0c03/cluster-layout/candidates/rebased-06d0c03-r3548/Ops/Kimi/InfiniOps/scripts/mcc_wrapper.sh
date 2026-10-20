#!/bin/bash
# Filter out flags unsupported by `mcc`.
ARGS=()
skip_next=0
linking=1
for arg in "$@"; do
    if [ $skip_next -eq 1 ]; then
        skip_next=0
        continue
    fi
    case "$arg" in
        -c|-E|-S)
            linking=0
            ARGS+=("$arg")
            ;;
        -pthread)
            ;;
        -B)
            skip_next=1
            ;;
        -B*)
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

MUSA_ROOT_DIR="${MUSA_ROOT:-${MUSA_HOME:-${MUSA_PATH:-/usr/local/musa}}}"

if command -v g++ >/dev/null 2>&1; then
    GXX_MAJOR="$(g++ -dumpversion | cut -d. -f1)"
    if [ -d "/usr/include/c++/${GXX_MAJOR}" ]; then
        ARGS=(
            "-isystem" "/usr/include/c++/${GXX_MAJOR}"
            "-isystem" "/usr/include/x86_64-linux-gnu/c++/${GXX_MAJOR}"
            "-isystem" "/usr/include/c++/${GXX_MAJOR}/backward"
            "${ARGS[@]}"
        )
    fi

    STDCPP_LIB="$(g++ -print-file-name=libstdc++.so)"
    if [ $linking -eq 1 ] && [ -f "${STDCPP_LIB}" ]; then
        ARGS=("-L$(dirname "${STDCPP_LIB}")" "${ARGS[@]}")
    fi
fi

exec "${MUSA_ROOT_DIR}/bin/mcc" "${ARGS[@]}"

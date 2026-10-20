#!/bin/bash
# Filter out flags unsupported by `mxcc`.
ARGS=()
skip_next=0
for arg in "$@"; do
    if [ $skip_next -eq 1 ]; then
        skip_next=0
        continue
    fi
    case "$arg" in
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

exec ${MACA_PATH}/mxgpu_llvm/bin/mxcc "${ARGS[@]}"

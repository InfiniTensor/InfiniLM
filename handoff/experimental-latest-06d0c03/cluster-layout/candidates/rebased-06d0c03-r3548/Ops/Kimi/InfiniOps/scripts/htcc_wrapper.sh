#!/bin/bash
# HPCC analog of mxcc_wrapper.sh.
# htcc only enables device compilation for `.cu`. Rewrite compile inputs.
# Place the `.cu` symlink next to the source so same-directory `#include`
# resolution (e.g. bindings `mul.h` vs C-API `generated/include/mul.h`) still
# works — unlike rewriting into `/tmp`.
ARGS=()
skip_next=0
has_compile=0
for arg in "$@"; do
    if [ $skip_next -eq 1 ]; then
        skip_next=0
        ARGS+=("$arg")
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
        -c)
            has_compile=1
            ARGS+=("$arg")
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

if [ "$has_compile" -eq 1 ]; then
    rewritten=()
    for arg in "${ARGS[@]}"; do
        case "$arg" in
            *.cc|*.cpp|*.cxx)
                if [ -f "$arg" ]; then
                    abs=$(readlink -f "$arg")
                    cu="${abs}.cu"
                    ln -sfn "$abs" "$cu"
                    rewritten+=("$cu")
                else
                    rewritten+=("$arg")
                fi
                ;;
            *)
                rewritten+=("$arg")
                ;;
        esac
    done
    ARGS=("${rewritten[@]}")
fi

HPCC_PATH="${HPCC_PATH:-/opt/hpcc}"
exec "${HPCC_PATH}/htgpu_llvm/bin/htcc" "${ARGS[@]}"

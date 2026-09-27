#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
suite=${1:-all}
build_dir=${NATIVE_TEST_BUILD_DIR:-"$root/build/native-tests"}
runtime_dir=${INFINILM_RUNTIME_DIR:-"$root/build/linux/x86_64/release"}
cxx=${CXX:-c++}

case "$suite" in
    all|analyzer|runtime|paged_attention|config) ;;
    *) printf 'Unknown test suite: %s\n' "$suite" >&2; exit 2 ;;
esac

flags=(-std=c++17 -O1 -g -pthread -fno-omit-frame-pointer
       -I"$root/csrc/infinicore/include")
case "${SANITIZER:-none}" in
    none) ;;
    address) flags+=(-fsanitize=address,undefined) ;;
    thread) flags+=(-fsanitize=thread) ;;
    *) printf 'SANITIZER must be none, address, or thread.\n' >&2; exit 2 ;;
esac
read -r -a extra_flags <<< "${CXXFLAGS:-}"
read -r -a extra_links <<< "${LDFLAGS:-}"
flags+=("${extra_flags[@]}" -UNDEBUG)
mkdir -p "$build_dir"

run_test() {
    local name=$1
    shift
    "$cxx" "${flags[@]}" "$root/test/native/test_$name.cc" "$@" \
        "${extra_links[@]}" -o "$build_dir/$name"
    "$build_dir/$name"
    printf 'PASS %s\n' "$name"
}

if [[ "$suite" == all || "$suite" == analyzer ]]; then
    run_test analyzer_concurrency
fi
if [[ "$suite" == analyzer ]]; then
    exit 0
fi

: "${INFINI_ROOT:?Set INFINI_ROOT to the matching installed Infini stack.}"
flags+=(-I"$root/csrc" -I"$root/csrc/infinicore/src"
        -I"$root/third_party/spdlog/include"
        -I"$root/third_party/json/single_include"
        -I"$INFINI_ROOT/include" -I"$INFINI_ROOT/include/infiniccl")
if [[ -d "${CUDA_HOME:-/usr/local/cuda}/include" ]]; then
    flags+=(-I"${CUDA_HOME:-/usr/local/cuda}/include")
fi
links=(-L"$runtime_dir" -L"$INFINI_ROOT/lib" -L"$INFINI_ROOT/lib64"
       -Wl,-rpath,"$runtime_dir" -linfinicore_runtime -linfinirt)
export LD_LIBRARY_PATH="$runtime_dir:$INFINI_ROOT/lib:$INFINI_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "$suite" == all || "$suite" == runtime ]]; then
    run_test runtime_safety "${links[@]}"
fi
if [[ "$suite" == all || "$suite" == paged_attention ]]; then
    run_test paged_attention_metadata \
        "$root/csrc/infinicore/src/ops/paged_attention_prefill/paged_attention_prefill.cc" \
        "${links[@]}"
fi
if [[ "$suite" == all || "$suite" == config ]]; then
    run_test config_parsing \
        "$root/csrc/config/config_factory.cpp" \
        "$root/csrc/config/model_config.cpp" \
        "$root/csrc/config/quant_config.cpp" \
        "$root/csrc/models/models_registry.cpp" \
        "$root/csrc/global_state/global_state.cpp" \
        "$root/csrc/layers/quantization/base_quantization.cpp" \
        "$root/csrc/layers/quantization/none_quantization.cpp" \
        "$root/csrc/layers/quantization/compressed_tensors.cpp" \
        "$root/csrc/layers/quantization/awq.cpp" \
        "$root/csrc/layers/quantization/awq_marlin.cpp" \
        "$root/csrc/layers/quantization/gptq.cpp" \
        "$root/csrc/layers/quantization/gptq_marlin.cpp" \
        "$root/csrc/layers/quantization/gptq_qy.cpp" \
        "$root/csrc/layers/quantization/mxfp4.cpp" \
        "${links[@]}"
fi

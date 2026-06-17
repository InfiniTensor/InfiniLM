#pragma once

#include <cstdlib>
#include <string>

namespace infinilm::engine {

/// When true, C++ ``PagedCompiler`` must not capture or replay monolithic prefill CUDA graphs
/// (torch compiled prefill owns the prefill path).
inline bool skip_cpp_prefill_graph() {
    const char *v = std::getenv("INFINI_PREFILL_COMPILE");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// When true, use native C++ ``PiecewisePrefillCompiler`` for prefill (HPCC v1 path).
inline bool native_piecewise_prefill_enabled() {
    const char *v = std::getenv("INFINI_PREFILL_NATIVE_CG");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// When true, skip native CG replay and call eager piecewise methods (bisect vs CG capture).
inline bool native_cg_replay_none() {
    const char *v = std::getenv("INFINI_NATIVE_CG_REPLAY_NONE");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// When true, attempt HCCL allreduce inside post-attn CG capture (often unsupported on HPCC).
/// Production path: graph records matmul+copy only; eager AR on ``ar_staging`` after replay.
inline bool piecewise_ar_in_graph() {
    const char *v = std::getenv("INFINI_PIECEWISE_AR_IN_GRAPH");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// When true with INFINI_NATIVE_CG_MAX_LAYERS=N, replay CG for layers 0..N-1 then eager tail N..L-1.
inline bool piecewise_eager_tail() {
    const char *v = std::getenv("INFINI_PIECEWISE_EAGER_TAIL");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// Output directory for per-layer hidden-state dumps (unset = disabled).
inline const char *dump_layer_hidden_dir() {
    return std::getenv("INFINI_DUMP_LAYER_HIDDEN_DIR");
}

/// Row index for hidden-state dumps (default 511).
inline size_t dump_layer_hidden_row() {
    static size_t cached = static_cast<size_t>(-1);
    if (cached == static_cast<size_t>(-1)) {
        const char *v = std::getenv("INFINI_DUMP_LAYER_HIDDEN_ROW");
        cached = (v != nullptr && v[0] != '\0')
                     ? static_cast<size_t>(std::strtoul(v, nullptr, 10))
                     : 511;
    }
    return cached;
}

/// Mode tag prefix for hidden-state dump filenames (e.g. PIECEWISE, GRAPH_DECODE).
inline const char *dump_layer_hidden_tag() {
    const char *v = std::getenv("INFINI_DUMP_LAYER_HIDDEN_TAG");
    return (v != nullptr && v[0] != '\0') ? v : "unknown";
}

} // namespace infinilm::engine

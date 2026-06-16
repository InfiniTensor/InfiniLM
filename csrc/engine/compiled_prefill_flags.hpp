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

/// When true, record row-parallel HCCL allreduce inside post-attn CG segments (vs legacy staging).
inline bool piecewise_ar_in_graph() {
    const char *v = std::getenv("INFINI_PIECEWISE_AR_IN_GRAPH");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

} // namespace infinilm::engine

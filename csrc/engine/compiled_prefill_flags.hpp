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

} // namespace infinilm::engine

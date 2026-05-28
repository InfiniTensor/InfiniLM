#pragma once

#include <cstdlib>
#include <string>

namespace infinilm::engine {

/// When true, C++ ``PagedCompiler`` must not capture or replay prefill CUDA graphs
/// (torch compiled prefill owns the prefill path).
inline bool skip_cpp_prefill_graph() {
    const char *v = std::getenv("INFINI_PREFILL_COMPILE");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

} // namespace infinilm::engine

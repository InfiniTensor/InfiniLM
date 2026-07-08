#pragma once

#include <cstdlib>
#include <string>

namespace infinilm::global_state {

/// When true, piecewise pre/post segments use AOTInductor kernels (M4 path).
inline bool piecewise_inductor_segment_enabled() {
    const char *v = std::getenv("INFINI_PIECEWISE_INDUCTOR_SEGMENT");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

} // namespace infinilm::global_state

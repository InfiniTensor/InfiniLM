#pragma once

#include <cstdlib>
#include <string>

namespace infinilm::global_state {

/// When true, piecewise pre/post segments use AOTInductor kernels (M4 path).
inline bool piecewise_inductor_segment_enabled() {
    const char *v = std::getenv("INFINI_PIECEWISE_INDUCTOR_SEGMENT");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// Production default ON when inductor segment is enabled; set to 0 to bisect.
inline bool scoped_inductor_pre_attn_enabled() {
    const char *v = std::getenv("INFINI_PIECEWISE_SCOPED_INDUCTOR");
    if (v == nullptr || v[0] == '\0') {
        return piecewise_inductor_segment_enabled();
    }
    return v[0] != '0' && std::string(v) != "false";
}

inline bool repro_skip_midchunk_eager() {
    const char *v = std::getenv("INFINI_PIECEWISE_REPRO_SKIP_MIDCHUNK_EAGER");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

inline bool repro_skip_final_inductor() {
    const char *v = std::getenv("INFINI_PIECEWISE_REPRO_SKIP_FINAL_INDUCTOR");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

/// Tail bucket eligible for inductor pre_attn inside hcGraph (B4 only).
inline bool bucket_is_inductor_eligible(size_t bucket) {
    constexpr size_t kInductorTailBucket = 4;
    return bucket == kInductorTailBucket;
}

} // namespace infinilm::global_state

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

/// Opt-in: skip host ``syncDevice()`` between QKV CG replay and eager RoPE.
/// Only relevant when ``PRE_ATTN_QKV_ONLY=1`` (Phase-2c split). Product fused
/// pre (LN+QKV+RoPE in one CG) does not need this fence. Default: sync ON.
inline bool skip_qkv_rope_sync() {
    const char *v = std::getenv("INFINI_PIECEWISE_SKIP_QKV_ROPE_SYNC");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

/// Kill-switch: capture/replay Phase-2c QKV-only pre CG + host-eager RoPE.
/// Default OFF → product fused pre ``[CG: LN+QKV+RoPE]``.
inline bool pre_attn_qkv_only() {
    const char *v = std::getenv("INFINI_PIECEWISE_PRE_ATTN_QKV_ONLY");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

/// Prefill pad-up CG (seq_len < graph_bucket). Default ON when unset;
/// ``INFINI_PIECEWISE_PAD_UP_CG=0`` restores legacy eager-only pad-up.
inline bool piecewise_pad_up_cg_enabled() {
    const char *v = std::getenv("INFINI_PIECEWISE_PAD_UP_CG");
    if (v != nullptr && v[0] == '0' && v[1] == '\0') {
        return false;
    }
    return true;
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

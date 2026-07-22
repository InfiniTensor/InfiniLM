#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

namespace infinilm::engine {
namespace {

inline const char *cudagraph_policy_cstr_() {
    const char *v = std::getenv("INFINI_CUDAGRAPH_POLICY");
    if (v == nullptr || v[0] == '\0') {
        return "";
    }
    if (std::strcmp(v, "eager") == 0) {
        return "eager";
    }
    if (std::strcmp(v, "full_and_piecewise") == 0) {
        return "full_and_piecewise";
    }
    // Reject unknown / track_b — legacy flag behavior.
    return "";
}

inline bool env_is_one_(const char *name) {
    const char *v = std::getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

} // namespace

/// ``INFINI_CUDAGRAPH_POLICY`` when set to a known value; empty = legacy envs only.
inline const char *cudagraph_policy() {
    return cudagraph_policy_cstr_();
}

/// When true, C++ ``PagedCompiler`` must not capture or replay monolithic prefill CUDA graphs
/// (torch compiled prefill owns the prefill path).
inline bool skip_cpp_prefill_graph() {
    if (std::strcmp(cudagraph_policy(), "eager") == 0) {
        return true;
    }
    const char *v = std::getenv("INFINI_PREFILL_COMPILE");
    if (v != nullptr && v[0] != '\0' && std::string(v) != "0") {
        return true;
    }
    // full_and_piecewise + PREFILL_NATIVE_CG=0: decode FULL only (no monolithic
    // prefill — MetaX HTC risk). Prefill native piecewise is handled separately.
    if (std::strcmp(cudagraph_policy(), "full_and_piecewise") == 0) {
        const char *native = std::getenv("INFINI_PREFILL_NATIVE_CG");
        const bool native_on =
            native != nullptr && native[0] != '\0' && std::string(native) != "0";
        // Unset defaults on under full_and_piecewise; explicit 0 → skip monolithic.
        if (native != nullptr && !native_on) {
            return true;
        }
    }
    // Decode-only CG for MiniCPM5 MoE: skip monolithic prefill (HTC mem risk).
    return env_is_one_("INFINI_DECODE_GRAPH_ONLY");
}

/// Skip monolithic full-forward decode CUDAGraph capture.
/// MiniCPM5 MoE CG-1: MoE+FA are host-breaks, but remaining ops in one capture still
/// HTC-fault on MetaX; use decode-piecewise instead (``native_piecewise_decode_enabled``).
/// Under ``full_and_piecewise``, monolithic FULL decode is the default (unless SKIP /
/// DECODE_GRAPH_ONLY). Explicit ``eager`` disables all decode CG.
inline bool skip_monolithic_decode_graph() {
    if (std::strcmp(cudagraph_policy(), "eager") == 0) {
        return true;
    }
    if (env_is_one_("INFINI_SKIP_MONOLITHIC_DECODE_CG")) {
        return true;
    }
    if (std::strcmp(cudagraph_policy(), "full_and_piecewise") == 0) {
        // FULL decode unless opt-out via DECODE_GRAPH_ONLY / SKIP (checked above).
        return env_is_one_("INFINI_DECODE_GRAPH_ONLY");
    }
    return env_is_one_("INFINI_DECODE_GRAPH_ONLY");
}

/// When true, use native C++ ``PiecewiseDecodeCompiler`` (per-layer pre / eager FA /
/// post-without-MoE / eager MoE). Default on when monolithic decode CG is skipped
/// (legacy). Under ``full_and_piecewise`` default off (monolithic FULL); under
/// ``eager`` always off.
inline bool native_piecewise_decode_enabled() {
    if (std::strcmp(cudagraph_policy(), "eager") == 0) {
        return false;
    }
    const char *v = std::getenv("INFINI_DECODE_PIECEWISE");
    if (v != nullptr) {
        return v[0] != '\0' && std::string(v) != "0";
    }
    if (std::strcmp(cudagraph_policy(), "full_and_piecewise") == 0) {
        return false;
    }
    return skip_monolithic_decode_graph();
}

/// When true, use native C++ ``PiecewisePrefillCompiler`` for prefill (HPCC v1 path).
/// ``full_and_piecewise`` defaults on; ``eager`` forces off; unset policy uses
/// ``INFINI_PREFILL_NATIVE_CG`` only.
inline bool native_piecewise_prefill_enabled() {
    if (std::strcmp(cudagraph_policy(), "eager") == 0) {
        return false;
    }
    const char *v = std::getenv("INFINI_PREFILL_NATIVE_CG");
    if (v != nullptr) {
        return v[0] != '\0' && std::string(v) != "0";
    }
    return std::strcmp(cudagraph_policy(), "full_and_piecewise") == 0;
}

/// When true, allow paged decode CUDAGraph capture/replay under tensor parallel (tp>1).
inline bool decode_cg_tp_enabled() {
    return env_is_one_("INFINI_DECODE_CG_TP");
}

} // namespace infinilm::engine

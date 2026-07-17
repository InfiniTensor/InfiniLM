#pragma once

#include <cstdlib>
#include <string>

namespace infinilm::engine {

/// When true, C++ ``PagedCompiler`` must not capture or replay monolithic prefill CUDA graphs
/// (torch compiled prefill owns the prefill path).
inline bool skip_cpp_prefill_graph() {
    const char *v = std::getenv("INFINI_PREFILL_COMPILE");
    if (v != nullptr && v[0] != '\0' && std::string(v) != "0") {
        return true;
    }
    // Decode-only CG for MiniCPM5 MoE: skip monolithic prefill (HTC mem risk).
    const char *d = std::getenv("INFINI_DECODE_GRAPH_ONLY");
    return d != nullptr && d[0] == '1' && d[1] == '\0';
}

/// Skip monolithic full-forward decode CUDAGraph capture.
/// MiniCPM5 MoE CG-1: MoE+FA are host-breaks, but remaining ops in one capture still
/// HTC-fault on MetaX; use decode-piecewise instead (``native_piecewise_decode_enabled``).
inline bool skip_monolithic_decode_graph() {
    const char *v = std::getenv("INFINI_SKIP_MONOLITHIC_DECODE_CG");
    if (v != nullptr && v[0] == '1' && v[1] == '\0') {
        return true;
    }
    const char *d = std::getenv("INFINI_DECODE_GRAPH_ONLY");
    return d != nullptr && d[0] == '1' && d[1] == '\0';
}

/// When true, use native C++ ``PiecewiseDecodeCompiler`` (per-layer pre / eager FA /
/// post-without-MoE / eager MoE). Default on when monolithic decode CG is skipped.
inline bool native_piecewise_decode_enabled() {
    const char *v = std::getenv("INFINI_DECODE_PIECEWISE");
    if (v != nullptr) {
        return v[0] != '\0' && std::string(v) != "0";
    }
    return skip_monolithic_decode_graph();
}

/// When true, use native C++ ``PiecewisePrefillCompiler`` for prefill (HPCC v1 path).
inline bool native_piecewise_prefill_enabled() {
    const char *v = std::getenv("INFINI_PREFILL_NATIVE_CG");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// When true, allow paged decode CUDAGraph capture/replay under tensor parallel (tp>1).
inline bool decode_cg_tp_enabled() {
    const char *v = std::getenv("INFINI_DECODE_CG_TP");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

} // namespace infinilm::engine

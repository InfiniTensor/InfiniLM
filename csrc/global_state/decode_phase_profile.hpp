#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace infinilm::global_state::decode_phase_profile {

/// Opt-in decode host profiler: ``INFINI_DECODE_HOST_PROFILE=1`` (default off).
inline bool enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_DECODE_HOST_PROFILE");
        cached = (raw != nullptr && raw[0] != '\0' && std::string(raw) != "0") ? 1 : 0;
    }
    return cached == 1;
}

inline bool recording() { return enabled(); }

inline double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

/// Legacy layer-phase stubs (eager-path leftovers; unused under FULL decode CG).
struct Counters {
    double attn_ms{0};
    double moe_ms{0};
    double dense_mlp_ms{0};
    double other_layer_ms{0};
    double eager_forward_ms{0};
};

inline Counters &counters() {
    static Counters c;
    return c;
}

inline void reset() { counters() = Counters{}; }

inline bool &decode_step_active() {
    static bool active = false;
    return active;
}

/// Last ``get_compiled`` bank vs reset split (filled by PagedCompiler when profiling).
struct GetCompiledSplit {
    double bank_h2d_ms{0};
    double reset_runtime_ms{0};
};

inline GetCompiledSplit &last_get_compiled_split() {
    static GetCompiledSplit s;
    return s;
}

/// One FULL-decode graph-hit step on the rank worker.
struct HostStep {
    double to_model_ms{0};
    double get_compiled_ms{0};
    double bank_h2d_ms{0};
    double reset_runtime_ms{0};
    double graph_run_ms{0};
    double launch_ms{0};
    double hostop_ms{0};
    double op_list_ms{0};
    double tail_sync_ms{0};
    double sample_ms{0};
    double d2h_ms{0};
    double step_ms{0};
    size_t device_segment_count{0};
    bool skipped_tail_sync{false};
};

inline constexpr size_t kWarmupSteps = 16;
inline constexpr size_t kLogEvery = 64;

struct HostAccum {
    size_t seen{0};
    size_t recorded{0};
    HostStep sum{};
    std::vector<double> step_ms;
    std::vector<double> to_model_ms;
    std::vector<double> get_compiled_ms;
    std::vector<double> bank_h2d_ms;
    std::vector<double> reset_runtime_ms;
    std::vector<double> graph_run_ms;
    std::vector<double> hostop_ms;
    std::vector<double> sample_ms;
    std::vector<double> d2h_ms;
    size_t device_segment_sum{0};
};

inline HostAccum &host_accum() {
    static HostAccum a;
    return a;
}

inline double percentile_sorted(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double idx = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

inline void push_ms(std::vector<double> &v, double x) { v.push_back(x); }

inline void record_host_step(const HostStep &s, bool log_rank0) {
    if (!enabled()) {
        return;
    }
    auto &a = host_accum();
    ++a.seen;
    if (a.seen <= kWarmupSteps) {
        return;
    }
    ++a.recorded;
    a.sum.to_model_ms += s.to_model_ms;
    a.sum.get_compiled_ms += s.get_compiled_ms;
    a.sum.bank_h2d_ms += s.bank_h2d_ms;
    a.sum.reset_runtime_ms += s.reset_runtime_ms;
    a.sum.graph_run_ms += s.graph_run_ms;
    a.sum.launch_ms += s.launch_ms;
    a.sum.hostop_ms += s.hostop_ms;
    a.sum.op_list_ms += s.op_list_ms;
    a.sum.tail_sync_ms += s.tail_sync_ms;
    a.sum.sample_ms += s.sample_ms;
    a.sum.d2h_ms += s.d2h_ms;
    a.sum.step_ms += s.step_ms;
    a.device_segment_sum += s.device_segment_count;
    push_ms(a.step_ms, s.step_ms);
    push_ms(a.to_model_ms, s.to_model_ms);
    push_ms(a.get_compiled_ms, s.get_compiled_ms);
    push_ms(a.bank_h2d_ms, s.bank_h2d_ms);
    push_ms(a.reset_runtime_ms, s.reset_runtime_ms);
    push_ms(a.graph_run_ms, s.graph_run_ms);
    push_ms(a.hostop_ms, s.hostop_ms);
    push_ms(a.sample_ms, s.sample_ms);
    push_ms(a.d2h_ms, s.d2h_ms);

    if (!log_rank0 || a.recorded % kLogEvery != 0) {
        return;
    }
    const double n = static_cast<double>(a.recorded);
    const double mean_seg = static_cast<double>(a.device_segment_sum) / n;
    spdlog::info(
        "decode_host_profile: n={} (skip_warmup={}) "
        "step_ms mean={:.3f} p50={:.3f} | "
        "to_model mean={:.3f} p50={:.3f} | "
        "get_compiled mean={:.3f} p50={:.3f} "
        "(bank_h2d mean={:.3f} p50={:.3f}; reset mean={:.3f} p50={:.3f}) | "
        "graph_run mean={:.3f} p50={:.3f} "
        "(launch mean={:.3f}; hostop mean={:.3f}; op_list mean={:.3f}; "
        "tail_sync mean={:.3f}; skipped_tail_last={}) | "
        "sample mean={:.3f} p50={:.3f} | d2h mean={:.3f} p50={:.3f} | "
        "device_segments mean={:.2f} last={}",
        a.recorded,
        kWarmupSteps,
        a.sum.step_ms / n,
        percentile_sorted(a.step_ms, 0.5),
        a.sum.to_model_ms / n,
        percentile_sorted(a.to_model_ms, 0.5),
        a.sum.get_compiled_ms / n,
        percentile_sorted(a.get_compiled_ms, 0.5),
        a.sum.bank_h2d_ms / n,
        percentile_sorted(a.bank_h2d_ms, 0.5),
        a.sum.reset_runtime_ms / n,
        percentile_sorted(a.reset_runtime_ms, 0.5),
        a.sum.graph_run_ms / n,
        percentile_sorted(a.graph_run_ms, 0.5),
        a.sum.launch_ms / n,
        a.sum.hostop_ms / n,
        a.sum.op_list_ms / n,
        a.sum.tail_sync_ms / n,
        s.skipped_tail_sync ? 1 : 0,
        a.sum.sample_ms / n,
        percentile_sorted(a.sample_ms, 0.5),
        a.sum.d2h_ms / n,
        percentile_sorted(a.d2h_ms, 0.5),
        mean_seg,
        s.device_segment_count);
}

} // namespace infinilm::global_state::decode_phase_profile

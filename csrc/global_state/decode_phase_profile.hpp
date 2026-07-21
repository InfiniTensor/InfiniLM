#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace infinilm::global_state::decode_phase_profile {

struct Counters {
    uint64_t decode_steps{0};
    double eager_forward_ms{0.0};
    double attn_ms{0.0};
    double moe_ms{0.0};
    /// Span-fuse graph path: host+GPU exclusive wall of Graph::run segments
    /// (post+MoE[+next_pre] folded). Zero on pure-eager MoE host-break path.
    double graph_run_ms{0.0};
    /// P8b: number of FA host-breaks / Graph::run calls this step (launch tax).
    uint64_t n_fa{0};
    uint64_t n_graph_runs{0};
    double dense_mlp_ms{0.0};
    double other_layer_ms{0.0};
    double sync_ms{0.0};
    double sample_ms{0.0};
};

inline bool enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_DECODE_PHASE_PROFILE");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

/// When set with PHASE_PROFILE, syncStream after FA / Graph::run so attn_ms /
/// graph_run_ms are exclusive GPU+host (default on). Set =0 for launch-only
/// host windows (GPU drain folds into sync_ms).
inline bool exclusive_sync() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_DECODE_PHASE_PROFILE_SYNC");
        if (raw == nullptr || raw[0] == '\0') {
            cached = 1; // default: exclusive when profiling
        } else {
            cached = (raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
        }
    }
    return cached == 1;
}

/// RankWorker sets this around decode forward so prefill layers are excluded.
inline bool &decode_step_active() {
    static thread_local bool active = false;
    return active;
}

inline bool recording() { return enabled() && decode_step_active(); }

inline Counters &counters() {
    static thread_local Counters c;
    return c;
}

inline void reset() { counters() = {}; }

inline double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

inline void log_step_if_due(size_t n_req = 1) {
    if (!enabled()) {
        return;
    }
    auto &c = counters();
    c.decode_steps++;
    // Per-step log (reset() clears prior step); keep every step for small cells.
    const double per_graph =
        c.n_graph_runs > 0 ? c.graph_run_ms / static_cast<double>(c.n_graph_runs) : 0.0;
    const double per_fa =
        c.n_fa > 0 ? c.attn_ms / static_cast<double>(c.n_fa) : 0.0;
    spdlog::info(
        "decode_phase_profile: steps={} n_req={} forward_ms={:.3f} attn_ms={:.3f} moe_ms={:.3f} "
        "graph_run_ms={:.3f} n_fa={} n_graph_runs={} per_fa_ms={:.3f} per_graph_ms={:.3f} "
        "dense_mlp_ms={:.3f} other_layer_ms={:.3f} sync_ms={:.3f} sample_ms={:.3f}",
        c.decode_steps,
        n_req,
        c.eager_forward_ms,
        c.attn_ms,
        c.moe_ms,
        c.graph_run_ms,
        c.n_fa,
        c.n_graph_runs,
        per_fa,
        per_graph,
        c.dense_mlp_ms,
        c.other_layer_ms,
        c.sync_ms,
        c.sample_ms);
}

} // namespace infinilm::global_state::decode_phase_profile

#pragma once

#include "forward_context.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace infinilm::global_state::ar_profile {

struct Counters {
    uint64_t fast_path{0};
    uint64_t copy_path{0};
    uint64_t bytes_copied{0};
    double wall_ms{0.0};
    uint64_t deferred_batches{0};
    uint64_t decode_graph_steps{0};
    double decode_pre_barrier_ms{0.0};
    double decode_post_sync_ms{0.0};
    double decode_graph_run_ms{0.0};
};

inline bool enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_AR_PROFILE");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

inline Counters &counters() {
    static thread_local Counters c;
    return c;
}

inline void reset() {
    counters() = {};
}

inline double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

inline void allreduce_tensor(infinicore::Tensor tensor, infinicclComm_t comm) {
    const bool profile = enabled();
    const double t0 = profile ? monotonic_ms() : 0.0;
    if (tensor->is_contiguous()) {
        if (profile) {
            counters().fast_path++;
        }
        infinicore::op::distributed::allreduce_(tensor, tensor, INFINICCL_SUM, comm);
    } else {
        if (profile) {
            counters().copy_path++;
            counters().bytes_copied += 2 * tensor->nbytes();
        }
        auto contiguous = tensor->contiguous();
        infinicore::op::distributed::allreduce_(contiguous, contiguous, INFINICCL_SUM, comm);
        tensor->copy_from(contiguous);
    }
    if (profile) {
        counters().wall_ms += monotonic_ms() - t0;
    }
}

inline void run_deferred_allreduces(const std::vector<DeferredAllreduce> &ops) {
    const bool profile = enabled();
    const double t0 = profile ? monotonic_ms() : 0.0;
    const auto fast0 = counters().fast_path;
    const auto copy0 = counters().copy_path;
    const auto bytes0 = counters().bytes_copied;
    for (const auto &op : ops) {
        if (op.comm != nullptr && op.tensor) {
            allreduce_tensor(op.tensor, op.comm);
        }
    }
    if (profile) {
        counters().deferred_batches++;
        const double batch_ms = monotonic_ms() - t0;
        const auto &c = counters();
        spdlog::info(
            "ar_profile: deferred batch n_ops={} batch_fast={} batch_copy={} batch_bytes={} "
            "batch_wall_ms={:.3f} cumulative_fast={} cumulative_copy={} cumulative_bytes={} "
            "cumulative_wall_ms={:.3f}",
            ops.size(),
            c.fast_path - fast0,
            c.copy_path - copy0,
            c.bytes_copied - bytes0,
            batch_ms,
            c.fast_path,
            c.copy_path,
            c.bytes_copied,
            c.wall_ms);
    }
}

inline void log_summary(const char *tag) {
    if (!enabled()) {
        return;
    }
    const auto &c = counters();
    const uint64_t total_ar = c.fast_path + c.copy_path;
    const double copy_pct = total_ar > 0 ? 100.0 * static_cast<double>(c.copy_path) / static_cast<double>(total_ar) : 0.0;
    spdlog::info(
        "ar_profile: summary tag={} fast_path={} copy_path={} copy_pct={:.1f} bytes_copied={} "
        "wall_ms={:.3f} deferred_batches={} decode_graph_steps={} decode_pre_barrier_ms={:.3f} "
        "decode_post_sync_ms={:.3f} decode_graph_run_ms={:.3f}",
        tag,
        c.fast_path,
        c.copy_path,
        copy_pct,
        c.bytes_copied,
        c.wall_ms,
        c.deferred_batches,
        c.decode_graph_steps,
        c.decode_pre_barrier_ms,
        c.decode_post_sync_ms,
        c.decode_graph_run_ms);
}

} // namespace infinilm::global_state::ar_profile

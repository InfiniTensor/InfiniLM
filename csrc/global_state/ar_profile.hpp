#pragma once

#include "../utils.hpp"
#include "forward_context.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace infinilm::global_state::ar_profile {

struct BarrierLabelStats {
    double wait_ms{0.0};
    uint64_t count{0};
};

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

inline std::unordered_map<std::string, BarrierLabelStats> &barrier_cumulative_stats() {
    static thread_local std::unordered_map<std::string, BarrierLabelStats> stats;
    return stats;
}

inline std::unordered_map<std::string, BarrierLabelStats> &barrier_chunk_stats() {
    static thread_local std::unordered_map<std::string, BarrierLabelStats> stats;
    return stats;
}

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
    barrier_cumulative_stats().clear();
    barrier_chunk_stats().clear();
}

inline void record_barrier_wait(const char *label, double ms) {
    if (!enabled() || label == nullptr || label[0] == '\0' || ms < 0.0) {
        return;
    }
    const std::string key(label);
    auto &cumulative = barrier_cumulative_stats()[key];
    cumulative.wait_ms += ms;
    cumulative.count++;
    auto &chunk = barrier_chunk_stats()[key];
    chunk.wait_ms += ms;
    chunk.count++;
}

inline void log_barrier_chunk_summary(const char *tag,
                                      std::optional<size_t> seq_len = std::nullopt,
                                      std::optional<size_t> n_req = std::nullopt) {
    if (!enabled() || tag == nullptr || tag[0] == '\0') {
        return;
    }
    auto &chunk = barrier_chunk_stats();
    if (chunk.empty()) {
        return;
    }
    std::vector<std::pair<std::string, BarrierLabelStats>> ranked;
    ranked.reserve(chunk.size());
    double total_ms = 0.0;
    for (const auto &[label, stats] : chunk) {
        ranked.emplace_back(label, stats);
        total_ms += stats.wait_ms;
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
        return a.second.wait_ms > b.second.wait_ms;
    });
    std::string labels;
    const size_t max_labels = 8;
    for (size_t i = 0; i < ranked.size() && i < max_labels; ++i) {
        if (i > 0) {
            labels += ",";
        }
        labels += ranked[i].first + ":" + std::to_string(ranked[i].second.wait_ms);
    }
    if (seq_len.has_value() && n_req.has_value()) {
        spdlog::info(
            "ar_profile: barrier_chunk tag={} seq_len={} n_req={} total_ms={:.3f} labels={}",
            tag,
            *seq_len,
            *n_req,
            total_ms,
            labels);
    } else if (n_req.has_value()) {
        spdlog::info(
            "ar_profile: barrier_chunk tag={} n_req={} total_ms={:.3f} labels={}",
            tag,
            *n_req,
            total_ms,
            labels);
    } else {
        spdlog::info(
            "ar_profile: barrier_chunk tag={} total_ms={:.3f} labels={}",
            tag,
            total_ms,
            labels);
    }
    chunk.clear();
}

inline double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

/// Allreduce hidden_states[1,bucket,H] for the first valid_len tokens via contiguous staging.
template <typename AllreduceFn>
inline void allreduce_hidden_valid_contiguous(infinicore::Tensor &hidden_states,
                                              size_t valid_len,
                                              infinicore::Tensor &ar_staging,
                                              AllreduceFn &&allreduce_fn) {
    if (valid_len == 0) {
        return;
    }
    const size_t bucket = hidden_states->size(1);
    const size_t hidden_size = hidden_states->size(2);
    if (valid_len == bucket && hidden_states->is_contiguous()) {
        allreduce_fn(hidden_states);
        return;
    }
    if (ar_staging && valid_len < bucket) {
        auto staging_tail = ar_staging->narrow({{1, valid_len, bucket - valid_len}});
        infinicore::Tensor staging_tail_ref = staging_tail;
        set_zeros(staging_tail_ref);
    }
    auto hidden_narrow = hidden_states->narrow({{1, 0, valid_len}});
    auto staging_narrow = ar_staging->narrow({{1, 0, valid_len}});
    staging_narrow->copy_from(hidden_narrow);
    auto ar_view = staging_narrow->view({1, valid_len, hidden_size});
    allreduce_fn(ar_view);
    hidden_narrow->copy_from(staging_narrow);
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

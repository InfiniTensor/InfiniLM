#pragma once

#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace infinilm::global_state::ar_profile {

inline bool enabled() { return false; }

struct Counters {
    double decode_pre_barrier_ms{0};
    double decode_graph_run_ms{0};
    double decode_post_sync_ms{0};
    size_t decode_graph_steps{0};
};

inline Counters &counters() {
    static Counters c;
    return c;
}

inline void log_barrier_chunk_summary(const char *,
                                      std::optional<size_t> = std::nullopt,
                                      size_t = 0) {}

/// Run AllReduce only over the leading valid_len tokens of a [B,S,H] hidden.
/// When valid_len < S and ar_staging is a contiguous [1,bucket,H] (or matching)
/// buffer, copy the valid slice into staging, reduce, and copy back — avoids
/// AllReduce on padded-bucket tails (FC4_Sum inflation).
template <typename Fn>
inline void allreduce_hidden_valid_contiguous(infinicore::Tensor &hidden,
                                              size_t valid_len,
                                              infinicore::Tensor &ar_staging,
                                              Fn &&fn) {
    if (!hidden) {
        return;
    }
    const auto shape = hidden->shape();
    if (shape.size() < 2) {
        fn(hidden);
        return;
    }
    const size_t seq_dim = 1;
    const size_t seq_len = shape[seq_dim];
    if (valid_len == 0 || valid_len >= seq_len || !ar_staging) {
        // Full tensor (or no staging): reduce in place.
        auto narrowed = (valid_len > 0 && valid_len < seq_len)
                            ? hidden->narrow({{seq_dim, 0, valid_len}})
                            : hidden;
        fn(narrowed);
        return;
    }
    // Pack valid tokens into contiguous staging, reduce, scatter back.
    auto valid = hidden->narrow({{seq_dim, 0, valid_len}});
    auto staging_valid = ar_staging->narrow({{seq_dim, 0, valid_len}});
    staging_valid->copy_from(valid);
    fn(staging_valid);
    valid->copy_from(staging_valid);
}

} // namespace infinilm::global_state::ar_profile

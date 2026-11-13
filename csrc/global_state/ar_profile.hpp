#pragma once

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

template <typename Fn>
inline void allreduce_hidden_valid_contiguous(infinicore::Tensor &hidden,
                                              size_t valid_len,
                                              infinicore::Tensor & /*ar_staging*/,
                                              Fn &&fn) {
    (void)valid_len;
    fn(hidden);
}

} // namespace infinilm::global_state::ar_profile

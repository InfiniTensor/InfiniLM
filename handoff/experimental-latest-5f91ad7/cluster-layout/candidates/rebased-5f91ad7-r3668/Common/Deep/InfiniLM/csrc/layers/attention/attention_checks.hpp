#pragma once

#include "../../execution/submit.hpp"
#include "runtime/tensor.hpp"
#include <base/flash_attn_with_kvcache.h>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace infinilm::layers::attention::detail {
using infinilm::runtime::Runtime;
using infinilm::runtime::Tensor;

inline int64_t check_kvcache(Runtime &runtime, const Tensor &out,
                             const Tensor &q, const Tensor &k_cache,
                             const Tensor &v_cache, const Tensor &lengths,
                             const Tensor &blocks, const std::optional<Tensor> &alibi,
                             size_t logical_bound) {
    runtime.assertOwnerThread();
    for (const auto &t : {out, q, k_cache, v_cache, lengths, blocks}) {
        infinilm::execution::detail::check_owner(runtime, t);
    }
    infinilm::execution::detail::check_owner(runtime, alibi);
    if (q->ndim() != 4 || q->size(1) != 1) {
        throw std::invalid_argument("MhaKVCache requires single-token decode");
    }
    if (logical_bound > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("logical KV upper bound exceeds int64 range");
    }
    return static_cast<int64_t>(logical_bound);
}

} // namespace infinilm::layers::attention::detail

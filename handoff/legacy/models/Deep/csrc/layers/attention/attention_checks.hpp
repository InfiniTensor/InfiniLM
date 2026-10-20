#pragma once

#include "runtime/tensor.hpp"
#include "../../execution/submit.hpp"
#if !defined(INFINICORE_MHA_OMITTED) || !defined(INFINICORE_MHA_KVCACHE_OMITTED)
#include <base/flash_attn_with_kvcache.h>
#endif
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::layers::attention::detail {
using infinilm::runtime::Tensor;
using infinilm::runtime::Runtime;
using infinilm::runtime::Device;

// LM request policy only; generic metadata and current-storage checks live in Ops.
// Keep the unqualified non-NVIDIA finite-window path closed without a device table.
inline void check_kvcache_window(const Tensor &out, const std::vector<int64_t> &window_size) {
    if (out->device().type() != Device::Type::kNvidia
        && (window_size.size() != 2 || window_size[0] != -1 || window_size[1] != -1))
        throw std::invalid_argument("unsupported MhaKVCache finite window");
}

inline int64_t check_kvcache(Runtime &runtime, const Tensor &out,
                             const Tensor &q, const Tensor &k_cache,
                             const Tensor &v_cache, const Tensor &lengths,
                             const Tensor &blocks, const std::optional<Tensor> &alibi,
                             size_t logical_bound, const std::vector<int64_t> &window_size) {
    runtime.assertOwnerThread();
    for (const auto &t : {out, q, k_cache, v_cache, lengths, blocks})
        infinilm::execution::detail::check_owner(runtime, t);
    infinilm::execution::detail::check_owner(runtime, alibi);
    if (q->ndim() != 4 || q->size(1) != 1)
        throw std::invalid_argument("MhaKVCache requires single-token decode");
    check_kvcache_window(out, window_size);
    if (logical_bound > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        throw std::invalid_argument("logical KV upper bound exceeds int64 range");
    return static_cast<int64_t>(logical_bound);
}

} // namespace infinilm::layers::attention::detail

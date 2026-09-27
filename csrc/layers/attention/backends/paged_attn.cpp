#include "paged_attn.hpp"

#include <stdexcept>

namespace infinilm::layers::attention::backends {

PagedAttentionImpl::PagedAttentionImpl(size_t num_heads,
                                       size_t head_size,
                                       float scale,
                                       size_t num_kv_heads,
                                       size_t layer_idx)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_(head_size) {}

infinicore::Tensor PagedAttentionImpl::forward(const AttentionLayer &,
                                               const infinicore::Tensor &,
                                               const infinicore::Tensor &,
                                               const infinicore::Tensor &,
                                               infinicore::Tensor &,
                                               const infinilm::global_state::AttentionMetadata &) const {
    throw std::runtime_error(
        "PagedAttention has been temporarily removed and may be restored in a future release.");
}

std::tuple<infinicore::Tensor, infinicore::Tensor> PagedAttentionImpl::do_kv_cache_update(const AttentionLayer &,
                                                                                          const infinicore::Tensor,
                                                                                          const infinicore::Tensor,
                                                                                          infinicore::Tensor &,
                                                                                          const infinicore::Tensor) const {
    throw std::runtime_error(
        "PagedAttention has been temporarily removed and may be restored in a future release.");
}
} // namespace infinilm::layers::attention::backends
